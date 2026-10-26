#!/usr/bin/env python3
"""Sample and report native unconditional / no-CFG LOCO external baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dss_external_evaluation/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dss_external_evaluation/xdg")

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["sample", "metrics", "plots"], required=True)
    parser.add_argument("--model", choices=["tabsyn", "tabdiff"], required=True)
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--exp-name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=["unconditional", "loco"], default="unconditional")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resample-rounds", type=int, default=1)
    parser.add_argument("--impute-condition", choices=["x_t", "x_0"], default="x_t")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--metric-sample-size", type=int, default=5000)
    parser.add_argument("--dcr-sample-size", type=int, default=1000)
    parser.add_argument("--plot-sample-size", type=int, default=0)
    parser.add_argument("--pair-metric-sample-size", type=int, default=0)
    args = parser.parse_args()
    if min(args.trials, args.batch_size, args.resample_rounds) < 1 or args.steps < 2:
        parser.error("Positive trials/batch-size/resample-rounds and steps >= 2 are required.")
    if min(args.metric_sample_size, args.plot_sample_size, args.pair_metric_sample_size) < 0 or args.dcr_sample_size < 2:
        parser.error("Sample limits must be nonnegative; DCR rows must be >= 2.")
    args.repo = (args.repo or ROOT / "external" / args.model).resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def read_schema(repo: Path, dataname: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    data = repo / "data" / dataname
    info = json.loads((data / "info.json").read_text())
    from scripts.tabsyn_impute import column_token_mapping, select_model_columns

    train = select_model_columns(pd.read_csv(data / "train.csv", low_memory=False), info)
    test = select_model_columns(pd.read_csv(data / "test.csv", low_memory=False), info)
    mapping = column_token_mapping(info)
    columns = [name for name in train if train[name].nunique(dropna=False) > 1]
    if not columns or train.empty or test.empty:
        raise ValueError("Nonempty train/test and nonconstant model columns are required.")
    schema = {
        "model_columns": columns,
        "numerical_columns": [name for name in columns if mapping[name][0] == "numerical"],
        "categorical_columns": [name for name in columns if mapping[name][0] == "categorical"],
        "constant_columns": [name for name in train if name not in columns],
        "schema_source": "trained upstream info.json; preserved column roles",
    }
    return info, train, test, schema


def seed_sampling(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unconditional_sampler(args: argparse.Namespace):
    """Load weights once; return the model's native unconditional sampler."""
    import torch

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    if args.model == "tabdiff":
        from scripts.tabdiff_impute import load_model, recover_frame, resolve_checkpoint

        checkpoint = resolve_checkpoint(args.repo, args.dataname, args.exp_name, None)
        diffusion, dataset, _, info = load_model(args.repo, args.dataname, checkpoint, device)

        def generate(rows: int) -> pd.DataFrame:
            with torch.no_grad():
                sample = diffusion.sample_all(rows, args.batch_size, keep_nan_samples=False)
            return recover_frame(sample, dataset, info)

        return generate, str(checkpoint)

    from scripts.tabsyn_impute import configure_imports, decode_latent, load_state

    configure_imports(args.repo)
    from tabsyn.diffusion_utils import sample
    from tabsyn.model import MLPDiffusion, Model
    from tabsyn.vae.model import Decoder_model, Model_VAE
    from utils_train import preprocess

    data = args.repo / "data" / args.dataname
    info = json.loads((data / "info.json").read_text())
    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        str(data), task_type=info["task_type"], inverse=True
    )
    vae_dir = args.repo / "tabsyn" / "vae" / "ckpt" / args.dataname
    train_z = torch.as_tensor(np.load(vae_dir / "train_z.npy")[:, 1:, :], dtype=torch.float32)
    token_dim = train_z.shape[2]
    flat = train_z.flatten(1)
    mean = flat.mean(dim=0).to(device)
    vae = Model_VAE(2, d_numerical, categories, token_dim, n_head=1, factor=32, bias=True).to(device)
    vae.load_state_dict(load_state(vae_dir / "model.pt", device))
    decoder = Decoder_model(2, d_numerical, categories, token_dim, 1, 32).to(device)
    decoder.load_weights(vae)
    decoder.eval()
    del vae
    checkpoint = args.repo / "tabsyn" / "ckpt" / args.dataname / "model.pt"
    diffusion = Model(denoise_fn=MLPDiffusion(flat.shape[1], 1024).to(device), hid_dim=flat.shape[1]).to(device)
    diffusion.load_state_dict(load_state(checkpoint, device))
    diffusion.eval()

    def generate(rows: int) -> pd.DataFrame:
        parts = []
        with torch.no_grad():
            for start in range(0, rows, args.batch_size):
                latent = sample(diffusion.denoise_fn_D, min(args.batch_size, rows - start),
                                flat.shape[1], num_steps=args.steps, device=device)
                parts.append(decode_latent(latent * 2 + mean, decoder, token_dim,
                                          num_inverse, cat_inverse, info, device))
        return pd.concat(parts, ignore_index=True)

    return generate, str(checkpoint)


def sample_results(args: argparse.Namespace) -> None:
    _, train, test, schema = read_schema(args.repo, args.dataname)
    columns = schema["model_columns"]
    targets = columns if args.protocol == "loco" else []
    names = [f"loco_{target}" for target in targets] or ["unconditional"]
    for name in names:
        if list((args.output_dir / name).glob("trial_*.csv")):
            raise FileExistsError(f"Saved trials already exist: {args.output_dir / name}. Use metrics/plots to reuse them.")
    truth = train if args.protocol == "unconditional" or args.split == "train" else test
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth.to_csv(args.output_dir / "evaluation_truth.csv", index=False)
    run = {
        **schema, "model": args.model, "dataname": args.dataname, "repo": str(args.repo),
        "sampling_protocol": args.protocol, "split_label": "train" if args.protocol == "unconditional" else args.split,
        "training_subset": "all", "rows": len(truth), "samples": args.trials,
        "seed": args.seed, "cfg": False, "w_num": 0.0, "w_cat": 0.0,
        "loco_columns": targets, "scenarios": names, "exp_name": args.exp_name,
        "train_reference_csv": str(args.repo / "data" / args.dataname / "train.csv"),
        "heldout_csv": str(args.repo / "data" / args.dataname / "test.csv"),
        "steps": args.steps if args.model == "tabsyn" else "checkpoint native sampler",
        "resample_rounds": args.resample_rounds, "impute_condition": args.impute_condition,
        "status": "sampling",
    }
    info_path = args.output_dir / "run_info.json"
    info_path.write_text(json.dumps(run, indent=2) + "\n")
    if args.protocol == "unconditional":
        generate, checkpoint = unconditional_sampler(args)
        run["checkpoint"] = checkpoint
        directory = args.output_dir / "unconditional"
        directory.mkdir(exist_ok=True)
        for trial in range(args.trials):
            seed_sampling(args.seed + trial)
            print(f"Native unconditional sample {trial + 1}/{args.trials}: {len(train)} rows", flush=True)
            generated = generate(len(train))
            if len(generated) != len(train) or set(train) - set(generated):
                raise ValueError("Native sample has incorrect row count or is missing model columns.")
            generated.loc[:, train.columns].to_csv(directory / f"trial_{trial:03d}.csv", index=False)
    else:
        for target in targets:
            directory = args.output_dir / f"loco_{target}"
            command = [sys.executable, "-u", str(ROOT / "scripts" / f"{args.model}_impute.py"),
                       "--dataname", args.dataname, f"--{args.model}-repo", str(args.repo),
                       "--split", args.split, "--mask-cols", target,
                       "--output-dir", str(directory), "--trials", str(args.trials),
                       "--batch-size", str(args.batch_size), "--resample-rounds", str(args.resample_rounds),
                       "--seed", str(args.seed), "--gpu", str(args.gpu)]
            if args.model == "tabdiff":
                command += ["--impute-condition", args.impute_condition]
                if args.exp_name:
                    command += ["--exp-name", args.exp_name]
            else:
                command += ["--steps", str(args.steps)]
            print(f"LOCO {args.model}/{args.split}/{target} (no CFG)", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            child_info = json.loads((directory / "run_info.json").read_text())
            run["checkpoint"] = child_info.get("checkpoint", child_info.get("diffusion_checkpoint"))
            mask = pd.DataFrame(False, index=truth.index, columns=columns)
            mask[target] = truth[target].notna()
            mask.to_csv(directory / "target_mask.csv", index=False)
            info_path.write_text(json.dumps(run, indent=2) + "\n")
    run["status"] = "complete"
    info_path.write_text(json.dumps(run, indent=2) + "\n")
    print(f"Completed sampling: {args.output_dir}", flush=True)


def heldout_dcr(train: pd.DataFrame, heldout: pd.DataFrame, generated: pd.DataFrame,
                schema: dict, rows: int, seed: int) -> float:
    from dss_imputer.generative_metrics import prepare_metric_tables
    from sdmetrics.single_table import DCROverfittingProtection

    count = min(rows, len(train), len(heldout), len(generated))
    real, synthetic, fitted, metadata = prepare_metric_tables(
        heldout.sample(count, random_state=seed).reset_index(drop=True),
        generated.sample(count, random_state=seed + 1).reset_index(drop=True),
        train.sample(count, random_state=seed + 2).reset_index(drop=True),
        numerical_columns=schema["numerical_columns"], categorical_columns=schema["categorical_columns"],
    )
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return float(DCROverfittingProtection.compute(fitted, synthetic, real, metadata,
                                                    num_rows_subsample=None, num_iterations=1))
    finally:
        np.random.set_state(state)


def metric_results(args: argparse.Namespace, run: dict) -> None:
    from dss_imputer.generative_metrics import (
        METRIC_PROTOCOL_VERSION, aggregate_metric_trials, evaluate_generative_quality,
        generative_metric_seed, prepare_metric_tables,
    )
    from scripts.eval_dss_imputer import context_integrity_records
    from scripts.recompute_dss_imputer_metrics import load_target_mask
    from sdmetrics.reports.single_table import QualityReport

    output = args.output_dir
    columns = run["model_columns"]
    truth_full = pd.read_csv(output / "evaluation_truth.csv", low_memory=False)
    truth = truth_full.loc[:, columns]
    train = pd.read_csv(run["train_reference_csv"], low_memory=False).loc[:, columns]
    heldout = pd.read_csv(run["heldout_csv"], low_memory=False).loc[:, columns]
    if run.get("status") != "complete" or len(truth) != run["rows"]:
        raise ValueError("Sampling is incomplete or truth does not match the saved row count.")
    unconditional = run["sampling_protocol"] == "unconditional"
    trial_rows, summaries, errors, integrity, shape_rows, pair_rows = [], [], [], [], [], []
    for scenario in run["scenarios"]:
        directory = output / scenario
        paths = sorted(directory.glob("trial_*.csv"))
        if len(paths) != run["samples"]:
            raise ValueError(f"Expected {run['samples']} trials in {directory}, found {len(paths)}")
        mask = None if unconditional else load_target_mask(directory, truth)
        scores_by_trial = []
        seed = generative_metric_seed(run["seed"] + 100_000, scenario)
        for trial_index, path in enumerate(paths):
            generated_full = pd.read_csv(path, low_memory=False)
            if len(generated_full) != run["rows"]:
                raise ValueError(f"Unexpected trial length: {path}")
            generated = generated_full.loc[:, columns]
            if not unconditional:
                checks = context_integrity_records(truth_full, generated_full, columns, mask, scenario, path.stem)
                integrity.extend(checks)
                if any(row["mismatches"] for row in checks):
                    raise ValueError(f"Known context changed: {path}")
                real_s, generated_s = truth, generated
            else:
                count = min(len(truth), len(generated), args.metric_sample_size or len(truth))
                real_s = truth.sample(count, random_state=seed).reset_index(drop=True)
                generated_s = generated.sample(count, random_state=(seed + 1) % (2**32)).reset_index(drop=True)
            print(f"Generative metrics: {scenario}/{path.name}", flush=True)
            scores, warnings = evaluate_generative_quality(
                real_s, generated_s, train, args.metric_sample_size, args.dcr_sample_size,
                seed, c2st_seed=generative_metric_seed(910_000, scenario),
                numerical_columns=run["numerical_columns"], categorical_columns=run["categorical_columns"],
                target_mask=mask, compute_dcr=False,
            )
            if unconditional or run["split_label"] == "test":
                try:
                    scores["dcr"] = heldout_dcr(train, heldout, generated, run, args.dcr_sample_size, seed)
                except Exception as exc:
                    warnings.append(f"dcr: {type(exc).__name__}: {exc}")
            if unconditional:
                r, g, _, metadata = prepare_metric_tables(
                    real_s, generated_s, numerical_columns=run["numerical_columns"],
                    categorical_columns=run["categorical_columns"],
                )
                report = QualityReport()
                report.real_correlation_threshold = report.real_association_threshold = 0.0
                report.generate(r, g, metadata, verbose=False)
                for label, sink in (("Column Shapes", shape_rows), ("Column Pair Trends", pair_rows)):
                    for record in report.get_details(label).to_dict("records"):
                        sink.append({"trial": trial_index, **record})
            trial_rows.append({"scenario": scenario, "trial": trial_index, **scores})
            scores_by_trial.append(scores)
            for warning in warnings:
                errors.append({"scenario": scenario, "trial": trial_index, "error": warning})
                print(f"WARNING: {warning}", flush=True)
        summaries.append({"scenario": scenario, "trials": len(paths), **aggregate_metric_trials(scores_by_trial)})
    pd.DataFrame(trial_rows).to_csv(output / "generative_metrics_by_trial.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "generative_metrics_summary.csv", index=False)
    pd.DataFrame(errors, columns=["scenario", "trial", "error"]).to_csv(output / "generative_metric_errors.csv", index=False)
    if integrity:
        pd.DataFrame(integrity).to_csv(output / "context_integrity.csv", index=False)
    if unconditional:
        pd.DataFrame(shape_rows).to_csv(output / "column_shapes_by_trial.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(output / "column_pairs_by_trial.csv", index=False)
    (output / "metric_run_info.json").write_text(json.dumps({
        "metric_protocol": METRIC_PROTOCOL_VERSION, "quality_reference": run["split_label"],
        "metric_sample_size": args.metric_sample_size, "dcr_sample_size": args.dcr_sample_size,
        "dcr_reference": "distinct prepared train/test; descriptive only for conditional samples",
        "train_loco_dcr": "omitted: real comparison would overlap fitted train",
        "unconditional_rowwise_metrics": "not applicable: independently generated rows have no paired ground truth",
        "model_schema": {name: run[name] for name in ("model_columns", "numerical_columns", "categorical_columns")},
    }, indent=2) + "\n")
    run.update({"metric_protocol": METRIC_PROTOCOL_VERSION, "generative_metrics": True,
                "metric_sample_size": args.metric_sample_size, "dcr_sample_size": args.dcr_sample_size})
    (output / "run_info.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Saved metrics: {output}", flush=True)


def plot_results(args: argparse.Namespace, run: dict) -> None:
    if run["sampling_protocol"] == "loco":
        from scripts.plot_dss_imputer_loco_quality import main as plot_loco

        plot_loco([
            "--results-dir", str(args.output_dir), "--output-dir", str(args.output_dir / "quality_plots"),
            "--prediction-source", "trial_000", "--plot-sample-size", str(args.plot_sample_size),
            "--metric-sample-size", str(args.pair_metric_sample_size),
            "--pair-plot-kind", "scatter", "--max-categories", "50",
        ])
        return
    import matplotlib.pyplot as plt
    from scripts.plot_dss_imputer_evaluation import plot_distribution

    real = pd.read_csv(args.output_dir / "evaluation_truth.csv", low_memory=False)
    generated = pd.read_csv(args.output_dir / "unconditional" / "trial_000.csv", low_memory=False)
    output = args.output_dir / "quality_plots"
    for column in run["model_columns"]:
        numerical = column in run["numerical_columns"]
        limits = None
        if numerical:
            divisor = 1e3 if column == "iops" else 1e6 if column == "lat" else 1
            values = pd.concat([real[column], generated[column]]).to_numpy(float) / divisor
            low, high = float(np.nanmin(values)), float(np.nanmax(values))
            padding = max((high - low) * .02, 1e-9)
            limits = (low - padding, high + padding)
        plot_distribution(real[column], generated[column], column, "numerical" if numerical else "categorical",
                          output / "distributions" / column, f"{run['model']} (unconditional)", "train",
                          args.plot_sample_size, 50, run["seed"], density_scale="linear",
                          display_limits=limits, histogram_columns={column}, max_categorical_values=50,
                          protocol_label="Unconditional")
    if {"iops", "lat"}.issubset(real.columns):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True, sharey=True)
        for ax, frame, label in zip(axes, (real, generated), ("Ground truth: train", f"{run['model']}: unconditional")):
            if args.plot_sample_size > 0 and len(frame) > args.plot_sample_size:
                frame = frame.sample(args.plot_sample_size, random_state=run["seed"])
            ax.scatter(frame.iops / 1e3, frame.lat / 1e6, s=.1, c="blue", marker="o",
                       alpha=.4, edgecolors="black", linewidth=.2, rasterized=True)
            ax.set(xlabel=r"IOPS ($10^3$)", ylabel=r"Latency ($\mu$s)", title=label)
            ax.grid(True, linestyle="--", alpha=.6)
        fig.tight_layout()
        fig.savefig(output / "iops_vs_lat.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    (output / "run_info.json").write_text(json.dumps({
        "prediction_source": "trial_000", "plot_sample_size": args.plot_sample_size,
        "split_label": "train", "sampling_protocol": "unconditional", "rows": run["rows"],
    }, indent=2) + "\n")
    print(f"Saved plots: {output}", flush=True)


def main() -> None:
    args = parse_args()
    if args.phase == "sample":
        sample_results(args)
    else:
        run = json.loads((args.output_dir / "run_info.json").read_text())
        if run["status"] != "complete" or run["model"] != args.model or run["dataname"] != args.dataname:
            raise ValueError("Results are incomplete or belong to a different model/dataset.")
        if args.phase == "metrics":
            metric_results(args, run)
        else:
            plot_results(args, run)


if __name__ == "__main__":
    main()
