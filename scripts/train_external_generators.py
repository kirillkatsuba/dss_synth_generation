#!/usr/bin/env python3
"""Prepare and run external diffusion tabular generators for DSS data.

Supported upstream implementations:
- TabDDPM: https://github.com/yandex-research/tab-ddpm
- TabDiff: https://github.com/minkaixu/tabdiff
- TabSyn: https://github.com/amazon-science/tabsyn
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPOS = {
    "tabddpm": {
        "url": "https://github.com/yandex-research/tab-ddpm.git",
        "path": Path("external/tab-ddpm"),
    },
    "tabdiff": {
        "url": "https://github.com/minkaixu/tabdiff.git",
        "path": Path("external/tabdiff"),
    },
    "tabsyn": {
        "url": "https://github.com/amazon-science/tabsyn.git",
        "path": Path("external/tabsyn"),
    },
}

DEFAULT_NUMERIC_TARGETS = ("iops",)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_name: str
    target: str
    columns: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    target_col_idx: int
    num_col_idx: list[int]
    cat_col_idx: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare DSS data and run TabDDPM, TabDiff, and TabSyn training/sampling."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tabddpm", "tabdiff", "tabsyn", "all"],
        default=["all"],
        help="Which external generators to run.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_NUMERIC_TARGETS),
        help=(
            "Target columns for upstream repos. They support one target per run, "
            "so the script prepares one dataset variant per target. For plain "
            "synthetic dataset generation, one target is enough because the output "
            "still contains all original columns."
        ),
    )
    parser.add_argument(
        "--real-train",
        type=Path,
        default=Path("dataset/pools/train_hdd_sequential.csv"),
        help="Real training CSV.",
    )
    parser.add_argument(
        "--real-test",
        type=Path,
        default=Path("dataset/pools/test_hdd_sequential.csv"),
        help="Real holdout CSV.",
    )
    parser.add_argument(
        "--dataset-prefix",
        default="dss_hdd_sequential",
        help="Base dataset name used inside upstream repos.",
    )
    parser.add_argument(
        "--synth-dir",
        type=Path,
        default=Path("synth_data"),
        help="Directory for final synthetic CSV copies.",
    )
    parser.add_argument(
        "--phase",
        nargs="+",
        choices=["clone", "prepare", "train", "sample", "all"],
        default=["all"],
        help="Pipeline phases to execute.",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="Python executable inside the relevant upstream environment.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for TabSyn/TabDiff.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string for TabDDPM config, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--tabddpm-steps",
        type=int,
        default=10000,
        help="TabDDPM training steps.",
    )
    parser.add_argument(
        "--tabddpm-timesteps",
        type=int,
        default=1000,
        help="TabDDPM diffusion timesteps.",
    )
    parser.add_argument(
        "--tabddpm-batch-size",
        type=int,
        default=4096,
        help="TabDDPM train batch size.",
    )
    parser.add_argument(
        "--tabdiff-exp-name",
        default="dss_tabdiff",
        help="Experiment name for TabDiff checkpoint/result folders.",
    )
    parser.add_argument(
        "--tabdiff-num-samples",
        type=int,
        default=None,
        help="Rows to sample with TabDiff. Defaults to TabDiff's real training size.",
    )
    parser.add_argument(
        "--tabsyn-steps",
        type=int,
        default=50,
        help="TabSyn sampling NFEs.",
    )
    parser.add_argument(
        "--skip-clone-if-present",
        action="store_true",
        default=True,
        help="Do not reclone repositories that already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of executing train/sample commands.",
    )
    parser.add_argument(
        "--clearml-project",
        default=None,
        help="If set, initialize a ClearML task under this project.",
    )
    parser.add_argument(
        "--clearml-task-name",
        default=None,
        help="ClearML task name. Defaults to a name derived from phase/models/targets.",
    )
    parser.add_argument(
        "--clearml-tags",
        nargs="*",
        default=[],
        help="Optional ClearML tags.",
    )
    return parser.parse_args()


def expand_models(models: list[str]) -> list[str]:
    if "all" in models:
        return ["tabddpm", "tabdiff", "tabsyn"]
    return list(dict.fromkeys(models))


def expand_phases(phases: list[str]) -> list[str]:
    if "all" in phases:
        return ["clone", "prepare", "train", "sample"]
    return list(dict.fromkeys(phases))


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    shown = " ".join(cmd)
    prefix = f"(cd {cwd} && {shown})" if cwd else shown
    print(prefix)
    if not dry_run:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)


def init_clearml(args: argparse.Namespace, models: list[str], phases: list[str]):
    if not args.clearml_project:
        return None

    try:
        from clearml import Task
    except ImportError as exc:
        raise RuntimeError(
            "ClearML logging requested, but package 'clearml' is not installed. "
            "Install it in the SLURM environment: pip install clearml"
        ) from exc

    task_name = args.clearml_task_name
    if not task_name:
        task_name = (
            f"dss_synth_{'-'.join(models)}_{'-'.join(phases)}_"
            f"{'-'.join(args.targets)}"
        )

    task = Task.init(project_name=args.clearml_project, task_name=task_name)
    if args.clearml_tags:
        task.add_tags(args.clearml_tags)
    task.connect(vars(args), name="train_external_generators_args")
    task.get_logger().report_text(
        "ClearML initialized. Console output from this wrapper and child "
        "training processes is captured in the task log."
    )
    return task


def upload_clearml_artifacts(task, paths: list[Path]) -> None:
    if task is None:
        return
    for path in paths:
        if path.exists():
            task.upload_artifact(name=str(path), artifact_object=str(path))


def ensure_repos(models: list[str], dry_run: bool, skip_if_present: bool) -> None:
    Path("external").mkdir(exist_ok=True)
    for model in models:
        repo = REPOS[model]
        repo_path = repo["path"]
        if repo_path.exists() and skip_if_present:
            print(f"{model}: using existing {repo_path}")
            continue
        run_command(["git", "clone", repo["url"], str(repo_path)], dry_run=dry_run)


def infer_spec(df: pd.DataFrame, dataset_prefix: str, target: str) -> DatasetSpec:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is absent from {list(df.columns)}")

    columns = list(df.columns)
    numeric_cols = [
        col
        for col in columns
        if col != target and pd.api.types.is_numeric_dtype(df[col])
    ]
    cat_cols = [col for col in columns if col != target and col not in numeric_cols]
    return DatasetSpec(
        dataset_name=f"{dataset_prefix}_{target}",
        target=target,
        columns=columns,
        num_cols=numeric_cols,
        cat_cols=cat_cols,
        target_col_idx=columns.index(target),
        num_col_idx=[columns.index(col) for col in numeric_cols],
        cat_col_idx=[columns.index(col) for col in cat_cols],
    )


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4))


def prepare_tabsyn_like_repo(
    repo_path: Path,
    spec: DatasetSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_val_path: bool,
) -> None:
    data_dir = repo_path / "data" / spec.dataset_name
    info_dir = repo_path / "data" / "Info"
    data_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / f"{spec.dataset_name}.csv"
    test_csv = data_dir / f"{spec.dataset_name}_test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    info = {
        "name": spec.dataset_name,
        "task_type": "regression",
        "header": "infer",
        "column_names": None,
        "num_col_idx": spec.num_col_idx,
        "cat_col_idx": spec.cat_col_idx,
        "target_col_idx": [spec.target_col_idx],
        "file_type": "csv",
        "data_path": f"data/{spec.dataset_name}/{spec.dataset_name}.csv",
        "test_path": f"data/{spec.dataset_name}/{spec.dataset_name}_test.csv",
    }
    if include_val_path:
        info["val_path"] = None

    write_json(info_dir / f"{spec.dataset_name}.json", info)


def prepare_tabddpm_repo(
    repo_path: Path,
    spec: DatasetSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Path:
    data_dir = repo_path / "data" / spec.dataset_name
    exp_dir = repo_path / "exp" / spec.dataset_name / "dss_ddpm"
    data_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    val_size = max(1, int(0.1 * len(train_df)))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]
    train_part = train_df.iloc[train_idx].reset_index(drop=True)
    val_part = train_df.iloc[val_idx].reset_index(drop=True)
    test_part = test_df.reset_index(drop=True)

    def save_split(frame: pd.DataFrame, split: str) -> None:
        if spec.num_cols:
            np.save(
                data_dir / f"X_num_{split}.npy",
                frame[spec.num_cols].to_numpy(dtype=np.float32),
                allow_pickle=True,
            )
        if spec.cat_cols:
            np.save(
                data_dir / f"X_cat_{split}.npy",
                frame[spec.cat_cols].fillna("nan").astype(str).to_numpy(),
                allow_pickle=True,
            )
        np.save(
            data_dir / f"y_{split}.npy",
            frame[spec.target].to_numpy(dtype=np.float32),
            allow_pickle=True,
        )

    save_split(train_part, "train")
    save_split(val_part, "val")
    save_split(test_part, "test")

    info = {
        "task_type": "regression",
        "name": spec.dataset_name,
        "id": f"{spec.dataset_name}--id",
        "train_size": len(train_part),
        "val_size": len(val_part),
        "test_size": len(test_part),
        "n_num_features": len(spec.num_cols),
        "n_cat_features": len(spec.cat_cols),
        "columns": spec.columns,
        "num_columns": spec.num_cols,
        "cat_columns": spec.cat_cols,
        "target": spec.target,
    }
    write_json(data_dir / "info.json", info)

    config_path = exp_dir / "config.toml"
    sample_count = len(train_df)
    config = f'''seed = 0
parent_dir = "exp/{spec.dataset_name}/dss_ddpm"
real_data_path = "data/{spec.dataset_name}/"
num_numerical_features = {len(spec.num_cols)}
model_type = "mlp"
device = "{args.device}"

[model_params]
d_in = {len(spec.num_cols) + len(spec.cat_cols) + 1}
is_y_cond = false
num_classes = 0

[model_params.rtdl_params]
d_layers = [256, 256]
dropout = 0.0

[diffusion_params]
num_timesteps = {args.tabddpm_timesteps}
gaussian_loss_type = "mse"
scheduler = "cosine"

[train.main]
steps = {args.tabddpm_steps}
lr = 0.001
weight_decay = 1e-5
batch_size = {args.tabddpm_batch_size}

[train.T]
seed = 0
normalization = "quantile"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

[sample]
num_samples = {sample_count}
batch_size = 10000
seed = 0

[eval.type]
eval_model = "simple"
eval_type = "synthetic"

[eval.T]
seed = 0
normalization = "__none__"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"
'''
    config_path.write_text(config)
    return config_path


def prepare_data(
    models: list[str],
    specs: list[DatasetSpec],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[tuple[str, str], Path | None]:
    configs: dict[tuple[str, str], Path | None] = {}
    for spec in specs:
        if "tabsyn" in models:
            prepare_tabsyn_like_repo(REPOS["tabsyn"]["path"], spec, train_df, test_df, include_val_path=False)
            configs[("tabsyn", spec.target)] = None
        if "tabdiff" in models:
            prepare_tabsyn_like_repo(REPOS["tabdiff"]["path"], spec, train_df, test_df, include_val_path=True)
            configs[("tabdiff", spec.target)] = None
        if "tabddpm" in models:
            config = prepare_tabddpm_repo(REPOS["tabddpm"]["path"], spec, train_df, test_df, args)
            configs[("tabddpm", spec.target)] = config
    return configs


def process_prepared_data(models: list[str], specs: list[DatasetSpec], args: argparse.Namespace) -> None:
    for spec in specs:
        if "tabsyn" in models:
            run_command(
                [args.python, "process_dataset.py", "--dataname", spec.dataset_name],
                cwd=REPOS["tabsyn"]["path"],
                dry_run=args.dry_run,
            )
        if "tabdiff" in models:
            run_command(
                [args.python, "process_dataset.py", "--dataname", spec.dataset_name],
                cwd=REPOS["tabdiff"]["path"],
                dry_run=args.dry_run,
            )


def train_models(
    models: list[str],
    specs: list[DatasetSpec],
    args: argparse.Namespace,
) -> None:
    for spec in specs:
        if "tabddpm" in models:
            config_path = Path("exp") / spec.dataset_name / "dss_ddpm" / "config.toml"
            run_command(
                [args.python, "scripts/pipeline.py", "--config", str(config_path), "--train"],
                cwd=REPOS["tabddpm"]["path"],
                dry_run=args.dry_run,
            )
        if "tabdiff" in models:
            run_command(
                [
                    args.python,
                    "main.py",
                    "--dataname",
                    spec.dataset_name,
                    "--mode",
                    "train",
                    "--exp_name",
                    args.tabdiff_exp_name,
                    "--gpu",
                    str(args.gpu),
                    "--no_wandb",
                ],
                cwd=REPOS["tabdiff"]["path"],
                dry_run=args.dry_run,
            )
        if "tabsyn" in models:
            run_command(
                [
                    args.python,
                    "main.py",
                    "--dataname",
                    spec.dataset_name,
                    "--method",
                    "vae",
                    "--mode",
                    "train",
                    "--gpu",
                    str(args.gpu),
                ],
                cwd=REPOS["tabsyn"]["path"],
                dry_run=args.dry_run,
            )
            run_command(
                [
                    args.python,
                    "main.py",
                    "--dataname",
                    spec.dataset_name,
                    "--method",
                    "tabsyn",
                    "--mode",
                    "train",
                    "--gpu",
                    str(args.gpu),
                ],
                cwd=REPOS["tabsyn"]["path"],
                dry_run=args.dry_run,
            )


def convert_tabddpm_sample(repo_path: Path, spec: DatasetSpec, output_path: Path) -> None:
    parent_dir = repo_path / "exp" / spec.dataset_name / "dss_ddpm"
    parts: list[pd.DataFrame] = []
    if spec.num_cols:
        x_num = np.load(parent_dir / "X_num_train.npy", allow_pickle=True)
        parts.append(pd.DataFrame(x_num, columns=spec.num_cols))
    if spec.cat_cols:
        x_cat = np.load(parent_dir / "X_cat_train.npy", allow_pickle=True)
        parts.append(pd.DataFrame(x_cat, columns=spec.cat_cols))
    y = np.load(parent_dir / "y_train.npy", allow_pickle=True)
    parts.append(pd.DataFrame({spec.target: y.reshape(-1)}))
    generated = pd.concat(parts, axis=1)
    generated = generated.reindex(columns=spec.columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(output_path, index=False)


def copy_latest_tabdiff_sample(repo_path: Path, spec: DatasetSpec, exp_name: str, output_path: Path) -> None:
    result_dir = repo_path / "tabdiff" / "result" / spec.dataset_name / exp_name
    samples = sorted(result_dir.glob("**/samples.csv"), key=lambda path: path.stat().st_mtime)
    if not samples:
        raise FileNotFoundError(f"No TabDiff samples found under {result_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(samples[-1], output_path)


def sample_models(models: list[str], specs: list[DatasetSpec], args: argparse.Namespace) -> None:
    args.synth_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if "tabddpm" in models:
            config_path = Path("exp") / spec.dataset_name / "dss_ddpm" / "config.toml"
            run_command(
                [args.python, "scripts/pipeline.py", "--config", str(config_path), "--sample"],
                cwd=REPOS["tabddpm"]["path"],
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                convert_tabddpm_sample(
                    REPOS["tabddpm"]["path"],
                    spec,
                    args.synth_dir / f"tabddpm_{spec.dataset_name}.csv",
                )

        if "tabdiff" in models:
            cmd = [
                args.python,
                "main.py",
                "--dataname",
                spec.dataset_name,
                "--mode",
                "test",
                "--exp_name",
                args.tabdiff_exp_name,
                "--gpu",
                str(args.gpu),
                "--no_wandb",
            ]
            if args.tabdiff_num_samples is not None:
                cmd += ["--num_samples_to_generate", str(args.tabdiff_num_samples)]
            run_command(cmd, cwd=REPOS["tabdiff"]["path"], dry_run=args.dry_run)
            if not args.dry_run:
                copy_latest_tabdiff_sample(
                    REPOS["tabdiff"]["path"],
                    spec,
                    args.tabdiff_exp_name,
                    args.synth_dir / f"tabdiff_{spec.dataset_name}.csv",
                )

        if "tabsyn" in models:
            output_path = args.synth_dir / f"tabsyn_{spec.dataset_name}.csv"
            run_command(
                [
                    args.python,
                    "main.py",
                    "--dataname",
                    spec.dataset_name,
                    "--method",
                    "tabsyn",
                    "--mode",
                    "sample",
                    "--gpu",
                    str(args.gpu),
                    "--steps",
                    str(args.tabsyn_steps),
                    "--save_path",
                    str(Path.cwd() / output_path),
                ],
                cwd=REPOS["tabsyn"]["path"],
                dry_run=args.dry_run,
            )


def collect_train_artifacts(models: list[str], specs: list[DatasetSpec]) -> list[Path]:
    artifacts: list[Path] = []
    for spec in specs:
        if "tabddpm" in models:
            run_dir = REPOS["tabddpm"]["path"] / "exp" / spec.dataset_name / "dss_ddpm"
            artifacts.extend([run_dir / "config.toml", run_dir / "loss.csv"])
        if "tabdiff" in models:
            run_dir = REPOS["tabdiff"]["path"] / "tabdiff" / "ckpt" / spec.dataset_name
            artifacts.extend(run_dir.glob("**/config.pkl"))
        if "tabsyn" in models:
            artifacts.append(REPOS["tabsyn"]["path"] / "data" / spec.dataset_name / "info.json")
    return artifacts


def main() -> None:
    args = parse_args()
    models = expand_models(args.models)
    phases = expand_phases(args.phase)
    task = init_clearml(args, models, phases)

    train_df = pd.read_csv(args.real_train)
    test_df = pd.read_csv(args.real_test)
    specs = [infer_spec(train_df, args.dataset_prefix, target) for target in args.targets]

    if "clone" in phases:
        ensure_repos(models, args.dry_run, args.skip_clone_if_present)

    configs: dict[tuple[str, str], Path | None] = {}
    if "prepare" in phases:
        configs = prepare_data(models, specs, train_df, test_df, args)
        process_prepared_data(models, specs, args)
        for key, path in configs.items():
            if path:
                print(f"Prepared config for {key}: {path}")
        upload_clearml_artifacts(task, [path for path in configs.values() if path])

    if "train" in phases:
        train_models(models, specs, args)
        upload_clearml_artifacts(task, collect_train_artifacts(models, specs))

    if "sample" in phases:
        sample_models(models, specs, args)
        artifacts = []
        for spec in specs:
            for model in models:
                artifacts.append(args.synth_dir / f"{model}_{spec.dataset_name}.csv")
        upload_clearml_artifacts(task, artifacts)

    if task is not None:
        print("ClearML task URL:", task.get_output_log_web_page())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
