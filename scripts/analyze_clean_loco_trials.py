#!/usr/bin/env python3
"""Recompute aligned LOCO trials, audit training, and export mean/sample-SD tables."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_loco_single_trial import (
    Source, align_sources, load_prediction, metric_record, NUMERICAL,
    generative_metric_seed, semantic_keys, numeric_column,
)
from scripts.export_loco_comparison import marginal_scores, marginal_spec
from dss_imputer.generative_metrics import prepare_metric_tables, quality_report_scores
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sdmetrics.reports.single_table import QualityReport
from threadpoolctl import threadpool_limits

DATASETS = ("hdd_sequential", "ssd_sequential", "ssd_random", "cache")
MODELS = ("hybrid", "tabdiff", "tabsyn")
FIELDS = (
    "target_overall_quality", "target_column_shapes", "target_column_pair_trends",
    "c2st", "alpha_precision", "beta_recall", "target_c2st",
    "target_alpha_precision", "target_beta_recall", "tv", "ks",
    "numeric_schema_target_overall_quality", "numeric_schema_target_column_shapes",
    "numeric_schema_target_column_pair_trends", "mse", "r2", "exact_category_accuracy",
    "outside_gt_support_fraction", "non_integer_fraction", "nonfinite_predictions",
    "below_gt_min_fraction", "above_gt_max_fraction", "generated_min", "generated_max",
    "overall_quality", "column_shapes", "column_pair_trends",
)
DISPLAY = FIELDS[:11]
LOWER = {"tv", "ks", "mse", "outside_gt_support_fraction", "non_integer_fraction",
         "nonfinite_predictions", "below_gt_min_fraction", "above_gt_max_fraction"}
LABELS = {"hybrid": "Hybrid (last)", "tabdiff": "TabDiff (native checkpoint)",
          "tabsyn": "TabSyn (native checkpoint)"}


def sources_for(downloads: Path, dataset: str, split: str) -> list[Source]:
    roots = [downloads / f"{dataset}_hybrid_mcar15_log_t200_clean_20261023_noes_s42" / f"{split}_last",
             *[downloads / "clean_20261022_r2" / model / dataset / f"loco_{split}"
               for model in MODELS[1:]]]
    sources = []
    for model, path in zip(MODELS, roots):
        info = json.loads((path / "run_info.json").read_text())
        truth = pd.read_csv(path / "evaluation_truth.csv", low_memory=False)
        if len(truth) != info["rows"]:
            raise ValueError(f"Truth row count differs from metadata: {path}")
        sources.append(Source(model, path, info, truth))
    if any(set(s.info["model_columns"]) != set(sources[0].info["model_columns"]) for s in sources):
        raise ValueError(f"Different modeled columns: {dataset}/{split}")
    return sources


def signature(files: list[Path], settings: dict) -> str:
    state = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    return hashlib.sha256(json.dumps([state, settings], sort_keys=True).encode()).hexdigest()


def numerical_schema_scores(real, generated, target, count, seed):
    """Sensitivity: numeric-valued columns evaluated as numbers, without rounding."""
    idx = (np.random.default_rng(seed).choice(len(real), count, replace=False)
           if len(real) > count else np.arange(len(real)))
    numeric = [c for c in real if numeric_column(real[c])]
    r, g, _, metadata = prepare_metric_tables(
        real.iloc[idx].reset_index(drop=True), generated.iloc[idx].reset_index(drop=True),
        numerical_columns=numeric, categorical_columns=[c for c in real if c not in numeric])
    report = QualityReport()
    report.real_correlation_threshold = 0.0
    report.real_association_threshold = 0.0
    report.generate(r, g, metadata, verbose=False)
    values, errors = quality_report_scores(report, [target], real.columns)
    return {"numeric_schema_" + k: v for k, v in values.items()
            if k in {"target_overall_quality", "target_column_shapes", "target_column_pair_trends"}}, errors


def process_group(job):
    downloads, output, dataset, split, count, seed, implementation = job
    threadpool_limits(1)
    sources = sources_for(Path(downloads), dataset, split)
    columns = sources[0].info["model_columns"]
    truth, audit = align_sources(sources, columns)
    # These exports should contain all identical rows, not a silently reduced intersection.
    if any(row["source_rows_not_compared"] for row in audit):
        raise ValueError(f"Incomplete alignment for {dataset}/{split}: {audit}")
    targets = sources[0].info["loco_columns"]
    if any(set(s.info["loco_columns"]) != set(targets) for s in sources):
        raise ValueError("LOCO scenario sets differ")
    out = Path(output) / dataset / split
    (out / "cache").mkdir(parents=True, exist_ok=True)
    records, errors, integrity, native = [], [], [], []
    files = [s.root / "evaluation_truth.csv" for s in sources]
    for source in sources:
        original = pd.read_csv(source.root / "generative_metrics_by_trial.csv")
        original["dataset"], original["split"], original["model"] = dataset, split, source.key
        native.append(original)
        for target in targets:
            available = sorted(p.name for p in (source.root / f"loco_{target}").glob("trial_*.csv"))
            if available != [f"trial_{i:03d}.csv" for i in range(5)]:
                raise ValueError(f"Expected exactly five trials: {source.root}/{target}: {available}")
            spec = marginal_spec(truth[target], target)
            for trial in range(5):
                key = dict(dataset=dataset, split=split, model=source.key, target=target, trial=trial)
                path = source.root / f"loco_{target}/trial_{trial:03d}.csv"
                cache = out / "cache" / f"{source.key}_{target}_{trial:03d}.json"
                settings = dict(implementation=implementation, count=count, seed=seed,
                                columns=columns, numeric=NUMERICAL)
                digest = signature([*files, path, source.root / "run_info.json"], settings)
                cached = json.loads(cache.read_text()) if cache.exists() else {}
                if cached.get("signature") != digest:
                    generated, checks = load_prediction(source, columns, target, trial, truth)
                    metric_seed = generative_metric_seed(seed, f"{dataset}/{split}/loco_{target}")
                    scores, issues = metric_record(truth, generated, target, count, metric_seed)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        sensitivity, sensitivity_errors = numerical_schema_scores(
                            truth, generated, target, count, metric_seed)
                    issues += ["numeric_schema/" + e for e in sensitivity_errors]
                    issues += ["numeric_schema/warning/" + str(w.message) for w in caught]
                    scores.update(sensitivity)
                    marginal = marginal_scores(truth[target], generated[target], spec)
                    scores.update(marginal)
                    if marginal.get("marginal_error"):
                        issues.append(marginal["marginal_error"])
                    if numeric_column(truth[target]):
                        actual = truth[target].to_numpy(float)
                        pred = pd.to_numeric(generated[target], errors="coerce").to_numpy(float)
                        scores.update(below_gt_min_fraction=float((pred < actual.min()).mean()),
                                      above_gt_max_fraction=float((pred > actual.max()).mean()),
                                      generated_min=float(np.nanmin(pred)), generated_max=float(np.nanmax(pred)))
                    scores["c2st_convergence_warning"] = any("ConvergenceWarning" in e for e in issues)
                    record = {**key, "rows": len(truth), "metric_rows": min(count, len(truth)),
                              "metric_seed": metric_seed, **scores}
                    cached = dict(signature=digest, record=record,
                                  errors=[{**key, "error": e} for e in issues],
                                  integrity=[{**key, **check} for check in checks])
                    temp = cache.with_suffix(".tmp")
                    temp.write_text(json.dumps(cached, indent=2, default=str))
                    temp.replace(cache)
                records.append(cached["record"])
                errors.extend(cached["errors"])
                integrity.extend(cached["integrity"])
            print(f"Completed {dataset}/{split}/{source.key}/loco_{target}", flush=True)
    pd.DataFrame(records).to_csv(out / "metrics_by_trial.csv", index=False)
    pd.concat(native, ignore_index=True).to_csv(out / "native_metrics_by_trial.csv", index=False)
    return records, errors, integrity, [{"dataset": dataset, "split": split, **r} for r in audit]


def aggregate_trials(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "split", "target", "model"]
    if frame.duplicated(keys + ["trial"]).any():
        raise ValueError("Duplicate trial records")
    rows = []
    for key, group in frame.groupby(keys, sort=False):
        for metric in FIELDS:
            values = pd.to_numeric(group.get(metric, pd.Series(np.nan, index=group.index)), errors="coerce")
            values = values[np.isfinite(values)]
            n = len(values)
            warning = bool(group.get("c2st_convergence_warning", pd.Series(False)).any())
            mean, std = (float(values.mean()) if n else np.nan), (float(values.std(ddof=1)) if n > 1 else np.nan)
            fmt = ".3g" if metric in {"mse", "generated_min", "generated_max"} else ".4f"
            formatted = f"{mean:{fmt}} +/- {std:{fmt}}" if n > 1 else "n/a"
            rows.append(dict(zip(keys, key)) | dict(metric=metric, mean=mean, std=std,
                        n=n, expected_trials=len(group), complete=n == len(group),
                        c2st_warning=warning and metric in {"c2st", "target_c2st"},
                        mean_std=formatted))
    return pd.DataFrame(rows)


def html_table(frame, fields):
    selected = frame[frame.metric.isin(fields)]
    cells = []
    for (target, model), group in selected.groupby(["target", "model"], sort=False):
        columns = [html.escape(target), html.escape(LABELS[model])]
        for field in fields:
            row = group[group.metric == field].iloc[0]
            eligible = selected[(selected.target == target) & (selected.metric == field)
                                & selected.complete & ~selected.c2st_warning]
            best = eligible["mean"].min() if field in LOWER else eligible["mean"].max()
            value = html.escape(row.mean_std.replace("+/-", "±"))
            if row.complete and not row.c2st_warning and np.isfinite(row["mean"]) and np.isclose(row["mean"], best, rtol=1e-9, atol=1e-12):
                value = f"<u>{value}</u>"
            if not row.complete or row.c2st_warning:
                value += f" * (n={row.n})"
            columns.append(value)
        cells.append("<tr>" + "".join(f"<td>{c}</td>" for c in columns) + "</tr>")
    headers = ["LOCO", "Model", *[f + (" ↓" if f in LOWER else " ↑") for f in fields]]
    return "<div class='scroll'><table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>" + "".join(cells) + "</table></div>"


def page(title, body):
    return ("<!doctype html><html lang='ru'><meta charset='utf-8'><title>" + html.escape(title)
            + "</title><style>body{font:15px system-ui;margin:30px;color:#202124}table{border-collapse:collapse}"
            "td,th{padding:8px;border:1px solid #ddd;white-space:nowrap}th{background:#eee}"
            "u{text-underline-offset:3px;font-weight:600}.scroll{overflow-x:auto}img{max-width:100%}"
            "p{max-width:1100px;line-height:1.5}</style><h1>" + html.escape(title) + "</h1>" + body + "</html>")


def training_audit(downloads: Path, output: Path):
    root = downloads / "dss_training_audit_clean_20261023_noes/models"
    rows, drift = [], []
    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    for i, dataset in enumerate(DATASETS):
        p = root / f"{dataset}_hybrid_mcar15_log_t200_clean_20261023_noes_s42"
        config = json.loads((p / "run_config.json").read_text())
        history = pd.read_csv(p / "history.csv")
        best = history.loc[history.validation_loss.idxmin()]
        last = history.iloc[-1]
        rows.append(dict(dataset=dataset, epochs=int(last.epoch), best_loss_epoch=int(best.epoch),
                         best_validation_loss=best.validation_loss,
                         **{k: last[k] for k in ["train_loss", "validation_loss", "train_numerical_loss",
                             "validation_numerical_loss", "train_categorical_loss", "validation_categorical_loss"]},
                         **{k: config.get(k) for k in ["trainable_parameters", "train_rows", "validation_rows",
                             "learning_rate", "batch_size", "diffusion_steps", "patience", "split_method"]}))
        for j, component in enumerate(["loss", "numerical_loss", "categorical_loss"]):
            ax = axes[i, j]
            ax.plot(history.epoch, history["train_" + component], label="Train", color="#157a6e")
            ax.plot(history.epoch, history["validation_" + component], label="Test (used as validation)", color="#b52b43")
            ax.axvline(best.epoch, color="#444444", ls=":", lw=.8)
            ax.set(title=f"{dataset}: {component}", xlabel="Epoch", ylabel="Loss")
            ax.grid(alpha=.2)
            if i == 0 and j == 0: ax.legend(fontsize=8)
        train = pd.read_csv(downloads / p.name / "train_last/evaluation_truth.csv")
        test = pd.read_csv(downloads / p.name / "test_last/evaluation_truth.csv")
        evaluation = json.loads((downloads / p.name / "test_last/run_info.json").read_text())
        modeled = set(evaluation["model_columns"])
        for col in train:
            a, b = semantic_keys(train[col]), semantic_keys(test[col])
            record = dict(dataset=dataset, column=col, train_unique=a.nunique(), test_unique=b.nunique(),
                          test_unseen_values=len(set(b) - set(a)), test_unseen_fraction=float((~b.isin(a)).mean()),
                          role=("excluded" if col not in modeled else
                                "numerical" if col in NUMERICAL else "categorical"))
            if numeric_column(train[col]):
                record.update(train_min=train[col].min(), train_max=train[col].max(),
                              test_min=test[col].min(), test_max=test[col].max())
            drift.append(record)
    fig.tight_layout(); fig.savefig(output / "training_losses.png", dpi=160); plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "training_audit.csv", index=False)
    pd.DataFrame(drift).to_csv(output / "train_test_support.csv", index=False)


def render(summary: pd.DataFrame, output: Path):
    body = "<p>Пять независимых сэмплов одного чекпоинта; mean ± sample std (ddof=1). Это не разброс по seed обучения.</p>"
    body += "<p>Генеративные метрики: одни и те же 5000 строк для всех моделей и trial; KS/TV и MSE/R²: все строки. "
    body += "Основная схема: iops, lat, n_jobs, iodepth числовые; остальные категориальные. "
    body += "Есть отдельная проверка чувствительности: все числовые по смыслу значения рассматриваются как числа. "
    body += "Ни округления предсказаний, ни удаления сгенерированных выбросов нет. Подчёркнуто лучшее среднее, не статистическая значимость.</p>"
    body += "<p><a href='summary_long.csv'>Средние, std, число trial</a> · <a href='metrics_by_trial.csv'>Каждый trial</a> · "
    body += "<a href='protocol.json'>Протокол</a> · <a href='training_audit.csv'>Обучение</a> · <a href='train_test_support.csv'>Поддержка train/test</a></p>"
    for dataset in DATASETS:
        body += f"<h2>{dataset}</h2>"
        for split in ("train", "test"):
            subset = summary[(summary.dataset == dataset) & (summary.split == split)]
            folder = output / dataset / split
            text = "<p><a href='../../index.html'>Все датасеты</a></p><h2>Generative + KS/TV</h2>" + html_table(subset, DISPLAY)
            text += "<h2>Чувствительность к типам колонок (все числовые значения как числа)</h2>" + html_table(subset, FIELDS[11:14])
            text += "<h2>Дополнительные точечные метрики, не критерий генеративного качества</h2>" + html_table(subset, FIELDS[14:18])
            (folder / "index.html").write_text(page(f"{dataset} / {split} / five trials", text))
            subset.to_csv(folder / "summary_long.csv", index=False)
            subset.pivot(index=["target", "model"], columns="metric", values="mean_std").to_csv(folder / "mean_std.csv")
            for target in subset.target.unique():
                dest = output / dataset / f"loco_{target}"
                dest.mkdir(exist_ok=True)
                g = subset[subset.target == target]
                (dest / f"{split}_metrics.html").write_text(page(f"{dataset} / LOCO {target} / {split}", html_table(g, DISPLAY)))
                g.to_csv(dest / f"{split}_metrics.csv", index=False)
            body += f"<p><a href='{dataset}/{split}/index.html'>{split}: все LOCO</a> · <a href='{dataset}/{split}/mean_std.csv'>mean ± std CSV</a></p>"
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        targets = summary[summary.dataset == dataset].target.drop_duplicates().tolist()
        for ax, split in zip(axes, ("train", "test")):
            sub = summary[(summary.dataset == dataset) & (summary.split == split) & (summary.metric == "target_overall_quality")]
            for j, model in enumerate(MODELS):
                v = sub[sub.model == model].set_index("target").reindex(targets)
                ax.errorbar(v["mean"], np.arange(len(targets)) + (j-1)*.19, xerr=v["std"],
                            fmt="o", ms=4, capsize=3, label=LABELS[model])
            ax.set(yticks=np.arange(len(targets)), yticklabels=targets, xlim=(-.03, 1.03),
                   xlabel="Target quality (mean +/- sample SD)", title=f"{dataset} / {split}")
            ax.grid(axis="x", alpha=.25); ax.invert_yaxis()
        axes[0].legend(fontsize=8, loc="lower left")
        fig.tight_layout(); fig.savefig(output / dataset / "target_quality.png", dpi=170); plt.close(fig)
        body += f"<img src='{dataset}/target_quality.png'>"
    body += "<h2>История обучения</h2><p>Красная линия измерена на test, который использовался для мониторинга и выбора best.</p><img src='training_losses.png'>"
    (output / "index.html").write_text(page("Clean data: Hybrid / TabDiff / TabSyn, 5 trials", body))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("/Users/kirill/Downloads"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/clean_loco_trials_comparison")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--metric-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__), ROOT / "dss_imputer/generative_metrics.py", ROOT / "dss_imputer/schema.py",
            ROOT / "scripts/compare_loco_single_trial.py", ROOT / "scripts/export_loco_comparison.py"]
    implementation = hashlib.sha256(b"".join(p.read_bytes() for p in code)).hexdigest()
    jobs = [(str(args.downloads), str(args.output_dir), dataset, split, args.metric_rows, args.seed, implementation)
            for dataset in DATASETS for split in ("train", "test")]
    records, errors, integrity, alignment = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_group, job) for job in jobs]
        for future in as_completed(futures):
            r, e, c, a = future.result()
            records.extend(r); errors.extend(e); integrity.extend(c); alignment.extend(a)
    frame = pd.DataFrame(records).sort_values(["dataset", "split", "target", "model", "trial"])
    if len(frame) != 960: raise ValueError(f"Expected 960 trials, got {len(frame)}")
    frame.to_csv(args.output_dir / "metrics_by_trial.csv", index=False)
    pd.DataFrame(errors, columns=["dataset", "split", "model", "target", "trial", "error"]).to_csv(args.output_dir / "metric_errors.csv", index=False)
    pd.DataFrame(integrity).to_csv(args.output_dir / "context_integrity.csv", index=False)
    pd.DataFrame(alignment).to_csv(args.output_dir / "alignment_audit.csv", index=False)
    summary = aggregate_trials(frame)
    summary.to_csv(args.output_dir / "summary_long.csv", index=False)
    training_audit(args.downloads, args.output_dir)
    render(summary, args.output_dir)
    protocol = dict(metric_rows=args.metric_rows, seed=args.seed, trials=5, std_ddof=1,
                    numerical_columns=NUMERICAL, implementation_sha256=implementation,
                    versions={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "sdmetrics"]},
                    fixed_subsample_across_trials=True, checkpoints="Hybrid last; native baseline checkpoint from alignment_audit.csv",
                    alpha_beta="naive minmax + onehot, not learned embedding; ties in discrete data affect interpretation",
                    c2st="SDMetrics LogisticDetection quality (higher better), not classifier accuracy; weak to nonlinear conditional mismatches",
                    dcr="Not ranked/recomputed: known context and overlapping train evaluation make privacy interpretation invalid",
                    tv="Exact PMF for categorical/integer <=128 levels; otherwise 50 GT quantile bins with overflow; no rounding",
                    limitations=["One training seed; std is sampling variability only", "Test monitored during training; not an untouched final holdout",
                                 "Joint scores diluted by copied context", "Numerical-schema sensitivity reported separately",
                                 "No controlled before/after cleaning claim: other training settings differ"])
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(f"Saved {len(frame)} trials and {len(summary)} aggregates to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
