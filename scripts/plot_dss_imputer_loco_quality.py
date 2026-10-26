#!/usr/bin/env python3
"""Compare every LOCO target with truth, conditioned on every known column."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cache_dir = Path(tempfile.gettempdir()) / "dss_imputer_loco_quality"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sdmetrics.column_pairs.statistical import CorrelationSimilarity, ContingencySimilarity
from sdmetrics.errors import ConstantInputError
from dss_imputer.generative_metrics import prepare_metric_tables

if __package__:
    from .plot_dss_imputer_evaluation import axis_label, categorical_display_values, plot_distribution
else:
    from plot_dss_imputer_evaluation import axis_label, categorical_display_values, plot_distribution


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prediction-source", choices=["trial_000", "point_estimate"], default="trial_000")
    parser.add_argument("--plot-sample-size", type=int, default=20000)
    parser.add_argument("--metric-sample-size", type=int, default=20000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--histogram-bins", type=int, default=50)
    parser.add_argument("--max-categories", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-plot-kind", choices=["auto", "scatter"], default="auto")
    parser.add_argument("--iops-limits", nargs=2, type=float)
    parser.add_argument("--lat-limits", nargs=2, type=float)
    args = parser.parse_args(argv)
    if args.bins < 2 or args.max_categories < 2 or args.histogram_bins < 2:
        parser.error("--bins, --max-categories and --histogram-bins must be >= 2")
    return args


def scale(series: pd.Series, column: str) -> pd.Series:
    divisor = 1e3 if column == "iops" else 1e6 if column == "lat" else 1.0
    return pd.to_numeric(series, errors="coerce") / divisor


def category_order(real: pd.Series, generated: pd.Series, maximum: int) -> list[str]:
    _, support = categorical_display_values(real, generated, maximum)
    return support


def grouped(series: pd.Series, order: list[str]) -> pd.Series:
    display, _ = categorical_display_values(series, series, max(series.nunique(dropna=False)+1, 2))
    values = pd.Series(display["ground_truth"], index=series.index)
    return values.where(values.isin(order), "Other") if "Other" in order else values


def shared_limits(real: pd.Series, generated: pd.Series) -> tuple[float, float]:
    values = pd.concat([real, generated]).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * 0.03, 0.01) if high > low else max(abs(low) * 0.05, 0.5)
    return low - pad, high + pad


def numeric_bins(series: pd.Series, count: int) -> tuple[pd.Series, list[str]]:
    values = series.to_numpy(dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, count + 1)))
    if len(edges) < 2:
        value = float(values[0])
        edges = np.array([value - 0.5, value + 0.5])
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    labels = [f"{i + 1}: {edges[i]:.3g}–{edges[i + 1]:.3g}" for i in range(len(edges) - 1)]
    return pd.cut(series, edges, labels=labels, include_lowest=True).astype(str), labels


def conditional_table(
    context: pd.Series,
    target: pd.Series,
    context_order: list[str],
    target_order: list[str],
) -> pd.DataFrame:
    counts = pd.crosstab(
        pd.Categorical(grouped(target, target_order), categories=target_order),
        pd.Categorical(grouped(context, context_order), categories=context_order),
        dropna=False,
    )
    return counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)


def plot_pair(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    context: str,
    target: str,
    numerical: set[str],
    output: Path,
    max_categories: int,
    bins: int,
    limits: dict[str, tuple[float, float]],
    kind: str = "auto",
) -> None:
    x_numeric, y_numeric = context in numerical, target in numerical
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=y_numeric)
    if kind == "scatter":
        values = {}
        ticks = {}
        bounds = {}
        for column in (context, target):
            if column in numerical:
                values[column] = [scale(real[column], column), scale(generated[column], column)]
                bounds[column] = limits.get(column, shared_limits(*values[column]))
                ticks[column] = None
            else:
                display, support = categorical_display_values(real[column], generated[column], max_categories)
                positions = {value: index for index, value in enumerate(support)}
                values[column] = [np.array([positions[value] for value in display[source]])
                                  for source in ("ground_truth", "model")]
                ticks[column] = support
                bounds[column] = (-0.6, len(support)-0.4)
        if ticks[target] is not None and len(ticks[target]) > 12:
            fig.set_size_inches(11, max(4.2, 1.5+0.22*len(ticks[target])))
        for index, ax in enumerate(axes):
            ax.scatter(values[context][index], values[target][index], s=0.8, alpha=0.35,
                       color="#2563eb" if index == 0 else "#dc2626", rasterized=True)
            ax.set(xlim=bounds[context], ylim=bounds[target], xlabel=axis_label(context),
                   ylabel=axis_label(target), title="Ground truth" if index == 0 else "Generated")
            if ticks[context] is not None:
                ax.set_xticks(np.arange(len(ticks[context])), ticks[context], rotation=35, ha="right")
            if ticks[target] is not None:
                ax.set_yticks(np.arange(len(ticks[target])), ticks[target])
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.tick_params(labelsize=9)
    elif x_numeric and y_numeric:
        x = scale(real[context], context)
        y_real, y_generated = scale(real[target], target), scale(generated[target], target)
        xlim = limits.get(context, shared_limits(x, x))
        ylim = limits.get(target, shared_limits(y_real, y_generated))
        for ax, values, name in zip(axes, (y_real, y_generated), ("Ground truth", "Generated")):
            ax.scatter(x, values, s=0.5, alpha=0.35, color="#2563eb", rasterized=True)
            ax.set(xlim=xlim, ylim=ylim, xlabel=axis_label(context), ylabel=axis_label(target), title=name)
            ax.grid(True, linestyle="--", alpha=0.35)
    elif not x_numeric and y_numeric:
        x_order = category_order(real[context], real[context], max_categories)
        y_real, y_generated = scale(real[target], target), scale(generated[target], target)
        ylim = limits.get(target, shared_limits(y_real, y_generated))
        x = grouped(real[context], x_order)
        for ax, values, name in zip(axes, (y_real, y_generated), ("Ground truth", "Generated")):
            frame = pd.DataFrame({"context": x, "target": values})
            sns.boxplot(data=frame, x="context", y="target", order=x_order, ax=ax,
                        color="#60a5fa", fliersize=0.7, linewidth=0.7)
            ax.set(ylim=ylim, xlabel=context, ylabel=axis_label(target), title=name)
            ax.tick_params(axis="x", labelrotation=35)
    else:
        y_order = category_order(real[target], generated[target], max_categories)
        if x_numeric:
            x, x_order = numeric_bins(scale(real[context], context), bins)
            x_label = axis_label(context)
        else:
            x_order = category_order(real[context], real[context], max_categories)
            x = grouped(real[context], x_order)
            x_label = context
        for index, (ax, frame, name) in enumerate(zip(axes, (real, generated), ("Ground truth", "Generated"))):
            table = conditional_table(x, frame[target], x_order, y_order)
            sns.heatmap(table, vmin=0, vmax=1, cmap="Blues", ax=ax, cbar=index == 1)
            ax.set(xlabel=x_label, ylabel=target, title=f"{name}: P({target} | {context})")
            ax.tick_params(axis="x", labelrotation=40)
            ax.tick_params(axis="y", labelrotation=0)
    fig.suptitle(f"LOCO {target} | context: {context}", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pair_metric(
    real: pd.DataFrame,
    generated: pd.DataFrame,
    context: str,
    target: str,
    numerical: set[str],
    bins: int,
) -> dict[str, object]:
    x_numeric, y_numeric = context in numerical, target in numerical
    try:
        numerical_names = [name for name in (context, target) if name in numerical]
        real, generated, _, _ = prepare_metric_tables(
            real[[context, target]], generated[[context, target]],
            numerical_columns=numerical_names,
            categorical_columns=[name for name in (context, target) if name not in numerical],
        )
        if x_numeric and y_numeric:
            breakdown = CorrelationSimilarity.compute_breakdown(
                real[[context, target]], generated[[context, target]], coefficient="Pearson"
            )
            if not np.isfinite(breakdown["score"]):
                raise ValueError(f"Nonfinite correlation score: {breakdown.get('error', breakdown.get('Error', ''))}")
            return {"metric": "CorrelationSimilarity", "score": breakdown["score"],
                    "real_correlation": breakdown["real"], "generated_correlation": breakdown["synthetic"], "error": ""}
        score = ContingencySimilarity.compute(
            real[[context, target]], generated[[context, target]],
            continuous_column_names=numerical_names,
            num_discrete_bins=bins,
        )
        if not np.isfinite(score):
            raise ValueError("Nonfinite contingency score")
        return {"metric": "ContingencySimilarity", "score": score,
                "real_correlation": np.nan, "generated_correlation": np.nan, "error": ""}
    except (ValueError, TypeError, KeyError, ZeroDivisionError, ConstantInputError) as exc:
        return {"metric": "CorrelationSimilarity" if x_numeric and y_numeric else "ContingencySimilarity",
                "score": np.nan, "real_correlation": np.nan, "generated_correlation": np.nan,
                "error": str(exc)}


def take_rows(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size <= 0 or len(frame) <= size:
        return frame
    indices = np.random.default_rng(seed).choice(len(frame), size=size, replace=False)
    return frame.iloc[np.sort(indices)]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or args.results_dir / "quality_plots"
    run_info = json.loads((args.results_dir / "run_info.json").read_text())
    truth_path = args.results_dir / "evaluation_truth.csv"
    if not truth_path.is_file():
        raise FileNotFoundError(f"Missing {truth_path}; rerun eval with --training-subset fit/validation")
    truth = pd.read_csv(truth_path, low_memory=False)
    columns = run_info["model_columns"]
    numerical = set(run_info["numerical_columns"])
    categorical = set(run_info["categorical_columns"])
    if set(columns) != numerical | categorical:
        raise ValueError("run_info model columns do not match numerical and categorical columns")
    integrity_path = args.results_dir / "context_integrity.csv"
    if integrity_path.is_file():
        integrity = pd.read_csv(integrity_path)
        if integrity["mismatches"].sum() != 0:
            raise ValueError(f"Known context changed; inspect {integrity_path}")
    limits = {}
    if args.iops_limits:
        limits["iops"] = tuple(args.iops_limits)
    if args.lat_limits:
        limits["lat"] = tuple(args.lat_limits)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    targets = run_info.get("loco_columns") or columns
    for target in targets:
        source_path = args.results_dir / f"loco_{target}" / f"{args.prediction_source}.csv"
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing {source_path}. Rerun eval with SAVE_FIRST_SAMPLE=1 for trial_000."
            )
        generated = pd.read_csv(source_path, low_memory=False)
        if len(truth) != len(generated):
            raise ValueError(f"Row count differs for {target}: {len(truth)} vs {len(generated)}")
        if not set(columns).issubset(truth.columns) or not set(columns).issubset(generated.columns):
            raise ValueError(f"Missing modeled columns in truth or {source_path}")
        real_checked, generated_checked, _, _ = prepare_metric_tables(
            truth[columns], generated[columns],
            numerical_columns=numerical, categorical_columns=categorical,
        )
        for context in columns:
            if context == target:
                continue
            actual, predicted = real_checked[context], generated_checked[context]
            if not (actual.eq(predicted) | (actual.isna() & predicted.isna())).all():
                raise ValueError(f"Known context {context} changed in {source_path}")
        valid_target = truth[target].notna() & generated[target].notna()
        target_indices = np.flatnonzero(valid_target.to_numpy())
        sampled_target = take_rows(pd.DataFrame({"idx": target_indices}), args.plot_sample_size, args.seed)["idx"]
        sample_truth = truth.iloc[sampled_target]
        sample_generated = generated.iloc[sampled_target]
        target_dir = output_dir / f"loco_{target}"
        print(f"Plotting {target}: {len(target_indices)} valid rows", flush=True)
        plot_distribution(
            sample_truth[target], sample_generated[target], target,
            "numerical" if target in numerical else "categorical",
            target_dir / "distribution", "Generated", run_info.get("split_label", "evaluation"),
            args.plot_sample_size, 50, args.seed, density_scale="linear",
            display_limits=limits.get(target), histogram_columns={target},
            histogram_bins=args.histogram_bins, max_categorical_values=args.max_categories,
        )
        for context in columns:
            if context == target:
                continue
            valid = valid_target & truth[context].notna()
            valid_indices = np.flatnonzero(valid.to_numpy())
            if not len(valid_indices):
                continue
            paired = pd.DataFrame({"idx": valid_indices})
            metric_indices = take_rows(paired, args.metric_sample_size, args.seed)["idx"].to_numpy()
            plot_indices = take_rows(paired, args.plot_sample_size, args.seed)["idx"].to_numpy()
            real_metric = truth.iloc[metric_indices][[context, target]].reset_index(drop=True).copy()
            generated_metric = generated.iloc[metric_indices][[context, target]].reset_index(drop=True).copy()
            for name in (context, target):
                if name in numerical:
                    real_metric[name] = pd.to_numeric(real_metric[name], errors="coerce")
                    generated_metric[name] = pd.to_numeric(generated_metric[name], errors="coerce")
            finite = real_metric.notna().all(axis=1) & generated_metric.notna().all(axis=1)
            real_metric, generated_metric = real_metric.loc[finite], generated_metric.loc[finite]
            result = pair_metric(real_metric, generated_metric, context, target, numerical, args.bins) if len(real_metric) else {
                "metric": "none", "score": np.nan, "real_correlation": np.nan,
                "generated_correlation": np.nan, "error": "No valid paired rows"}
            records.append({"target": target, "context": context,
                            "pair_type": ("num" if context in numerical else "cat") + "-" + ("num" if target in numerical else "cat"),
                            "n_rows": len(real_metric), "prediction_source": args.prediction_source, **result})
            plot_pair(
                truth.iloc[plot_indices][[context, target]].reset_index(drop=True),
                generated.iloc[plot_indices][[context, target]].reset_index(drop=True),
                context, target, numerical, target_dir / "pairs" / f"{context}.png",
                args.max_categories, args.bins, limits,
                args.pair_plot_kind,
            )
        pd.DataFrame(records).to_csv(output_dir / "pairwise_metrics.csv", index=False)
    (output_dir / "run_info.json").write_text(json.dumps({
        "source_results_dir": str(args.results_dir), "prediction_source": args.prediction_source,
        "checkpoint": run_info.get("checkpoint"), "checkpoint_epoch": run_info.get("checkpoint_epoch"),
        "split_label": run_info.get("split_label"), "training_subset": run_info.get("training_subset"),
        "plot_sample_size": args.plot_sample_size, "metric_sample_size": args.metric_sample_size,
        "bins": args.bins, "max_categories": args.max_categories, "targets": targets,
        "pair_plot_kind": args.pair_plot_kind,
    }, indent=2))
    print(f"Saved {len(records)} pairwise scores to {output_dir / 'pairwise_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
