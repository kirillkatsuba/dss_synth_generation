#!/usr/bin/env python3
"""Aligned single-trial LOCO metrics and three-model Matplotlib panels."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dss_single_trial/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dss_single_trial/xdg")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from dss_imputer.generative_metrics import (
    METRIC_PROTOCOL_VERSION, evaluate_generative_quality, generative_metric_seed,
)
from dss_imputer.schema import _value_key
from scripts.eval_dss_imputer import context_integrity_records

MODELS = ("hybrid", "tabsyn", "tabdiff")
LABELS = {"hybrid": "Hybrid Imputer (last)", "tabsyn": "TabSyn", "tabdiff": "TabDiff"}
COLORS = {"truth": "#2563eb", "hybrid": "#d97706", "tabsyn": "#059669", "tabdiff": "#dc2626"}
NUMERICAL = ("iops", "lat", "n_jobs", "iodepth")
HYBRID = {
    "hdd_sequential": ("hdd_hybrid_mcar15_log_s42", "loco_all_train_last", "loco_all_test_last"),
    "ssd_sequential": ("ssd_sequential_hybrid_mcar15_log_t200_full_s42", "train_last", "test_last"),
    "ssd_random": ("ssd_random_hybrid_mcar15_log_t200_full_s42", "train_last", "test_last"),
    "cache": ("cache_hybrid_mcar15_log_t200_full_s42", "train_last", "test_last"),
}
SCORES = ["target_overall_quality", "target_column_shapes", "target_column_pair_trends",
          "c2st", "alpha_precision", "beta_recall", "target_c2st", "mse", "r2"]
HEADERS = ["Target overall", "Target shape", "Target pairs", "C2ST joint", "Alpha joint",
           "Beta joint", "C2ST target", "MSE (raw)", "R2"]


@dataclass
class Source:
    key: str
    root: Path
    info: dict
    truth: pd.DataFrame
    indexer: np.ndarray | None = None


def numeric_column(values: pd.Series) -> bool:
    return bool(pd.to_numeric(values, errors="coerce").notna().eq(values.notna()).all())


def semantic_keys(values: pd.Series) -> pd.Series:
    # Report columns have homogeneous types; canonicalize each distinct level once.
    codes, unique = pd.factorize(values, sort=False, use_na_sentinel=False)
    keys = np.asarray([_value_key(value) for value in unique], dtype=object)
    return pd.Series(keys[codes], index=values.index)


def canonical_rows(frame: pd.DataFrame, columns: list[str], numeric: set[str]) -> pd.DataFrame:
    result = pd.DataFrame(index=range(len(frame)))
    for column in columns:
        values = frame[column].reset_index(drop=True)
        result[column] = (pd.to_numeric(values, errors="raise").astype(np.float32)
                          if column in numeric else semantic_keys(values))
    return result


def row_index(frame: pd.DataFrame) -> pd.MultiIndex:
    hashes = pd.util.hash_pandas_object(frame, index=False)
    occurrence = hashes.groupby(hashes, sort=False).cumcount()
    return pd.MultiIndex.from_arrays([hashes.to_numpy(), occurrence.to_numpy()])


def align_sources(sources: list[Source], columns: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    reference = sources[0].truth
    numeric = {column for column in columns if numeric_column(reference[column])}
    canonical = [canonical_rows(source.truth, columns, numeric) for source in sources]
    reference_index = row_index(canonical[0])
    mappings = [row_index(frame).get_indexer(reference_index) for frame in canonical]
    keep = np.logical_and.reduce([indices >= 0 for indices in mappings])
    if not keep.any():
        raise ValueError("No common GT rows after float32-compatible alignment.")
    selected = np.flatnonzero(keep)
    rows = []
    for source, frame, indices in zip(sources, canonical, mappings):
        source.indexer = indices[keep]
        pd.testing.assert_frame_equal(canonical[0].iloc[selected].reset_index(drop=True),
                                      frame.iloc[source.indexer].reset_index(drop=True))
        precision = {}
        for column in numeric:
            a = reference.iloc[selected][column].to_numpy(float)
            b = source.truth.iloc[source.indexer][column].to_numpy(float)
            precision[column] = float(np.nanmax(np.abs(a - b)))
        rows.append({"model": source.key, "source_rows": len(source.truth),
                     "common_rows": len(selected), "source_rows_not_compared": len(source.truth) - len(selected),
                     "max_gt_precision_difference": json.dumps(precision),
                     "result_root": str(source.root), "checkpoint": source.info.get("checkpoint"),
                     "checkpoint_epoch": source.info.get("checkpoint_epoch"),
                     "training_subset": source.info.get("training_subset"),
                     "stored_protocol": source.info.get("metric_protocol"),
                     "alignment": "modeled GT columns rounded to float32 for identity only; duplicate occurrence keys"})
    return reference.iloc[selected][columns].reset_index(drop=True), rows


def group_sources(root: Path, tag: str, dataset: str, split: str, variant: str) -> list[Source]:
    if variant == "cache_t400":
        hybrid_path = root / "dss_imputer/cache_hybrid_mcar15_log_t400_e1000_full_s42/test_last"
    else:
        run, train, test = HYBRID[dataset]
        hybrid_path = root / "dss_imputer" / run / (train if split == "train" else test)
    directories = [hybrid_path, *(root / "external_generators" / tag / model / dataset / f"loco_{split}"
                                 for model in MODELS[1:])]
    sources = []
    for key, directory in zip(MODELS, directories):
        info = json.loads((directory / "run_info.json").read_text())
        truth = pd.read_csv(directory / "evaluation_truth.csv", low_memory=False)
        if len(truth) != info["rows"]:
            raise ValueError(f"GT row count differs from metadata: {directory}")
        sources.append(Source(key, directory, info, truth))
    columns = sources[0].info["model_columns"]
    for source in sources:
        if set(source.info["model_columns"]) != set(columns):
            raise ValueError(f"Different model columns: {source.root}")
    return sources


def load_prediction(source: Source, columns: list[str], target: str, trial: int,
                    shared_truth: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    path = source.root / f"loco_{target}" / f"trial_{trial:03d}.csv"
    generated = pd.read_csv(path, low_memory=False)
    if len(generated) != len(source.truth):
        raise ValueError(f"Trial and own GT differ in length: {path}")
    mask = np.zeros((len(source.truth), len(columns)), dtype=bool)
    mask[:, columns.index(target)] = True
    checks = context_integrity_records(source.truth, generated, columns, mask, f"loco_{target}", path.stem)
    if any(row["mismatches"] for row in checks):
        raise ValueError(f"Known context changed relative to own GT: {path}")
    result = generated.iloc[source.indexer][columns].reset_index(drop=True)
    # Known context is identical semantically; use the common GT precision for joint metrics.
    for column in columns:
        if column != target:
            result[column] = shared_truth[column].to_numpy()
    return result, [{"model": source.key, **row} for row in checks]


def metric_cache_signature(source: Source, sources: list[Source], target: str,
                           aligned_rows: int, metric_rows: int, implementation: str) -> str:
    trial_file = source.root / f"loco_{target}/trial_000.csv"
    fingerprint = {"source": str(trial_file), "size": trial_file.stat().st_size,
                   "mtime": trial_file.stat().st_mtime_ns, "metric_rows": metric_rows,
                   "aligned_rows": aligned_rows, "implementation": implementation,
                   "truth": [(str(item.root), (item.root / "evaluation_truth.csv").stat().st_mtime_ns)
                             for item in sources]}
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()


def metric_record(real: pd.DataFrame, generated: pd.DataFrame, target: str, count: int,
                  seed: int) -> tuple[dict, list[str]]:
    numeric = [column for column in real if column in NUMERICAL]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores, errors = evaluate_generative_quality(
            real, generated, real, count, 1000, seed, c2st_seed=seed,
            numerical_columns=numeric, categorical_columns=[column for column in real if column not in numeric],
            target_mask=np.column_stack([np.full(len(real), column == target) for column in real]),
            compute_dcr=False,
        )
    errors.extend(sorted({f"warning/{item.category.__name__}: {str(item.message).splitlines()[0]}" for item in caught}))
    # Target-specific internal details are redundant in the comparison table.
    scores = {name: value for name, value in scores.items() if not name.startswith(f"target_{target}_")}
    scores["numeric_values"] = numeric_column(real[target])
    scores["mse"], scores["r2"] = np.nan, np.nan
    if scores["numeric_values"]:
        expected = pd.to_numeric(real[target], errors="raise").to_numpy(float)
        predicted = pd.to_numeric(generated[target], errors="coerce").to_numpy(float)
        valid = np.isfinite(expected)
        scores["deterministic_rows"] = int(valid.sum())
        scores["nonfinite_predictions"] = int((valid & ~np.isfinite(predicted)).sum())
        if scores["nonfinite_predictions"]:
            errors.append("MSE/R2 unavailable: nonfinite prediction on an observed target.")
        else:
            scores["mse"] = float(mean_squared_error(expected[valid], predicted[valid]))
            scores["r2"] = float(r2_score(expected[valid], predicted[valid]))
        if np.allclose(expected[valid], np.rint(expected[valid]), atol=0, rtol=0):
            scores["non_integer_fraction"] = float(np.mean(~np.isclose(predicted[valid], np.rint(predicted[valid]), atol=1e-6, rtol=0)))
    else:
        scores["deterministic_rows"] = 0
    if target not in numeric:
        actual = semantic_keys(real[target])
        predicted = semantic_keys(generated[target])
        scores["exact_category_accuracy"] = float(actual.eq(predicted).mean())
        scores["outside_gt_support_fraction"] = float((~predicted.isin(set(actual))).mean())
    return scores, errors


def scaled(values: pd.Series, column: str) -> np.ndarray:
    divisor = 1e3 if column == "iops" else 1e6 if column == "lat" else 1
    return pd.to_numeric(values, errors="raise").to_numpy(float) / divisor


def label(column: str) -> str:
    return {"iops": r"IOPS ($10^3$)", "lat": r"Latency / $10^6$"}.get(column, column)


def limits(values: list[np.ndarray], dataset: str, column: str, window: bool) -> tuple[float, float]:
    if window and dataset == "hdd_sequential" and column in {"iops", "lat"}:
        return (0, 14) if column == "iops" else (0, 100)
    joined = np.concatenate(values)
    joined = joined[np.isfinite(joined)]
    low, high = float(joined.min()), float(joined.max())
    pad = .035 * (high - low) if high > low else max(abs(low) * .05, .5)
    return low - pad, high + pad


def category_values(real: pd.Series, predictions: list[pd.Series],
                    max_categories: int | None = None) -> tuple[list[np.ndarray], list[str]]:
    all_keys = [semantic_keys(values) for values in [real, *predictions]]
    support = list(dict.fromkeys(all_keys[0]))
    display = dict(zip(all_keys[0], real.astype(str)))
    observed = set(support)
    grouped = max_categories is not None and len(support) > max_categories
    if grouped:
        support = all_keys[0].value_counts().index[:max_categories].tolist()
    support.sort(key=(lambda key: float(display[key])) if numeric_column(real) else lambda key: display[key])
    novel = any(not set(keys).issubset(observed) for keys in all_keys[1:])
    mapping = {key: index for index, key in enumerate(support)}
    if grouped:
        mapping.update({key: len(support) for key in observed - set(support)})
    unknown = len(support) + int(grouped)
    arrays = [keys.map(mapping).fillna(unknown).to_numpy(float) for keys in all_keys]
    labels = [display[key] for key in support] + (["Other GT"] if grouped else []) + (["Outside GT"] if novel else [])
    return arrays, labels


def save_distribution(real: pd.DataFrame, predictions: dict[str, pd.DataFrame], target: str,
                      output: Path, title: str, logarithmic: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.6), sharex=True, sharey=True)
    numeric = numeric_column(real[target])
    if numeric:
        values = [scaled(frame[target], target) for frame in [real, *predictions.values()]]
        excluded = [int((~np.isfinite(v) | (v <= 0 if logarithmic else False)).sum()) for v in values]
        clean = [v[np.isfinite(v) & ((v > 0) if logarithmic else True)] for v in values]
        joined = np.concatenate(clean)
        if logarithmic:
            bins = np.linspace(np.log10(joined.min()) - .015, np.log10(joined.max()) + .015, 51)
            plot_edges = 10 ** bins
            density = [np.histogram(np.log10(v), bins=bins, density=True)[0] for v in clean]
        else:
            low, high = float(joined.min()), float(joined.max())
            padding = max((high - low) * .015, 1e-9)
            bins = plot_edges = np.linspace(low - padding, high + padding, 51)
            density = [np.histogram(v, bins=bins, density=True)[0] for v in clean]
        for index, (model, ax) in enumerate(zip(MODELS, axes)):
            for data_index, color, name in ((0, COLORS["truth"], "Ground truth"),
                                             (index + 1, COLORS[model], LABELS[model])):
                ax.stairs(density[data_index], plot_edges, fill=True, alpha=.25, color=color, label=name, linewidth=1)
            if logarithmic:
                ax.set_xscale("log")
            ax.set_xlabel(label(target) + (" (log scale)" if logarithmic else ""))
            if excluded[0] or excluded[index + 1]:
                ax.text(.02, .95, f"Not displayable: GT {excluded[0]}, model {excluded[index + 1]}",
                        transform=ax.transAxes, va="top", fontsize=7)
        axes[0].set_ylabel("Density per log10 unit" if logarithmic else "Density")
    else:
        arrays, categories = category_values(real[target], [frame[target] for frame in predictions.values()])
        positions = np.arange(len(categories))
        for index, (model, ax) in enumerate(zip(MODELS, axes)):
            for data_index, shift, color, name in ((0, -.2, COLORS["truth"], "Ground truth"),
                                                   (index + 1, .2, COLORS[model], LABELS[model])):
                probabilities = np.bincount(arrays[data_index].astype(int), minlength=len(categories)) / len(real)
                ax.bar(positions + shift, probabilities, width=.4, alpha=.65, color=color, label=name)
            ax.set_xticks(positions, categories, rotation=35, ha="right")
            ax.set_xlabel(target)
        axes[0].set_ylabel("Probability")
    for model, ax in zip(MODELS, axes):
        ax.set_title(LABELS[model], fontsize=11)
        ax.grid(True, linestyle="--", alpha=.3)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title + f" | n={len(real):,}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def save_pair(real: pd.DataFrame, predictions: dict[str, pd.DataFrame], target: str,
              context: str, dataset: str, output: Path, title: str, window: bool = True) -> None:
    values, categories = {}, {}
    for column in (context, target):
        if numeric_column(real[column]):
            values[column] = [scaled(frame[column], column) for frame in [real, *predictions.values()]]
            categories[column] = None
        else:
            values[column], categories[column] = category_values(real[column], [frame[column] for frame in predictions.values()])
    bounds = {column: ((-.6, len(categories[column]) - .4) if categories[column] is not None
                       else limits(values[column], dataset, column, window)) for column in (context, target)}
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 6.6), sharex=True, sharey=True)
    for index, model in enumerate(MODELS):
        for row, source, color in ((0, 0, COLORS["truth"]), (1, index + 1, COLORS[model])):
            ax = axes[row, index]
            x = values[context][source]
            y = values[target][source]
            ax.scatter(x, y, s=.5, alpha=.35, color=color, linewidths=0, rasterized=True)
            ax.set_xlim(*bounds[context]); ax.set_ylim(*bounds[target])
            ax.set_title(f"{LABELS[model]} / {'Ground truth' if row == 0 else 'Prediction'}", fontsize=10)
            if row == 1:
                ax.set_xlabel(label(context))
            if index == 0:
                ax.set_ylabel(label(target))
            for axis, column in (("x", context), ("y", target)):
                if categories[column] is not None:
                    positions = np.arange(len(categories[column]))
                    if axis == "x":
                        ax.set_xticks(positions, categories[column], rotation=25, ha="right")
                    else:
                        ax.set_yticks(positions, categories[column])
            clipped = np.isfinite(x) & np.isfinite(y) & ((x < bounds[context][0]) | (x > bounds[context][1]) |
                                                        (y < bounds[target][0]) | (y > bounds[target][1]))
            if clipped.any():
                ax.text(.02, .97, f"Outside window: {clipped.mean():.1%}", transform=ax.transAxes,
                        va="top", fontsize=7, bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
            ax.grid(True, linestyle="--", alpha=.25)
    fig.suptitle(title + f" | context: {context} | n={len(real):,}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def save_conditional_pair(real: pd.DataFrame, predictions: dict[str, pd.DataFrame], target: str,
                          context: str, dataset: str, output: Path, title: str) -> None:
    """Categorical panels show conditional frequencies, not arbitrary label correlations."""
    x_numeric, y_numeric = context in NUMERICAL, target in NUMERICAL
    if x_numeric and y_numeric:
        save_pair(real, predictions, target, context, dataset, output, title)
        return
    frames = [real, *predictions.values()]
    values, supports = {}, {}
    for column in (context, target):
        if column in NUMERICAL:
            values[column] = [scaled(frame[column], column) for frame in frames]
        else:
            values[column], supports[column] = category_values(real[column], [frame[column] for frame in predictions.values()], 12)
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 7), sharex=True, sharey=True)
    if y_numeric:
        bounds = limits(values[target], dataset, target, True)
        for index, model in enumerate(MODELS):
            for row, source in ((0, 0), (1, index + 1)):
                ax = axes[row, index]
                groups = [values[target][source][values[context][source] == code] for code in range(len(supports[context]))]
                present = [code for code, group in enumerate(groups) if len(group)]
                ax.boxplot([groups[code] for code in present], positions=present, widths=.6,
                           patch_artist=True, showfliers=False,
                           boxprops={"facecolor": COLORS["truth"] if row == 0 else COLORS[model], "alpha": .55},
                           medianprops={"color": "black"})
                ax.set_xticks(range(len(supports[context])), supports[context], rotation=35, ha="right")
                ax.set_xlim(-.6, len(supports[context]) - .4); ax.set_ylim(*bounds)
                if row == 1: ax.set_xlabel(context)
                if index == 0: ax.set_ylabel(label(target))
        description = "Conditional boxplots; whiskers 1.5 IQR; outlier points hidden (all rows used)"
    else:
        if x_numeric:
            x = values[context][0]
            edges = np.unique(np.quantile(x[np.isfinite(x)], np.linspace(0, 1, 13)))
            codes = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
            x_labels = [f"{a:.3g}..{b:.3g}" for a, b in zip(edges[:-1], edges[1:])]
        else:
            codes, x_labels = values[context][0].astype(int), supports[context]
        for index, model in enumerate(MODELS):
            for row, source in ((0, 0), (1, index + 1)):
                ax = axes[row, index]
                table = np.zeros((len(supports[target]), len(x_labels)))
                np.add.at(table, (values[target][source].astype(int), codes), 1)
                totals = table.sum(axis=0, keepdims=True)
                table = np.divide(table, totals, out=np.zeros_like(table), where=totals > 0)
                ax.imshow(table, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="Blues")
                ax.set_xticks(range(len(x_labels)), x_labels, rotation=55, ha="right", fontsize=7)
                ax.set_yticks(range(len(supports[target])), supports[target])
                if row == 1: ax.set_xlabel(label(context))
                if index == 0: ax.set_ylabel(target)
        description = f"P({target} | {context}); color scale 0 to 1; numerical context in GT quantile bins"
    for index, model in enumerate(MODELS):
        for row in (0, 1):
            axes[row, index].set_title(f"{LABELS[model]} / {'Ground truth' if row == 0 else 'Prediction'}", fontsize=10)
    fig.suptitle(title + f" | context: {context} | n={len(real):,}\n" + description, fontsize=11)
    fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)


def formatted_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=["target", "model", *SCORES]).copy()
    result["model"] = result.model.map(LABELS)
    for column in SCORES:
        result[column] = result[column].map(lambda value: "n/a" if pd.isna(value) else
                                            (f"{value:.3g}" if column == "mse" else f"{value:.4f}"))
    flagged = frame.get("c2st_convergence_warning", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    for column in ("c2st", "target_c2st"):
        result.loc[flagged, column] += "*"
    return result.rename(columns=dict(zip(SCORES, HEADERS)))


def save_metric_figures(frame: pd.DataFrame, output: Path, title: str) -> None:
    display = formatted_table(frame)
    fig, ax = plt.subplots(figsize=(17, max(3, .31 * (len(display) + 2))))
    ax.axis("off")
    ax.set_title(title + " | trial_000 (no averaging)", fontsize=13, pad=12)
    if frame.get("c2st_convergence_warning", pd.Series(dtype=bool)).any():
        fig.text(.02, .015, "* C2ST estimator reached its iteration limit; see metric_errors.csv. Interpret with caution.", fontsize=8)
    table = ax.table(cellText=display.to_numpy(), colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.4)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#e5e7eb"); cell.set_text_props(weight="bold")
        elif row % 6 in (1, 2, 3):
            cell.set_facecolor("#f4f7fa")
    fig.tight_layout(); fig.savefig(output / "metrics_table.png", dpi=170); plt.close(fig)
    fields = ["target_column_shapes", "target_column_pair_trends", "c2st",
              "alpha_precision", "beta_recall", "r2"]
    titles = ["Column shape", "Target-context trends", "Joint C2ST quality", "Joint alpha precision", "Joint beta recall", "R2 (raw units)"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    targets = list(dict.fromkeys(frame.target))
    x = np.arange(len(targets))
    for ax, field, heading in zip(axes.flat, fields, titles):
        for index, model in enumerate(MODELS):
            scores = frame[frame.model == model].set_index("target")[field].reindex(targets)
            ax.bar(x + (index - 1) * .25, scores, width=.25, color=COLORS[model], label=LABELS[model])
        ax.set_xticks(x, targets, rotation=40, ha="right"); ax.set_title(heading, fontsize=11)
        if field != "r2": ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title + " | one stochastic trial", fontsize=13)
    fig.tight_layout(); fig.savefig(output / "metrics_comparison.png", dpi=170); plt.close(fig)


def html_page(title: str, body: str) -> str:
    return "<!doctype html><html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>" + \
        f"<title>{html.escape(title)}</title><style>body{{font:15px Arial,sans-serif;margin:24px auto;max-width:1500px;padding:0 18px;color:#222}}" + \
        "img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:12px}td,th{padding:6px 10px;border:1px solid #ddd}" + \
        "th{background:#eee}a{color:#1759a0}details{margin:15px 0}section{border-top:1px solid #ccc;margin-top:25px}nav{line-height:2}</style>" + \
        f"<h1>{html.escape(title)}</h1>{body}</html>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("/Users/kirill/edu/reports"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/loco_single_trial_comparison")
    parser.add_argument("--tag", default="uncond_all_20261015")
    parser.add_argument("--datasets", nargs="+", default=list(HYBRID), choices=list(HYBRID))
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--metric-sample-size", type=int, default=2000)
    parser.add_argument("--phase", choices=["all", "metrics", "plots"], default="all")
    parser.add_argument("--skip-existing-plots", action="store_true", help="Resume plotting unchanged inputs; existing PNGs are not validated.")
    parser.add_argument("--include-cache400", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.trial != 0 or args.metric_sample_size < 2:
        parser.error("This report deliberately uses trial_000 and at least two metric rows.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    def needs_plot(path: Path) -> bool:
        return not (args.skip_existing_plots and path.is_file())
    groups = [(dataset, split, "main") for dataset in args.datasets for split in args.splits]
    if args.include_cache400 and "cache" in args.datasets and "test" in args.splits:
        groups.append(("cache", "test", "cache_t400"))
    metric_rows, audits, integrity, error_rows, pages = [], [], [], [], []
    metric_source_hash = hashlib.sha256((ROOT / "dss_imputer/generative_metrics.py").read_bytes()).hexdigest()
    for dataset, split, variant in groups:
        sources = group_sources(args.reports_root, args.tag, dataset, split, variant)
        columns = sources[0].info["model_columns"]
        real, alignment = align_sources(sources, columns)
        audits.extend({"dataset": dataset, "split": split, "variant": variant, **record} for record in alignment)
        group = args.output_dir / dataset / split / variant
        group.mkdir(parents=True, exist_ok=True)
        title = f"{dataset} / {split}" + (" / Hybrid t400, epoch 1000" if variant == "cache_t400" else "")
        body = f"<p><a href='../../../index.html'>All datasets</a></p><p>Trial 000 only. Matched rows: {len(real):,}. " + \
               f"Generative metric rows: {min(len(real), args.metric_sample_size):,}. MSE/R2 and plots: all matched rows.</p>"
        body += "<p>Columns: Hybrid last, TabSyn, TabDiff. Pair panels: GT on top, predictions below. " + \
                "HDD IOPS/latency window: 0–14 and 0–100; full-range versions also provided for their pair.</p>"
        body += "<p>Common metric schema: numerical iops, lat, n_jobs, iodepth; remaining columns categorical. " + \
                "No rounding of predictions. MSE/R2 also shown for numeric-valued categorical targets; not applicable to text categories.</p>"
        body += "<p>Generative scores: higher is better. C2ST is SDMetrics LogisticDetection quality, not classifier accuracy. " + \
                "Joint scores include copied context; target scores isolate the recovered column. " + \
                "Alpha/beta are naive embedding-based estimates and require caution for discrete marginals. " + \
                "There is no confidence interval from one trial. MSE: lower is better; R2: higher is better. " + \
                "An asterisk marks C2ST estimates associated with an estimator convergence warning.</p>"
        body += "<p><a href='metrics.csv'>Metrics CSV</a> | <a href='../../../metric_errors.csv'>Metric warnings/errors</a></p><img loading='lazy' src='metrics_table.png'><img loading='lazy' src='metrics_comparison.png'>"
        body += "<nav>LOCO: " + " | ".join(f"<a href='#{column}'>{column}</a>" for column in columns) + "</nav>"
        group_rows = []
        for target in columns:
            started = time.monotonic()
            directory = group / f"loco_{target}"
            directory.mkdir(exist_ok=True)
            predictions = {}
            for source in sources:
                prediction, checks = load_prediction(source, columns, target, args.trial, real)
                predictions[source.key] = prediction
                integrity.extend({"dataset": dataset, "split": split, "variant": variant, **record} for record in checks)
                cache = directory / f"metrics_{source.key}.json"
                signature = metric_cache_signature(source, sources, target, len(real),
                                                   args.metric_sample_size, metric_source_hash)
                saved = json.loads(cache.read_text()) if cache.is_file() else None
                if saved and saved["signature"] == signature:
                    scores, errors = saved["scores"], saved["errors"]
                elif args.phase == "plots":
                    raise FileNotFoundError(f"Matching cached metrics required for plots-only: {cache}")
                else:
                    print(f"Metrics {title} / {target} / {source.key}", flush=True)
                    scores, errors = metric_record(real, prediction, target, args.metric_sample_size,
                                                  generative_metric_seed(42, f"{dataset}/{split}/{target}"))
                    cache.write_text(json.dumps({"signature": signature, "scores": scores, "errors": errors}, indent=2) + "\n")
                record = {"dataset": dataset, "split": split, "variant": variant, "target": target,
                          "model": source.key, "trial": 0, "rows": len(real),
                          "metric_rows": min(len(real), args.metric_sample_size),
                          "c2st_convergence_warning": any("ConvergenceWarning" in error for error in errors), **scores}
                metric_rows.append(record); group_rows.append(record)
                error_rows.extend({"dataset": dataset, "split": split, "variant": variant,
                                   "target": target, "model": source.key, "error": error} for error in errors)
            if args.phase != "metrics":
                if needs_plot(directory / "distribution.png"):
                    save_distribution(real, predictions, target, directory / "distribution.png", title + f" / LOCO {target}")
                if target in {"iops", "lat"} and needs_plot(directory / "distribution_log.png"):
                    save_distribution(real, predictions, target, directory / "distribution_log.png", title + f" / LOCO {target}", True)
                for context in columns:
                    if context != target:
                        if needs_plot(directory / f"pair_{context}.png"):
                            save_pair(real, predictions, target, context, dataset, directory / f"pair_{context}.png", title + f" / LOCO {target}")
                        if (target not in NUMERICAL or context not in NUMERICAL) and needs_plot(directory / f"conditional_{context}.png"):
                            save_conditional_pair(real, predictions, target, context, dataset,
                                                  directory / f"conditional_{context}.png", title + f" / LOCO {target}")
                if dataset == "hdd_sequential" and target in {"iops", "lat"} and needs_plot(directory / "pair_iops_lat_full.png"):
                    context = "lat" if target == "iops" else "iops"
                    save_pair(real, predictions, target, context, dataset, directory / "pair_iops_lat_full.png", title + f" / LOCO {target} (full range)", False)
            body += f"<section id='{target}'><h2>LOCO {target}</h2><img loading='lazy' src='loco_{target}/distribution.png'>"
            if target in {"iops", "lat"}:
                body += f"<img loading='lazy' src='loco_{target}/distribution_log.png'>"
            for context in columns:
                if context != target:
                    body += f"<details><summary>Context: {context}</summary>"
                    if target not in NUMERICAL or context not in NUMERICAL:
                        body += f"<img loading='lazy' src='loco_{target}/conditional_{context}.png'>"
                    body += f"<img loading='lazy' src='loco_{target}/pair_{context}.png'></details>"
            if dataset == "hdd_sequential" and target in {"iops", "lat"}:
                body += f"<details><summary>IOPS / latency: full range</summary><img loading='lazy' src='loco_{target}/pair_iops_lat_full.png'></details>"
            body += "</section>"
            print(f"Completed {title} / {target}: {time.monotonic() - started:.1f}s", flush=True)
            pd.DataFrame(metric_rows).to_csv(args.output_dir / "metrics_all.csv", index=False)
        frame = pd.DataFrame(group_rows)
        frame.to_csv(group / "metrics.csv", index=False)
        formatted_table(frame).to_csv(group / "metrics_formatted.csv", index=False)
        if args.phase != "metrics":
            save_metric_figures(frame, group, title)
            (group / "index.html").write_text(html_page(title, body))
        pages.append((title, group.relative_to(args.output_dir).as_posix() + "/index.html"))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "metrics_all.csv", index=False)
    pd.concat([metrics[["dataset", "split", "variant"]], formatted_table(metrics)], axis=1).to_csv(
        args.output_dir / "metrics_formatted_all.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output_dir / "alignment_audit.csv", index=False)
    pd.DataFrame(integrity).to_csv(args.output_dir / "context_integrity.csv", index=False)
    pd.DataFrame(error_rows, columns=["dataset", "split", "variant", "target", "model", "error"]).to_csv(
        args.output_dir / "metric_errors.csv", index=False)
    main_only = metrics[metrics.variant == "main"]
    train, test = main_only[main_only.split == "train"], main_only[main_only.split == "test"]
    gap = train.merge(test, on=["dataset", "target", "model"], suffixes=("_train", "_test"))
    for metric in SCORES:
        gap[f"{metric}_test_minus_train"] = gap[f"{metric}_test"] - gap[f"{metric}_train"]
    gap.to_csv(args.output_dir / "train_test_gap.csv", index=False)
    notes = ["Saved trial_000 only; no sample averaging, no sampling, no training.",
             "Raw source reports are read-only; all outputs are written to this report directory.",
             "MSE/R2 use all aligned rows in original units, not log space or scaled plot units.",
             "Text categories have no meaningful MSE/R2 and are reported as n/a.",
             "Generative metrics use the same fixed subset for all models; no std can be estimated from one trial.",
             "C2ST is LogisticDetection quality (higher is better), a linear discriminator, not proof of conditional correctness. Joint alpha/beta include identical context; target alpha/beta on discrete support may be degenerate.",
             "Quality and alpha/beta/C2ST use the explicit common Hybrid metric schema for every model.",
             "Context checked against own saved GT before float32-compatible row alignment. Only known context precision is harmonized; predictions are not rounded.",
             "Row identity uses all modeled GT values at float32 precision plus duplicate occurrence. Metadata/IDs unavailable in external GT are not invented.",
             "HDD Hybrid was trained on an internal fit subset, while external models use full train. Train comparison uses common rows, not equal training sets.",
             "External TabSyn/TabDiff checkpoints are those recorded in source metadata, not necessarily last; Hybrid is always last.",
             "DCR is not used: conditional copied context confounds privacy interpretation, and train references overlap.",
             "Some Hybrid best checkpoints were selected on test historically; this report does not use best.",
             "IOPS plot units are /1e3, latency plot units /1e6. No assumption of physical latency units beyond stored scaling."]
    (args.output_dir / "protocol.json").write_text(json.dumps({
        "metric_protocol": METRIC_PROTOCOL_VERSION, "trial": 0, "metric_sample_size": args.metric_sample_size,
        "plots_rows": "all matched", "notes": notes, "source_reports": str(args.reports_root),
        "groups": [{"dataset": d, "split": s, "variant": v} for d, s, v in groups],
    }, indent=2) + "\n")
    links = "".join(f"<li><a href='{path}'>{html.escape(title)}</a></li>" for title, path in pages)
    body = "<p>Hybrid last, TabSyn, TabDiff. Same saved trial_000; train and test are separate.</p><ul>" + links + "</ul>"
    body += "<p><a href='metrics_all.csv'>All detailed metrics CSV</a> | <a href='metrics_formatted_all.csv'>Readable summary CSV</a> | <a href='train_test_gap.csv'>Train/test gaps</a> | " + \
            "<a href='alignment_audit.csv'>Row alignment</a> | <a href='metric_errors.csv'>Metric warnings/errors</a> | <a href='protocol.json'>Protocol</a></p>"
    body += "<h2>Method and limitations</h2><ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in notes) + "</ul>"
    (args.output_dir / "index.html").write_text(html_page("Single-trial LOCO comparison", body))
    print(f"Saved {len(metrics)} model-target rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
