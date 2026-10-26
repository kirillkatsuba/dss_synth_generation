#!/usr/bin/env python3
"""Four-panel LOCO galleries with a shared GT, marginal TV/KS and ranked tables."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_loco_single_trial import (
    COLORS, HEADERS as BASE_HEADERS, HYBRID, NUMERICAL,
    SCORES as BASE_SCORES, align_sources, category_values, generative_metric_seed,
    group_sources, html_page, label, limits, load_prediction, metric_cache_signature,
    metric_record, numeric_column, scaled, semantic_keys,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ORDER = ("truth", "hybrid", "tabdiff", "tabsyn")
MODEL_ORDER = ORDER[1:]
NAMES = {"truth": "Ground truth", "hybrid": "Ours (Hybrid, last)",
         "tabdiff": "TabDiff", "tabsyn": "TabSyn"}
SCORES = [*BASE_SCORES[:-2], "tv", "ks", *BASE_SCORES[-2:]]
HEADERS = {**dict(zip(BASE_SCORES, BASE_HEADERS)), "tv": "TV", "ks": "KS"}
LOWER_BETTER = {"tv", "ks", "mse"}
MARGINAL_PROTOCOL = "full_rows_pmf_or_gt_quantile50_with_overflow_ks_numeric_v1"


def marginal_spec(real: pd.Series, target: str) -> dict:
    if not numeric_column(real):
        return {"tv_method": "empirical_pmf", "ks_applicable": False}
    values = pd.to_numeric(real, errors="raise").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Marginal reference must have finite, observed values.")
    levels = len(np.unique(values))
    discrete = target not in NUMERICAL or levels == 1 or (
        levels <= 128 and np.equal(values, np.rint(values)).all())
    if discrete:
        return {"tv_method": "empirical_pmf", "ks_applicable": True}
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 51)))
    quantiles[-1] = np.nextafter(quantiles[-1], np.inf)
    return {"tv_method": "gt_quantile_histogram", "ks_applicable": True,
            "edges": np.r_[-np.inf, quantiles, np.inf]}


def marginal_scores(real: pd.Series, generated: pd.Series, spec: dict) -> dict:
    result = {"tv": np.nan, "ks": np.nan, "tv_method": spec["tv_method"],
              "marginal_rows": len(real), "tv_bin_count": len(spec.get("edges", [])) - 1}
    if len(real) != len(generated) or not len(real):
        raise ValueError("Marginal metrics require equally sized, nonempty samples.")
    if real.isna().any() or generated.isna().any():
        result["marginal_error"] = "Missing values; marginal metrics not computed."
        return result
    if spec["ks_applicable"]:
        expected = pd.to_numeric(real, errors="raise").to_numpy(float)
        predicted = pd.to_numeric(generated, errors="coerce").to_numpy(float)
        if not np.isfinite(predicted).all():
            result["marginal_error"] = "Nonfinite numerical predictions; marginal metrics not computed."
            return result
        result["ks"] = float(ks_2samp(expected, predicted, method="asymp").statistic)
    if spec["tv_method"] == "empirical_pmf":
        a = semantic_keys(real).value_counts(normalize=True)
        b = semantic_keys(generated).value_counts(normalize=True)
        support = a.index.union(b.index)
        result["tv"] = float(.5 * np.abs(a.reindex(support, fill_value=0) - b.reindex(support, fill_value=0)).sum())
        result["tv_bin_count"] = len(support)
    else:
        a = np.histogram(expected, bins=spec["edges"])[0] / len(expected)
        b = np.histogram(predicted, bins=spec["edges"])[0] / len(predicted)
        result["tv"] = float(.5 * np.abs(a - b).sum())
    return result


def best_scores(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(False, index=frame.index, columns=SCORES)
    keys = [key for key in ("dataset", "split", "variant", "target") if key in frame]
    groups = frame.groupby(keys, sort=False, dropna=False).groups.values() if keys else [frame.index]
    for indices in groups:
        for field in SCORES:
            if field not in frame:
                continue
            values = pd.to_numeric(frame.loc[indices, field], errors="coerce")
            valid = np.isfinite(values)
            if field in {"c2st", "target_c2st"} and "c2st_convergence_warning" in frame:
                valid &= ~frame.loc[indices, "c2st_convergence_warning"].fillna(False).astype(bool)
            if not valid.any():
                continue
            best = values[valid].min() if field in LOWER_BETTER else values[valid].max()
            result.loc[indices, field] = valid & np.isclose(values, best, rtol=1e-9, atol=1e-12)
    return result


def pretty_value(value: float, field: str, warning: bool = False) -> str:
    if pd.isna(value) or not np.isfinite(float(value)):
        return "n/a"
    text = f"{value:.3g}" if field == "mse" else f"{value:.4f}"
    return text + ("*" if warning and field in {"c2st", "target_c2st"} else "")


def display_table(frame: pd.DataFrame, fields: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = SCORES if fields is None else fields
    flags = best_scores(frame)
    result = pd.DataFrame(index=frame.index)
    result["LOCO"] = frame.target
    result["Model"] = frame.model.map(NAMES)
    for field in fields:
        heading = HEADERS[field] + (" (lower)" if field in LOWER_BETTER else " (higher)")
        result[heading] = [pretty_value(row.get(field, np.nan), field, row.get("c2st_convergence_warning", False))
                           for row in frame.to_dict("records")]
    return result, flags


def table_html(frame: pd.DataFrame, fields: list[str] | None = None, emphasis: str = "bold") -> str:
    fields = SCORES if fields is None else fields
    display, flags = display_table(frame, fields)
    rows = ["<table><thead><tr>" + "".join(f"<th>{html.escape(name)}</th>" for name in display) + "</tr></thead><tbody>"]
    for index, row in display.iterrows():
        cells = []
        for column, value in enumerate(row):
            text = html.escape(str(value))
            if column >= 2 and flags.loc[index, fields[column - 2]]:
                text = f"<u>{text}</u>" if emphasis == "underline" else f"<strong style='color:#116134'>{text}</strong>"
            cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(rows) + "</tbody></table>"


def save_table(frame: pd.DataFrame, output: Path, title: str,
               fields: list[str] | None = None, emphasis: str = "bold") -> None:
    fields = SCORES if fields is None else fields
    display, flags = display_table(frame, fields)
    fig, ax = plt.subplots(figsize=(max(14, 1.6 * len(fields) + 3.4), max(2.5, .32 * (len(frame) + 3))))
    ax.axis("off")
    ax.set_title(title + (" | best per LOCO underlined" if emphasis == "underline" else " | best per LOCO in bold"), fontsize=13, pad=14)
    headings = [name.replace(" (lower)", "\n(lower)").replace(" (higher)", "\n(higher)") for name in display]
    widths = [.085, .12] + [.795 / len(fields)] * len(fields)
    table = ax.table(cellText=display.to_numpy(), colLabels=headings, colWidths=widths, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(7.7); table.scale(1, 1.65)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#e5e7eb"); cell.set_text_props(weight="bold"); cell.set_height(cell.get_height() * 1.55)
        elif row % 6 in (1, 2, 3):
            cell.set_facecolor("#f5f7fa")
        if emphasis == "bold" and row > 0 and column >= 2 and flags.iloc[row - 1][fields[column - 2]]:
            cell.set_text_props(weight="bold", color="#116134")
    lower = "TV, KS, MSE" if "mse" in fields else "TV, KS"
    style = "underlined" if emphasis == "underline" else "bold"
    fig.text(.02, .02, f"{lower}: lower is better. Other scores: higher is better. Ties are {style}. * C2ST convergence warning; not ranked.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    if emphasis == "underline":
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Draw actual underlines beneath the winning text, without requiring TeX.
        for row in range(len(frame)):
            for column, field in enumerate(fields, start=2):
                if flags.iloc[row][field]:
                    bounds = table[row + 1, column].get_text().get_window_extent(renderer)
                    x0, y = fig.transFigure.inverted().transform((bounds.x0, bounds.y0 - 1))
                    x1, _ = fig.transFigure.inverted().transform((bounds.x1, bounds.y0 - 1))
                    fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color="black", linewidth=.8))
    fig.savefig(output, dpi=170); plt.close(fig)


def metric_matrices(frame: pd.DataFrame, output: Path, title: str) -> str:
    output.mkdir(exist_ok=True)
    flags = best_scores(frame)
    body = "<p><a href='../index.html'>Dataset report</a></p><p>Best per LOCO in bold; ties included. Warnings (*) are not ranked.</p>"
    for field in SCORES:
        matrix = frame.pivot(index="target", columns="model", values=field).reindex(
            index=frame.target.drop_duplicates(), columns=MODEL_ORDER)
        matrix.to_csv(output / f"{field}.csv")
        matrix_flags = frame.assign(best=flags[field]).pivot(index="target", columns="model", values="best").reindex_like(matrix)
        texts = [[pretty_value(matrix.loc[target, model], field,
                               bool(frame[(frame.target == target) & (frame.model == model)].iloc[0].get("c2st_convergence_warning", False)))
                  for model in MODEL_ORDER] for target in matrix.index]
        heading = HEADERS[field] + (" (lower is better)" if field in LOWER_BETTER else " (higher is better)")
        fig, ax = plt.subplots(figsize=(9, max(2.5, .42 * (len(matrix) + 3))))
        ax.axis("off"); ax.set_title(title + " / " + heading, fontsize=11, pad=15)
        table = ax.table(cellText=[[target, *row] for target, row in zip(matrix.index, texts)],
                         colLabels=["LOCO", *[NAMES[model] for model in MODEL_ORDER]],
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#dddddd")
            if row == 0:
                cell.set_facecolor("#e5e7eb"); cell.set_text_props(weight="bold")
            elif column and matrix_flags.iloc[row - 1, column - 1]:
                cell.set_text_props(weight="bold", color="#116134")
        fig.tight_layout(); fig.savefig(output / f"{field}.png", dpi=170); plt.close(fig)
        body += f"<h2>{heading}</h2><p><a href='{field}.csv'>CSV</a></p><img loading='lazy' src='{field}.png'>"
    (output / "index.html").write_text(html_page(title + " / metrics", body))
    return "<p><a href='tables/index.html'>Separate tables for every metric</a></p>"


def panel_axes(title: str, pair: bool = False) -> tuple:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True) if pair else \
        plt.subplots(1, 4, figsize=(16, 3.7), sharex=True, sharey=True)
    for key, ax in zip(ORDER, axes.flat):
        ax.set_title(NAMES[key], fontsize=11)
    fig.suptitle(title, fontsize=12)
    return fig, dict(zip(ORDER, axes.flat))


def distribution(real: pd.DataFrame, predictions: dict, target: str, output: Path,
                 title: str, logarithmic: bool = False) -> None:
    frames = {"truth": real, **predictions}
    fig, axes = panel_axes(title + f" | n={len(real):,}")
    if numeric_column(real[target]):
        values = {key: scaled(frames[key][target], target) for key in ORDER}
        clean = {key: x[np.isfinite(x) & ((x > 0) if logarithmic else True)] for key, x in values.items()}
        pooled = np.concatenate(list(clean.values()))
        if logarithmic:
            lo, hi = np.log10(pooled.min()), np.log10(pooled.max())
            edges = np.linspace(lo - .015, hi + .015, 51)
            display_edges = 10 ** edges
        else:
            lo, hi = float(pooled.min()), float(pooled.max())
            pad = max((hi - lo) * .015, 1e-9)
            edges = display_edges = np.linspace(lo - pad, hi + pad, 51)
        for key, ax in axes.items():
            x = np.log10(clean[key]) if logarithmic else clean[key]
            counts = np.histogram(x, bins=edges)[0]
            density = counts / max(1, len(x)) / np.diff(edges)
            ax.stairs(density, display_edges, fill=True, alpha=.55, color=COLORS[key], linewidth=.9)
            if logarithmic: ax.set_xscale("log")
            excluded = len(real) - len(x)
            if excluded:
                ax.text(.02, .96, f"Not displayed: {excluded}", transform=ax.transAxes, va="top", fontsize=8)
            ax.set_xlabel(label(target) + (" (log scale)" if logarithmic else ""))
        axes["truth"].set_ylabel("Density per log10 unit" if logarithmic else "Density")
    else:
        arrays, categories = category_values(real[target], [predictions[key][target] for key in MODEL_ORDER])
        for (key, ax), values in zip(axes.items(), arrays):
            probabilities = np.bincount(values.astype(int), minlength=len(categories)) / len(real)
            ax.bar(np.arange(len(categories)), probabilities, color=COLORS[key], alpha=.7)
            ax.set_xticks(range(len(categories)), categories, rotation=35, ha="right"); ax.set_xlabel(target)
        axes["truth"].set_ylabel("Probability")
    for ax in axes.values(): ax.grid(True, linestyle="--", alpha=.25)
    fig.tight_layout(); fig.savefig(output, dpi=170); plt.close(fig)


def pair_values(real: pd.DataFrame, predictions: dict, target: str, context: str, categorical: bool = False) -> tuple:
    values, supports = {}, {}
    frames = {"truth": real, **predictions}
    for column in (context, target):
        is_numeric = column in NUMERICAL if categorical else numeric_column(real[column])
        if is_numeric:
            values[column] = {key: scaled(frames[key][column], column) for key in ORDER}
        else:
            arrays, supports[column] = category_values(real[column], [predictions[key][column] for key in MODEL_ORDER], 12 if categorical else None)
            values[column] = dict(zip(ORDER, arrays))
    return values, supports


def scatter_pair(real: pd.DataFrame, predictions: dict, target: str, context: str,
                 dataset: str, output: Path, title: str, window: bool = True) -> None:
    values, supports = pair_values(real, predictions, target, context)
    bounds = {column: ((-.6, len(supports[column]) - .4) if column in supports else
                       limits(list(values[column].values()), dataset, column, window)) for column in (context, target)}
    fig, axes = panel_axes(title + f" | context: {context} | n={len(real):,}", True)
    for key, ax in axes.items():
        x, y = values[context][key], values[target][key]
        ax.scatter(x, y, s=.5, alpha=.35, color=COLORS[key], linewidths=0, rasterized=True)
        ax.set(xlim=bounds[context], ylim=bounds[target], xlabel=label(context), ylabel=label(target))
        for column, axis in ((context, "x"), (target, "y")):
            if column in supports:
                ticks = range(len(supports[column]))
                if axis == "x": ax.set_xticks(ticks, supports[column], rotation=30, ha="right")
                else: ax.set_yticks(ticks, supports[column])
        outside = (x < bounds[context][0]) | (x > bounds[context][1]) | (y < bounds[target][0]) | (y > bounds[target][1])
        if outside.any():
            ax.text(.02, .97, f"Outside window: {outside.mean():.1%}", transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        ax.grid(True, linestyle="--", alpha=.25)
    fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)


def conditional_pair(real: pd.DataFrame, predictions: dict, target: str, context: str,
                     dataset: str, output: Path, title: str) -> None:
    values, supports = pair_values(real, predictions, target, context, True)
    fig, axes = panel_axes(title + f" | context: {context} | n={len(real):,}", True)
    if target in NUMERICAL:
        bounds = limits(list(values[target].values()), dataset, target, True)
        for key, ax in axes.items():
            groups = [values[target][key][values[context][key] == code] for code in range(len(supports[context]))]
            present = [code for code, group in enumerate(groups) if len(group)]
            ax.boxplot([groups[code] for code in present], positions=present, widths=.6,
                       patch_artist=True, showfliers=False, medianprops={"color": "black"},
                       boxprops={"facecolor": COLORS[key], "alpha": .55})
            ax.set_xticks(range(len(supports[context])), supports[context], rotation=35, ha="right")
            ax.set(xlim=(-.6, len(supports[context]) - .4), ylim=bounds, xlabel=context, ylabel=label(target))
        caption = "Conditional boxplots; whiskers 1.5 IQR; outlier points hidden, all rows used"
    else:
        if context in NUMERICAL:
            x = values[context]["truth"]
            edges = np.unique(np.quantile(x, np.linspace(0, 1, 13)))
            if len(edges) == 1:
                codes, labels = np.zeros(len(x), dtype=int), [f"{edges[0]:.3g}"]
            else:
                codes = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
                labels = [f"{a:.3g}..{b:.3g}" for a, b in zip(edges[:-1], edges[1:])]
        else:
            codes, labels = values[context]["truth"].astype(int), supports[context]
        for key, ax in axes.items():
            table = np.zeros((len(supports[target]), len(labels)))
            np.add.at(table, (values[target][key].astype(int), codes), 1)
            totals = table.sum(axis=0, keepdims=True)
            table = np.divide(table, totals, out=np.zeros_like(table), where=totals > 0)
            ax.imshow(table, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(supports[target])), supports[target])
            ax.set(xlabel=label(context), ylabel=target)
        caption = f"P({target} | {context}); color scale 0 to 1; rare GT levels grouped only for display"
    fig.text(.5, .012, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(output, dpi=140); plt.close(fig)


def export_minimal_report(source_root: Path, output_root: Path,
                          datasets: list[str], splits: list[str]) -> None:
    """Package existing scatter/distribution figures, with no inference or new metrics."""
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("The minimal report must be separate from the source report.")
    metrics = pd.read_csv(source_root / "metrics_all.csv")
    metrics = metrics[(metrics.variant == "main") & metrics.dataset.isin(datasets) & metrics.split.isin(splits)]
    fields = [field for field in SCORES if field not in {"mse", "r2"}]
    if metrics.empty:
        raise ValueError("No matching main-run metrics in the source report.")
    output_root.mkdir(parents=True, exist_ok=True)
    navigation, scenarios, copied = [], 0, 0
    for dataset in datasets:
        dataset_metrics = metrics[metrics.dataset == dataset]
        if dataset_metrics.empty:
            raise ValueError(f"Missing dataset: {dataset}")
        targets = list(dataset_metrics.target.drop_duplicates())
        navigation.append(f"<h2>{html.escape(dataset)}</h2><ul>")
        for target in targets:
            directory = output_root / dataset / f"loco_{target}"
            directory.mkdir(parents=True, exist_ok=True)
            body = "<p><a href='../../index.html'>All datasets</a></p>"
            body += "<p>GT / Ours / TabDiff / TabSyn. Saved trial 000, no averaging. Best metric values underlined within each split. "
            body += "TV/KS: lower is better; quality scores: higher is better. Text-category KS is n/a. * C2ST warning: not ranked.</p>"
            body += "<p>TV/KS and plots use all matched rows; other generative metrics use the same fixed subset. "
            body += "Continuous TV uses shared GT quantile bins including tails; categorical TV uses exact frequencies. "
            body += "Joint scores include copied context; marginal scores alone do not establish conditional correctness.</p>"
            for split in splits:
                frame = dataset_metrics[(dataset_metrics.target == target) & (dataset_metrics.split == split)]
                if set(frame.model) != set(MODEL_ORDER) or len(frame) != len(MODEL_ORDER):
                    raise ValueError(f"Expected exactly three models for {dataset}/{target}/{split}")
                frame = frame.set_index("model").loc[list(MODEL_ORDER)].reset_index()
                source = source_root / dataset / split / "main" / f"loco_{target}"
                shutil.copy2(source / "distribution.png", directory / f"{split}_distribution.png")
                copied += 1
                for context in targets:
                    if context != target:
                        shutil.copy2(source / f"pair_{context}.png", directory / f"{split}_scatter_vs_{context}.png")
                        copied += 1
                frame[["model", *fields]].to_csv(directory / f"{split}_metrics.csv", index=False)
                save_table(frame, directory / f"{split}_metrics.png", f"{dataset} / LOCO {target} / {split}",
                           fields=fields, emphasis="underline")
                body += f"<h2>{split}</h2><p>Rows: {int(frame.rows.iloc[0]):,}; generative metric rows: {int(frame.metric_rows.iloc[0]):,}.</p>"
                body += table_html(frame, fields=fields, emphasis="underline")
                body += f"<p><a href='{split}_metrics.png'>Table PNG</a> | <a href='{split}_metrics.csv'>Metrics CSV</a></p>"
                body += f"<h3>1D: {target}</h3><img loading='lazy' src='{split}_distribution.png'>"
                for context in targets:
                    if context != target:
                        body += f"<h3>Scatter: {target} vs {context}</h3><img loading='lazy' src='{split}_scatter_vs_{context}.png'>"
                scenarios += 1
            (directory / "index.html").write_text(html_page(f"{dataset} / LOCO {target}", body))
            navigation.append(f"<li><a href='{dataset}/loco_{target}/index.html'>LOCO {target}</a></li>")
        navigation.append("</ul>")
    introduction = "<p>Only 1D distributions, pairwise scatter plots, generative metrics and KS/TV. Train/test files are inside each LOCO folder.</p>"
    introduction += "<p>One common GT per four-panel figure: GT, Ours (Hybrid last), TabDiff, TabSyn. Saved trial 000. "
    introduction += "External models retain their recorded sampling checkpoints. HDD train uses the shared 66,218-row subset.</p>"
    (output_root / "index.html").write_text(html_page("LOCO: distributions, scatter and generative metrics", introduction + "".join(navigation)))
    print(f"Saved {scenarios} train/test LOCO scenarios, {copied} plot files and {scenarios} underlined tables to {output_root}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("/Users/kirill/edu/reports"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--minimal", action="store_true", help="Only distributions/scatter and underlined generative tables, organized dataset/LOCO.")
    parser.add_argument("--source-report", type=Path, default=ROOT / "reports/loco_single_trial_four_panel")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "reports/loco_single_trial_comparison")
    parser.add_argument("--tag", default="uncond_all_20261015")
    parser.add_argument("--datasets", nargs="+", choices=list(HYBRID), default=list(HYBRID))
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--metric-sample-size", type=int, default=2000)
    parser.add_argument("--include-cache400", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing-plots", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir or ROOT / ("reports/loco_scatter_report" if args.minimal else "reports/loco_single_trial_four_panel")
    if args.minimal:
        export_minimal_report(args.source_report, args.output_dir, args.datasets, args.splits)
        return
    if args.metric_sample_size < 2: parser.error("Metric sample size must be at least two.")
    if args.output_dir.resolve() in {args.cache_root.resolve(), args.reports_root.resolve()}:
        parser.error("Use a separate output directory to preserve source reports.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    groups = [(dataset, split, "main") for dataset in args.datasets for split in args.splits]
    if args.include_cache400 and "cache" in args.datasets and "test" in args.splits:
        groups.append(("cache", "test", "cache_t400"))
    implementation = hashlib.sha256((ROOT / "dss_imputer/generative_metrics.py").read_bytes()).hexdigest()
    all_rows, audits, checks_all, errors_all, links = [], [], [], [], []
    notes = [
        "Trial 000 only, no averaging. Panel order: GT, Ours (Hybrid last), TabDiff, TabSyn. GT shown once.",
        "All matched rows for plots, marginal TV/KS and MSE/R2; other generative metrics use the same fixed subset of up to 2000 rows (or the configured cap).",
        "TV = 0.5 sum |p-q|. Exact empirical PMFs for categorical columns and integer numerical columns with <=128 GT levels; no prediction rounding.",
        "For continuous values TV is a binned approximation, not an exact continuous TV: shared GT 50-quantile edges plus underflow/overflow bins. Bin definitions are saved per LOCO.",
        "KS = sup |F_real-F_generated| on raw numeric values, including numeric-valued categories. KS is n/a for text categories, never based on arbitrary label codes.",
        "Best values are bold within each dataset/split/variant/LOCO: TV, KS, MSE minimized, other displayed scores maximized. Ties use unrounded values (rtol=1e-9, atol=1e-12).",
        "C2ST estimates with a convergence warning (*) are retained but excluded from winner highlighting. Near-one linear C2ST is not proof of correct conditional sampling.",
        "CSV cannot preserve bold text: bold winners appear in HTML/PNG, and metrics_best_flags.csv records machine-readable winner flags.",
        "Joint alpha/beta include copied context; target-context scores and pair plots remain important. Discrete target-only alpha/beta can be degenerate.",
        "HDD train comparison is the common 66218-row internal fit subset, not identical historical training sets. Other datasets use their full saved train/test rows.",
        "Hybrid uses last. External samples use their recorded source checkpoints (including TabDiff best EMA), not an invented last checkpoint.",
        "Known context is verified against each source GT before float32-compatible row matching; predictions are not modified. Inputs and the previous report are not overwritten.",
        "HDD pair windows retain IOPS/1000 in [0,14], latency/1000000 in [0,100], with clipped fractions shown; full-range IOPS/latency pairs are also supplied. Metrics are never clipped.",
        "Other GT and Outside GT are display-only category groups. Outside GT means absent from this reference split, not necessarily physically invalid or unseen in training.",
    ]
    for dataset, split, variant in groups:
        sources = group_sources(args.reports_root, args.tag, dataset, split, variant)
        columns = sources[0].info["model_columns"]
        real, audit = align_sources(sources, columns)
        audits.extend({"dataset": dataset, "split": split, "variant": variant, **row} for row in audit)
        relative = Path(dataset) / split / variant
        group = args.output_dir / relative
        group.mkdir(parents=True, exist_ok=True)
        title = f"{dataset} / {split}" + (" / Hybrid t400 epoch1000" if variant != "main" else "")
        sections, group_rows = [], []
        for target in columns:
            started = time.monotonic()
            directory = group / f"loco_{target}"
            directory.mkdir(exist_ok=True)
            predictions, target_rows = {}, []
            spec = marginal_spec(real[target], target)
            for source in sources:
                prediction, checks = load_prediction(source, columns, target, 0, real)
                predictions[source.key] = prediction
                checks_all.extend({"dataset": dataset, "split": split, "variant": variant, **row} for row in checks)
                signature = metric_cache_signature(source, sources, target, len(real), args.metric_sample_size, implementation)
                saved = None
                for cache in (directory / f"metrics_{source.key}.json", args.cache_root / relative / f"loco_{target}/metrics_{source.key}.json"):
                    if cache.is_file():
                        candidate = json.loads(cache.read_text())
                        if candidate.get("signature") == signature:
                            saved = candidate
                            break
                if saved:
                    scores, errors = saved["scores"].copy(), list(saved["errors"])
                else:
                    print(f"Computing uncached metrics: {title} / {target} / {source.key}", flush=True)
                    scores, errors = metric_record(real, prediction, target, args.metric_sample_size,
                                                  generative_metric_seed(42, f"{dataset}/{split}/{target}"))
                scores.update(marginal_scores(real[target], prediction[target], spec))
                if "marginal_error" in scores: errors.append(scores["marginal_error"])
                (directory / f"metrics_{source.key}.json").write_text(json.dumps(
                    {"signature": signature, "scores": scores, "errors": errors, "marginal_protocol": MARGINAL_PROTOCOL}, indent=2) + "\n")
                record = {"dataset": dataset, "split": split, "variant": variant, "target": target,
                          "model": source.key, "trial": 0, "rows": len(real),
                          "metric_rows": min(len(real), args.metric_sample_size),
                          "c2st_convergence_warning": any("ConvergenceWarning" in error for error in errors), **scores}
                all_rows.append(record); group_rows.append(record); target_rows.append(record)
                errors_all.extend({"dataset": dataset, "split": split, "variant": variant,
                                   "target": target, "model": source.key, "error": error} for error in errors)
            serialized = {**spec, "protocol": MARGINAL_PROTOCOL, "rows": len(real)}
            if "edges" in spec:
                serialized["edges"] = [float(edge) if np.isfinite(edge) else ("-inf" if edge < 0 else "inf") for edge in spec["edges"]]
            (directory / "marginal_protocol.json").write_text(json.dumps(serialized, indent=2) + "\n")
            task_title = title + f" / LOCO {target}"
            if not args.skip_existing_plots or not (directory / "distribution.png").is_file():
                distribution(real, predictions, target, directory / "distribution.png", task_title)
            if target in {"iops", "lat"} and (not args.skip_existing_plots or not (directory / "distribution_log.png").is_file()):
                distribution(real, predictions, target, directory / "distribution_log.png", task_title, True)
            for context in columns:
                if context == target: continue
                if not args.skip_existing_plots or not (directory / f"pair_{context}.png").is_file():
                    scatter_pair(real, predictions, target, context, dataset, directory / f"pair_{context}.png", task_title)
                if (target not in NUMERICAL or context not in NUMERICAL) and (not args.skip_existing_plots or not (directory / f"conditional_{context}.png").is_file()):
                    conditional_pair(real, predictions, target, context, dataset, directory / f"conditional_{context}.png", task_title)
            if dataset == "hdd_sequential" and target in {"iops", "lat"} and (not args.skip_existing_plots or not (directory / "pair_iops_lat_full.png").is_file()):
                scatter_pair(real, predictions, target, "lat" if target == "iops" else "iops", dataset,
                             directory / "pair_iops_lat_full.png", task_title + " (full range)", False)
            target_frame = pd.DataFrame(target_rows).set_index("model").loc[list(MODEL_ORDER)].reset_index()
            target_frame.to_csv(directory / "metrics.csv", index=False)
            save_table(target_frame, directory / "metrics_table.png", task_title)
            body = "<p><a href='../index.html'>Dataset report</a></p><p>GT, Ours, TabDiff, TabSyn. Trial 000. Best per metric in bold.</p>"
            body += "<p>TV/KS/MSE: lower is better; other scores: higher is better. Text-category KS is n/a. " + \
                    "* C2ST estimator convergence warning; retained but not ranked. Continuous TV is a GT-quantile histogram approximation, not KDE.</p>"
            body += table_html(target_frame) + "<p><a href='metrics.csv'>Metrics CSV</a> | <a href='marginal_protocol.json'>TV/KS protocol</a></p>"
            body += "<h2>1D distribution</h2><img src='distribution.png'>"
            if target in {"iops", "lat"}: body += "<img loading='lazy' src='distribution_log.png'>"
            for context in columns:
                if context == target: continue
                body += f"<h2>Context: {context}</h2>"
                if target not in NUMERICAL or context not in NUMERICAL:
                    body += f"<img loading='lazy' src='conditional_{context}.png'>"
                body += f"<img loading='lazy' src='pair_{context}.png'>"
            if dataset == "hdd_sequential" and target in {"iops", "lat"}:
                body += "<h2>IOPS / latency, full range</h2><img loading='lazy' src='pair_iops_lat_full.png'>"
            (directory / "index.html").write_text(html_page(task_title, body))
            sections.append(f"<li><a href='loco_{target}/index.html'>LOCO {target}</a> | <a href='loco_{target}/metrics_table.png'>Table PNG</a></li>")
            print(f"Saved {task_title} | rows={len(real):,} | {time.monotonic() - started:.1f}s", flush=True)
        frame = pd.DataFrame(group_rows)
        frame["model_order"] = frame.model.map({key: index for index, key in enumerate(MODEL_ORDER)})
        frame = frame.sort_values(["target", "model_order"], kind="stable").drop(columns="model_order").reset_index(drop=True)
        frame.to_csv(group / "metrics.csv", index=False)
        save_table(frame, group / "metrics_table.png", title)
        body = "<p><a href='../../../index.html'>All datasets</a></p><p>Trial 000; shared GT; train/test kept separate.</p>"
        body += f"<p>Matched rows: {len(real):,}. TV/KS and MSE/R2 use all rows; other generative metrics: up to {args.metric_sample_size:,} rows.</p>"
        body += "<ul>" + "".join(sections) + "</ul>" + metric_matrices(frame, group / "tables", title)
        body += table_html(frame) + "<p>* C2ST convergence warning: value retained, not ranked. Text-category KS/MSE/R2 are n/a.</p>"
        body += "<p><a href='metrics.csv'>Detailed CSV</a> | <a href='metrics_table.png'>Table PNG</a></p>"
        (group / "index.html").write_text(html_page(title, body))
        links.append(f"<li><a href='{relative.as_posix()}/index.html'>{title}</a></li>")
    metrics = pd.DataFrame(all_rows)
    metrics["model_order"] = metrics.model.map({key: index for index, key in enumerate(MODEL_ORDER)})
    metrics = metrics.sort_values(["dataset", "split", "variant", "target", "model_order"], kind="stable").drop(columns="model_order").reset_index(drop=True)
    metrics.to_csv(args.output_dir / "metrics_all.csv", index=False)
    display, flags = display_table(metrics)
    pd.concat([metrics[["dataset", "split", "variant"]], display], axis=1).to_csv(args.output_dir / "metrics_formatted_all.csv", index=False)
    pd.concat([metrics[["dataset", "split", "variant", "target", "model"]], flags.add_prefix("best_")], axis=1).to_csv(args.output_dir / "metrics_best_flags.csv", index=False)
    pd.DataFrame(audits).to_csv(args.output_dir / "alignment_audit.csv", index=False)
    pd.DataFrame(checks_all).to_csv(args.output_dir / "context_integrity.csv", index=False)
    pd.DataFrame(errors_all, columns=["dataset", "split", "variant", "target", "model", "error"]).to_csv(args.output_dir / "metric_errors.csv", index=False)
    (args.output_dir / "protocol.json").write_text(json.dumps({"marginal_protocol": MARGINAL_PROTOCOL,
        "trial": 0, "metric_sample_size": args.metric_sample_size, "source_reports": str(args.reports_root),
        "panel_order": ORDER, "notes": notes}, indent=2) + "\n")
    body = "<p>GT, Ours, TabDiff, TabSyn. Four datasets, train/test separately. Hybrid last; saved trial 000.</p><ul>" + "".join(links) + "</ul>"
    body += "<p><a href='metrics_all.csv'>All metrics</a> | <a href='metrics_formatted_all.csv'>Compact CSV</a> | <a href='metrics_best_flags.csv'>Winner flags</a> | <a href='metric_errors.csv'>Warnings</a></p>"
    body += "<h2>Protocol</h2><ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in notes) + "</ul>"
    (args.output_dir / "index.html").write_text(html_page("LOCO: one shared GT and three models", body))
    print(f"Saved {len(metrics)} metric rows to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
