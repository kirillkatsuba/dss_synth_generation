#!/usr/bin/env python3
"""Add readable focus tables and train/test diagnostics to the five-trial report."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_clean_loco_trials import (
    DATASETS, MODELS, LABELS, NUMERICAL, page, html_table, semantic_keys,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def context_audit(downloads, output):
    rows, distribution_rows = [], []
    for dataset in DATASETS:
        root = downloads / f"{dataset}_hybrid_mcar15_log_t200_clean_20261023_noes_s42"
        train = pd.read_csv(root / "train_last/evaluation_truth.csv")
        test = pd.read_csv(root / "test_last/evaluation_truth.csv")
        info = json.loads((root / "test_last/run_info.json").read_text())
        configurations = [c for c in info["model_columns"] if c not in {"iops", "lat"}]
        for target in info["loco_columns"]:
            context = [c for c in configurations if c != target]
            a = train[context].apply(semantic_keys)
            b = test[context].apply(semantic_keys)
            train_keys = pd.util.hash_pandas_object(a, index=False)
            test_keys = pd.util.hash_pandas_object(b, index=False)
            rows.append(dict(dataset=dataset, target=target, configuration_columns=" ".join(context),
                             train_configurations=train_keys.nunique(), test_configurations=test_keys.nunique(),
                             test_unseen_configuration_fraction=float((~test_keys.isin(train_keys)).mean())))
        for col in info["model_columns"]:
            a, b = semantic_keys(train[col]), semantic_keys(test[col])
            pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
            support = pa.index.union(pb.index)
            categorical = col not in NUMERICAL
            record = dict(dataset=dataset, column=col, role="categorical" if categorical else "numerical",
                          unseen_test_fraction=float((~b.isin(a)).mean()) if categorical else np.nan,
                          train_test_tv=float(.5 * (pa.reindex(support, fill_value=0) - pb.reindex(support, fill_value=0)).abs().sum()) if categorical else np.nan)
            if pd.api.types.is_numeric_dtype(train[col]):
                record["train_test_ks"] = ks_2samp(train[col], test[col], method="asymp").statistic
            distribution_rows.append(record)
    pd.DataFrame(rows).to_csv(output / "context_shift.csv", index=False)
    pd.DataFrame(distribution_rows).to_csv(output / "train_test_marginals.csv", index=False)


def monitor_plot(downloads, output):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharey=True)
    records = []
    root = downloads / "dss_training_audit_clean_20261023_noes/models"
    for i, dataset in enumerate(DATASETS):
        history = pd.read_csv(root / f"{dataset}_hybrid_mcar15_log_t200_clean_20261023_noes_s42/generative_history.csv")
        for j, scenario in enumerate(["loco_iops", "loco_lat", "mcar_15"]):
            group = history[history.scenario == scenario].sort_values("epoch")
            ax = axes[i, j]
            for metric, color in [("target_column_shapes", "#157a6e"), ("target_column_pair_trends", "#b52b43"),
                                  ("beta_recall", "#7854a8")]:
                ax.plot(group.epoch, group[metric], label=metric, color=color)
                best = group.loc[group[metric].idxmax()]
                records.append(dict(dataset=dataset, scenario=scenario, metric=metric,
                                    best_epoch=int(best.epoch), best_score=best[metric],
                                    final_epoch=int(group.iloc[-1].epoch), final_score=group.iloc[-1][metric]))
            ax.set(title=f"{dataset} / {scenario}", xlabel="Epoch", ylim=(0, 1.02))
            ax.grid(alpha=.2)
            if i == 0 and j == 0: ax.legend(fontsize=7)
    fig.suptitle("Generative monitoring on test | 1024 fixed rows, one sample per checkpoint", fontsize=13)
    fig.tight_layout(); fig.savefig(output / "generative_monitor.png", dpi=170); plt.close(fig)
    pd.DataFrame(records).to_csv(output / "generative_monitor.csv", index=False)


def boundary_audit(downloads, output):
    records = []
    for dataset in DATASETS:
        hybrid = downloads / f"{dataset}_hybrid_mcar15_log_t200_clean_20261023_noes_s42"
        train = pd.read_csv(hybrid / "train_last/evaluation_truth.csv")
        for split in ("train", "test"):
            roots = [hybrid / f"{split}_last", *[downloads / "clean_20261022_r2" / model / dataset / f"loco_{split}" for model in MODELS[1:]]]
            for model, root in zip(MODELS, roots):
                truth = pd.read_csv(root / "evaluation_truth.csv")
                for target in ("iops", "lat"):
                    lower, upper = float(train[target].min()), float(train[target].max())
                    for trial in range(5):
                        prediction = pd.read_csv(root / f"loco_{target}/trial_{trial:03d}.csv", usecols=[target])[target].to_numpy(float)
                        record = dict(dataset=dataset, split=split, model=model, target=target, trial=trial,
                                      rows=len(prediction), train_min=lower, train_max=upper,
                                      test_max=float(truth[target].max()),
                                      near_train_min_fraction=float(np.isclose(prediction, lower, rtol=1e-6, atol=1e-8).mean()),
                                      near_train_max_fraction=float(np.isclose(prediction, upper, rtol=1e-6, atol=1e-8).mean()),
                                      truth_near_train_max_fraction=float(np.isclose(truth[target], upper, rtol=1e-6, atol=1e-8).mean()),
                                      above_truth_max_fraction=float((prediction > truth[target].max()).mean()))
                        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "boundary_mass_by_trial.csv", index=False)
    frame.groupby(["dataset", "split", "model", "target"])[["near_train_min_fraction", "near_train_max_fraction", "truth_near_train_max_fraction", "above_truth_max_fraction"]].agg(["mean", "std", "count"]).to_csv(output / "boundary_mass_summary.csv")


def focus_table(summary, output, split):
    sub = summary[(summary.split == split) & summary.target.isin(["iops", "lat"])]
    fields = ["target_column_shapes", "target_column_pair_trends", "alpha_precision", "beta_recall", "ks", "tv"]
    rows, winners = [], []
    for dataset in DATASETS:
        for target in ["iops", "lat"]:
            for model in MODELS:
                group = sub[(sub.dataset == dataset) & (sub.target == target) & (sub.model == model)].set_index("metric")
                texts = [dataset, target, model]
                flags = [False]*3
                for metric in fields:
                    r = group.loc[metric]
                    competitors = sub[(sub.dataset == dataset) & (sub.target == target) & (sub.metric == metric)]
                    eligible = competitors[competitors.complete & ~competitors.c2st_warning]
                    best = eligible["mean"].min() if metric in {"ks", "tv"} else eligible["mean"].max()
                    texts.append(r.mean_std.replace("+/-", "±"))
                    flags.append(bool(r.complete and np.isclose(r["mean"], best, rtol=1e-9, atol=1e-12)))
                rows.append(texts); winners.append(flags)
    fig, ax = plt.subplots(figsize=(18, 10)); ax.axis("off")
    ax.set_title(f"{split.upper()} / LOCO IOPS and latency / mean +/- sample SD, 5 trials", pad=12)
    table = ax.table(cellText=rows, colLabels=["Dataset", "LOCO", "Model", "Shape ↑", "Pairs ↑", "Alpha joint ↑", "Beta joint ↑", "KS ↓", "TV ↓"],
                     loc="center", cellLoc="center", colWidths=[.12, .055, .07]+[.125]*6)
    table.auto_set_font_size(False); table.set_fontsize(8.1); table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0: cell.set_facecolor("#e6e8eb"); cell.set_text_props(weight="bold")
        elif ((r-1)//3) % 2 == 0: cell.set_facecolor("#f5f7fa")
    fig.text(.02, .025, "Underlined: best mean per dataset and LOCO, not statistical significance. Exact PMF / quantile-bin TV. Joint scores include copied context.", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1)); fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for r, flags in enumerate(winners, 1):
        for c, flag in enumerate(flags):
            if flag:
                b = table[r, c].get_text().get_window_extent(renderer)
                (x0, y), (x1, _) = fig.transFigure.inverted().transform([(b.x0, b.y0-1), (b.x1, b.y0-1)])
                fig.add_artist(Line2D([x0, x1], [y, y], transform=fig.transFigure, color="black", lw=.7))
    fig.savefig(output / f"{split}_iops_lat_mean_std.png", dpi=190); plt.close(fig)
    sub.to_csv(output / f"{split}_iops_lat_mean_std.csv", index=False)


def metric_winners(summary, output):
    fields = ["target_column_shapes", "target_column_pair_trends", "target_overall_quality", "ks", "tv",
              "alpha_precision", "beta_recall", "numeric_schema_target_overall_quality"]
    records = []
    for (dataset, split, target, metric), group in summary[summary.metric.isin(fields)].groupby(["dataset", "split", "target", "metric"]):
        valid = group[group.complete & ~group.c2st_warning & group["mean"].notna()]
        if len(valid) != len(MODELS): continue
        best = valid["mean"].min() if metric in {"ks", "tv"} else valid["mean"].max()
        for _, row in valid.iterrows():
            records.append(dict(dataset=dataset, split=split, target=target, metric=metric, model=row.model,
                                best_mean=bool(np.isclose(row["mean"], best, rtol=1e-9, atol=1e-12))))
    wins = pd.DataFrame(records)
    wins.to_csv(output / "best_mean_by_scenario.csv", index=False)
    wins.groupby(["dataset", "split", "metric", "model"]).best_mean.agg(["sum", "count"]).to_csv(output / "best_mean_counts.csv")
    wide = summary[summary.metric.isin(fields)].pivot(index=["dataset", "target", "model", "metric"], columns="split", values="mean")
    wide["test_minus_train"] = wide.test - wide.train
    wide.to_csv(output / "train_test_metric_gap.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("/Users/kirill/Downloads"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/clean_loco_trials_comparison")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    context_audit(args.downloads, args.output_dir)
    monitor_plot(args.downloads, args.output_dir)
    boundary_audit(args.downloads, args.output_dir)
    if args.diagnostics_only: return
    summary = pd.read_csv(args.output_dir / "summary_long.csv")
    for split in ["train", "test"]: focus_table(summary, args.output_dir, split)
    metric_winners(summary, args.output_dir)
    body = "<p><a href='index.html'>Все LOCO и метрики</a></p><p>Подчёркнуто лучшее среднее; это не проверка статистической значимости.</p>"
    for split in ["test", "train"]:
        body += f"<h2>{split}</h2><img src='{split}_iops_lat_mean_std.png'>"
    body += "<h2>Мониторинг генеративного качества</h2><img src='generative_monitor.png'>"
    body += "<p><a href='context_shift.csv'>Новые комбинации настроек</a> · <a href='train_test_marginals.csv'>Сдвиги распределений</a></p>"
    (args.output_dir / "focus.html").write_text(page("IOPS / latency: пять trial и диагностика обучения", body))
    index = args.output_dir / "index.html"
    navigation = ("<nav id='analysis-links'><a href='analysis.md'>Подробный разбор и выводы</a> · "
                  "<a href='focus.html'>Сводные таблицы IOPS / latency</a></nav>")
    content = index.read_text()
    if "id='analysis-links'" not in content:
        index.write_text(content.replace("<h1>", navigation + "<h1>", 1))


if __name__ == "__main__":
    main()
