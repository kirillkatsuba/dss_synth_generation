#!/usr/bin/env python3
"""Clean train clouds in log(IOPS), log(latency), preserving original row values.

Detector follows remove_outliers in HSE-LAMBDA/digital-twin at commit
884cd9a0c334dfb96ab4291a4c734d3ae848ccdf, scripts/calculate_summary_stats.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dss_outliers_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "dss_outliers_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.training_data import config_path, dataset_paths

KEYS = ["id", "io_type"]
VALUES = ["iops", "lat"]


def cloud_seed(seed: int, key: tuple) -> int:
    digest = hashlib.sha256(json.dumps([str(value) for value in key]).encode()).digest()
    return (seed + int.from_bytes(digest[:4], "big")) % 2**32


def clean_clouds(frame: pd.DataFrame, contamination: float = .05, seed: int = 42,
                 epsilon: float = 1e-6, min_rows: int = 20,
                 zero_policy: str = "keep") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < contamination <= .5:
        raise ValueError("contamination must be in (0, 0.5].")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    if min_rows < 5 or not 0 <= seed < 2**32 or zero_policy not in {"keep", "drop"}:
        raise ValueError("Invalid min_rows, seed, or zero_policy.")
    missing = set(KEYS + VALUES) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"Expected nonempty data with columns {KEYS + VALUES}; missing: {sorted(missing)}")
    if frame[KEYS].isna().any().any():
        raise ValueError("Cloud identifiers must not be missing.")
    values = frame[VALUES].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("IOPS/latency must be finite and nonnegative; invalid measurements require an explicit policy.")
    with np.errstate(over="raise", invalid="raise"):
        log_values = np.log(values + epsilon)
    zero_rows = (values == 0).any(axis=1)
    keep = np.ones(len(frame), dtype=bool)
    reasons = np.full(len(frame), "inlier", dtype=object)
    distances = np.full(len(frame), np.nan)
    summaries = []
    context = [column for column in frame if column not in KEYS + VALUES]
    for key, positions in frame.groupby(KEYS, sort=True, dropna=False).indices.items():
        positions = np.asarray(positions, dtype=int)
        if context and frame.iloc[positions][context].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Cloud {key} contains multiple contexts; refusing to mix clouds.")
        excluded = positions[zero_rows[positions]] if zero_policy == "drop" else np.array([], dtype=int)
        keep[excluded], reasons[excluded] = False, "zero_policy"
        selected = positions[keep[positions]]
        xy = log_values[selected]
        status, message = "fit", ""
        if len(selected) < min_rows:
            status = "skipped_small_cloud"
        elif np.linalg.matrix_rank(xy - xy.mean(axis=0)) < 2:
            status = "skipped_rank_deficient"
        else:
            caught = []
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    detector = EllipticEnvelope(contamination=contamination, random_state=cloud_seed(seed, key))
                    labels = detector.fit_predict(xy)
                    distance = detector.mahalanobis(xy)
                if np.linalg.matrix_rank(detector.covariance_) < 2 or not np.isfinite(distance).all():
                    status = "skipped_unstable_covariance"
                else:
                    distances[selected] = distance
                    removed = selected[labels == -1]
                    keep[removed], reasons[removed] = False, "elliptic_outlier"
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                status, message = "skipped_fit_error", f"{type(exc).__name__}: {exc}"
            notices = sorted({f"{item.category.__name__}: {str(item.message)}" for item in caught})
            message = " | ".join([part for part in [message, *notices] if part])
        if status != "fit":
            reasons[selected] = status
        removed_distances = distances[positions[~keep[positions]]]
        finite_distances = removed_distances[np.isfinite(removed_distances)]
        summaries.append({**dict(zip(KEYS, key)), "rows_before": len(positions),
                          "rows_after": int(keep[positions].sum()), "rows_removed": int((~keep[positions]).sum()),
                          "removed_fraction": float((~keep[positions]).mean()),
                          "zero_measurements": int(zero_rows[positions].sum()),
                          "zero_measurements_kept": int((zero_rows[positions] & keep[positions]).sum()),
                          "detector_rows": len(selected), "status": status, "warning_or_error": message,
                          "max_removed_distance": float(finite_distances.max()) if len(finite_distances) else np.nan})
    decisions = frame[KEYS].reset_index(drop=True).copy()
    decisions.insert(0, "source_row", np.arange(len(frame)))
    decisions["keep"], decisions["reason"] = keep, reasons
    decisions["zero_measurement"] = zero_rows
    decisions["mahalanobis_distance_squared"] = distances
    return frame.iloc[np.flatnonzero(keep)].copy(), decisions, pd.DataFrame(summaries)


def axis_bounds(values: np.ndarray, logarithmic: bool) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    pad = max((high - low) * .035, .1)
    return (low - pad if logarithmic else max(0, low - pad), high + pad)


def comparison_axes(axes, frame: pd.DataFrame, keep: np.ndarray, epsilon: float,
                    logarithmic: bool = False, window: bool = False,
                    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None) -> None:
    values = frame[VALUES].to_numpy(float)
    xy = np.log(values + epsilon) if logarithmic else values / np.array([1e3, 1e6])
    if bounds is None:
        bounds = [(0, 14), (0, 100)] if window else [axis_bounds(xy[:, index], logarithmic) for index in (0, 1)]
    ranges = np.asarray(bounds, dtype=float)
    if ranges.shape != (2, 2) or not np.isfinite(ranges).all() or (ranges[:, 0] >= ranges[:, 1]).any():
        raise ValueError("Plot bounds must be two finite, increasing axis ranges.")
    visible = ((xy[:, 0] >= bounds[0][0]) & (xy[:, 0] <= bounds[0][1]) &
               (xy[:, 1] >= bounds[1][0]) & (xy[:, 1] <= bounds[1][1]))
    for index, ax in enumerate(axes):
        shown = np.ones(len(frame), dtype=bool) if index == 0 else keep
        ax.scatter(xy[shown, 0], xy[shown, 1], s=.6, alpha=.4, color="#2563eb", linewidths=0, rasterized=True)
        if index == 0 and (~keep).any():
            ax.scatter(xy[~keep, 0], xy[~keep, 1], s=1.7, alpha=.65, color="#dc2626", linewidths=0,
                       label=f"Removed in window: {(~keep & visible).sum():,}", rasterized=True)
            ax.legend(fontsize=8, markerscale=2, frameon=False)
        ax.set(xlim=bounds[0], ylim=bounds[1],
               xlabel="ln(IOPS + eps)" if logarithmic else r"IOPS ($10^3$)",
               ylabel="ln(Latency + eps)" if logarithmic else r"Latency / $10^6$")
        outside = shown & ((xy[:, 0] < bounds[0][0]) | (xy[:, 0] > bounds[0][1]) |
                           (xy[:, 1] < bounds[1][0]) | (xy[:, 1] > bounds[1][1]))
        if outside.any():
            ax.text(.02, .96, f"Outside window: {outside.sum():,} ({outside.sum()/shown.sum():.1%})",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        ax.set_title(f"{'Original' if index == 0 else 'Cleaned'} (n={shown.sum():,})", fontsize=11)
        ax.grid(True, linestyle="--", alpha=.3)


def save_comparison(frame: pd.DataFrame, keep: np.ndarray, output: Path, dataset: str,
                    epsilon: float, logarithmic: bool = False, window: bool = False,
                    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    comparison_axes(axes, frame, keep, epsilon, logarithmic, window, bounds)
    fig.suptitle(f"{dataset} / train | removed {(~keep).sum():,} / {len(frame):,} ({(~keep).mean():.2%})", fontsize=12)
    fig.tight_layout(); fig.savefig(output, dpi=220); plt.close(fig)


def save_cloud_examples(frame: pd.DataFrame, decisions: pd.DataFrame, summary: pd.DataFrame,
                        output: Path, dataset: str, epsilon: float, count: int) -> None:
    selected = summary.sort_values(["zero_measurements", "rows_removed", "max_removed_distance"],
                                   ascending=False, na_position="last").head(count)
    fig, axes = plt.subplots(len(selected), 2, figsize=(11, 3.3 * len(selected)), squeeze=False)
    for row, (_, cloud) in enumerate(selected.iterrows()):
        mask = (frame.id == cloud.id) & (frame.io_type == cloud.io_type)
        positions = np.flatnonzero(mask)
        comparison_axes(axes[row], frame.iloc[positions], decisions.keep.to_numpy()[positions], epsilon)
        for index in (0, 1):
            axes[row, index].set_title(f"{cloud.id} / {cloud.io_type}\n" + axes[row, index].get_title(), fontsize=9)
    fig.suptitle(f"{dataset}: individual clouds (same axes within each pair)", fontsize=12)
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dss_imputer/datasets.json")
    parser.add_argument("--datasets", nargs="+", default=list(json.loads((ROOT / "configs/dss_imputer/datasets.json").read_text())))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/outlier_cleaning")
    parser.add_argument("--contamination", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--min-cloud-rows", type=int, default=20)
    parser.add_argument("--zero-policy", choices=["keep", "drop"], default="keep")
    parser.add_argument("--cloud-examples", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepared-config", type=Path,
                        help="Write shared training config: cleaned train, unchanged original test.")
    args = parser.parse_args()
    configured = json.loads(args.config.read_text())
    if not set(args.datasets) <= set(configured) or args.cloud_examples < 0:
        parser.error("Unknown dataset or negative cloud-examples count.")
    # Validate every source before creating outputs; never overwrite input CSVs.
    sources = {dataset: dataset_paths(args.config, dataset) for dataset in args.datasets}
    inputs = {path for paths in sources.values() for path in paths.values()}
    generated = set()
    for dataset in args.datasets:
        generated.update((args.output_dir / dataset / name).resolve() for name in
                         ("train_clean.csv", "removed_rows.csv", "row_decisions.csv", "cloud_summary.csv", "run_info.json"))
    if generated & inputs or (args.prepared_config and args.prepared_config.resolve() in inputs | generated | {args.config.resolve()}):
        parser.error("Output paths must not overwrite input CSVs or cleaning artifacts.")
    if args.prepared_config and args.prepared_config.exists() and not args.overwrite:
        raise FileExistsError(f"Prepared config exists: {args.prepared_config}; use --overwrite or another path.")
    for dataset in args.datasets:
        destination = args.output_dir / dataset
        if destination.exists() and any(destination.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output exists: {destination}; choose another directory or --overwrite.")
    links, totals, prepared = [], [], {}
    for dataset in args.datasets:
        source = sources[dataset]["train_csv"]
        frame = pd.read_csv(source, low_memory=False)
        print(f"Cleaning {dataset}: {len(frame):,} train rows, cloud keys {KEYS}", flush=True)
        cleaned, decisions, summary = clean_clouds(frame, args.contamination, args.seed, args.epsilon,
                                                  args.min_cloud_rows, args.zero_policy)
        keep = decisions.keep.to_numpy()
        directory = args.output_dir / dataset
        directory.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(directory / "train_clean.csv", index=False)
        removed = frame.iloc[np.flatnonzero(~keep)].copy()
        removed.insert(0, "source_row", np.flatnonzero(~keep))
        removed.to_csv(directory / "removed_rows.csv", index=False)
        decisions.to_csv(directory / "row_decisions.csv", index=False)
        summary.to_csv(directory / "cloud_summary.csv", index=False)
        save_comparison(frame, keep, directory / "before_after.png", dataset, args.epsilon)
        save_comparison(frame, keep, directory / "before_after_log.png", dataset, args.epsilon, True)
        if dataset == "hdd_sequential":
            save_comparison(frame, keep, directory / "before_after_window.png", dataset, args.epsilon, window=True)
        if args.cloud_examples:
            save_cloud_examples(frame, decisions, summary, directory / "cloud_examples.png", dataset, args.epsilon, args.cloud_examples)
        record = {"dataset": dataset, "source_train": str(source.resolve()),
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "source_test": str(sources[dataset]["test_csv"]),
                  "test_sha256": hashlib.sha256(sources[dataset]["test_csv"].read_bytes()).hexdigest(),
                  "rows_before": len(frame), "rows_after": len(cleaned), "rows_removed": len(removed),
                  "removed_fraction": len(removed) / len(frame), "clouds": len(summary),
                  "skipped_clouds": int(summary.status.ne("fit").sum()),
                  "clouds_with_warnings": int(summary.warning_or_error.ne("").sum()),
                  "zero_rows_before": int(decisions.zero_measurement.sum()),
                  "zero_rows_after": int((decisions.zero_measurement & decisions.keep).sum()),
                  "contamination": args.contamination, "seed": args.seed, "epsilon": args.epsilon,
                  "zero_policy": args.zero_policy, "min_cloud_rows": args.min_cloud_rows,
                  "detector_columns": ["log(iops + eps)", "log(lat + eps)"], "group_columns": KEYS,
                  "original_values_preserved": True, "test_modified": False}
        (directory / "run_info.json").write_text(json.dumps(record, indent=2) + "\n")
        prepared[dataset] = {"train_csv": config_path(directory / "train_clean.csv"),
                             "test_csv": config_path(sources[dataset]["test_csv"])}
        totals.append(record)
        links.append(f"<h2>{dataset}</h2><p>Train: {len(frame):,} to {len(cleaned):,}; removed {len(removed):,} ({len(removed)/len(frame):.2%}). "
                     f"Skipped clouds: {record['skipped_clouds']}; zero rows retained: {record['zero_rows_after']}.</p>"
                     f"<p><a href='{dataset}/train_clean.csv'>Clean train CSV</a> | <a href='{dataset}/cloud_summary.csv'>Cloud audit</a></p>")
        for name in ["before_after_window.png"] if dataset == "hdd_sequential" else []:
            links.append(f"<h3>Established window (not the full range)</h3><img src='{dataset}/{name}'>")
        for name, title in [("before_after.png", "All rows, full range"), ("before_after_log.png", "Log coordinates used by the detector")]:
            links.append(f"<h3>{title}</h3><img src='{dataset}/{name}'>")
        if args.cloud_examples:
            links.append(f"<h3>Individual clouds</h3><img src='{dataset}/cloud_examples.png'>")
        print(f"Finished {dataset}: removed {len(removed):,}; skipped {record['skipped_clouds']} clouds; zeros kept {record['zero_rows_after']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(totals).to_csv(args.output_dir / "summary.csv", index=False)
    heading = "<h1>Outlier cleaning: original left, cleaned right</h1>"
    heading += "<p>Train only. Original files/test unchanged. Each id + io_type cloud fitted separately in log(iops + eps), log(lat + eps). "
    heading += f"EllipticEnvelope contamination={args.contamination}, seed={args.seed}, eps={args.epsilon}, zero policy={args.zero_policy}. "
    heading += "Red points on the left are removed rows. All other column values are preserved; full-range panels use identical axes. "
    heading += "Skipped/unstable clouds are retained and recorded, not silently filtered. The contamination fraction is configured, not a learned anomaly prevalence.</p>"
    page = "<!doctype html><meta charset='utf-8'><title>Cloud outlier cleaning</title><style>body{font:15px Arial;max-width:1400px;margin:24px auto;padding:0 16px}img{max-width:100%}h2{border-top:1px solid #ccc;padding-top:20px}p{line-height:1.5}</style>"
    (args.output_dir / "index.html").write_text(page + heading + "".join(links))
    if args.prepared_config:
        args.prepared_config.parent.mkdir(parents=True, exist_ok=True)
        args.prepared_config.write_text(json.dumps(prepared, indent=2) + "\n")
        print(f"Prepared training config: {args.prepared_config}", flush=True)
    print(f"Report: {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
