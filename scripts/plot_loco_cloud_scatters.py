#!/usr/bin/env python3
"""Plot ground-truth and LOCO scatter pairs for fixed exported clouds."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dss_clouds_mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.export_loco_clouds import validate_alignment

FEATURES = ["iops", "lat", "block_size", "n_jobs", "iodepth",
            "read_fraction", "io_type", "raid", "n_disks"]
CATEGORICAL_AXES = {"io_type", "raid"}
COLORS = {"ground_truth": "#2563eb", "prediction": "#dc2626"}
METADATA = {"source_row", "cloud_id", "cloud_name"}


def load_aligned_cloud(truth_path: Path, prediction_path: Path, target: str):
    truth = pd.read_csv(truth_path, low_memory=False)
    prediction = pd.read_csv(prediction_path, low_memory=False)
    for frame in (truth, prediction):
        if "source_row" not in frame or frame.source_row.duplicated().any():
            raise ValueError("Cloud CSV must contain unique source_row values.")
        if not len(frame):
            raise ValueError("Cloud CSV is empty.")
    if set(truth.source_row) != set(prediction.source_row):
        raise ValueError(f"Cloud membership differs: {prediction_path}")
    truth = truth.sort_values("source_row").reset_index(drop=True)
    prediction = prediction.set_index("source_row").loc[truth.source_row].reset_index()
    validate_alignment(truth, prediction.loc[:, truth.columns], target)
    reference = f"ground_truth_{target}"
    if reference not in prediction:
        raise ValueError(f"Missing reference column {reference}: {prediction_path}")
    pd.testing.assert_series_equal(prediction[reference], truth[target],
                                   check_names=False, check_dtype=False, check_exact=True)
    if truth.cloud_id.nunique() != 1:
        raise ValueError("Per-cloud CSV contains multiple cloud IDs.")
    return truth, prediction


def display_axis(series_list: list[pd.Series], column: str):
    if column in CATEGORICAL_AXES:
        if any(series.isna().any() for series in series_list):
            raise ValueError(f"Missing categorical values in {column}")
        support = sorted(set().union(*(set(s.astype(str)) for s in series_list)))
        mapping = {value: index for index, value in enumerate(support)}
        values = [s.astype(str).map(mapping).to_numpy(dtype=float) for s in series_list]
        return values, (-0.6, len(support)-0.4), support
    divisor = {"iops": 1e3, "lat": 1e6}.get(column, 1)
    values = [pd.to_numeric(s, errors="raise").to_numpy(dtype=float)/divisor for s in series_list]
    joined = np.concatenate(values)
    if not np.isfinite(joined).all():
        raise ValueError(f"Nonfinite values in {column}; no scatter rows are silently dropped.")
    low, high = float(joined.min()), float(joined.max())
    padding = max((high-low)*0.08, abs(low)*0.015, 0.05 if divisor != 1 else 0.5)
    return values, (low-padding, high+padding), None


def axis_label(column):
    return {"iops": "IOPS / 1000", "lat": "Latency / 1e6"}.get(column, column)


def draw_scatter(ax, x, y, xcolumn, ycolumn, xlimits, ylimits, xticks, yticks, title, color):
    ax.scatter(x, y, s=9, color=color, alpha=0.5, linewidths=0)
    ax.set(xlim=xlimits, ylim=ylimits, xlabel=axis_label(xcolumn),
           ylabel=axis_label(ycolumn), title=title)
    if xticks is not None:
        ax.set_xticks(np.arange(len(xticks)), xticks, rotation=45, ha="right")
    if yticks is not None:
        ax.set_yticks(np.arange(len(yticks)), yticks)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_target_pairs(truth, prediction, target, columns, destination, dpi):
    contexts = [column for column in columns if column != target]
    if not contexts:
        raise ValueError("At least one known column is required.")
    row_count = (len(contexts)+1)//2
    fig, axes = plt.subplots(row_count, 4, figsize=(15.2, row_count*3.5), squeeze=False)
    ys, ylimits, yticks = display_axis([truth[target], prediction[target]], target)
    for index, context in enumerate(contexts):
        xs, xlimits, xticks = display_axis([truth[context], prediction[context]], context)
        row, column = index//2, (index%2)*2
        for source_index, source in enumerate(("ground_truth", "prediction")):
            draw_scatter(axes[row, column+source_index], xs[source_index], ys[source_index],
                         context, target, xlimits, ylimits, xticks, yticks,
                         f"{'Ground truth' if source_index == 0 else 'LOCO prediction'}\n{target} vs {context}",
                         COLORS[source])
    for index in range(len(contexts)*2, axes.size):
        axes.flat[index].set_visible(False)
    identifiers = truth.id.unique() if "id" in truth else []
    identifier = str(identifiers[0]) if len(identifiers) == 1 else f"IDs: {len(identifiers)}"
    fig.suptitle(f"Cloud {int(truth.cloud_id.iloc[0]):02d} | LOCO {target} | {destination.parent.name} | n={len(truth)}\n{identifier}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def plot_iops_latency(truth, predictions, destination, dpi):
    names = ["Ground truth", "LOCO iops", "LOCO lat"]
    frames = [truth, predictions["iops"], predictions["lat"]]
    xs, xlimits, _ = display_axis([frame.iops for frame in frames], "iops")
    ys, ylimits, _ = display_axis([frame.lat for frame in frames], "lat")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for index, ax in enumerate(axes):
        draw_scatter(ax, xs[index], ys[index], "iops", "lat", xlimits, ylimits, None, None,
                     names[index], COLORS["ground_truth" if index == 0 else "prediction"])
    fig.suptitle(f"Cloud {int(truth.cloud_id.iloc[0]):02d} | {destination.parent.name} | n={len(truth)}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def write_index(output, records, focused):
    sections = []
    for prediction in sorted({record["prediction"] for record in records}):
        subset = [record for record in records if record["prediction"] == prediction]
        targets = [target for target in FEATURES if any(record["target"] == target for record in subset)]
        header = "".join(f"<th>{html.escape(target)}</th>" for target in targets)
        rows = []
        for cloud in sorted({record["cloud"] for record in subset}):
            group = [record for record in subset if record["cloud"] == cloud]
            links = {record["target"]: record["file"] for record in group}
            focus = focused.get((prediction, cloud))
            extra = f'<a href="{html.escape(focus)}">IOPS / latency</a>' if focus else ""
            cells = "".join(f'<td><a href="{html.escape(links[target])}">{html.escape(target)}</a></td>'
                            if target in links else "<td></td>" for target in targets)
            rows.append(f"<tr><th>Cloud {cloud:02d}</th><td>{html.escape(group[0]['experiment_id'])}</td><td>{extra}</td>{cells}</tr>")
        sections.append(f"<h2>{html.escape(prediction)}</h2><div class='scroll'><table><thead><tr><th>Cloud</th><th>Experiment</th><th>IOPS / latency</th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")
    (output / "index.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>LOCO cloud scatters</title><style>body{font:14px system-ui,sans-serif;margin:24px;color:#20252a}"
        "h1{font-size:24px}h2{font-size:18px}.scroll{overflow:auto}table{border-collapse:collapse}"
        "th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}a{color:#1461ad}</style>"
        "<h1>LOCO cloud scatter comparisons</h1>" + "".join(sections) + "</html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clouds-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", nargs="+", default=FEATURES)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--clouds", nargs="+", type=int, help="Optional subset for preview.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate the same report in place.")
    args = parser.parse_args()
    if args.dpi <= 0 or len(set(args.columns)) != len(args.columns):
        raise ValueError("DPI must be positive and columns unique.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise ValueError("Output directory must be empty, or use --overwrite for the same report.")
        previous = json.loads((args.output_dir / "run_info.json").read_text())
        if previous["clouds_dir"] != str(args.clouds_dir.resolve()) or previous["columns"] != args.columns:
            raise ValueError("Overwrite requires the same source folder and columns.")
        if previous.get("selected_clouds") != args.clouds:
            raise ValueError("Overwrite requires the same selected clouds.")
    jobs = []
    for scenario in sorted(args.clouds_dir.glob("loco_*")):
        target = scenario.name.removeprefix("loco_")
        if target not in args.columns:
            continue
        for path in sorted(scenario.glob("*/cloud_*.csv")):
            cloud = int(path.stem.removeprefix("cloud_"))
            if args.clouds and cloud not in args.clouds:
                continue
            truth, prediction = load_aligned_cloud(args.clouds_dir / "ground_truth" / path.name, path, target)
            for column in args.columns:
                display_axis([truth[column], prediction[column]], column)
            jobs.append((target, cloud, path, truth, prediction))
    if not jobs:
        raise FileNotFoundError("No requested per-cloud LOCO prediction CSVs found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, paired = [], {}
    for index, (target, cloud, path, truth, prediction) in enumerate(jobs, 1):
        destination = args.output_dir / f"loco_{target}" / path.parent.name / f"cloud_{cloud:02d}_pairs.png"
        plot_target_pairs(truth, prediction, target, args.columns, destination, args.dpi)
        records.append({"target": target, "cloud": cloud, "prediction": path.parent.name,
                        "rows": len(truth), "pairs": len(args.columns)-1,
                        "experiment_id": str(truth.id.iloc[0]) if "id" in truth else "",
                        "file": str(destination.relative_to(args.output_dir)),
                        "source_csv": str(path.resolve()),
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if target in {"iops", "lat"}:
            paired.setdefault((path.parent.name, cloud), {"truth": truth})[target] = prediction
        print(f"[{index}/{len(jobs)}] {destination.relative_to(args.output_dir)}", flush=True)
    focused = {}
    for (prediction, cloud), group in paired.items():
        if not {"iops", "lat"}.issubset(group):
            continue
        destination = args.output_dir / "iops_latency" / prediction / f"cloud_{cloud:02d}.png"
        plot_iops_latency(group["truth"], group, destination, args.dpi)
        focused[prediction, cloud] = str(destination.relative_to(args.output_dir))
    pd.DataFrame(records).to_csv(args.output_dir / "plot_manifest.csv", index=False)
    write_index(args.output_dir, records, focused)
    (args.output_dir / "run_info.json").write_text(json.dumps({
        "clouds_dir": str(args.clouds_dir.resolve()), "columns": args.columns,
        "selected_clouds": args.clouds,
        "scatter_only": True, "pair_sheets": len(records), "pairs_per_sheet": len(args.columns)-1,
        "iops_latency_comparisons": len(focused), "row_selection": "All rows in each saved cloud; no subsampling.",
        "scales": "Linear, shared GT/prediction limits from all values; no quantile clipping.",
        "categorical_axes": sorted(CATEGORICAL_AXES), "category_positions": "Shared labeled ticks; no jitter.",
        "numeric_axes": "Numeric-valued features retain their values, including integer-valued categories.",
        "divisors": {"iops": 1e3, "lat": 1e6},
        "excluded_columns": ["id", "load_type", "device_type", "offset"],
        "sampling_performed": False, "metrics_computed": False,
    }, indent=2))
    archive = args.output_dir.with_suffix(".zip")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, str(Path(args.output_dir.name) / path.relative_to(args.output_dir)))
    print(f"Index: {args.output_dir / 'index.html'}\nArchive: {archive}", flush=True)


if __name__ == "__main__":
    main()
