#!/usr/bin/env python3
"""Plot tighter windows from saved cleaning masks without rerunning the detector."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.clean_dataset_outliers import save_comparison
import numpy as np
import pandas as pd

WINDOWS = {
    "ssd_sequential": ((0., 70.), (0., 100.)),
    "ssd_random": ((0., 350.), (0., 6.)),
    "cache": ((0., 400.), (0., 5.)),
}


def plot_saved(source_root: Path, dataset: str, bounds: tuple) -> dict:
    directory = source_root / dataset
    info = json.loads((directory / "run_info.json").read_text())
    source = Path(info["source_train"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != info["source_sha256"]:
        raise ValueError(f"Original CSV changed since cleaning: {source}")
    frame = pd.read_csv(source, low_memory=False)
    decisions = pd.read_csv(directory / "row_decisions.csv")
    if len(frame) != len(decisions) or not np.array_equal(decisions.source_row, np.arange(len(frame))):
        raise ValueError(f"Invalid saved row mapping: {dataset}")
    if decisions.keep.dtype != bool or len(frame) != info["rows_before"] or decisions.keep.sum() != info["rows_after"]:
        raise ValueError(f"Saved mask does not match cleaning metadata: {dataset}")
    keep = decisions.keep.to_numpy()
    save_comparison(frame, keep, directory / "before_after_zoom.png", dataset, info["epsilon"], bounds=bounds)
    xy = frame[["iops", "lat"]].to_numpy(float) / [1e3, 1e6]
    inside = ((xy[:, 0] >= bounds[0][0]) & (xy[:, 0] <= bounds[0][1]) &
              (xy[:, 1] >= bounds[1][0]) & (xy[:, 1] <= bounds[1][1]))
    record = {"dataset": dataset, "x_limits": bounds[0], "y_limits": bounds[1],
              "original_visible": int(inside.sum()), "cleaned_visible": int((inside & keep).sum()),
              "original_outside": int((~inside).sum()), "cleaned_outside": int((~inside & keep).sum()),
              "detector_rerun": False}
    print(record, flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "reports/outlier_cleaning")
    parser.add_argument("--datasets", nargs="+", choices=list(WINDOWS), default=list(WINDOWS))
    parser.add_argument("--view", nargs=5, action="append", metavar=("DATASET", "XMIN", "XMAX", "YMIN", "YMAX"),
                        help="Override a dataset window in IOPS/1000 and latency/1000000 plot units.")
    args = parser.parse_args()
    windows = dict(WINDOWS)
    for dataset, xmin, xmax, ymin, ymax in args.view or []:
        if dataset not in windows:
            parser.error(f"Unknown dataset: {dataset}")
        try:
            values = np.asarray([xmin, xmax, ymin, ymax], dtype=float)
        except ValueError:
            parser.error("Axis limits must be numbers.")
        if not np.isfinite(values).all() or not values[0] < values[1] or not values[2] < values[3]:
            parser.error("Axis limits must be finite and increasing.")
        windows[dataset] = (tuple(values[:2]), tuple(values[2:]))
    records = [plot_saved(args.source_root, dataset, windows[dataset]) for dataset in args.datasets]
    directory = args.source_root / "zoom"
    directory.mkdir(exist_ok=True)
    (directory / "plot_windows.json").write_text(json.dumps(records, indent=2) + "\n")
    body = "<!doctype html><meta charset='utf-8'><title>Outlier cleaning: zoom</title>"
    body += "<style>body{font:15px Arial;max-width:1300px;margin:24px auto;padding:0 16px}img{max-width:100%}</style>"
    body += "<h1>Original left / cleaned right: tighter windows</h1><p>Display only; cleaning masks and CSVs unchanged. Counts outside each window are annotated.</p>"
    for dataset in args.datasets:
        body += f"<h2>{dataset}</h2><img src='../{dataset}/before_after_zoom.png'>"
    (directory / "index.html").write_text(body)
    print(f"Gallery: {directory / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
