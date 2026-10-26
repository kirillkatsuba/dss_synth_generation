#!/usr/bin/env python3
"""Export saved LOCO predictions using fixed ground-truth cloud memberships."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.explore_iops_latency_clouds import load_annotations


def validate_alignment(truth: pd.DataFrame, prediction: pd.DataFrame, target: str) -> None:
    if target not in truth:
        raise ValueError(f"Unknown LOCO target: {target}")
    if len(prediction) != len(truth) or list(prediction.columns) != list(truth.columns):
        raise ValueError("Prediction rows/columns differ from ground truth; refusing positional alignment.")
    for column in truth.columns:
        if column == target:
            continue
        expected, actual = truth[column], prediction[column]
        if pd.api.types.is_numeric_dtype(expected):
            actual = pd.to_numeric(actual, errors="raise")
            matches = np.isclose(expected, actual, rtol=1e-12, atol=1e-9, equal_nan=True)
        else:
            matches = expected.eq(actual) | (expected.isna() & actual.isna())
        if not np.asarray(matches).all():
            raise ValueError(f"Known context differs in {column}; source-row alignment is not verified.")


def export_groups(frame, labels, groups, directory, target=None, truth=None):
    annotated = frame.copy()
    if target is not None:
        reference_name = f"ground_truth_{target}"
        if reference_name in annotated:
            raise ValueError(f"Reserved reference column already exists: {reference_name}")
        annotated[reference_name] = truth[target].to_numpy()
    annotated.insert(0, "source_row", np.arange(len(frame)))
    annotated.insert(1, "cloud_id", labels)
    annotated.insert(2, "cloud_name", [groups.get(i, {}).get("name", "") for i in labels])
    directory.mkdir(parents=True, exist_ok=True)
    annotated.loc[labels > 0].to_csv(directory / "all_clouds.csv", index=False)
    for identifier in groups:
        annotated.loc[labels == identifier].to_csv(directory / f"cloud_{identifier:02d}.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    truth = pd.read_csv(args.data_csv, low_memory=False)
    info = json.loads((args.results_dir / "run_info.json").read_text())
    saved_truth = pd.read_csv(args.results_dir / "evaluation_truth.csv", low_memory=False)
    pd.testing.assert_frame_equal(truth, saved_truth, check_dtype=False, check_exact=True)
    if info["rows"] != len(truth):
        raise ValueError("Saved evaluation row count does not match ground truth.")
    source_hash = hashlib.sha256(args.data_csv.read_bytes()).hexdigest()
    labels, groups = load_annotations(args.annotations, {"sha256": source_hash, "rows": range(len(truth))})
    scenarios = sorted(args.results_dir.glob("loco_*"))
    predictions = []
    # Validate every input before creating the exports; never infer cloud labels from predictions.
    for scenario in scenarios:
        if not scenario.is_dir():
            continue
        target = scenario.name.removeprefix("loco_")
        paths = sorted(scenario.glob("trial_*.csv"))
        if (scenario / "point_estimate.csv").is_file():
            paths.append(scenario / "point_estimate.csv")
        if not paths:
            raise FileNotFoundError(f"No saved predictions in {scenario}")
        for path in paths:
            frame = pd.read_csv(path, low_memory=False)
            validate_alignment(truth, frame, target)
        predictions.extend((target, path) for path in paths)
    if not predictions:
        raise FileNotFoundError("No LOCO prediction CSVs were found.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty to avoid mixing export versions.")
    export_groups(truth, labels, groups, args.output_dir / "ground_truth")
    manifest = []
    for target, path in predictions:
        frame = pd.read_csv(path, low_memory=False)
        destination = args.output_dir / f"loco_{target}" / path.stem
        export_groups(frame, labels, groups, destination, target, truth)
        manifest.append({"scenario": f"loco_{target}", "prediction": path.stem,
                         "source_csv": str(path.resolve()), "rows": int((labels > 0).sum()),
                         "clouds": len(groups), "directory": str(destination.relative_to(args.output_dir))})
        print(f"Exported loco_{target}/{path.stem}: {len(groups)} clouds", flush=True)
    pd.DataFrame(manifest).to_csv(args.output_dir / "manifest.csv", index=False)
    (args.output_dir / "run_info.json").write_text(json.dumps({
        "results_dir": str(args.results_dir.resolve()), "checkpoint": info.get("checkpoint"),
        "checkpoint_epoch": info.get("checkpoint_epoch"), "source_csv": str(args.data_csv.resolve()),
        "source_sha256": source_hash, "annotations": str(args.annotations.resolve()),
        "source_rows": len(truth), "exported_rows_per_prediction": int((labels > 0).sum()),
        "clouds": len(groups), "loco_targets": sorted({target for target, _ in predictions}),
        "membership": "Fixed ground-truth annotations, not reclustered predictions.",
        "alignment": "Exact saved truth match, full row counts, known-context validation for every prediction.",
        "source_row": "Zero-based row index in source CSV, excluding header.",
        "reference_column": "ground_truth_<target>; original target column contains the prediction.",
        "sampling_performed": False,
    }, indent=2))
    archive = args.output_dir.with_suffix(".zip")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in sorted(args.output_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, arcname=str(Path(args.output_dir.name) / path.relative_to(args.output_dir)))
    print(f"Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
