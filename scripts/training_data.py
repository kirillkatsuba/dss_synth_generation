#!/usr/bin/env python3
"""Resolve shared train/test CSVs; relative paths are repository-relative."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dataset_paths(config: Path, dataset: str) -> dict[str, Path]:
    records = json.loads(config.read_text())
    if dataset not in records:
        raise ValueError(f"Unknown dataset {dataset!r} in {config}")
    paths = {}
    for field in ("train_csv", "test_csv"):
        value = records[dataset].get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing {field} for {dataset!r} in {config}")
        path = (ROOT / value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[field] = path
    if paths["train_csv"] == paths["test_csv"]:
        raise ValueError("Training and test CSVs must be different files.")
    return paths


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-config", type=Path,
                        help="Shared train/test path config, e.g. dataset/cleaned/datasets.json.")
    parser.add_argument("--dataset", help="Dataset key in --dataset-config.")


def resolve_dataset_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser,
                              train_field: str, test_field: str) -> argparse.Namespace:
    if (args.dataset_config is None) != (args.dataset is None):
        parser.error("--dataset-config and --dataset must be supplied together.")
    if args.dataset_config is not None:
        paths = dataset_paths(args.dataset_config, args.dataset)
        setattr(args, train_field, paths["train_csv"])
        setattr(args, test_field, paths["test_csv"])
    return args


def config_path(path: Path) -> str:
    """Keep configs portable when their data lives inside the repository."""
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--field", choices=["train_csv", "test_csv"], required=True)
    args = parser.parse_args()
    print(dataset_paths(args.config, args.dataset)[args.field])


if __name__ == "__main__":
    main()
