#!/usr/bin/env python3
"""Print non-constant CSV columns, one name per line."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {args.csv}")

        excluded = set(args.exclude)
        first_values: dict[str, str | None] = {
            column: None for column in reader.fieldnames if column not in excluded
        }
        nonconstant: set[str] = set()

        for row in reader:
            for column in first_values:
                value = row[column]
                if first_values[column] is None:
                    first_values[column] = value
                elif value != first_values[column]:
                    nonconstant.add(column)

    for column in reader.fieldnames:
        if column in nonconstant and column not in excluded:
            print(column)


if __name__ == "__main__":
    main()
