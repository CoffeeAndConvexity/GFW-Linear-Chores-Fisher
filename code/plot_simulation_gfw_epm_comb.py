# -*- coding: utf-8 -*-
"""
Plot GFW vs EPM vs COMB simulation results.

The script reads CSV files produced by run_simulation.py and saves three plots
for each input file:
  - running time
  - number of iterations
  - solved-instance ratio
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = {
    "GFW": {"label": "GFW", "color": "darkgreen", "marker": "^"},
    "EPM": {"label": "EPM", "color": "darkorange", "marker": "*"},
    "COMB": {"label": "COMB", "color": "royalblue", "marker": "o"},
}

METRICS = {
    "runningtime": {
        "prefix": "runningtime",
        "ylabel": "Running time in seconds",
        "filename_prefix": "rt",
    },
    "iteration": {
        "prefix": "iteration",
        "ylabel": "# iterations to reach CE",
        "filename_prefix": "ni",
    },
    "solved": {
        "prefix": "solved",
        "ylabel": "Ratio of solved instances",
        "filename_prefix": "sr",
    },
}

VARIANTS = {
    "approx": [("a1", "Approximate", "dashed")],
    "exact": [("e", "Exact", "solid")],
    "both": [("a1", "Approximate", "dashed"), ("e", "Exact", "solid")],
}


def clean_name(path: Path) -> str:
    stem = path.stem.strip()
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem.strip("_")


def has_comb_columns(csv_path: Path) -> bool:
    try:
        columns = pd.read_csv(csv_path, nrows=0).columns
    except Exception:
        return False
    return any(col.startswith("runningtime_COMB_") for col in columns)


def discover_csv_files(data_dir: Path) -> list[Path]:
    csv_files = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        if "welfare" in csv_path.stem:
            continue
        if has_comb_columns(csv_path):
            csv_files.append(csv_path)
    return csv_files


def values_for_metric(data: pd.DataFrame, metric: str, algorithm: str, suffix: str, num_seeds: int) -> pd.Series | None:
    column = f"{METRICS[metric]['prefix']}_{algorithm}_{suffix}"
    if column not in data.columns:
        return None

    values = pd.to_numeric(data[column], errors="coerce")
    if metric == "solved":
        values = values / num_seeds
    return values


def plot_metric(
    data: pd.DataFrame,
    csv_path: Path,
    output_dir: Path,
    *,
    metric: str,
    variant: str,
    num_seeds: int,
) -> Path | None:
    plotted_any = False
    plt.figure(figsize=(8, 5.5))

    for algorithm, style in ALGORITHMS.items():
        for suffix, suffix_label, linestyle in VARIANTS[variant]:
            values = values_for_metric(data, metric, algorithm, suffix, num_seeds)
            if values is None:
                continue

            label = style["label"] if variant != "both" else f"{style['label']}: {suffix_label}"
            plt.plot(
                data["size"],
                values,
                label=label,
                marker=style["marker"],
                color=style["color"],
                linestyle=linestyle,
                linewidth=2.5,
                markersize=10,
            )
            plotted_any = True

    if not plotted_any:
        plt.close()
        return None

    plt.xticks(fontsize=12, rotation=15)
    plt.yticks(fontsize=12)
    plt.xlabel("Size of instances", fontsize=14)
    plt.ylabel(METRICS[metric]["ylabel"], fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{METRICS[metric]['filename_prefix']}_{clean_name(csv_path)}_{variant}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_csv(csv_path: Path, output_dir: Path, *, variant: str, num_seeds: int) -> list[Path]:
    data = pd.read_csv(csv_path)
    if "size" not in data.columns:
        raise ValueError(f"{csv_path} does not contain a 'size' column.")

    output_paths = []
    for metric in METRICS:
        output_path = plot_metric(
            data,
            csv_path,
            output_dir,
            metric=metric,
            variant=variant,
            num_seeds=num_seeds,
        )
        if output_path is not None:
            output_paths.append(output_path)
    return output_paths


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "data"
    default_output_dir = script_dir.parent / "figures"

    parser = argparse.ArgumentParser(
        description="Plot simulation results comparing GFW, EPM, and COMB."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="CSV files to plot. If omitted, all non-welfare CSVs with COMB columns in ../data are used.",
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="both")
    args = parser.parse_args()

    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")

    csv_files = args.csv_files or discover_csv_files(args.data_dir)
    if not csv_files:
        raise ValueError(f"No simulation CSV files with COMB columns found in {args.data_dir}.")

    for csv_path in csv_files:
        output_paths = plot_csv(csv_path, args.output_dir, variant=args.variant, num_seeds=args.num_seeds)
        print(f"{csv_path}:")
        for output_path in output_paths:
            print(f"  saved {output_path}")


if __name__ == "__main__":
    main()
