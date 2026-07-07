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
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


X_ROTATION = 15

CSV_NAMES = [
    "uniform_5x5.10x10....30x30_100.csv",
    # "uniform_5x5.50x50....300x300_100.csv",
    # "exponential_5x5.50x50....300x300_100.csv",
    # "lognormal_5x5.50x50....300x300_100.csv",
    # "truncnormal_5x5.50x50....300x300_100.csv",
    # "uniform_10x100.200x100....1000x100_100.csv",
    # "uniform_100x10.100x200....100x1000_100.csv",
]

ALGORITHMS = {
    "GFW": {"label": "GFW", "color": "darkgreen", "marker": "^", "order": 0.8},
    "EPM": {"label": "EPM", "color": "darkorange", "marker": "*", "order": 0.6},
    "COMB": {"label": "COMB", "color": "royalblue", "marker": "o", "order": 0.5},
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
    "approx": [("a1", "approx", "dashed")],
    "exact": [("e", "exact", "solid")],
    "both": [("a1", "approx", "dashed"), ("e", "exact", "solid")],
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


def values_for_metric(
    data: pd.DataFrame,
    metric: str,
    algorithm: str,
    suffix: str,
    num_seeds: int,
) -> tuple[pd.Series | None, pd.Series | None]:
    column = f"{METRICS[metric]['prefix']}_{algorithm}_{suffix}"
    std_column = f"{column}-std"
    if column not in data.columns:
        return None, None

    values = pd.to_numeric(data[column], errors="coerce")
    std = pd.to_numeric(data[std_column], errors="coerce") if std_column in data.columns else None
    if metric == "solved":
        values = values / num_seeds
        if std is not None:
            std = std / num_seeds
    return values, std

def plot_csv_pdf(csv_path: Path, output_dir: Path, *, variant: str, num_seeds: int) -> Path | None:
    
    data = pd.read_csv(csv_path)
    if "size" not in data.columns:
        raise ValueError(f"{csv_path} does not contain a 'size' column.")

    data_size = data["size"]
    new_data_size = []
    for i in range(len(data_size)):
        ds = list(data_size[i].split("x"))
        if ds[0] == ds[1]:
            new_data_size.append(f"{ds[0]}")
        else:
            new_data_size.append(data_size[i])
    x_axis = new_data_size
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
    fig.subplots_adjust(
        wspace=-0.5,  # horizontal gap
        hspace=-0.5   # vertical gap
    )
    metric_order = ["iteration", "runningtime", "solved"]
    plotted_any = False

    for ax, metric in zip(axes, metric_order):
        metric_plotted = False
        for algorithm, style in ALGORITHMS.items():
            for suffix, suffix_label, linestyle in VARIANTS[variant]:
                values, std = values_for_metric(data, metric, algorithm, suffix, num_seeds)
                if values is None:
                    continue
                
                if suffix == "e":
                    ax.plot(x_axis, 
                        values, 
                        label=f"{style['label']}: {suffix_label}", 
                        color=style["color"], 
                        marker=style["marker"],
                        markersize=12, 
                        linestyle=linestyle, 
                        linewidth=3,
                        zorder=2 + style["order"])
                    ax.fill_between(x_axis, 
                                    values - std if std is not None else values, 
                                    values + std if std is not None else values,
                                    color=style["color"],
                                    alpha=0.2,
                                    linewidth=3,
                                    edgecolor="none",
                                    zorder=0 + style["order"]
                    )
                else:
                    ax.errorbar(x_axis, 
                                values, 
                                yerr=std if std is not None else None,
                                label=f"{style['label']}: {suffix_label}",
                                color=style["color"],
                                marker=style["marker"],
                                markersize=12,
                                linestyle=linestyle,
                                linewidth=3,
                                elinewidth=2,        # error bar line width
                                capsize=3.5,           # cap length
                                capthick=2,          # cap thickness
                                alpha=0.6,
                                zorder=1 + style["order"]
                    )
                metric_plotted = True
                plotted_any = True

        # ax.set_title(METRICS[metric]["ylabel"], fontsize=16)
        ax.tick_params(axis="both", labelsize=18, rotation=X_ROTATION)
        # ax.set_xlabel("Size of instances", fontsize=16)
        if ax == axes[2]:
            ax.set_ylim([-0.1, 1.1])
        if metric_plotted and ax == axes[0]:
            ax.legend(fontsize=16, loc="upper left")

    # axes[0].set_ylabel("Value", fontsize=16)
    fig.tight_layout()

    if not plotted_any:
        plt.close(fig)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clean_name(csv_path)}_{'vs'.join(ALGORITHMS.keys())}.pdf"
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_csv(csv_path: Path, output_dir: Path, *, variant: str, num_seeds: int) -> list[Path]:
    data = pd.read_csv(csv_path)
    if "size" not in data.columns:
        raise ValueError(f"{csv_path} does not contain a 'size' column.")

    output_paths = []
    pdf_path = plot_csv_pdf(csv_path, output_dir, variant=variant, num_seeds=num_seeds)
    if pdf_path is not None:
        output_paths.append(pdf_path)

    return output_paths


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    print(f"Script directory: {script_dir}")
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

    csv_files = [args.data_dir / name for name in CSV_NAMES]

    for csv_path in csv_files:
        output_paths = plot_csv(csv_path, args.output_dir, variant=args.variant, num_seeds=args.num_seeds)
        print(f"{csv_path}:")
        for output_path in output_paths:
            print(f"  saved {output_path}")


if __name__ == "__main__":
    main()
