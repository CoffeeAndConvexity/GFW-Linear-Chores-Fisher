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


X_ROTATION = 0

CSV_NAMES = [
    # "uniform_5x5.10x10....30x30_100.csv",
    # "lognormal_5x5.10x10....30x30_100.csv",
    # "truncnormal_5x5.10x10....30x30_100.csv",
    # "exponential_5x5.10x10....30x30_100.csv",
    # "randint_5x5.10x10....30x30_100.csv",
    # "uniform_5x5.50x50....300x300_100.csv",
    # "exponential_5x5.50x50....300x300_100.csv",
    # "lognormal_5x5.50x50....300x300_100.csv",
    # "truncnormal_5x5.50x50....300x300_100.csv",
    # "randint_5x5.50x50....300x300_100.csv",
    # ["uniform_10x100.500x100....3000x100_100.csv", 
    #  "exponential_10x100.500x100....3000x100_100.csv", 
    #  "lognormal_10x100.500x100....3000x100_100.csv", 
    #  "truncnormal_10x100.500x100....3000x100_100.csv",
    #  "randint_10x100.500x100....3000x100_100.csv"],
    # ["uniform_100x10.100x500....100x3000_100.csv",
    #  "exponential_100x10.100x500....100x3000_100.csv",
    #  "lognormal_100x10.100x500....100x3000_100.csv",
    #  "truncnormal_100x10.100x500....100x3000_100.csv",
    #  "randint_100x10.100x500....100x3000_100.csv"],
    "bidding_5x5.50x50....300x300_100.csv",
    "biddingwithnoise_5x5.50x50....300x300_100.csv",
]

ALGORITHMS = {
    "GFW": {"label": "GFW", "color": ["darkgreen", "darkblue", "darkred", "darkcyan", "darkmagenta"], "marker": "^", "order": 0.8},
    "EPM": {"label": "EPM", "color": ["darkorange", "orange", "gold", "darkgoldenrod", "peru"], "marker": "*", "order": 0.6},
    # "COMB": {"label": "COMB", "color": ["royalblue", "deepskyblue", "dodgerblue", "cornflowerblue", "mediumblue"], "marker": "o", "order": 0.5},
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
    if isinstance(path, list):
        # concatenate the word before "_" in each path's stem
        stem = "_".join([re.match(r"([A-Za-z]+)_", p.stem).group(1) for p in path])

        p1 = path[0]
        stem1 = p1.stem.strip()
        # take the part between the first "_" and the last "_" in stem1
        match = re.search(r"_(.*)_", stem1)
        if match:
            stem += f"_{match.group(1)}"
    else:
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
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
    fig.subplots_adjust(
        wspace=-0.5,  # horizontal gap
        hspace=-0.5   # vertical gap
    )
    metric_order = ["iteration", "runningtime", "solved"]
    plotted_any = False

    original_csv_path = csv_path
    print(f"Plotting CSV file: {csv_path}")
    for idx, csv_path in enumerate(csv_path if isinstance(csv_path, list) else [csv_path]):
        value_generating_distribution = re.match(r"([A-Za-z]+)_", csv_path.stem).group(1)
        print(csv_path)
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

        for ax, metric in zip(axes, metric_order):
            for algorithm, style in ALGORITHMS.items():
                for suffix, suffix_label, linestyle in VARIANTS[variant]:
                    values, std = values_for_metric(data, metric, algorithm, suffix, num_seeds)
                    if values is None:
                        continue
                    
                    if suffix == "e":
                        ax.plot(x_axis, values, 
                            label=f"{style['label']}: {suffix_label}", 
                            color=style["color"][idx], marker=style["marker"], markersize=12, 
                            linestyle=linestyle, linewidth=3, zorder=2 + style["order"])
                        ax.fill_between(x_axis, values - std if std is not None else values, 
                                        values + std if std is not None else values,
                                        color=style["color"][idx], alpha=0.2, linewidth=3,
                                        edgecolor="none", zorder=0 + style["order"])
                    elif suffix == "a1":
                        ax.errorbar(x_axis, values, yerr=std if std is not None else None,
                                    label=f"{style['label']}: {suffix_label}",
                                    color=style["color"][idx],
                                    marker=style["marker"], markersize=12,
                                    linestyle=linestyle, linewidth=3, elinewidth=2, 
                                    capsize=3.5, capthick=2, alpha=0.6, zorder=1 + style["order"]
                        )
                    
                    plotted_any = True

            # ax.set_title(METRICS[metric]["ylabel"], fontsize=16)
            ax.tick_params(axis="both", labelsize=18, rotation=X_ROTATION)
            # ax.set_xlabel("Size of instances", fontsize=16)
            if ax == axes[2]:
                ax.set_ylim([-0.1, 1.1])
            if ax == axes[0]:
                ax.legend(fontsize=16, loc="upper left")

    # axes[0].set_ylabel("Value", fontsize=16)
    fig.tight_layout()

    if not plotted_any:
        plt.close(fig)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{clean_name(original_csv_path)}_{'vs'.join(ALGORITHMS.keys())}.pdf"
    print(f"Saving plot to {output_path}")
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    return output_path

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

    csv_files = []
    for names in CSV_NAMES:
        if isinstance(names, list):
            new_list = []
            for name in names:
                new_list.append(args.data_dir / name)
            csv_files.append(new_list)
        else:
            csv_files.append(args.data_dir / names)

    for csv_path in csv_files:
        output_paths = plot_csv_pdf(csv_path, args.output_dir, variant=args.variant, num_seeds=args.num_seeds)


if __name__ == "__main__":
    main()
