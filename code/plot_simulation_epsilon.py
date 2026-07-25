# -*- coding: utf-8 -*-
"""
Plot number of iterations vs epsilon.

The script reads CSV files produced by run_simulation_epsilon.py and saves one
figure comparing algorithms/variants on the iteration metric.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FixedFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np
import pandas as pd


X_ROTATION = 0

SIZE = (300, 300)
EPSILON_LIST = [5e-1, 5e-2, 5e-3, 5e-4]
NUM_SEEDS = 100

ALGORITHMS = {
    "GFW": {
        "label": "GFW",
        "color": ["darkgreen", "darkblue", "darkred", "darkcyan", "darkmagenta"],
        "marker": "^",
        "order": 0.8,
    },
    "EPM": {
        "label": "EPM",
        "color": ["darkorange", "orange", "gold", "darkgoldenrod", "peru"],
        "marker": "*",
        "order": 0.6,
    },
    # "COMB": {"label": "COMB", "color": ["royalblue", "deepskyblue", "dodgerblue", "cornflowerblue", "mediumblue"], "marker": "o", "order": 0.5},
}

VARIANTS = {
    "approx": [("a1", "approx", "dashed")],
    "exact": [("e", "exact", "solid")],
    "both": [("a1", "approx", "dashed"), ("e", "exact", "solid")],
}


def csv_path_for_epsilon(
    data_dir: Path,
    distribution: str,
    size: tuple[int, int],
    num_seeds: int,
    epsilon: float,
) -> Path:
    epsilon_str = f"{epsilon:g}"
    return data_dir / f"{distribution}_{size[0]}x{size[1]}_{num_seeds}_{epsilon_str}.csv"


def read_iteration_stats(
    csv_paths: dict[float, Path],
    size_label: str,
) -> tuple[dict[str, dict[str, dict[str, list[float]]]], dict[str, dict[str, list[float]]]]:
    results: dict[str, dict[str, dict[str, list[float]]]] = {}
    solved_results: dict[str, dict[str, list[float]]] = {}
    for algorithm in ALGORITHMS:
        results[algorithm] = {
            "a1": {"values": [], "std": []},
            "e": {"values": [], "std": []},
        }
        solved_results[algorithm] = {
            "a1": [],
            "e": [],
        }

    for csv_path in csv_paths.values():
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        data = pd.read_csv(csv_path)
        if "size" not in data.columns:
            raise ValueError(f"{csv_path} does not contain a 'size' column.")

        row = data.loc[data["size"].astype(str) == size_label]
        if row.empty:
            row = data.iloc[[0]]

        for algorithm in ALGORITHMS:
            for suffix in ("a1", "e"):
                col = f"iteration_{algorithm}_{suffix}"
                std_col = f"{col}-std"

                value = (
                    pd.to_numeric(row[col], errors="coerce").iloc[0]
                    if col in data.columns
                    else np.nan
                )
                std = (
                    pd.to_numeric(row[std_col], errors="coerce").iloc[0]
                    if std_col in data.columns
                    else np.nan
                )

                results[algorithm][suffix]["values"].append(value)
                results[algorithm][suffix]["std"].append(std)

                solved_col = f"solved_{algorithm}_{suffix}"
                solved_value = (
                    pd.to_numeric(row[solved_col], errors="coerce").iloc[0]
                    if solved_col in data.columns
                    else np.nan
                )
                solved_results[algorithm][suffix].append(solved_value / NUM_SEEDS if np.isfinite(solved_value) else np.nan)

    return results, solved_results


def choose_manual_y_ticks(values: np.ndarray, yscale: str) -> list[float]:
    finite_values = values[np.isfinite(values)]
    if yscale == "log":
        finite_values = finite_values[finite_values > 0]
    if finite_values.size == 0:
        return []

    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))

    if yscale == "log":
        nice_ticks = np.array(
            [
                1, 2, 3, 4, 5, 6, 7, 8, 9,
                10, 12, 15, 20, 25, 30, 40, 50, 60, 80,
                100, 120, 150, 200, 250, 300, 400, 500, 600, 800,
                1000,
            ],
            dtype=float,
        )
        lower = y_min * 0.9
        upper = y_max * 1.1
        ticks = nice_ticks[(nice_ticks >= lower) & (nice_ticks <= upper)]
        if ticks.size == 0:
            ticks = np.array([y_min, np.sqrt(y_min * y_max), y_max], dtype=float)
        elif ticks.size > 3:
            targets = np.geomspace(y_min, y_max, 3)
            selected: list[float] = []
            for target in targets:
                nearest = float(ticks[np.argmin(np.abs(np.log(ticks) - np.log(target)))])
                if nearest not in selected:
                    selected.append(nearest)
            ticks = np.array(selected, dtype=float)
        return ticks.tolist()

    lower = np.floor(y_min)
    upper = np.ceil(y_max)
    if upper <= lower:
        return [lower]
    return np.linspace(lower, upper, 3, dtype=float).tolist()


def plot_iterations_vs_epsilon(
    epsilons: list[float],
    results: dict[str, dict[str, dict[str, list[float]]]],
    solved_results: dict[str, dict[str, list[float]]],
    output_path: Path,
    variant: str,
    xmode: str,
    xscale: str,
    yscale: str,
) -> None:
    fig, (ax_iter, ax_solved) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    epsilon_array = np.asarray(epsilons, dtype=float)
    x_axis = epsilon_array if xmode == "epsilon" else 1.0 / epsilon_array
    x_tick_labels = [f"{e:g}" for e in epsilons] if xmode == "epsilon" else [f"{(1.0 / e):g}" for e in epsilons]
    plotted_y_values: list[float] = []

    plotted_any_iter = False
    plotted_any_solved = False
    exact_solved_reference: dict[str, float] = {}
    for algorithm, style in ALGORITHMS.items():
        for suffix, suffix_label, linestyle in VARIANTS[variant]:
            y_values = np.asarray(results[algorithm][suffix]["values"], dtype=float)
            y_std = np.asarray(results[algorithm][suffix]["std"], dtype=float)
            y_solved = np.asarray(solved_results[algorithm][suffix], dtype=float)

            valid = np.isfinite(y_values)
            color = style["color"][0]

            if suffix == "a1" and valid.any():
                x = x_axis[valid]
                y = y_values[valid]
                std = y_std[valid]
                plotted_y_values.extend(y.tolist())

                if suffix == "e":
                    ax_iter.plot(
                        x,
                        y,
                        label=f"{style['label']}: {suffix_label}",
                        color=color,
                        marker=style["marker"],
                        markersize=12,
                        linestyle=linestyle,
                        linewidth=3,
                        zorder=2 + style["order"],
                    )
                    safe_std = np.where(np.isfinite(std), std, 0.0)
                    ax_iter.fill_between(
                        x,
                        y - safe_std,
                        y + safe_std,
                        color=color,
                        alpha=0.2,
                        linewidth=3,
                        edgecolor="none",
                        zorder=0 + style["order"],
                    )
                else:
                    safe_std = np.where(np.isfinite(std), std, np.nan)
                    ax_iter.errorbar(
                        x,
                        y,
                        yerr=safe_std,
                        label=f"{style['label']}: {suffix_label}",
                        color=color,
                        marker=style["marker"],
                        markersize=12,
                        linestyle=linestyle,
                        linewidth=3,
                        elinewidth=2,
                        capsize=3.5,
                        capthick=2,
                        alpha=0.6,
                        zorder=1 + style["order"],
                    )
                plotted_any_iter = True

            valid_solved = np.isfinite(y_solved)
            if suffix == "a1" and valid_solved.any():
                x_solved = x_axis[valid_solved]
                y_solved_valid = y_solved[valid_solved]
                ax_solved.plot(
                    x_solved,
                    y_solved_valid,
                    label=f"{style['label']}: {suffix_label}",
                    color=color,
                    marker=style["marker"],
                    markersize=12,
                    linestyle=linestyle,
                    linewidth=3,
                    alpha=0.85,
                )
                plotted_any_solved = True

            y_exact_solved = np.asarray(solved_results[algorithm]["e"], dtype=float)
            valid_exact = np.isfinite(y_exact_solved)
            if valid_exact.any():
                exact_solved_reference[algorithm] = float(np.mean(y_exact_solved[valid_exact]))

    if not plotted_any_iter and not plotted_any_solved:
        plt.close(fig)
        raise ValueError("No valid series were found to plot.")

    manual_y_ticks = choose_manual_y_ticks(np.asarray(plotted_y_values, dtype=float), yscale)

    for ax in (ax_iter, ax_solved):
        ax.set_xscale(xscale)
        ax.set_xlabel("epsilon" if xmode == "epsilon" else "1 / epsilon", fontsize=18)
        ax.xaxis.set_major_locator(FixedLocator(x_axis))
        ax.xaxis.set_major_formatter(FixedFormatter(x_tick_labels))

        if xscale == "log":
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator())

        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.35)
        ax.tick_params(axis="both", labelsize=18, rotation=X_ROTATION)

    ax_iter.set_yscale(yscale)
    ax_iter.set_ylabel("# iterations to reach CE", fontsize=18)
    if manual_y_ticks:
        ax_iter.yaxis.set_major_locator(FixedLocator(manual_y_ticks))
        ax_iter.yaxis.set_major_formatter(FixedFormatter([f"{tick:g}" for tick in manual_y_ticks]))

    if yscale == "log":
        ax_iter.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax_iter.yaxis.set_minor_formatter(NullFormatter())
    else:
        if not manual_y_ticks:
            ax_iter.yaxis.set_major_locator(MaxNLocator(nbins=10))
        ax_iter.yaxis.set_minor_locator(AutoMinorLocator())

    ax_solved.set_yscale("linear")
    ax_solved.set_ylabel("Ratio of solved instances", fontsize=18)
    ax_solved.set_ylim(-0.05, 1.05)
    solved_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax_solved.yaxis.set_major_locator(FixedLocator(solved_ticks))
    ax_solved.yaxis.set_major_formatter(FixedFormatter([f"{tick:g}" for tick in solved_ticks]))
    ax_solved.yaxis.set_minor_locator(AutoMinorLocator())

    for algorithm, exact_ratio in exact_solved_reference.items():
        style = ALGORITHMS[algorithm]
        ax_solved.axhline(
            y=exact_ratio,
            color=style["color"][0],
            linestyle=":",
            linewidth=2,
            alpha=0.9,
            label=f"{style['label']}: exact CE",
        )

    ax_iter.legend(fontsize=16, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir.parent / "data"
    default_output_dir = script_dir.parent / "figures"

    parser = argparse.ArgumentParser(
        description="Plot number of iterations vs epsilon."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--distribution", type=str, default="lognormal")
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--size", type=str, default=f"{SIZE[0]}x{SIZE[1]}")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="approx")
    parser.add_argument("--xmode", choices=["epsilon", "inv_epsilon"], default="inv_epsilon")
    parser.add_argument("--xscale", choices=["linear", "log"], default="log")
    parser.add_argument("--yscale", choices=["linear", "log"], default="log")
    parser.add_argument("--epsilons", type=float, nargs="*", default=EPSILON_LIST)
    args = parser.parse_args()

    size_tokens = args.size.lower().split("x")
    if len(size_tokens) != 2:
        raise ValueError("--size must be formatted as NxM, e.g. 300x300")

    size_tuple = (int(size_tokens[0]), int(size_tokens[1]))
    epsilons = list(args.epsilons)

    csv_paths = {
        epsilon: csv_path_for_epsilon(
            args.data_dir,
            args.distribution,
            size_tuple,
            args.num_seeds,
            epsilon,
        )
        for epsilon in epsilons
    }

    results, solved_results = read_iteration_stats(csv_paths, size_label=args.size)
    epsilon_part = "_".join(f"{e:g}" for e in epsilons)
    xmode_label = "epsilon"
    output_path = (
        args.output_dir
        / f"{args.distribution}_{args.size}_{args.num_seeds}_iterations_vs_{xmode_label}_{args.xscale}x_{args.yscale}y_{epsilon_part}_{'vs'.join(ALGORITHMS.keys())}.pdf"
    )

    print("Using CSV files:")
    for epsilon, path in csv_paths.items():
        print(f"  epsilon={epsilon:g}: {path}")

    plot_iterations_vs_epsilon(
        epsilons=epsilons,
        results=results,
        solved_results=solved_results,
        output_path=output_path,
        variant=args.variant,
        xmode=args.xmode,
        xscale=args.xscale,
        yscale=args.yscale,
    )
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
