"""Regenerate every artifact in ``figures/`` from the committed CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_simulation import plot_csv_pdf
from plot_simulation_epsilon import (
    ALGORITHMS as EPSILON_ALGORITHMS,
    plot_iterations_vs_epsilon,
    read_iteration_stats,
)
from plot_welfare import plot_welfare


SMALL_COMPARISONS = [
    "uniform_5x5.10x10....30x30_100.csv",
    "exponential_5x5.10x10....30x30_100.csv",
    "lognormal_5x5.10x10....30x30_100.csv",
    "truncnormal_5x5.10x10....30x30_100.csv",
    "randint_5x5.10x10....30x30_100.csv",
]

LARGE_SQUARE_COMPARISONS = [
    "uniform_5x5.50x50....300x300_100.csv",
    "exponential_5x5.50x50....300x300_100.csv",
    "lognormal_5x5.50x50....300x300_100.csv",
    "truncnormal_5x5.50x50....300x300_100.csv",
    "randint_5x5.50x50....300x300_100.csv",
    "uniformha_5x5.50x50....300x300_100.csv",
    "bidding_5x5.50x50....300x300_100.csv",
    "biddingwithnoise_5x5.50x50....300x300_100.csv",
]

RECTANGULAR_GROUPS = [
    [
        f"{distribution}_10x100.500x100....3000x100_100.csv"
        for distribution in ("uniform", "exponential", "lognormal", "truncnormal", "randint")
    ],
    [
        f"{distribution}_100x10.100x500....100x3000_100.csv"
        for distribution in ("uniform", "exponential", "lognormal", "truncnormal", "randint")
    ],
]

WELFARE_FILES = [
    "exponential_5x5.10x10....80x80_welfare_10.csv",
    "lognormal_5x5.10x10....80x80_welfare_10.csv",
    "randint10_3x3.4x4....50x50_welfare_1.csv",
    "randint10_5x5.10x10....80x80_welfare_10.csv",
    "uniform_3x3.4x4....50x50_welfare_1.csv",
    "uniform_5x5.10x10....80x80_welfare_10.csv",
]

EPSILON_DISTRIBUTIONS = ["uniform", "lognormal", "truncnormal"]
EPSILONS = [0.5, 0.05, 0.005, 0.0005]


def require_files(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required input is missing: {missing[0]}")


def reproduce(data_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for name in SMALL_COMPARISONS + LARGE_SQUARE_COMPARISONS:
        csv_path = data_dir / name
        require_files([csv_path])
        output = plot_csv_pdf(csv_path, output_dir, variant="both", num_seeds=100)
        if output is not None:
            outputs.append(output)

    for names in RECTANGULAR_GROUPS:
        csv_paths = [data_dir / name for name in names]
        require_files(csv_paths)
        output = plot_csv_pdf(csv_paths, output_dir, variant="both", num_seeds=100)
        if output is not None:
            outputs.append(output)

    for name in WELFARE_FILES:
        csv_path = data_dir / name
        require_files([csv_path])
        outputs.append(plot_welfare(csv_path, output_dir))

    size = (300, 300)
    num_seeds = 100
    for distribution in EPSILON_DISTRIBUTIONS:
        csv_paths = {
            epsilon: data_dir / f"{distribution}_{size[0]}x{size[1]}_{num_seeds}_{epsilon:g}.csv"
            for epsilon in EPSILONS
        }
        require_files(list(csv_paths.values()))
        results, solved_results = read_iteration_stats(
            csv_paths,
            size_label=f"{size[0]}x{size[1]}",
            num_seeds=num_seeds,
        )
        epsilon_part = "_".join(f"{epsilon:g}" for epsilon in EPSILONS)
        output = output_dir / (
            f"{distribution}_{size[0]}x{size[1]}_{num_seeds}_iterations_vs_epsilon_"
            f"logx_logy_{epsilon_part}_{'vs'.join(EPSILON_ALGORITHMS.keys())}.pdf"
        )
        plot_iterations_vs_epsilon(
            epsilons=EPSILONS,
            results=results,
            solved_results=solved_results,
            output_path=output,
            variant="approx",
            xmode="epsilon",
            xscale="log",
            yscale="log",
        )
        outputs.append(output)

    return outputs


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "figures")
    args = parser.parse_args()

    outputs = reproduce(args.data_dir, args.output_dir)
    print(f"Generated {len(outputs)} figures in {args.output_dir}")


if __name__ == "__main__":
    main()
