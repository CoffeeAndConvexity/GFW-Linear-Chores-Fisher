from __future__ import annotations

import argparse
import random
from collections.abc import Iterable
from typing import Sequence

import numpy as np

from balance_allocation import balance_allocation


def _rounded(values: Sequence[float]) -> list[float]:
    return [round(v, 6) for v in values]


def _print_matrix(name: str, matrix: Sequence[Sequence[float]]) -> None:
    print(f"{name}=")
    for row in matrix:
        print("  ", _rounded(row))


def _mpb_edges(p: Sequence[float], d: Sequence[Sequence[float]]) -> set[tuple[int, int]]:
    n_agents = len(d)
    n_chores = len(p)
    eps = 1e-12
    mpb = [min(d[i][j] / p[j] for j in range(n_chores)) for i in range(n_agents)]
    return {
        (i, j)
        for i in range(n_agents)
        for j in range(n_chores)
        if abs((d[i][j] / p[j]) - mpb[i]) <= eps * max(1.0, abs(mpb[i]))
    }


def _money_flows(x: Sequence[Sequence[float]], p: Sequence[float]) -> list[list[float]]:
    return [[x[i][j] * p[j] for j in range(len(p))] for i in range(len(x))]


def _column_sums(x: Sequence[Sequence[float]]) -> list[float]:
    return [sum(row[j] for row in x) for j in range(len(x[0]))]


def _print_instance(name: str, p: Sequence[float], d: Sequence[Sequence[float]]) -> None:
    print(name)
    print(f"n_agents={len(d)}, n_chores={len(p)}")
    _print_matrix("disutilities", d)
    print("prices=", _rounded(p))
    print("MPB_edges=", sorted(_mpb_edges(p, d)))

    x, e, s_set = balance_allocation(p, d)

    _print_matrix("allocation", x)
    print("column_sums=", _rounded(_column_sums(x)))
    print("earnings=", _rounded(e))
    _print_matrix("money_flows", _money_flows(x, p))
    print("S=", sorted(s_set))
    
    print("-" * 60)

def _random_instances(
    *,
    seed: int,
    count: int,
) -> Iterable[tuple[str, list[float], list[list[float]]]]:
    
    rng = random.Random(seed)

    for index in range(1, count + 1):
        n_agents = 10
        n_chores = 10
        d = [
            [rng.randint(1, 10) for _ in range(n_chores)]
            for _ in range(n_agents)
        ]
        p = np.amin(d, axis=0).tolist()  # Set prices to the minimum disutility for each chore
        
        yield f"Generating random instance {index} with {n_agents} agents and {n_chores} chores...", p, d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the balance-allocation test instances and their full outputs."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-count", type=int, default=30)
    args = parser.parse_args()

    print("---------")

    for name, p, d in _random_instances(seed=args.seed, count=args.random_count):
        _print_instance(name, p, d)


if __name__ == "__main__":
    # print("This script generates random instances of the balance-allocation problem and prints their details.")
    main()
