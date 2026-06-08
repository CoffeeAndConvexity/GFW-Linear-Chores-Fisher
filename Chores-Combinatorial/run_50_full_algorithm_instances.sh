#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -u - <<'PY'
from __future__ import annotations

import random
from time import perf_counter
from statistics import mean, median

from full_algorithm import full_algorithm


NUM_INSTANCES = 50
SEED = 42
MIN_AGENTS = 20
MAX_AGENTS = 50
MIN_CHORES = 20
MAX_CHORES = 50
LOW = 1
HIGH = 1001
TOL = 1e-6
MAX_ITER = 10_000


def make_instance(
    rng: random.Random,
    n_agents: int,
    n_chores: int,
) -> list[list[float]]:

    return [
        [float(rng.randint(LOW, HIGH)) for _ in range(n_chores)]
        for _ in range(n_agents)
    ]


def main() -> None:
    rng = random.Random(SEED)
    iterations: list[int] = []
    elapsed_times: list[float] = []
    failures: list[tuple[int, int, int, str]] = []
    start_all = perf_counter()

    print(f"Running {NUM_INSTANCES} random full-algorithm instances")
    print(
        "settings: "
        f"seed={SEED}, agents=[{MIN_AGENTS},{MAX_AGENTS}], "
        f"chores=[{MIN_CHORES},{MAX_CHORES}], values=[{LOW},{HIGH}], tol={TOL}"
    )
    print("-" * 72)

    for index in range(1, NUM_INSTANCES + 1):
        n_agents = rng.randint(MIN_AGENTS, MAX_AGENTS)
        n_chores = rng.randint(MIN_CHORES, MAX_CHORES)
        d = make_instance(rng, n_agents, n_chores)

        start = perf_counter()
        try:
            _, _, e, _, n_iter = full_algorithm(d, tol=TOL, max_iter=MAX_ITER)
            elapsed = perf_counter() - start
            gap = max(e) - min(e)
            iterations.append(n_iter)
            elapsed_times.append(elapsed)
            print(
                f"instance={index:02d} size={n_agents}x{n_chores} "
                f"status=solved iterations={n_iter} "
                f"final_gap={gap:.3e} seconds={elapsed:.2f}"
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = perf_counter() - start
            failures.append((index, n_agents, n_chores, f"{type(exc).__name__}: {exc}"))
            print(
                f"instance={index:02d} size={n_agents}x{n_chores} "
                f"status=failed seconds={elapsed:.2f} "
                f"reason={type(exc).__name__}: {exc}"
            )

    print("-" * 72)
    print("Summary")
    print(f"  solved={len(iterations)}")
    print(f"  failed={len(failures)}")

    if iterations:
        print(f"  min_iterations={min(iterations)}")
        print(f"  max_iterations={max(iterations)}")
        print(f"  mean_iterations={mean(iterations):.2f}")
        print(f"  median_iterations={median(iterations):.2f}")
        print(f"  total_seconds={perf_counter() - start_all:.2f}")
        print(f"  mean_seconds={mean(elapsed_times):.2f}")
        print("  iterations_by_solved_instance=", iterations)

    if failures:
        print("  failed_instances=")
        for index, n_agents, n_chores, reason in failures:
            print(f"    instance={index:02d} size={n_agents}x{n_chores} reason={reason}")


if __name__ == "__main__":
    main()
PY
