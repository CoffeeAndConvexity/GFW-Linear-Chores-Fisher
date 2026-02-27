from __future__ import annotations

import argparse
import random
from typing import Sequence

from full_algorithm import full_algorithm


def ce_support_violations(
    d: Sequence[Sequence[float]],
    p: Sequence[float],
    x: Sequence[Sequence[float]],
    *,
    x_tol: float = 1e-9,
    ratio_tol: float = 1e-8,
) -> list[tuple[int, int, float, float]]:
    """
    Return CE-support violations as tuples:
    (agent_i, chore_j, ratio_ij, min_ratio_i)
    where x_ij > x_tol but ratio_ij is not close to min_ratio_i.
    """
    n_agents = len(d)
    n_chores = len(p)
    violations: list[tuple[int, int, float, float]] = []

    for i in range(n_agents):
        min_ratio = min(d[i][jj] / p[jj] for jj in range(n_chores))
        for j in range(n_chores):
            if x[i][j] > x_tol:
                ratio = d[i][j] / p[j]
                if abs(ratio - min_ratio) > ratio_tol * max(1.0, abs(min_ratio)):
                    violations.append((i, j, ratio, min_ratio))
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full_algorithm on random instances and report CE support violations."
    )
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-agents", type=int, default=2)
    parser.add_argument("--max-agents", type=int, default=4)
    parser.add_argument("--min-chores", type=int, default=2)
    parser.add_argument("--max-chores", type=int, default=4)
    parser.add_argument("--low", type=float, default=0.5)
    parser.add_argument("--high", type=float, default=5.0)
    parser.add_argument("--digits", type=int, default=3)
    args = parser.parse_args()

    if args.num_instances <= 0:
        raise ValueError("--num-instances must be positive.")
    if args.min_agents <= 0 or args.max_agents < args.min_agents:
        raise ValueError("Invalid agent bounds.")
    if args.min_chores <= 0 or args.max_chores < args.min_chores:
        raise ValueError("Invalid chore bounds.")
    if args.high <= args.low:
        raise ValueError("--high must be greater than --low.")

    random.seed(args.seed)

    solved = 0
    failed = 0
    with_violations = 0

    for t in range(1, args.num_instances + 1):
        n_agents = random.randint(args.min_agents, args.max_agents)
        n_chores = random.randint(args.min_chores, args.max_chores)
        d = [
            [round(random.uniform(args.low, args.high), args.digits) for _ in range(n_chores)]
            for _ in range(n_agents)
        ]

        print(f"Instance {t}")
        print(f"n_agents={n_agents}, n_chores={n_chores}")
        print("d=")
        for row in d:
            print("  ", row)

        try:
            x, p, e, s_set, n_iter = full_algorithm(d)
            solved += 1

            print("status=solved")
            print("p=")
            print("  ", [round(v, 6) for v in p])
            print("x=")
            for row in x:
                print("  ", [round(v, 6) for v in row])
            print("e=")
            print("  ", [round(v, 6) for v in e])
            print(f"S={sorted(s_set)}, iters={n_iter}")

            violations = ce_support_violations(d, p, x)
            if violations:
                with_violations += 1
                print("ce_support=violated")
                print("violations (i, j, d_ij/p_j, min_ratio_i)=")
                for i, j, ratio, min_ratio in violations:
                    print(
                        "  ",
                        (i, j, round(ratio, 8), round(min_ratio, 8)),
                    )
            else:
                print("ce_support=ok")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("status=failed")
            print(f"reason={type(exc).__name__}: {exc}")

        print("-" * 60)

    print("Summary")
    print(f"  solved={solved}")
    print(f"  failed={failed}")
    print(f"  solved_with_ce_support_violations={with_violations}")


if __name__ == "__main__":
    main()

