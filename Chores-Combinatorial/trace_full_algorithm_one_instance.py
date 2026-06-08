from __future__ import annotations

import argparse
import random
from typing import Sequence

from allocation_update import allocation_update
from balance_allocation import balance_allocation
from price_update import price_update


def rounded(values: Sequence[float]) -> list[float]:
    return [round(v, 6) for v in values]


def print_matrix(name: str, matrix: Sequence[Sequence[float]]) -> None:
    print(f"{name}=")
    for row in matrix:
        print("  ", rounded(row))


def money_flows(x: Sequence[Sequence[float]], p: Sequence[float]) -> list[list[float]]:
    return [[x[i][j] * p[j] for j in range(len(p))] for i in range(len(x))]


def earning_gap(e: Sequence[float]) -> float:
    return max(e) - min(e)


def mpb_edges(p: Sequence[float], d: Sequence[Sequence[float]]) -> set[tuple[int, int]]:
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


def ce_violations(
    d: Sequence[Sequence[float]],
    p: Sequence[float],
    x: Sequence[Sequence[float]],
    e: Sequence[float],
    *,
    tol: float,
) -> list[str]:
    n_agents = len(d)
    n_chores = len(p)
    violations: list[str] = []

    if any(price <= 0 for price in p):
        violations.append("some prices are non-positive")

    for j in range(n_chores):
        allocated = sum(x[i][j] for i in range(n_agents))
        if abs(allocated - 1.0) > tol:
            violations.append(f"chore {j} has total allocation {allocated:.12g}, not 1")

    for i in range(n_agents):
        computed = sum(x[i][j] * p[j] for j in range(n_chores))
        if abs(computed - e[i]) > tol:
            violations.append(
                f"agent {i} earning is {e[i]:.12g}, but allocation pays {computed:.12g}"
            )

    edges = mpb_edges(p, d)
    for i in range(n_agents):
        for j in range(n_chores):
            if x[i][j] > tol and (i, j) not in edges:
                violations.append(
                    f"agent {i} receives chore {j} with x={x[i][j]:.12g}, but it is not MPB"
                )

    if earning_gap(e) > tol:
        violations.append(f"earnings are not equal; gap={earning_gap(e):.12g}")

    return violations


def print_ce_validation(
    d: Sequence[Sequence[float]],
    p: Sequence[float],
    x: Sequence[Sequence[float]],
    e: Sequence[float],
    *,
    tol: float,
) -> None:
    
    violations = ce_violations(d, p, x, e, tol=tol)
    
    print("\n CE validation")
    if not violations:
        print("  valid=True")
        return

    print("  valid=False")
    print("  violations=")
    for violation in violations:
        print("   -", violation)


def make_instance(
    *,
    seed: int,
    n_agents: int,
    n_chores: int,
    low: int,
    high: int,
) -> list[list[float]]:
    rng = random.Random(seed)
    return [
        [float(rng.randint(low, high)) for _ in range(n_chores)]
        for _ in range(n_agents)
    ]


def print_state(
    *,
    label: str,
    d: Sequence[Sequence[float]],
    p: Sequence[float],
    x: Sequence[Sequence[float]],
    e: Sequence[float],
    s_set: set[int],
) -> None:
    print(label)
    print("prices=", rounded(p))
    print("earnings=", rounded(e))
    print("earning_gap=", round(earning_gap(e), 12))
    print("S=", sorted(s_set))
    print("MPB_edges=", sorted(mpb_edges(p, d)))
    print_matrix("allocation", x)
    print_matrix("money_flows", money_flows(x, p))


def trace_full_algorithm(
    d: Sequence[Sequence[float]],
    *,
    tol: float,
    max_iter: int,
) -> None:
    print_matrix("disutilities", d)
    n_agents = len(d)
    n_chores = len(d[0])

    p = [min(d[i][j] for i in range(n_agents)) for j in range(n_chores)]
    x, e, s_set = balance_allocation(p, d)

    print_state(label="Initial state after balance_allocation", d=d, p=p, x=x, e=e, s_set=s_set)

    iteration = 0
    while earning_gap(e) > tol:
        if iteration >= max_iter:
            print(f"status=failed: reached max_iter={max_iter}")
            return

        print("-" * 80)
        print(f"Iteration {iteration + 1}: input")
        print_state(label="before price_update", d=d, p=p, x=x, e=e, s_set=s_set)

        try:
            p, e, gamma, gamma_set = price_update(e, p, s_set, d)
        except Exception as exc:  # noqa: BLE001
            if "Gamma(S) contains all chores" in str(exc):
                print("price_update found Gamma(S) = all chores")
                print(
                    "paper note: this is terminal under the maintained "
                    "invariants; there is no outside chore to define gamma"
                )
                print("trying balance_allocation at the current prices")
                try:
                    x, e, s_set = balance_allocation(p, d)
                except Exception as balance_exc:  # noqa: BLE001
                    print("balance_allocation recovery failed")
                    print(f"reason={type(balance_exc).__name__}: {balance_exc}")
                    return

                print_state(label="after recovery balance_allocation", d=d, p=p, x=x, e=e, s_set=s_set)
                if earning_gap(e) <= tol:
                    print("Gamma(S)=all chores and earnings are equal within tolerance")
                    break

                print("status=failed: Gamma(S) covered all chores but earnings are still unequal")
                return

            print("price_update failed")
            print(f"reason={type(exc).__name__}: {exc}")
            return

        print(f"price_update output: gamma={round(gamma, 12)}, Gamma(S)={sorted(gamma_set)}")
        print_state(label="after price_update", d=d, p=p, x=x, e=e, s_set=s_set)

        try:
            x, e, s_set = allocation_update(x, p, s_set, d, e, gamma_set)
        except Exception as exc:  # noqa: BLE001
            print("allocation_update failed")
            print(f"reason={type(exc).__name__}: {exc}")
            return

        print("allocation_update output")
        print_state(label="after allocation_update", d=d, p=p, x=x, e=e, s_set=s_set)
        iteration += 1

    print("-" * 80)
    print(f"status=solved, iterations={iteration}")
    print_state(label="final state", d=d, p=p, x=x, e=e, s_set=s_set)

    print_ce_validation(d, p, x, e, tol=tol)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace the full combinatorial algorithm on one generated instance."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-agents", type=int, default=10)
    parser.add_argument("--n-chores", type=int, default=10)
    parser.add_argument("--low", type=int, default=1)
    parser.add_argument("--high", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()

    d = make_instance(
        seed=args.seed,
        n_agents=args.n_agents,
        n_chores=args.n_chores,
        low=args.low,
        high=args.high,
    )
    trace_full_algorithm(d, tol=args.tol, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
