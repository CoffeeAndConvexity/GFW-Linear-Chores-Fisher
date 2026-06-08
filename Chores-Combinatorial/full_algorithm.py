from __future__ import annotations

from typing import Sequence

from allocation_update import allocation_update
from balance_allocation import balance_allocation
from price_update import price_update


def full_algorithm(
    d: Sequence[Sequence[float]],
    *,
    tol: float = 1e-7,
    max_iter: int = 10_000,
) -> tuple[list[list[float]], list[float], list[float], set[int], int]:
    """
    Implement Algorithm 4 (Full Algorithm).

    Steps:
    1) Initialize prices by p_j = min_i d_ij.
    2) (x, e, S) <- Balance-allocation(p).
    3) While earnings are not equal:
         - Price-update(x, p, S)
         - Allocation-update(x, p, S)
    4) Return (x, p).

    Returns:
        (x, p, e, S, n_iter)
    """
    n_agents = len(d)
    if n_agents == 0:
        raise ValueError("d must contain at least one agent row.")
    n_chores = len(d[0])
    if n_chores == 0:
        raise ValueError("d must contain at least one chore column.")
    if any(len(row) != n_chores for row in d):
        raise ValueError("All rows in d must have the same number of columns.")
    if any(v <= 0 for row in d for v in row):
        raise ValueError("All disutility values d[i][j] must be strictly positive.")
    if tol < 0:
        raise ValueError("tol must be non-negative.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    p = [min(d[i][j] for i in range(n_agents)) for j in range(n_chores)]
    x, e, s_set = balance_allocation(p, d)

    def _earning_gap(values: Sequence[float]) -> float:
        return max(values) - min(values)

    n_iter = 0
    while _earning_gap(e) > tol:
        if n_iter >= max_iter:
            raise RuntimeError(
                f"full_algorithm did not converge within max_iter={max_iter} "
                f"(current gap={_earning_gap(e):.3e})."
            )

        try:
            p, e, _, gamma_set = price_update(e, p, s_set, d)
        except ValueError as exc:
            if "Gamma(S) contains all chores" not in str(exc):
                raise

            # In the paper this case is terminal: if Gamma(S) covers every
            # chore, there is no outside chore whose price can be changed to
            # create a new MPB edge. With a numerical solver this can appear
            # when earnings are already equal up to solver precision.
            x, e, s_set = balance_allocation(p, d)
            if _earning_gap(e) <= tol:
                break
            raise RuntimeError(
                "Gamma(S) contains all chores while earnings are unequal; "
                "this violates the paper's iteration invariant."
            ) from exc

        x, e, s_set = allocation_update(x, p, s_set, d, e, gamma_set)
        n_iter += 1

    return x, p, e, s_set, n_iter
