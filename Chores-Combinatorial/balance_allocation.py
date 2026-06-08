from __future__ import annotations

from typing import Sequence


def balance_allocation(
    p: Sequence[float],
    d: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[float], set[int]]:
    """
    Implement Algorithm 3: Balance-allocation(p).

    Maximizes prod_i e_i over MPB edges with sum_i x_ij = 1 for each chore j,
    where e_i = sum_j x_ij * p_j.

    Returns:
        x: allocation matrix with shape (n_agents, n_chores)
        e: agent earnings induced by x, with e_i = sum_j p_j * x_ij
        S: set of agents with the lowest earnings
    """
    n_agents = len(d)
    if n_agents <= 0:
        raise ValueError("d must contain at least one agent row.")
    if len(p) == 0:
        raise ValueError("p must contain at least one chore price.")
    n_chores = len(p)
    if any(len(row) != n_chores for row in d):
        raise ValueError("Each row of d must have len(p) entries.")
    if any(price <= 0 for price in p):
        raise ValueError("All prices in p must be strictly positive.")
    if any(disutility <= 0 for row in d for disutility in row):
        raise ValueError("All disutility values d[i][j] must be strictly positive.")

    eps = 1e-12
    mpb = [min(d[i][j] / p[j] for j in range(n_chores)) for i in range(n_agents)]
    mpb_edges = {
        (i, j)
        for i in range(n_agents)
        for j in range(n_chores)
        if abs((d[i][j] / p[j]) - mpb[i]) <= eps * max(1.0, abs(mpb[i]))
    }
    try:
        import cvxpy as cp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cvxpy is required for balance_allocation. Install it with: pip install cvxpy"
        ) from exc

    x_var = cp.Variable((n_agents, n_chores), nonneg=True)
    p_const = cp.Constant([float(v) for v in p])
    e_expr = x_var @ p_const

    constraints = [cp.sum(x_var, axis=0) == 1]
    constraints.extend(
        x_var[i, j] == 0
        for i in range(n_agents)
        for j in range(n_chores)
        if (i, j) not in mpb_edges
    )
    objective = cp.Maximize(cp.geo_mean(e_expr))
    problem = cp.Problem(objective, constraints) # type: ignore

    # SCS is widely available with cvxpy and handles this convex program.
    problem.solve(solver=cp.SCS, eps=1e-9, max_iters=100_000, verbose=False)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise ValueError(f"CVXPY failed to solve balance-allocation: status={problem.status}")
    if x_var.value is None:
        raise ValueError("CVXPY returned no solution for x.")

    x = [
        [max(0.0, float(x_var.value[i, j])) for j in range(n_chores)]
        for i in range(n_agents)
    ]
    e = [sum(x[i][j] * p[j] for j in range(n_chores)) for i in range(n_agents)]

    min_surplus = min(e)
    surplus_eps = 1e-8
    s_set = {i for i, surplus in enumerate(e) if abs(surplus - min_surplus) <= surplus_eps}

    return x, e, s_set
