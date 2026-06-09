from __future__ import annotations

import time
from typing import Iterable, Literal, Sequence

import numpy as np

import gurobipy as gp

from utils import APPROXIMATE_THR, EXACT_THR, E2TOL, E3TOL, eps_approx_eq, gurobi_license_params


env = gp.Env(params=gurobi_license_params)

SolverName = str


def _rescale_prices_and_earnings(
    p: Sequence[float],
    e: Sequence[float],
    n_agents: int,
    *,
    tol: float = 1e-8,
) -> tuple[list[float], list[float]]:
    e_arr = np.asarray(e, dtype=float)
    total_earnings = float(np.sum(e_arr))
    target_total = float(n_agents)
    if total_earnings <= 0:
        raise ValueError("Total earnings must be positive for rescaling.")

    if abs(total_earnings - target_total) <= tol * max(1.0, target_total):
        return [float(v) for v in p], [float(v) for v in e]

    scale = target_total / total_earnings
    p_scaled = np.asarray(p, dtype=float) * scale
    e_scaled = e_arr * scale
    return p_scaled.tolist(), e_scaled.tolist()


def _validate_dimensions(
    d: Sequence[Sequence[float]],
    p: Sequence[float],
) -> None:
    n_agents = len(d)
    if n_agents == 0:
        raise ValueError("d must contain at least one agent row.")
    n_chores = len(d[0])
    if n_chores == 0:
        raise ValueError("d must contain at least one chore column.")

    for row in d:
        if len(row) != n_chores:
            raise ValueError("All rows in d must have the same number of columns.")

    if len(p) != n_chores:
        raise ValueError("len(p) must match the number of chores in d.")
    if any(price <= 0 for price in p):
        raise ValueError("All prices in p must be strictly positive.")
    if any(disutility <= 0 for row in d for disutility in row):
        raise ValueError("All disutility values d[i][j] must be strictly positive.")


def price_update(
    e: Sequence[float],
    p: Sequence[float],
    S: Iterable[int],
    d: Sequence[Sequence[float]],
) -> tuple[list[float], list[float], float, set[int]]:
    """
    Implement Algorithm 1: Price-Update(x, p, S).

    Returns:
        (new_p, new_e, gamma, gamma_set)
        where gamma_set is Γ(S).
    """
    n_agents = len(d)
    n_chores = len(p)

    _validate_dimensions(d, p)

    s_set = set(S)
    if not s_set:
        raise ValueError("S must be non-empty.")
    if any(i < 0 or i >= n_agents for i in s_set):
        raise ValueError("S contains an out-of-range agent index.")

    d_arr = np.asarray(d, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    
    # Compute minimum price-to-value ratio for each agent in S
    s_list = sorted(s_set)
    mpb = {}
    for i in s_list:
        mpb[i] = float(np.min(d_arr[i] / p_arr))

    eps = 1e-12
    gamma_set: set[int] = set()
    for i in s_list:
        for j in range(n_chores):
            ratio = d_arr[i, j] / p_arr[j]
            if abs(ratio - mpb[i]) <= eps * max(1.0, abs(mpb[i])):
                gamma_set.add(j)

    outside = [j for j in range(n_chores) if j not in gamma_set]
    if not outside:
        raise ValueError("Gamma(S) contains all chores; gamma is undefined in the formula.")

    gamma = max((mpb[i] * p_arr[j]) / d_arr[i, j] for i in s_list for j in outside)

    new_p = p_arr.copy()
    new_p[list(gamma_set)] *= gamma
    
    new_e = np.asarray(e, dtype=float)
    new_e[np.array(s_list)] *= gamma

    return new_p.tolist(), new_e.tolist(), float(gamma), gamma_set


def balance_allocation(
    p: Sequence[float],
    d: Sequence[Sequence[float]],
    *,
    solver: SolverName = "cvxpy",
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


def _validate_allocation_update_inputs(
    x: Sequence[Sequence[float]],
    p: Sequence[float],
    s_set: set[int],
    d: Sequence[Sequence[float]],
    e: Sequence[float],
    gamma_set: set[int],
) -> tuple[int, int]:
    n_agents = len(x)
    if n_agents == 0:
        raise ValueError("x must contain at least one agent row.")
    n_chores = len(p)
    if n_chores == 0:
        raise ValueError("p must contain at least one chore price.")
    if len(d) != n_agents:
        raise ValueError("len(d) must match len(x).")
    if any(len(row) != n_chores for row in x):
        raise ValueError("Each row of x must have len(p) entries.")
    if any(len(row) != n_chores for row in d):
        raise ValueError("Each row of d must have len(p) entries.")
    if len(e) != n_agents:
        raise ValueError("len(e) must match the number of agents in x.")
    if not s_set:
        raise ValueError("S must be non-empty.")
    if any(i < 0 or i >= n_agents for i in s_set):
        raise ValueError("S contains an out-of-range agent index.")
    if any(j < 0 or j >= n_chores for j in gamma_set):
        raise ValueError("gamma_set contains an out-of-range chore index.")
    if any(price <= 0 for price in p):
        raise ValueError("All prices in p must be strictly positive.")
    if any(disutility <= 0 for row in d for disutility in row):
        raise ValueError("All disutility values d[i][j] must be strictly positive.")
    return n_agents, n_chores


def allocation_update(
    x: Sequence[Sequence[float]],
    p: Sequence[float],
    S: Iterable[int],
    d: Sequence[Sequence[float]],
    e: Sequence[float],
    gamma_set: Iterable[int],
) -> tuple[list[list[float]], list[float], set[int]]:
    """
    Implement Algorithm 2: Allocation-Update(x, p, S).

    Returns:
        (x, e, S)
    """
    s_set = set(S)
    gamma = set(gamma_set)
    n_agents, n_chores = _validate_allocation_update_inputs(x, p, s_set, d, e, gamma)

    eps = 1e-12
    x_state = [[float(x[i][j]) for j in range(n_chores)] for i in range(n_agents)]
    e_state = [float(v) for v in e]

    mpb = {i: min(d[i][j] / p[j] for j in range(n_chores)) for i in s_set}

    edges_by_chore: dict[int, list[int]] = {}
    for i in sorted(s_set):
        for j in range(n_chores):
            if j in gamma:
                continue
            ratio = d[i][j] / p[j]
            if abs(ratio - mpb[i]) <= eps * max(1.0, abs(mpb[i])):
                edges_by_chore.setdefault(j, []).append(i)

    j_set = set(edges_by_chore.keys())

    outside = [i for i in range(n_agents) if i not in s_set]
    if not outside:
        return x_state, e_state, s_set

    e_max = max(e_state[i] for i in s_set)
    e_min = min(e_state[i] for i in outside)
    if sum(p[j] for j in j_set) > (e_min - e_max) / 2.0:
        return balance_allocation(p, d)

    for j in sorted(j_set):
        i_pick = edges_by_chore[j][0]
        for i_out in outside:
            if x_state[i_out][j] > eps:
                moved = x_state[i_out][j]
                x_state[i_out][j] = 0.0
                e_state[i_out] -= moved * p[j]

        x_state[i_pick][j] = 1.0
        e_state[i_pick] += p[j]

    return x_state, e_state, s_set


def full_algorithm(
    d: Sequence[Sequence[float]],
    *,
    tol: float = 1e-7,
    max_iter: int = 10_000,
    solver: SolverName = "cvxpy",
) -> tuple[list[list[float]], list[float], list[float], set[int], int]:
    """
    Implement Algorithm 4 (Full Algorithm).

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

    d_arr = np.asarray(d, dtype=float)
    p = [float(np.min(d_arr[:, j])) for j in range(n_chores)]
    x, e, s_set = balance_allocation(p, d, solver=solver)
    p, e = _rescale_prices_and_earnings(p, e, n_agents)

    def _earning_gap(values: Sequence[float]) -> float:
        return float(np.max(values) - np.min(values))

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

            x, e, s_set = balance_allocation(p, d, solver=solver)
            p, e = _rescale_prices_and_earnings(p, e, n_agents)
            if _earning_gap(e) <= tol:
                break
            raise RuntimeError(
                "Gamma(S) contains all chores while earnings are unequal; "
                "this violates the paper's iteration invariant."
            ) from exc

        x, e, s_set = allocation_update(x, p, s_set, d, e, gamma_set)
        p, e = _rescale_prices_and_earnings(p, e, n_agents)
        n_iter += 1

    return x, p, e, s_set, n_iter


def combinatorial_metrics(
    N: int,
    M: int,
    D: Sequence[Sequence[float]],
    B: Sequence[float],
    *,
    tol: float = 1e-7,
    max_iter: int = 10_000,
    print_eq: bool = False,
    solver: SolverName = "cvxpy",
    return_eq: bool = False,
) -> tuple[int, int, bool, bool, float | None, float | None] | tuple[int, int, bool, bool, float | None, float | None, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """
    Run the combinatorial algorithm and report GFW/EPM-style metrics:
    (num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e).

    The approximation/exactness checks use eps_approx_eq with
    APPROXIMATE_THR, EXACT_THR, E2TOL, and E3TOL from utils.py.
    """
    D_np = np.asarray(D, dtype=float)
    B_np = np.asarray(B, dtype=float)

    if D_np.shape != (N, M):
        raise ValueError(f"D must have shape ({N}, {M}).")
    if B_np.shape != (N,):
        raise ValueError(f"B must have shape ({N},).")
    if np.any(D_np <= 0):
        raise ValueError("All disutility values in D must be strictly positive.")
    if np.any(B_np <= 0):
        raise ValueError("All budgets in B must be strictly positive.")
    if tol < 0:
        raise ValueError("tol must be non-negative.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    p = [float(np.min(D_np[:, j])) for j in range(M)]
    x, e, s_set = balance_allocation(p, D_np.tolist(), solver=solver)
    p, e = _rescale_prices_and_earnings(p, e, N)

    max_num_iter = max_iter
    num_LMO_a1, num_LMO_e = max_num_iter, max_num_iter
    solved_a1, solved_e = False, False
    running_time_a1, running_time_e = None, None

    num_LMO = 0
    running_time = 0.0

    def _earning_gap(values: Sequence[float]) -> float:
        values_arr = np.asarray(values, dtype=float)
        return float(np.max(values_arr) - np.min(values_arr))

    def _evaluate_current_state() -> None:
        nonlocal solved_a1, solved_e, num_LMO_a1, num_LMO_e, running_time_a1, running_time_e

        eps = eps_approx_eq(
            N,
            M,
            D_np,
            B_np,
            np.asarray(p, dtype=float),
            np.asarray(x, dtype=float),
            E2Tol=E2TOL,
            E3Tol=E3TOL,
            ignore_print=True,
        )

        if isinstance(eps, str):
            return

        if eps <= APPROXIMATE_THR and num_LMO_a1 == max_num_iter:
            solved_a1 = True
            num_LMO_a1 = num_LMO
            running_time_a1 = running_time

        if eps <= EXACT_THR and num_LMO_e == max_num_iter:
            solved_e = True
            num_LMO_e = num_LMO
            running_time_e = running_time

    _evaluate_current_state()

    while _earning_gap(e) > tol and num_LMO < max_num_iter and not solved_e:
        iter_start = time.time()

        try:
            p, e, _, gamma_set = price_update(e, p, s_set, D_np.tolist())
        except ValueError as exc:
            if "Gamma(S) contains all chores" not in str(exc):
                raise

            x, e, s_set = balance_allocation(p, D_np.tolist(), solver=solver)
            p, e = _rescale_prices_and_earnings(p, e, N)
            running_time += time.time() - iter_start
            num_LMO += 1
            _evaluate_current_state()
            break

        x, e, s_set = allocation_update(x, p, s_set, D_np.tolist(), e, gamma_set)
        p, e = _rescale_prices_and_earnings(p, e, N)

        running_time += time.time() - iter_start
        num_LMO += 1

        _evaluate_current_state()

    if return_eq and solved_e:
        p_arr = np.asarray(p, dtype=float)
        x_arr = np.asarray(x, dtype=float)
        u_arr = np.sum(p_arr * x_arr, axis=1)
        return num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e, (p_arr, x_arr, u_arr)
    elif return_eq:
        return num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e, None

    return num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e
