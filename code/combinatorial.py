from __future__ import annotations

import time
from typing import Iterable, Literal, Sequence

import numpy as np

import gurobipy as gp

from utils import APPROXIMATE_THR, EXACT_THR, E2TOL, E3TOL, eps_approx_eq, gurobi_license_params


env = gp.Env(params=gurobi_license_params)

SolverName = str


class _BalanceAllocationModel:
    def __init__(self, n_agents: int, n_chores: int, *, solver: SolverName, warm_start: bool) -> None:
        try:
            import cvxpy as cp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "cvxpy is required for balance_allocation. Install it with: pip install cvxpy"
            ) from exc

        self.cp = cp
        self.n_agents = n_agents
        self.n_chores = n_chores
        self.solver_name = str(solver).strip().lower()
        self.warm_start = bool(warm_start)

        self.x_var = cp.Variable((n_agents, n_chores), nonneg=True)
        self.p_param = cp.Parameter(n_chores, nonneg=True)
        self.edge_mask = cp.Parameter((n_agents, n_chores), nonneg=True)

        e_expr = self.x_var @ self.p_param
        constraints = [cp.sum(self.x_var, axis=0) == 1, self.x_var <= self.edge_mask]
        objective = cp.Maximize(cp.geo_mean(e_expr))
        self.problem = cp.Problem(objective, constraints)  # type: ignore

    def solve(self, p, mpb_edges, *, x_init=None) -> tuple[np.ndarray, np.ndarray, set[int]]:
        p_arr = np.asarray(p, dtype=float)
        mask_arr = np.asarray(mpb_edges, dtype=float)

        self.p_param.value = np.maximum(p_arr, 0.0)
        self.edge_mask.value = np.maximum(mask_arr, 0.0)

        if self.warm_start and x_init is not None:
            x_init_arr = np.asarray(x_init, dtype=float)
            if x_init_arr.shape == (self.n_agents, self.n_chores):
                self.x_var.value = np.minimum(np.maximum(x_init_arr, 0.0), self.edge_mask.value)

        if self.solver_name in {"cvxpy", "scs"}:
            self.problem.solve(
                solver=self.cp.SCS,
                eps=1e-9,
                max_iters=100_000,
                verbose=False,
                warm_start=self.warm_start,
            )
        elif self.solver_name in {"cvxpy_strict", "scs_strict"}:
            self.problem.solve(
                solver=self.cp.SCS,
                eps=1e-11,
                max_iters=300_000,
                verbose=False,
                warm_start=self.warm_start,
            )
        elif self.solver_name == "gurobi":
            self.problem.solve(
                solver=self.cp.GUROBI,
                env=env,
                verbose=False,
                warm_start=self.warm_start,
            )
        elif self.solver_name == "mosek":
            self.problem.solve(
                solver=self.cp.MOSEK,
                verbose=False,
                warm_start=self.warm_start,
            )
        else:
            raise ValueError(
                f"Unknown COMB solver '{self.solver_name}'. Use one of: cvxpy/scs, cvxpy_strict/scs_strict, gurobi, mosek."
            )

        if self.problem.status not in {self.cp.OPTIMAL, self.cp.OPTIMAL_INACCURATE}:
            raise ValueError(f"CVXPY failed to solve balance-allocation: status={self.problem.status}")
        if self.x_var.value is None:
            raise ValueError("CVXPY returned no solution for x.")

        x = np.asarray(self.x_var.value, dtype=float)
        e = np.sum(p_arr * x, axis=1)

        min_surplus = float(np.min(e))
        surplus_eps = 1e-8
        s_set = {i for i, surplus in enumerate(e) if abs(float(surplus) - min_surplus) <= surplus_eps}

        return x, e, s_set


def _rescale_prices_and_earnings(p, e, n_agents, *, tol: float = 1e-8):
    
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

    return p_scaled, e_scaled


def price_update(e, p, S, D):
    """
    Implement Price-Update(x, p, S).

    Returns:
        (new_p, new_e, gamma, gamma_set) where gamma_set is Γ(S).
    """
    n_agents, n_chores = len(D), len(D[0])
    s_set = set(S)
    
    # Compute minimum pain-per-buck ratio for each agent in S
    s_list = sorted(s_set)
    mpb = {i: np.min(D[i] / p) for i in s_list}

    # Construct the set Γ(S) and its complement
    eps = 1e-8
    gamma_set: set[int] = set()
    for i in s_list:
        for j in range(n_chores):
            if abs(D[i, j] / p[j] - mpb[i]) <= eps:
                gamma_set.add(j)
    outside = [j for j in range(n_chores) if j not in gamma_set]
    if not outside:
        raise ValueError("Gamma(S) contains all chores; gamma is undefined in the formula.")

    gamma = max((mpb[i] * p[j]) / D[i, j] for i in s_list for j in outside)

    new_p = p.copy()
    new_p[np.array(list(gamma_set))] *= gamma
    
    new_e = e.copy()
    new_e[np.array(s_list)] *= gamma

    return new_p, new_e, float(gamma), gamma_set


def balance_allocation(
    p, D, *, solver="cvxpy", x_init=None, warm_start: bool = True, model: _BalanceAllocationModel | None = None,
) -> tuple[list[list[float]], list[float], set[int]]:
    """
    Implement Balance-allocation(p).

    Maximizes prod_i e_i over MPB edges with sum_i x_ij = 1 for each chore j, where e_i = sum_j x_ij * p_j.

    Returns:
        x: allocation matrix with shape (n_agents, n_chores)
        e: agent earnings induced by x, with e_i = sum_j p_j * x_ij
        S: set of agents with the lowest earnings
    """
    n_agents = len(D)
    n_chores = len(p)

    eps = 1e-8
    mpb = np.amin(D / p, axis=1)
    mpb_edges = np.where(np.abs(D / p - mpb[:, None]) <= eps, 1, 0)

    reusable_model = model
    if reusable_model is None:
        reusable_model = _BalanceAllocationModel(
            n_agents,
            n_chores,
            solver=solver,
            warm_start=warm_start,
        )

    return reusable_model.solve(p, mpb_edges, x_init=x_init)


def allocation_update(
    x,
    p,
    S,
    D,
    e,
    gamma_set: Iterable[int],
    *,
    solver="cvxpy",
    warm_start: bool = True,
    model: _BalanceAllocationModel | None = None,
):
    """
    Implement Allocation-Update(x, p, S).

    Returns:
        (x, e, S)
    """
    s_set = set(S)
    gamma = set(gamma_set)
    n_agents, n_chores = len(x), len(p)

    eps = 1e-8
    mpb = {i: min(D[i] / p) for i in s_set}
    edges_by_chore: dict[int, list[int]] = {}
    for i in sorted(s_set):
        for j in range(n_chores):
            if j in gamma:
                continue
            if abs(D[i, j] / p[j] - mpb[i]) <= eps:
                edges_by_chore.setdefault(j, []).append(i)

    j_set = set(edges_by_chore.keys())

    outside = [i for i in range(n_agents) if i not in s_set]
    if not outside:
        return x, e, s_set

    e_max = max(e[i] for i in s_set)
    e_min = min(e[i] for i in outside)
    if sum(p[j] for j in j_set) > (e_min - e_max) / 2.0:
        return balance_allocation(p, D, solver=solver, x_init=x, warm_start=warm_start, model=model)
    else:
        for j in sorted(j_set):
            i_pick = edges_by_chore[j][0]
            for i_out in outside:
                if x[i_out, j] > eps:
                    moved = x[i_out, j]
                    x[i_out, j] = 0.0
                    e[i_out] -= moved * p[j]

            x[i_pick, j] = 1.0
            e[i_pick] += p[j]

    return x, e, s_set


def combinatorial_metrics(
    N, M, D_np, B_np,
    *,
    tol: float = 1e-7,
    max_iter: int = 10_000,
    print_eq: bool = False,
    solver: SolverName = "cvxpy",
    warm_start: bool = True,
    return_eq: bool = False,
) -> tuple[int, int, bool, bool, float | None, float | None] | tuple[int, int, bool, bool, float | None, float | None, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """
    Run the combinatorial algorithm and report GFW/EPM-style metrics:
    (num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e).

    The approximation/exactness checks use eps_approx_eq with
    APPROXIMATE_THR, EXACT_THR, E2TOL, and E3TOL from utils.py.
    """
    persistent_model = _BalanceAllocationModel(
        N,
        M,
        solver=solver,
        warm_start=warm_start,
    )

    p = np.amin(D_np, axis=0)
    x, e, s_set = balance_allocation(p, D_np, solver=solver, warm_start=warm_start, model=persistent_model)
    p, e = _rescale_prices_and_earnings(p, e, N)

    max_num_iter = max_iter
    num_CMO_a1, num_CMO_e = max_num_iter, max_num_iter
    solved_a1, solved_e = False, False
    running_time_a1, running_time_e = None, None

    num_CMO = 0
    running_time = 0.0

    def _earning_gap(values):
        values_arr = np.asarray(values, dtype=float)
        return float(np.max(values_arr) - np.min(values_arr))

    def _evaluate_current_state() -> None:
        nonlocal solved_a1, solved_e, num_CMO_a1, num_CMO_e, running_time_a1, running_time_e

        eps = eps_approx_eq(
            N, M, D_np, B_np,
            np.asarray(p, dtype=float),
            np.asarray(x, dtype=float),
            E2Tol=E2TOL,
            E3Tol=E3TOL,
            ignore_print=True,
        )

        if isinstance(eps, str):
            return

        if eps <= APPROXIMATE_THR and num_CMO_a1 == max_num_iter:
            solved_a1 = True
            num_CMO_a1 = num_CMO
            running_time_a1 = running_time

        if eps <= EXACT_THR and num_CMO_e == max_num_iter:
            solved_e = True
            num_CMO_e = num_CMO
            running_time_e = running_time

    _evaluate_current_state()

    while _earning_gap(e) > tol and num_CMO < max_num_iter and not solved_e:

        iter_start = time.time()

        try:
            p, e, _, gamma_set = price_update(e, p, s_set, D_np)
        except ValueError as exc:
            print("Caught ValueError during price_update:", exc)
            if "Gamma(S) contains all chores" not in str(exc):
                raise

            x, e, s_set = balance_allocation(
                p,
                D_np,
                solver=solver,
                x_init=x,
                warm_start=warm_start,
                model=persistent_model,
            )
            
            p, e = _rescale_prices_and_earnings(p, e, N)
            running_time += time.time() - iter_start
            num_CMO += 1
            _evaluate_current_state()
            break

        # print("p:", p, type(p))
        # print("e:", e, type(e))

        x, e, s_set = allocation_update(
            x,
            p,
            s_set,
            D_np,
            e,
            gamma_set,
            solver=solver,
            warm_start=warm_start,
            model=persistent_model,
        )
        p, e = _rescale_prices_and_earnings(p, e, N)

        # print("-> p:", p)
        # print("-> e:", e)

        running_time += time.time() - iter_start
        num_CMO += 1

        _evaluate_current_state()

    if return_eq and solved_e:
        u = np.sum(D_np * x, axis=1)
        return num_CMO_a1, num_CMO_e, solved_a1, solved_e, running_time_a1, running_time_e, (p, x, u)
    elif return_eq:
        return num_CMO_a1, num_CMO_e, solved_a1, solved_e, running_time_a1, running_time_e, None

    return num_CMO_a1, num_CMO_e, solved_a1, solved_e, running_time_a1, running_time_e
