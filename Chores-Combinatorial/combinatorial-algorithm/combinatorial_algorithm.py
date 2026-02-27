from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class AlgorithmState:
    x: list[list[float]]
    p: list[float]
    e: list[float]
    s_set: set[int]
    iterations: int


class CombinatorialAlgorithm:
    """
    Stateful implementation of Algorithms 1-4 from:
    "Competitive Equilibrium with Chores: Combinatorial Algorithm and Hardness".
    """

    def __init__(
        self,
        d: Sequence[Sequence[float]],
        *,
        tol: float = 1e-8,
        max_iter: int = 10_000,
    ) -> None:
        self.d = [list(map(float, row)) for row in d]
        self.tol = tol
        self.max_iter = max_iter
        self._validate_disutilities()

        self.n_agents = len(self.d)
        self.n_chores = len(self.d[0])

        # Algorithm 4 initialization: p_j <- min_i d_ij.
        self.p = [min(self.d[i][j] for i in range(self.n_agents)) for j in range(self.n_chores)]

        # Algorithm 3 initialization.
        self.x, self.e, self.s_set = self.balance_allocation(self.p)

    def _validate_disutilities(self) -> None:
        if not self.d:
            raise ValueError("d must contain at least one agent row.")
        n_chores = len(self.d[0])
        if n_chores == 0:
            raise ValueError("d must contain at least one chore column.")
        if any(len(row) != n_chores for row in self.d):
            raise ValueError("All rows in d must have the same number of columns.")
        if any(v <= 0 for row in self.d for v in row):
            raise ValueError("All disutility values d[i][j] must be strictly positive.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")

    def _mpb(self, i: int) -> float:
        # MPB_i = min_j d_ij / p_j
        return min(self.d[i][j] / self.p[j] for j in range(self.n_chores))

    def _earning_gap(self) -> float:
        return max(self.e) - min(self.e)

    def balance_allocation(self, p: Sequence[float]) -> tuple[list[list[float]], list[float], set[int]]:
        """
        Algorithm 3: x <- argmax prod_i e_i, where e_i = sum_j x_ij p_j, x >= 0, sum_i x_ij = 1.
        """
        try:
            import cvxpy as cp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "cvxpy is required for balance_allocation. Install with: pip install cvxpy"
            ) from exc

        x_var = cp.Variable((self.n_agents, self.n_chores), nonneg=True)
        p_const = cp.Constant([float(v) for v in p])
        e_expr = x_var @ p_const
        constraints = [cp.sum(x_var, axis=0) == 1]
        objective = cp.Maximize(cp.geo_mean(e_expr))
        problem = cp.Problem(objective, constraints) # type: ignore
        problem.solve(solver=cp.SCS, verbose=False)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError(f"CVXPY failed in balance_allocation: status={problem.status}")
        if x_var.value is None:
            raise ValueError("CVXPY returned no solution.")

        x = [
            [max(0.0, float(x_var.value[i, j])) for j in range(self.n_chores)]
            for i in range(self.n_agents)
        ]
        e = [sum(x[i][j] * p[j] for j in range(self.n_chores)) for i in range(self.n_agents)]
        min_e = min(e)
        eps = 1e-12
        s_set = {i for i, val in enumerate(e) if abs(val - min_e) <= eps}
        return x, e, s_set

    def price_update(self) -> tuple[float, set[int]]:
        """
        Algorithm 1: updates global p and e, returns (gamma, Gamma(S)).
        """
        eps = 1e-12
        mpb = [self._mpb(i) for i in range(self.n_agents)]

        gamma_set: set[int] = set()
        for i in self.s_set:
            for j in range(self.n_chores):
                ratio = self.d[i][j] / self.p[j]
                if abs(ratio - mpb[i]) <= eps * max(1.0, abs(mpb[i])):
                    gamma_set.add(j)

        outside = [j for j in range(self.n_chores) if j not in gamma_set]
        if not outside:
            raise ValueError("Gamma(S) contains all chores; gamma is undefined.")

        gamma = min((mpb[i] * self.p[j]) / self.d[i][j] for i in self.s_set for j in outside)

        for j in gamma_set:
            self.p[j] *= gamma
        for i in self.s_set:
            self.e[i] *= gamma

        return gamma, gamma_set

    def allocation_update(self, gamma_set: set[int]) -> None:
        """
        Algorithm 2: updates global x/e/S (and may call balance_allocation).
        """
        eps = 1e-12
        mpb = [self._mpb(i) for i in range(self.n_agents)]

        # New MPB edges from S to [m]\Gamma(S).
        edges_by_chore: dict[int, list[int]] = {}
        for i in sorted(self.s_set):
            for j in range(self.n_chores):
                if j in gamma_set:
                    continue
                ratio = self.d[i][j] / self.p[j]
                if abs(ratio - mpb[i]) <= eps * max(1.0, abs(mpb[i])):
                    edges_by_chore.setdefault(j, []).append(i)
        j_set = set(edges_by_chore.keys())

        outside = [i for i in range(self.n_agents) if i not in self.s_set]
        if not outside:
            return

        e_max = max(self.e[i] for i in self.s_set)
        e_min = min(self.e[i] for i in outside)

        if sum(self.p[j] for j in j_set) > (e_min - e_max) / 2.0:
            self.x, self.e, self.s_set = self.balance_allocation(self.p)
            return

        for j in sorted(j_set):
            i_pick = edges_by_chore[j][0]

            # Follow the paper update: remove allocations on j from outside agents.
            for i_out in outside:
                if self.x[i_out][j] > eps:
                    moved = self.x[i_out][j]
                    self.x[i_out][j] = 0.0
                    self.e[i_out] -= moved * self.p[j]

            # Set x_ij <- 1 for the chosen i in S.
            current = self.x[i_pick][j]
            self.x[i_pick][j] = 1.0
            self.e[i_pick] += (1.0 - current) * self.p[j]

        min_e = min(self.e)
        self.s_set = {i for i, val in enumerate(self.e) if abs(val - min_e) <= eps}

    def run(self) -> AlgorithmState:
        """
        Algorithm 4.
        """
        it = 0
        while self._earning_gap() > self.tol:
            if it >= self.max_iter:
                raise RuntimeError(
                    f"Did not converge within max_iter={self.max_iter} "
                    f"(gap={self._earning_gap():.3e})."
                )
            _, gamma_set = self.price_update()
            self.allocation_update(gamma_set)
            it += 1

        return AlgorithmState(
            x=[row[:] for row in self.x],
            p=self.p[:],
            e=self.e[:],
            s_set=set(self.s_set),
            iterations=it,
        )


def solve_chores_ceei(
    d: Sequence[Sequence[float]],
    *,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> AlgorithmState:
    algo = CombinatorialAlgorithm(d, tol=tol, max_iter=max_iter)
    return algo.run()

