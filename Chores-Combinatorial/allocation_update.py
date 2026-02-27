from __future__ import annotations

from typing import Iterable, Sequence

from balance_allocation import balance_allocation


def _validate_inputs(
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

    Args:
        x: Current allocation matrix, shape (n_agents, n_chores).
        p: Current prices after the price-update step.
        S: Agent index set S.
        d: Disutility matrix, shape (n_agents, n_chores).
        e: agent earnings induced by x, with e_i = sum_j p_j * x_ij
        gamma_set: Gamma(S) from the preceding price-update step.

    Returns:
        (p, x, e, S)
    """
    s_set = set(S)
    gamma = set(gamma_set)
    n_agents, n_chores = _validate_inputs(x, p, s_set, d, e, gamma)
    eps = 1e-12
    x_state = [[float(x[i][j]) for j in range(n_chores)] for i in range(n_agents)]
    e_state = [float(v) for v in e]

    # MPB_i = min_j d_ij / p_j (minimum pain per buck).
    mpb = [min(d[i][j] / p[j] for j in range(n_chores)) for i in range(n_agents)]

    # E = new MPB edges from S to [m] \ Gamma(S).
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
        p, x, S = balance_allocation(p, n_agents)
        return p, x, e_state, S

    # For each j in J, move chore j according to the pseudocode updates on x and e.
    print("For each j in J, move chore j according to the pseudocode updates on x and e.")
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
