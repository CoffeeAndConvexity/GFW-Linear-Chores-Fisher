from __future__ import annotations

from typing import Iterable, Sequence


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

    The routine follows the pseudocode:
    1) Γ(S) <- {j | ∃ i ∈ S such that d_ij / p_j = MPB_i}
    2) γ <- max_{i∈S, j∉Γ(S)} (MPB_i * p_j / d_ij)
    3) For j ∈ Γ(S): p_j <- γ * p_j
    4) For i ∈ S: e_i <- γ * e_i

    Args:`r`n        e: Current agent earnings, length n_agents.`r`n        p: Current chore prices, length m.`r`n        S: Iterable of agent indices.`r`n        d: Disutility matrix with shape (n_agents, n_chores).

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

    # MPB_i = min_j d_ij / p_j (minimum pain per buck), only needed for i in S.
    mpb = {i: min(d[i][j] / p[j] for j in range(n_chores)) for i in s_set}

    # Γ(S) = chores that are MPB-tight for at least one i in S.
    eps = 1e-12
    gamma_set: set[int] = set()
    for i in s_set:
        for j in range(n_chores):
            ratio = d[i][j] / p[j]
            if abs(ratio - mpb[i]) <= eps * max(1.0, abs(mpb[i])):
                gamma_set.add(j)

    outside = [j for j in range(n_chores) if j not in gamma_set]
    if not outside:
        raise ValueError("Gamma(S) contains all chores; gamma is undefined in the formula.")

    gamma = max((mpb[i] * p[j]) / d[i][j] for i in s_set for j in outside)

    new_p = list(p)
    for j in gamma_set:
        new_p[j] = gamma * new_p[j]

    new_e = list(e)
    for i in s_set:
        new_e[i] = gamma * new_e[i]

    return new_p, new_e, gamma, gamma_set
