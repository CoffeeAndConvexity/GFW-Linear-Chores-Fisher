import numpy as np

"""
Set up 'Approximate' and 'Exact' tolerances
"""

APPROXIMATE_THR = 1e-2
EXACT_THR = 1e-9
E2TOL = 1e-9
E3TOL = 1e-9



def eps_approx_eq(N, M, D, B, p, x, E2Tol=E2TOL, E3Tol=E3TOL, report_all=False, ignore_print=False):

    def equal_budget_eps(B, p, x):
        B_ = np.sum(p * x, axis=1)
        eps1 = max(1 - min(B_ / B), 1 - 1 / max(B_ / B))

        return eps1

    def optimal_bundle_eps(N, D, p, x):
        eps2 = 0
        for i in range(N):
            beta = max(p / D[i])
            min_disuti = sum(p * x[i]) / beta
            eps_i = 1 - min_disuti / sum(D[i] * x[i])
            if eps_i > eps2:
                eps2 = eps_i

        return eps2

    def market_clearance_eps(M, x):
        eps3 = max(np.abs(np.sum(x, axis=0) - np.ones(M)))

        return eps3

    eps1 = equal_budget_eps(B, p, x)
    eps2 = optimal_bundle_eps(N, D, p, x)
    eps3 = market_clearance_eps(M, x)

    if report_all:
        if not ignore_print:
            print()
            print("===============================================================")
            print("(E1) Equal Budget \t\t (E2) Optimal Bundle \t\t (E3) Market Clearance")
            print("---------------------------------------------------------------")
            print(f"{eps1} \t\t {eps2} \t\t {eps3}")
            print("---------------------------------------------------------------")

    if eps2 > E2Tol:
        return f"(E2 - optimal_bundle) is not exactly satisfied (eps2 = {eps2})."

    if eps3 > E3Tol:
        return f"(E3 - market_clearance) is not exactly satisfied (eps3 = {eps3})."

    return eps1