# -*- coding: utf-8 -*-
"""
Chores CEEI Experiments
"""

import numpy as np
from scipy.stats import truncnorm
from sklearn.cluster import KMeans
import gurobipy as gp
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import time

# Create an environment with your WLS license
params = {
"WLSACCESSID": 'bd6082d8-df30-497b-86f3-f189f173114f',
"WLSSECRET": '33a99b27-d33d-4640-ab8e-6b4db5d020bd',
"LICENSEID": 2446748,
}
env = gp.Env(params=params)

print("- cvxpy version:", cp.__version__, "-")
print("- cvxpy installed solvers:", cp.installed_solvers())


"""
Set up 'Approximate' and 'Exact' tolerances
"""

APPROXIMATE_THR = 0.01
EXACT_THR = 1e-6
E2TOL = 1e-6
E3TOL = 1e-6

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


"""
Codes for Greedy Frank Wolfe (GFW)
"""

def polytope_dual(N, M, D, B):
    A = list()
    # use Gurobi, not include the nonnegative constraints
    for i in range(N):
        D_i_T = np.zeros(shape=(M, N))
        D_i_T[:, i] = - D[i]
        A_i = np.concatenate([np.identity(M), D_i_T], axis=1)
        A.append(A_i)
    A = np.concatenate(A, axis=0)
    b = np.zeros(shape=(M * N))
    C = np.concatenate([np.ones(M), np.zeros(N)]).reshape(1, -1)
    d = np.array(sum(B)).reshape(1, )

    return A, b, C, d

def feasible_point(M, N, D, B, random=False, random_seed=2024):  ###
    if random:
        np.random.seed(random_seed)
        p_0 = np.random.uniform(size=M)
        p_0 = p_0 / sum(p_0) * sum(B)
        beta_0 = np.amax(p_0 / D, axis=1) + np.random.uniform(size=N)
    else:
        p_0 = (sum(B) / M) * np.ones(shape=M)
        beta_0 = np.amax(p_0 / D, axis=1)

    return p_0, beta_0

def LMO(c, A, b, C=None, d=None, return_dual=False):
    m = gp.Model(env=env)
    N = len(c)

    M = A.shape[1] - len(c)
    p = m.addMVar(M)
    beta = m.addMVar(len(c))

    m.setObjective(c @ beta)
    m.addConstr(A[:, :M] @ p + A[:, M:] @ beta <= b)
    if C is not None and d is not None:
        m.addConstr(C[:, :M] @ p + C[:, M:] @ beta == d)
    # m.PStart = x0

    m.Params.LogToConsole = 0
    # m.Params.Method = 1
    m.optimize()

    if return_dual:
        return p.X, beta.X, np.array(m.Pi)
    else:
        return p.X, beta.X

def GFW(N, M, D, B, return_eq=False):  # Greedy Frank Wolfe

    # create the dual polyhedron
    A, b, C, d = polytope_dual(N, M, D, B)

    # initialize
    MAX_NUM_ITER = 80
    num_LMO_a1, num_LMO_e = MAX_NUM_ITER, MAX_NUM_ITER
    solved_a1, solved_e = False, False
    running_time_a1, running_time_e = None, None

    # counters
    num_LMO = 0
    running_time = 0

    # analysis
    solve_LP_time = 0
    solve_LP_num = 0

    p, beta = feasible_point(M, N, D, B)
    beta_ = beta  # 'beta_' keeps the last beta values

    for k in range(MAX_NUM_ITER):

        # main
        LP_start = GFW_start = time.time()

        p, beta, gamma = LMO(B / beta, A, b, C, d, return_dual=True)

        solve_LP_time += time.time() - LP_start
        solve_LP_num += 1
        num_LMO += 1

        x = -gamma[: N * M].reshape(N, M)
        x = sum(B) / sum(B * beta / beta_) * x
        beta_ = beta

        running_time += time.time() - GFW_start

        # evaluation
        eps = eps_approx_eq(N, M, D, B, p, x, ignore_print=True)

        if type(eps) is str:  # then this step should not be considered as one candidate for approximate and exact equilibrium
            if k == MAX_NUM_ITER - 1:  # if reach the maximum number of iterations, then we terminate the algorithm
                break 
            else:  # otherwise, continue to the next iteration 
                continue 

        if type(eps) is not str:
            if eps <= APPROXIMATE_THR and num_LMO_a1 == MAX_NUM_ITER:  # the second condition ensures that 'num_LMO_1' has not been updated
                solved_a1 = True
                num_LMO_a1 = num_LMO
                running_time_a1 = running_time

            if eps <= EXACT_THR:
                solved_e = True
                num_LMO_e = num_LMO
                running_time_e = running_time
                break

    # print("Average LP solving time:", solve_LP_time / solve_LP_num)
    if return_eq:
        if solved_e:
            print("p:\n", np.round(p, 3))
            print("x:\n", np.round(x, 3))
            print("u:\n", np.round(B / beta, 3))
    return num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e


"""
Codes for Exterior Point Methods (EPM)
"""

def poly_ext_primal(N, M, D, B):
    A = list()
    A.append(np.zeros(shape=(N, N + M * N)))
    for i in range(N):
        A[0][i, i] = -1
        A[0][i, N + i * M: N + i * M + M] = D[i]
    # show add x is nonnegative constraint when solving QP using some solver
    A = np.concatenate(A, axis=0)
    b = np.zeros(N)
    C = np.zeros(shape=(M, N + M * N))
    for j in range(M):
        C[j, N + j: N + M * N: M] = 1
    d = np.ones(M)

    return A, b, C, d

def QMO(ux0, N, A, b, C=None, d=None, solver='OSQP'):
    if solver == 'OSQP':
        ux = cp.Variable(len(ux0), nonneg=True)
        obj = cp.Minimize(sum((ux[:N] - ux0[:N]) ** 2))
        constraints = [A @ ux <= b, ]
        if C is not None and d is not None:
            constraints.append(C @ ux == d)
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver='OSQP', max_iter=100000, verbose=False)
        except:
            return ux0, np.float64(10000000)  # should change this
        # print(prob.status)

        return ux.value, prob.value

    elif solver == 'GUROBI':
        m = gp.Model(env=env)

        m.Params.LogToConsole = 0
        # m.Params.FeasibilityTol = 1e-9
        # m.Params.OptimalityTol = 1e-9
        m.Params.BarConvTol = 0
        # m.Params.BarCorrectors = 10000

        u = m.addMVar(N)
        x = m.addMVar(A.shape[1] - N)
        u0 = ux0[:N]
        m.setObjective((u - u0) @ (u - u0))
        m.addConstr(A[:, :N] @ u + A[:, N:] @ x <= b)
        if C is not None and d is not None:
            m.addConstr(C[:, :N] @ u + C[:, N:] @ x == d)

        m.optimize()

        obj = m.getObjective()

        return np.concatenate([u.X, x.X]), obj.getValue()

def find_X(N, M, u, A, b, C=None, d=None):
    m = gp.Model(env=env)

    x = m.addMVar(N * M)
    eps = m.addVar()

    m.setObjective(eps)
    m.addConstr(A[:, N:] @ x - eps <= u)
    if C is not None and d is not None:
        m.addConstr(C[:, N:] @ x == d)

    m.Params.LogToConsole = 0
    m.Params.FeasibilityTol = 1e-6
    m.Params.OptimalityTol = 1e-6
    m.optimize()
    # m.printQuality()

    if eps.X <= 1e-12:  # high accuracy required on finding a feasible (or over-allocated) allocation
        return x.X.reshape(N, M)
    else:
        return "There is no feasible allocation corresponding to given disutility.", eps.X

def u_is_feasible(N, M, u, A, b, C=None, d=None):
    X = find_X(N, M, u, A, b, C, d)
    if type(X) is np.ndarray:
        return True, X
    else:
        return False, None

def EPM(N, M, D, B, QMO_solver='best', print_quality=False, print_progress=False, ignore_print=False, return_eq=False):

    # construct extended (u, x) polyhedral
    A, b, C, d = poly_ext_primal(N, M, D, B)

    # pick an initial (infeasible) point (near 0)
    u = u0 = np.amin(D, axis=1) / (N + 1) ** 2  # or / (N + 1)

    # initialize
    MAX_NUM_ITER = 50
    MAX_FEASIBILITY_TOL = 1e-6
    num_QMO_a1, num_QMO_e = MAX_NUM_ITER, MAX_NUM_ITER
    solved_a1, solved_e = False, False
    running_time_a1, running_time_e = None, None

    # counters
    num_QMO = 0
    running_time = 0

    # analysis
    solve_QP_time = 0
    solve_QP_num = 0

    for k in range(MAX_NUM_ITER):

        # main
        QP_start = EPM_start = time.time()

        # choose the solver which performs best on this iteration
        if QMO_solver == 'best':
            solver_list = ['OSQP', 'GUROBI', 'MOSEK']
            res_dict = {
                'solver': [],
                'feasibility tolerance (ineq)': [],
                'feasibility tolerance (eq)': [],
                'objective value': [],
                'feasible objective value': []
            }
            u_obj_list = []
            for solver in solver_list:
                u_, min_dist = QMO(np.concatenate([u, np.zeros(M * N)]),
                            N,
                            np.concatenate([A, np.concatenate([-np.identity(N), np.zeros((N, M * N))], axis=1)], axis=0),
                            np.concatenate([b, -u]),
                            C,
                            d,
                            solver=solver
                )

                feasibility_tol_ineq = np.maximum(A @ u_ - b, 0)
                if C is not None and d is not None:
                    feasibility_tol_eq = np.abs(C @ u_ - d)
                res_dict['solver'].append(solver)
                res_dict['feasibility tolerance (ineq)'].append(max(feasibility_tol_ineq))
                res_dict['feasibility tolerance (eq)'].append(max(feasibility_tol_eq))
                res_dict['objective value'].append(min_dist)
                if max(max(feasibility_tol_ineq), max(feasibility_tol_eq)) > MAX_FEASIBILITY_TOL:
                    res_dict['feasible objective value'].append(np.inf)  # only consider feasible solution
                else:
                    res_dict['feasible objective value'].append(min_dist)
                u_obj_list.append((u_, min_dist))
            # print(type(res_dict['objective value'][0]))
            res_dict[f'obj value vs {solver_list[0]}'] = res_dict['objective value'] - res_dict['objective value'][0]

            selected_solver = solver_list[np.argmin(res_dict['feasible objective value'])]
            u_, min_dist = u_obj_list[np.argmin(res_dict['feasible objective value'])]
            if print_quality:
                res_table = pd.DataFrame(res_dict)
                print(selected_solver)

        # choose one specific solver
        else:
            ux_, min_dist = QMO(np.concatenate([u, np.zeros(M * N)]),
                            N,
                            np.concatenate([A, np.concatenate([-np.identity(N), np.zeros((N, M * N))], axis=1)], axis=0),
                            np.concatenate([b, -u]),
                            C,
                            d,
                            solver=QMO_solver
            )
            solve_QP_time += time.time() - QP_start
            solve_QP_num += 1

        u_, x = ux_[:N], ux_[N:].reshape(N, M)
        a = u_ - u
        a = N * a / sum(a * u_)
        u = 1 / a

        running_time += time.time() - EPM_start
        num_QMO += 1

        # evaluation
        c = a.reshape(-1, 1) * D
        p = np.amin(c, axis=0)
        try:
            eps = eps_approx_eq(N, M, D, B, p, x)
        except:
            print("--- Note ---")
            break

        next_u_is_feasible, x = u_is_feasible(N, M, u, A, b, C, d)

        if type(eps) is str and next_u_is_feasible:  # a, p are not precise, choose: report unsolved
            break
        if type(eps) is not str and eps > 1:
            break
        if type(eps) is not str:
            if eps <= APPROXIMATE_THR and num_QMO_a1 == MAX_NUM_ITER:
                solved_a1 = True
                num_QMO_a1 = num_QMO
                running_time_a1 = running_time
        if print_progress:
            print(eps)

        if next_u_is_feasible:
            c = a.reshape(-1, 1) * D
            p = np.amin(c, axis=0)
            if ignore_print:
                eps = eps_approx_eq(N, M, D, B, p, x, ignore_print=True)
            elif print_progress:
                eps = eps_approx_eq(N, M, D, B, p, x, report_all=True)
            else:
                eps = eps_approx_eq(N, M, D, B, p, x)

            if type(eps) is not str and eps <= APPROXIMATE_THR and num_QMO_a1 > num_QMO:
                solved_a1 = True
                num_QMO_a1 = num_QMO
                running_time_a1 = running_time

            if type(eps) is not str and eps <= EXACT_THR:
                solved_e = True
                num_QMO_e = num_QMO
                running_time_e = running_time

            if print_progress:
                print()
            break

    # print("Average QP solving time:", solve_QP_time / solve_QP_num)
    if return_eq:
        if solved_e:
            print("p:\n", np.round(p, 3))
            print("x:\n", np.round(x, 3))
            print("u:\n", np.round(u, 3))
    return num_QMO_a1, num_QMO_e, solved_a1, solved_e, running_time_a1, running_time_e


"""
run, save, and plot
"""

def run_GFW_vs_EPM(N, M, random_generating_method='uniform', num_seeds=10, num_iter_max_cap=80, running_time_max_cap=100, return_eq=False):

    def average(lst):  # return the average of a list
        num_nonNone = 0
        sum = 0

        for e in lst:
            if e is not None:
                num_nonNone += 1
                sum += e

        if num_nonNone > 0:
            return sum / num_nonNone
        else:
            return np.inf

    seeds = range(num_seeds)  # set how many instances we want to try for one size
    num_LMO_list = [[], []]
    running_time_GFW_list = [[], []]
    num_ins_GFW_solved = [0, 0]
    data_GFW = {
        'size': f'N*M = {N}*{M}',
        'num-iter-a1': None,
        'num-iter-e': None,
        'solved-a1': None,
        'solved-e': None,
        'running-time-a1': None,
        'running-time-e': None,
    }
    num_QMO_list = [[], []]
    running_time_EPM_list = [[], []]
    num_ins_EPM_solved = [0, 0]
    data_EPM = {
        'size': f'N*M = {N}*{M}',
        'num-iter-a1': None,
        'num-iter-e': None,
        'solved-a1': None,
        'solved-e': None,
        'running-time-a1': None,
        'running-time-e': None,
    }

    print(f"-*-*-*- SIZE: {N}*{M} -*-*-*-")
    print("PROGRESS: ", end ="")

    for s in seeds:
        np.random.seed(s)
        if random_generating_method == 'randint':
            D = np.random.randint(low=1, high=1001, size=(N, M))
        elif random_generating_method == 'uniform':
            D = np.random.uniform(size=(N, M))
        elif random_generating_method == 'lognormal':
            D = np.random.lognormal(size=(N, M))
        elif random_generating_method == 'truncnormal':
            D = truncnorm.rvs(a=1e-3, b=10, size=(N, M), random_state=s)
        elif random_generating_method == 'exponential':
            D = np.random.exponential(size=(N, M))

        B = np.ones(shape=N)

        '''
        Run the Greedy Frank Wolfe algorithm
        '''

        res_GFW = GFW(N, M, D, B, return_eq=return_eq)
        res_EPM = EPM(N, M, D, B, QMO_solver='GUROBI', ignore_print=True, return_eq=return_eq)

        if s + 1 < num_seeds:
            print("o", end ="")
        else:
            print("o")

        for i in range(2):
            num_ins_GFW_solved[i] += res_GFW[2 + i]
            num_ins_EPM_solved[i] += res_EPM[2 + i]
            if res_GFW[2 + i]:
                num_LMO_list[i].append(res_GFW[i])
                running_time_GFW_list[i].append(res_GFW[4 + i])
            if res_EPM[2 + i]:
                num_QMO_list[i].append(res_EPM[i])
                running_time_EPM_list[i].append(res_EPM[4 + i])

    data_GFW['num-iter-a1'] = min(average(num_LMO_list[0]), num_iter_max_cap)
    data_GFW['num-iter-e'] = min(average(num_LMO_list[1]), num_iter_max_cap)
    data_GFW['solved-a1'] = num_ins_GFW_solved[0]
    data_GFW['solved-e'] = num_ins_GFW_solved[1]
    data_GFW['running-time-a1'] = min(average(running_time_GFW_list[0]), running_time_max_cap)
    data_GFW['running-time-e'] = min(average(running_time_GFW_list[1]), running_time_max_cap)

    data_EPM['num-iter-a1'] = min(average(num_QMO_list[0]), num_iter_max_cap)
    data_EPM['num-iter-e'] = min(average(num_QMO_list[1]), num_iter_max_cap)
    data_EPM['solved-a1'] = num_ins_EPM_solved[0]
    data_EPM['solved-e'] = num_ins_EPM_solved[1]
    data_EPM['running-time-a1'] = min(average(running_time_EPM_list[0]), running_time_max_cap)
    data_EPM['running-time-e'] = min(average(running_time_EPM_list[1]), running_time_max_cap)

    # print(f"STATS: {data_GFW['num-iter-e']}/{data_GFW['solved-e']}/{data_GFW['running-time-e']} vs {data_EPM['num-iter-e']}/{data_EPM['solved-e']}/{data_EPM['running-time-e']}")

    return data_GFW, data_EPM

def run_and_save(size_list=[2, 50, 100], random_generating_method='uniform', num_seeds=10, download_csv=False):

    dict_ = {
        'x': list(),
        'i_y1': list(),
        'i_y2': list(),
        'i_z1': list(),
        'i_z2': list(),
        's_y1': list(),
        's_y2': list(),
        's_z1': list(),
        's_z2': list(),
        'r_y1': list(),
        'r_y2': list(),
        'r_z1': list(),
        'r_z2': list()
    }

    for size in size_list:
        N = size
        M = size
        dict_['x'].append(size)
        res_GFW, res_EPM = run_GFW_vs_EPM(N, M, random_generating_method, num_seeds)

        dict_['i_y1'].append(res_GFW['num-iter-a1'])
        dict_['s_y1'].append(res_GFW['solved-a1'])
        dict_['r_y1'].append(res_GFW['running-time-a1'])
        dict_['i_y2'].append(res_GFW['num-iter-e'])
        dict_['s_y2'].append(res_GFW['solved-e'])
        dict_['r_y2'].append(res_GFW['running-time-e'])

        dict_['i_z1'].append(res_EPM['num-iter-a1'])
        dict_['s_z1'].append(res_EPM['solved-a1'])
        dict_['r_z1'].append(res_EPM['running-time-a1'])
        dict_['i_z2'].append(res_EPM['num-iter-e'])
        dict_['s_z2'].append(res_EPM['solved-e'])
        dict_['r_z2'].append(res_EPM['running-time-e'])

    df = pd.DataFrame.from_dict(dict_)
    df.to_csv(f'{random_generating_method}.csv')

    return dict_

def plot_and_save(data, random_generating_method, num_seeds=10, download_fig=False):

    plt.figure()
    plt.plot(data['x'], data['r_y1'], label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['r_y2'], label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['r_z1'], label=f'EPM: Approximate', marker='*', color='darkorange', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['r_z2'], label=f'EPM: Exact', marker='*', color='darkorange', linewidth=2.5, markersize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("Running time in seconds", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"rt_{random_generating_method}.png")


    plt.figure()
    plt.plot(data['x'], data['i_y1'], label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['i_y2'], label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['i_z1'], label=f'EPM: Approximate', marker='*', color='darkorange', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['i_z2'], label=f'EPM: Exact', marker='*', color='darkorange', linewidth=2.5, markersize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("# iterations to reach CE", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"ni_{random_generating_method}.png")

    plt.figure()
    plt.plot(data['x'], np.array(data['s_y1']) / num_seeds, label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], np.array(data['s_y2']) / num_seeds, label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)
    plt.plot(data['x'], np.array(data['s_z1']) / num_seeds, label=f'EPM: Approximate', marker='*', color='darkorange', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], np.array(data['s_z2']) / num_seeds, label=f'EPM: Exact', marker='*', color='darkorange', linewidth=2.5, markersize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("Ratio of solved instances", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"sr_{random_generating_method}.png")


# if __name__ == "__main__": 

#     size_list = [2, 50, 100, 150, 200, 250, 300]
#     rgm_list = ['uniform', 'lognormal', 'truncnormal', 'exponential', 'randint']
#     rgm_list = ['exponential', 'randint']

#     for rgm in rgm_list:
#         print(f"================== {rgm} ==================")
#         data = run_and_save(size_list=size_list, random_generating_method=rgm, num_seeds=100)
#         plot_and_save(data, random_generating_method=rgm, num_seeds=100)


"""
Note:
1.   'randint (1-10)' 102*102 EPM -> simple because all equilibrium p = 1;
2.   We can change p0 to help GFW;
"""





"""
Experiments on AAMAS bidding data
"""

def distance_matrix_among_papers(D): 

    M = D.shape[1]
    X = D.T 
    distance_matrix = np.zeros((M, M))
    for j in range(M): 
        for j_ in range(M): 
            distance_matrix[j][j_] = sum((X[j] - X[j_]) ** 2)

    return distance_matrix 


def run_GFW_vs_EPM_on_bidding_data(N, M, D, distance_matrix, with_noise=True, num_seeds=10, num_iter_max_cap=80, running_time_max_cap=100, return_eq=False):

    def average(lst):  # return the average of a list
        num_nonNone = 0
        sum = 0

        for e in lst:
            if e is not None:
                num_nonNone += 1
                sum += e

        if num_nonNone > 0:
            return sum / num_nonNone
        else:
            return np.inf

    seeds = range(num_seeds)  # set how many instances we want to try for one size
    num_LMO_list = [[], []]
    running_time_GFW_list = [[], []]
    num_ins_GFW_solved = [0, 0]
    data_GFW = {
        'size': f'N*M = {N}*{M}',
        'num-iter-a1': None,
        'num-iter-e': None,
        'solved-a1': None,
        'solved-e': None,
        'running-time-a1': None,
        'running-time-e': None,
    }
    num_QMO_list = [[], []]
    running_time_EPM_list = [[], []]
    num_ins_EPM_solved = [0, 0]
    data_EPM = {
        'size': f'N*M = {N}*{M}',
        'num-iter-a1': None,
        'num-iter-e': None,
        'solved-a1': None,
        'solved-e': None,
        'running-time-a1': None,
        'running-time-e': None,
    }

    print(f"-*-*-*- SIZE: {N}*{M} -*-*-*-")
    print("PROGRESS: ", end ="")

    for s in seeds:
        # randomly pick a paper 
        np.random.seed(s)
        j = np.random.randint(D.shape[1])

        # find M-nearest papers
        M_nearest_neighbors = np.argsort(distance_matrix[j])[:M]

        # select reviewers with most responses 
        reviewer_num_response = np.ones(D.shape[0], dtype=int)
        for i in range(D.shape[0]):
            for j in M_nearest_neighbors: 
                if D[i][j] != 5: 
                    reviewer_num_response[i] += 1 
        M_top_response_reviewers = np.flip(np.argsort(reviewer_num_response))[:M] 

        # attain sampled D
        D_sampled = D[np.ix_(M_top_response_reviewers, M_nearest_neighbors)]

        if with_noise:
            np.random.seed(s)
            noise = np.random.normal(size=(N, M)) 
            D_sampled = np.maximum(D_sampled + noise, 1e-3)

        # all budgets are set to 1 
        B = np.ones(shape=N)

        '''
        Run the Greedy Frank Wolfe algorithm
        '''

        res_GFW = GFW(N, M, D_sampled, B, return_eq=return_eq)
        res_EPM = EPM(N, M, D_sampled, B, QMO_solver='GUROBI', ignore_print=True, return_eq=return_eq)

        if s + 1 < num_seeds:
            print(f"{s}o", end ="")
        else:
            print(f"{s}o")

        for i in range(2):
            num_ins_GFW_solved[i] += res_GFW[2 + i]
            num_ins_EPM_solved[i] += res_EPM[2 + i]
            if res_GFW[2 + i]:
                num_LMO_list[i].append(res_GFW[i])
                running_time_GFW_list[i].append(res_GFW[4 + i])
            if res_EPM[2 + i]:
                num_QMO_list[i].append(res_EPM[i])
                running_time_EPM_list[i].append(res_EPM[4 + i])

    data_GFW['num-iter-a1'] = min(average(num_LMO_list[0]), num_iter_max_cap)
    data_GFW['num-iter-e'] = min(average(num_LMO_list[1]), num_iter_max_cap)
    data_GFW['solved-a1'] = num_ins_GFW_solved[0]
    data_GFW['solved-e'] = num_ins_GFW_solved[1]
    data_GFW['running-time-a1'] = min(average(running_time_GFW_list[0]), running_time_max_cap)
    data_GFW['running-time-e'] = min(average(running_time_GFW_list[1]), running_time_max_cap)

    data_EPM['num-iter-a1'] = min(average(num_QMO_list[0]), num_iter_max_cap)
    data_EPM['num-iter-e'] = min(average(num_QMO_list[1]), num_iter_max_cap)
    data_EPM['solved-a1'] = num_ins_EPM_solved[0]
    data_EPM['solved-e'] = num_ins_EPM_solved[1]
    data_EPM['running-time-a1'] = min(average(running_time_EPM_list[0]), running_time_max_cap)
    data_EPM['running-time-e'] = min(average(running_time_EPM_list[1]), running_time_max_cap)

    # print(f"STATS: {data_GFW['num-iter-e']}/{data_GFW['solved-e']}/{data_GFW['running-time-e']} vs {data_EPM['num-iter-e']}/{data_EPM['solved-e']}/{data_EPM['running-time-e']}")

    return data_GFW, data_EPM


# def run_GFW_vs_EPM_on_bidding_data(D_):

#     def average(lst):  # return the average of a list
#         num_nonNone = 0
#         sum = 0

#         for e in lst:
#             if e is not None:
#                 num_nonNone += 1
#                 sum += e

#         if num_nonNone > 0:
#             return sum / num_nonNone
#         else:
#             return np.inf

#     N, M = D_.shape
#     num_LMO_list = [[], []]
#     running_time_GFW_list = [[], []]
#     num_ins_GFW_solved = [0, 0]
#     data_GFW = {
#         'size': f'N*M = {N}*{M}',
#         'num-iter-a1': None,
#         'num-iter-e': None,
#         'solved-a1': None,
#         'solved-e': None,
#         'running-time-a1': None,
#         'running-time-e': None,
#     }
#     num_QMO_list = [[], []]
#     running_time_EPM_list = [[], []]
#     num_ins_EPM_solved = [0, 0]
#     data_EPM = {
#         'size': f'N*M = {N}*{M}',
#         'num-iter-a1': None,
#         'num-iter-e': None,
#         'solved-a1': None,
#         'solved-e': None,
#         'running-time-a1': None,
#         'running-time-e': None,
#     }

#     B = np.ones(shape=N)

#     '''
#     Run the Greedy Frank Wolfe algorithm
#     '''

#     res_GFW = GFW(N, M, D_, B)
#     # res_GFW = GFW(N, M, D_, B, return_eq=True)
#     res_EPM = EPM(N, M, D_, B, QMO_solver='GUROBI', ignore_print=True)
#     # res_EPM = EPM(N, M, D_, B, QMO_solver='GUROBI', ignore_print=True, return_eq=True)

#     for i in range(2):
#         num_ins_GFW_solved[i] += res_GFW[2 + i]
#         num_ins_EPM_solved[i] += res_EPM[2 + i]
#         if res_GFW[2 + i] and res_EPM[2 + i]:
#             num_LMO_list[i].append(res_GFW[i])
#             num_QMO_list[i].append(res_EPM[i])
#             running_time_GFW_list[i].append(res_GFW[4 + i])
#             running_time_EPM_list[i].append(res_EPM[4 + i])

#     data_GFW['num-iter-a1'] = average(num_LMO_list[0])
#     data_GFW['num-iter-e'] = average(num_LMO_list[1])
#     data_GFW['solved-a1'] = num_ins_GFW_solved[0]
#     data_GFW['solved-e'] = num_ins_GFW_solved[1]
#     data_GFW['running-time-a1'] = average(running_time_GFW_list[0])
#     data_GFW['running-time-e'] = average(running_time_GFW_list[1])

#     data_EPM['num-iter-a1'] = average(num_QMO_list[0])
#     data_EPM['num-iter-e'] = average(num_QMO_list[1])
#     data_EPM['solved-a1'] = num_ins_EPM_solved[0]
#     data_EPM['solved-e'] = num_ins_EPM_solved[1]
#     data_EPM['running-time-a1'] = average(running_time_EPM_list[0])
#     data_EPM['running-time-e'] = average(running_time_EPM_list[1])

#     return data_GFW, data_EPM

# def cluster_and_sort(D, n_clusters=60, random_state=2024):  # sort groups by frequency

#     X = D.T
#     clusters = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(X)
#     unique, counts = np.unique(clusters.labels_, return_counts=True)
#     # print(unique, counts)
#     count_sorted_groups = np.argsort(counts)

#     return clusters, count_sorted_groups

# def get_sampled_bidding_data(D, clusters, selected_label, show_selected=True):

#     N, M = D.shape
#     selected_reviewers = []
#     selected_papers = []
#     for j in range(M):
#         if clusters.labels_[j] in selected_label:
#             selected_papers.append(j)
#             for i in range(N):
#                 if D[i][j] != 5:
#                     selected_reviewers.append(i)
#     selected_reviewers = np.unique(selected_reviewers)
#     selected_papers = np.unique(selected_papers)
#     selected_reviewers = np.random.choice(selected_reviewers, size=len(selected_papers), replace=False)
#     selected_papers = np.sort(selected_papers)
#     selected_reviewers = np.sort(selected_reviewers)

#     if show_selected:
#         print()
#         print("selected papers", selected_papers)
#         print("selected reviewers", selected_reviewers)

#     D_sampled = D[np.ix_(selected_reviewers, selected_papers)]

#     return D_sampled

"""
plot and save
"""

# def run_and_save_sampled_bidding_data(D, cut_list=[9, 30, 50, 55, 58, 60], with_noise=True, download_csv=False, show_selected=True):

#     clusters, count_sorted_groups = cluster_and_sort(D)

#     dict_ = {
#         'x': list(),
#         'i_y1': list(),
#         'i_y2': list(),
#         'i_z1': list(),
#         'i_z2': list(),
#         's_y1': list(),
#         's_y2': list(),
#         's_z1': list(),
#         's_z2': list(),
#         'r_y1': list(),
#         'r_y2': list(),
#         'r_z1': list(),
#         'r_z2': list()
#     }

#     if with_noise:
#         print("================== Sampled Bidding Data with Uniform Noise ==================")
#     else:
#         print("================== Sampled Bidding Data without Noise ==================")

#     for i in cut_list:
#         selected_label = count_sorted_groups[:i + 1]
#         D_sampled = get_sampled_bidding_data(D, clusters, selected_label, show_selected)
#         size = D_sampled.shape[0]
#         dict_['x'].append(size)
#         print(f"================== Cut: {i}; Size: {size} ==================")
#         N, M = D_sampled.shape
#         if with_noise:
#             noise = np.random.uniform(size=(N, M))
#             res_GFW_bidding_data, res_EPM_bidding_data = run_GFW_vs_EPM_on_bidding_data(np.maximum(D_sampled + noise, 1e-3))
#         else:
#             res_GFW_bidding_data, res_EPM_bidding_data = run_GFW_vs_EPM_on_bidding_data(D_sampled)

#         dict_['i_y1'].append(res_GFW_bidding_data['num-iter-a1'])
#         dict_['s_y1'].append(res_GFW_bidding_data['solved-a1'])
#         dict_['r_y1'].append(res_GFW_bidding_data['running-time-a1'])
#         dict_['i_y2'].append(res_GFW_bidding_data['num-iter-e'])
#         dict_['s_y2'].append(res_GFW_bidding_data['solved-e'])
#         dict_['r_y2'].append(res_GFW_bidding_data['running-time-e'])

#         dict_['i_z1'].append(res_EPM_bidding_data['num-iter-a1'])
#         dict_['s_z1'].append(res_EPM_bidding_data['solved-a1'])
#         dict_['r_z1'].append(res_EPM_bidding_data['running-time-a1'])
#         dict_['i_z2'].append(res_EPM_bidding_data['num-iter-e'])
#         dict_['s_z2'].append(res_EPM_bidding_data['solved-e'])
#         dict_['r_z2'].append(res_EPM_bidding_data['running-time-e'])

#     df = pd.DataFrame.from_dict(dict_)

#     if with_noise:
#         file_name = "bidding_data_with_noise"
#     else:
#         file_name = "bidding_data_without_noise"
#     df.to_csv(f"{file_name}.csv")

#     return dict_, file_name

def run_and_save_bidding_data(D, distance_matrix, size_list=[2, 50, 100], num_seeds=10, with_noise=True):

    dict_ = {
        'x': list(),
        'i_y1': list(),
        'i_y2': list(),
        'i_z1': list(),
        'i_z2': list(),
        's_y1': list(),
        's_y2': list(),
        's_z1': list(),
        's_z2': list(),
        'r_y1': list(),
        'r_y2': list(),
        'r_z1': list(),
        'r_z2': list()
    }

    for size in size_list:
        N = size
        M = size
        dict_['x'].append(size)
        
        res_GFW, res_EPM = run_GFW_vs_EPM_on_bidding_data(N, M, D, distance_matrix, with_noise=with_noise, num_seeds=num_seeds)

        dict_['i_y1'].append(res_GFW['num-iter-a1'])
        dict_['s_y1'].append(res_GFW['solved-a1'])
        dict_['r_y1'].append(res_GFW['running-time-a1'])
        dict_['i_y2'].append(res_GFW['num-iter-e'])
        dict_['s_y2'].append(res_GFW['solved-e'])
        dict_['r_y2'].append(res_GFW['running-time-e'])

        dict_['i_z1'].append(res_EPM['num-iter-a1'])
        dict_['s_z1'].append(res_EPM['solved-a1'])
        dict_['r_z1'].append(res_EPM['running-time-a1'])
        dict_['i_z2'].append(res_EPM['num-iter-e'])
        dict_['s_z2'].append(res_EPM['solved-e'])
        dict_['r_z2'].append(res_EPM['running-time-e'])

    df = pd.DataFrame.from_dict(dict_)
    if with_noise:
        df.to_csv(f'bidding_data_with_noise.csv')
    else:
        df.to_csv(f'bidding_data.csv')

    return dict_

def plot_and_save_bidding_data(data, num_seeds=10, with_noise=True):

    plt.figure()
    plt.plot(data['x'], data['r_y1'], label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['r_y2'], label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['r_z1'], label=f'EPM: Approximate', marker='*', color='darkorange', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['r_z2'], label=f'EPM: Exact', marker='*', color='darkorange', linewidth=2.5, markersize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("Running time in seconds", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    if with_noise:
        plt.savefig(f"rt_bidding_data_with_noise.png")
    else:
        plt.savefig(f"rt_bidding_data.png")


    plt.figure()
    plt.plot(data['x'], data['i_y1'], label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['i_y2'], label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['i_z1'], label=f'EPM: Approximate', marker='*', color='darkorange', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], data['i_z2'], label=f'EPM: Exact', marker='*', color='darkorange', linewidth=2.5, markersize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("# iterations to reach CE", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    if with_noise:
        plt.savefig(f"ni_bidding_data_with_noise.png")
    else:
        plt.savefig(f"ni_bidding_data.png")

    plt.figure()
    plt.plot(data['x'], np.array(data['s_y1']) / num_seeds, label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], np.array(data['s_y2']) / num_seeds, label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)
    plt.plot(data['x'], np.array(data['s_z1']) / num_seeds, label=f'EPM: Approximate', marker='*', color='darkorange', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['x'], np.array(data['s_z2']) / num_seeds, label=f'EPM: Exact', marker='*', color='darkorange', linewidth=2.5, markersize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("Ratio of solved instances", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    if with_noise:
        plt.savefig(f"sr_bidding_data_with_noise.png")
    else:
        plt.savefig(f"sr_bidding_data.png")


if __name__ == "__main__": 
    df = pd.read_csv('./bidding-data.csv')

    dict_bidder_index = dict()
    i = 0
    for bidder in df['Bidder']:
        if bidder not in dict_bidder_index.keys():
            dict_bidder_index[bidder] = i
            i += 1

    N, M = len(np.unique(df['Bidder'])), max(df['Submission'])
    print(f"N = {N}, M = {M}")
    B = np.ones(shape=N)

    dict_pref_value = {
        'yes': 1,
        'maybe': 3,
        'no response': 5,
        'no': 7,
        'conflict': 7 * M + 1  # optimal price bound?
    }

    dict_pref_value_1 = {
        'yes': 0.1,
        'maybe': 1,
        'no response': 10,
        'no': 100,
        'conflict': 100 * M + 1  # optimal price bound?
    }

    dict_pref_value_2 = {
        'yes': 1,
        'maybe': np.sqrt(3),
        'no response': np.sqrt(5),
        'no': np.sqrt(7),
        'conflict': np.sqrt(7) * M + 1  # optimal price bound?
    }

    D = dict_pref_value['no response'] * np.ones(shape=(N, M))
    for row in list(df.itertuples(index=False, name=None)):
        D[dict_bidder_index[row[0]]][row[1] - 1] = dict_pref_value[row[2]]

    distance_matrix = distance_matrix_among_papers(D)

    # unique, counts = np.unique(D, return_counts=True)
    # print(unique, counts)

    size_list = [2, 50, 100, 150, 200, 250, 300]

    data = run_and_save_bidding_data(D, distance_matrix, size_list=size_list, num_seeds=100, with_noise=True)
    plot_and_save_bidding_data(data, num_seeds=100, with_noise=True)

    # data, file_name = run_and_save_sampled_bidding_data(D, with_noise=True, download_csv=True, show_selected=False)
    # plot_and_save_bidding_data(data, file_name, download_fig=True)