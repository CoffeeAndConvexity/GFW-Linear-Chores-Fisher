# -*- coding: utf-8 -*-
"""
Chores CEEI Experiments
"""

import numpy as np
np.seterr(divide='ignore')
from scipy.stats import truncnorm
from sklearn.cluster import KMeans
import gurobipy as gp
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import time

'''
Set up Gurobi environment with WLS license (or use a local license if you have one)
'''

# Create an environment with your WLS license
params = {
    "WLSACCESSID": '0f33ff11-ef92-485a-97f0-af4a28654d6b',
    "WLSSECRET": '880ecd7e-6fe7-4566-bb77-2590cc321d04',
    "LICENSEID": 2546016,
}
env = gp.Env(params=params)

print("- cvxpy version:", cp.__version__, "-")
print("- cvxpy installed solvers:", cp.installed_solvers())


"""
Set up 'Approximate' and 'Exact' tolerances
"""

APPROXIMATE_THR = 1e-3
EXACT_THR = 1e-10
E2TOL = 1e-10
E3TOL = 1e-10

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

# Linear Minimization Oracle (LMO)
def LMO(c, A, b, C=None, d=None, 
        warm_start_primal=None,
        model_prev_iter=None, 
        return_dual=False, 
        return_model=False): 

    if model_prev_iter is not None:
        m = model_prev_iter
        m.setObjective(c @ m.getVars()[-len(c):])
    else: 
        m = gp.Model(env=env)
        N = len(c)

        M = A.shape[1] - len(c)
        p = m.addMVar(M)
        beta = m.addMVar(len(c))

        m.setObjective(c @ beta)
        m.addConstr(A[:, :M] @ p + A[:, M:] @ beta <= b)
        if C is not None and d is not None:
            m.addConstr(C[:, :M] @ p + C[:, M:] @ beta == d)
    
    if warm_start_primal is not None:
        m_vars = m.getVars()
        for i in range(len(m_vars)):
            m_vars[i].Start = warm_start_primal[i]

    m.Params.LogToConsole = 0
    # m.Params.Method = 1
    m.update()
    m.optimize()

    # return the optimal value and solution
    p, beta = m.getVars()[:A.shape[1] - len(c)], m.getVars()[-len(c):]
    p_value, beta_value = np.array([p[i].X for i in range(len(p))]), np.array([beta[i].X for i in range(len(beta))])
    
    if return_dual and return_model:
        return p_value, beta_value, np.array(m.Pi), m
    elif return_dual:
        return p_value, beta_value, np.array(m.Pi)
    elif return_model:
        return p_value, beta_value, m
    else:
        return p_value, beta_value


'''
Greedy Frank Wolfe - Main Algorithm
    - The algorithm ternimates when an exact CE is found, or the maximum number of iterations is reached
    - At each iteration, we check whether an approximate CE or exact CE is found
    - Running time does not include the time for evaluation
'''
def GFW(N, M, D, B, print_eq=False, max_iter=80):  

    # create the dual polyhedron
    A, b, C, d = polytope_dual(N, M, D, B)

    # initialize
    MAX_NUM_ITER = max_iter
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

    m = None  # the model from the last iteration, used for warm start

    for k in range(MAX_NUM_ITER):

        # main
        LP_start = GFW_start = time.time()

        p, beta, gamma, m = LMO(B / beta, A, b, C, d,
                                warm_start_primal=np.concatenate([p, beta]), 
                                model_prev_iter=m, 
                                return_dual=True, 
                                return_model=True)

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
            if k == MAX_NUM_ITER - 1: 
                break  # reach maximum number of iterations, terminate
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
                break  # reach exact CE, terminate

    # print("Average LP solving time:", solve_LP_time / solve_LP_num)
    if print_eq:
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
    # add x is nonnegative constraint when solving QP using some solver
    A = np.concatenate(A, axis=0)
    b = np.zeros(N)
    C = np.zeros(shape=(M, N + M * N))
    for j in range(M):
        C[j, N + j: N + M * N: M] = 1
    d = np.ones(M)

    return A, b, C, d

def QMO(ux0, N, 
        A, b,
        A_, b_,  
        C=None, d=None, 
        solver='OSQP', 
        warm_start_primal=None,
        model_prev_iter=None, 
        return_model=False):

    if solver == 'OSQP':  # Warm start has not been implemented for OSQP
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

        if model_prev_iter is not None:
            m = model_prev_iter
            m.setObjective((m.getVars()[:N] - ux0[:N]) @ (m.getVars()[:N] - ux0[:N]))
            # update the constraints on u, by constraint index
            for idx, con in enumerate(m.getConstrs()[N:2*N]): 
                con.RHS = b_[idx]
        else:
            m = gp.Model(env=env)

            u = m.addMVar(N)
            x = m.addMVar(A.shape[1] - N)
            u0 = ux0[:N]
            m.setObjective((u - u0) @ (u - u0))  # need to update u0 in each iteration
            m.addConstr(A[:, :N] @ u + A[:, N:] @ x <= b)
            m.addConstr(A_[:, :N] @ u + A_[:, N:] @ x <= b_)  # need to update RHS in each iteration
            if C is not None and d is not None:
                m.addConstr(C[:, :N] @ u + C[:, N:] @ x == d)
        
        if warm_start_primal is not None:
            m_vars = m.getVars()
            for i in range(len(m_vars)):
                m_vars[i].PStart = warm_start_primal[i]

        m.update()
        m.Params.LogToConsole = 0
        # m.Params.FeasibilityTol = 1e-9
        # m.Params.OptimalityTol = 1e-9
        m.Params.BarConvTol = 0
        # m.Params.BarCorrectors = 10000
        m.optimize()

        obj = m.getObjective()

        u_, x = m.getVars()[:N], m.getVars()[N:]
        u__value, x_value = np.array([u_[i].X for i in range(len(u_))]), np.array([x[i].X for i in range(len(x))])

        if return_model:
            return np.concatenate([u__value, x_value]), obj.getValue(), m
        else:
            return np.concatenate([u__value, x_value]), obj.getValue()

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

def EPM(N, M, D, B, 
        QMO_solver='best', 
        print_quality=False, 
        print_progress=False, 
        ignore_print=False, 
        print_eq=False):

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

    m = None  # the model from the last iteration, used for warm start

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
                            solver=solver, 
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
            x_ws = np.zeros(shape=(N, M))  # used for warm start
            for i in range(N):
                for j in range(M):
                    x_ws[i, j] = u[i] / (M * D[i, j]) 

            ux_, min_dist, m = QMO(np.concatenate([u, np.zeros(M * N)]),  # used to construct the objective
                            N,
                            A, b,
                            np.concatenate([-np.identity(N), np.zeros((N, M * N))], axis=1), -u, 
                            C, d,
                            solver=QMO_solver, 
                            warm_start_primal=np.concatenate([u, x_ws.flatten()]), 
                            model_prev_iter=m,
                            return_model=True
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

        # THIS CASE SHOULD NOT HAPPEN
        # If there is NaN in u, terminate - choose: report unsolved
        if np.sum(np.isnan(u)) > 0:
            print("There is NaN in u, terminate.")
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
    if print_eq:
        if solved_e:
            print("p:\n", np.round(p, 3))
            print("x:\n", np.round(x, 3))
            print("u:\n", np.round(u, 3))

    return num_QMO_a1, num_QMO_e, solved_a1, solved_e, running_time_a1, running_time_e


"""
run, save, and plot
"""

def run_GFW_vs_EPM(N, M, random_generating_method='uniform', num_seeds=10, num_iter_max_cap=80, running_time_max_cap=100, print_eq=False):

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
    full_num_LMO_list = [[], []]
    full_running_time_GFW_list = [[], []]
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

        res_GFW = GFW(N, M, D, B, print_eq=print_eq)
        res_EPM = EPM(N, M, D, B, QMO_solver='GUROBI', ignore_print=True, print_eq=print_eq)

        if res_EPM[3]:  
            if s + 1 < num_seeds:
                print("o", end ="", flush=True)
            else:
                print("o")
        else:
            if s + 1 < num_seeds:
                print("_", end ="", flush=True)
            else:
                print("_")

        for i in range(2):
            num_ins_GFW_solved[i] += res_GFW[2 + i]
            num_ins_EPM_solved[i] += res_EPM[2 + i]
            # if res_GFW[2 + i]:
            #     num_LMO_list[i].append(res_GFW[i])
            #     running_time_GFW_list[i].append(res_GFW[4 + i])
            # if res_EPM[2 + i]:
            #     num_QMO_list[i].append(res_EPM[i])
            #     running_time_EPM_list[i].append(res_EPM[4 + i])
            """
            We switch to only count the iterations and running time for instances that both algorithms can solve
            We also record full GFW stats for instances that GFW can solve; we use this to plot the GFW curves in case EPM cannot solve any instance
            """
            if res_GFW[2 + i] and res_EPM[2 + i]:
                num_LMO_list[i].append(res_GFW[i])
                running_time_GFW_list[i].append(res_GFW[4 + i])
                num_QMO_list[i].append(res_EPM[i])
                running_time_EPM_list[i].append(res_EPM[4 + i])
            if res_GFW[2 + i]:
                full_num_LMO_list[i].append(res_GFW[i])
                full_running_time_GFW_list[i].append(res_GFW[4 + i])


    data_GFW['solved-a1'] = num_ins_GFW_solved[0]
    data_GFW['solved-e'] = num_ins_GFW_solved[1]
    data_EPM['solved-a1'] = num_ins_EPM_solved[0]
    data_EPM['solved-e'] = num_ins_EPM_solved[1]


    if data_EPM['solved-a1'] > 0.05 * num_seeds:  # if EPM can solve at least 5% of the instances to approximate CE, use the instances that both algorithms can solve to compute the stats
        data_GFW['num-iter-a1'] = min(average(num_LMO_list[0]), num_iter_max_cap)
        data_GFW['running-time-a1'] = min(average(running_time_GFW_list[0]), running_time_max_cap)
        data_EPM['num-iter-a1'] = min(average(num_QMO_list[0]), num_iter_max_cap)
        data_EPM['running-time-a1'] = min(average(running_time_EPM_list[0]), running_time_max_cap)
    else:  # otherwise, use all instances that GFW can solve to compute the stats
        data_GFW['num-iter-a1'] = min(average(full_num_LMO_list[0]), num_iter_max_cap)
        data_GFW['running-time-a1'] = min(average(full_running_time_GFW_list[0]), running_time_max_cap)
        data_EPM['num-iter-a1'] = None
        data_EPM['running-time-a1'] = None

    if data_EPM['solved-e'] > 0.05 * num_seeds:  # if EPM can solve at least 5% of the instances to exact CE, use the instances that both algorithms can solve to compute the stats
        data_GFW['num-iter-e'] = min(average(num_LMO_list[1]), num_iter_max_cap)
        data_GFW['running-time-e'] = min(average(running_time_GFW_list[1]), running_time_max_cap)
        data_EPM['num-iter-e'] = min(average(num_QMO_list[1]), num_iter_max_cap)
        data_EPM['running-time-e'] = min(average(running_time_EPM_list[1]), running_time_max_cap)
    else:  # otherwise, use all instances that GFW can solve to compute the stats
        data_GFW['num-iter-e'] = min(average(full_num_LMO_list[1]), num_iter_max_cap)
        data_GFW['running-time-e'] = min(average(full_running_time_GFW_list[1]), running_time_max_cap)
        data_EPM['num-iter-e'] = None
        data_EPM['running-time-e'] = None

    print(f"STATS: {data_GFW['num-iter-e']}/{data_GFW['solved-e']}/{data_GFW['running-time-e']} vs {data_EPM['num-iter-e']}/{data_EPM['solved-e']}/{data_EPM['running-time-e']}")

    return data_GFW, data_EPM

def run_and_save(size_list=[2, 50, 100], random_generating_method='uniform', num_seeds=10, download_csv=False):

    dict_ = {
        'size': list(),
        'iteration_GFW_a1': list(),
        'iteration_GFW_e': list(),
        'iteration_EPM_a1': list(),
        'iteration_EPM_e': list(),
        'solved_GFW_a1': list(),
        'solved_GFW_e': list(),
        'solved_EPM_a1': list(),
        'solved_EPM_e': list(),
        'runningtime_GFW_a1': list(),
        'runningtime_GFW_e': list(),
        'runningtime_EPM_a1': list(),
        'runningtime_EPM_e': list()
    }

    for size in size_list:
        N = size
        M = size
        dict_['size'].append(size)
        res_GFW, res_EPM = run_GFW_vs_EPM(N, M, random_generating_method, num_seeds, num_iter_max_cap=800, running_time_max_cap=1000)

        dict_['iteration_GFW_a1'].append(res_GFW['num-iter-a1'])
        dict_['solved_GFW_a1'].append(res_GFW['solved-a1'])
        dict_['runningtime_GFW_a1'].append(res_GFW['running-time-a1'])
        dict_['iteration_GFW_e'].append(res_GFW['num-iter-e'])
        dict_['solved_GFW_e'].append(res_GFW['solved-e'])
        dict_['runningtime_GFW_e'].append(res_GFW['running-time-e'])

        dict_['iteration_EPM_a1'].append(res_EPM['num-iter-a1'])
        dict_['solved_EPM_a1'].append(res_EPM['solved-a1'])
        dict_['runningtime_EPM_a1'].append(res_EPM['running-time-a1'])
        dict_['iteration_EPM_e'].append(res_EPM['num-iter-e'])
        dict_['solved_EPM_e'].append(res_EPM['solved-e'])
        dict_['runningtime_EPM_e'].append(res_EPM['running-time-e'])

    df = pd.DataFrame.from_dict(dict_)
    df.to_csv(f'{random_generating_method}_warm_start.csv')

    return dict_

def plot_and_save(data, random_generating_method, num_seeds=10, download_fig=False):

    plt.figure()
    plt.plot(data['size'], data['runningtime_GFW_a1'], label=f'GFW: Approximate', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['size'], data['runningtime_GFW_e'], label=f'GFW: Exact', marker='^', color='g', linewidth=2.5, markersize=18)
    plt.plot(data['size'], data['runningtime_EPM_a1'], label=f'EPM: Approximate', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['size'], data['runningtime_EPM_e'], label=f'EPM: Exact', marker='*', color='orange', linewidth=2.5, markersize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("Running time in seconds", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"rt_{random_generating_method}_warm_start.png")


    plt.figure()
    plt.plot(data['size'], data['iteration_GFW_a1'], label=f'GFW: Approximate', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['size'], data['iteration_GFW_e'], label=f'GFW: Exact', marker='^', color='g', linewidth=2.5, markersize=18)
    plt.plot(data['size'], data['iteration_EPM_a1'], label=f'EPM: Approximate', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['size'], data['iteration_EPM_e'], label=f'EPM: Exact', marker='*', color='orange', linewidth=2.5, markersize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("# iterations to reach CE", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"ni_{random_generating_method}_warm_start.png")

    plt.figure()
    plt.plot(data['size'], np.array(data['solved_GFW_a1']) / num_seeds, label=f'GFW: Approximate', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['size'], np.array(data['solved_GFW_e']) / num_seeds, label=f'GFW: Exact', marker='^', color='g', linewidth=2.5, markersize=18)
    plt.plot(data['size'], np.array(data['solved_EPM_a1']) / num_seeds, label=f'EPM: Approximate', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['size'], np.array(data['solved_EPM_e']) / num_seeds, label=f'EPM: Exact', marker='*', color='orange', linewidth=2.5, markersize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("Ratio of solved instances", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"sr_{random_generating_method}_warm_start.png")


if __name__ == "__main__": 

    # size_list = [2, 50, 100, 150, 200, 250, 300]
    size_list = [5, 100, 200, 300, 400]
    # rgm_list = ['uniform', 'lognormal', 'truncnormal', 'exponential', 'randint']
    rgm_list = ['lognormal', 'truncnormal', 'exponential', 'randint', 'uniform', ]

    for rgm in rgm_list:
        print(f"================== {rgm} ==================")
        data = run_and_save(size_list=size_list, random_generating_method=rgm, num_seeds=100)
        # plot_and_save(data, random_generating_method=rgm, num_seeds=100)