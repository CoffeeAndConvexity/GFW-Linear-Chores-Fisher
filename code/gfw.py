import numpy as np
import time
import gurobipy as gp

import sys
import io
import os

from utils import APPROXIMATE_THR, EXACT_THR, E2TOL, E3TOL, eps_approx_eq, gurobi_license_params

'''
Set up Gurobi environment with WLS license (or use a local license if you have one)
'''

# Create an environment with your WLS license parameters
env = gp.Env(params=gurobi_license_params)

print("- Gurobi version:", gp.gurobi.version(), "-")

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
        LP_solve_method="default", 
        return_dual=False, 
        return_model=False): 

    starting_time = time.time()
    
    if model_prev_iter is not None:
        m = model_prev_iter
        m.setObjective(c @ m.getVars()[-len(c):])  # update objective
    else: 
        env.setParam("LogFile", "temp_gfw.log")
        m = gp.Model(env=env)
        N = len(c)

        M = A.shape[1] - len(c)
        p = m.addMVar(M)
        beta = m.addMVar(len(c))

        m.setObjective(c @ beta)  # initial objective
        m.addConstr(A[:, :M] @ p + A[:, M:] @ beta <= b)
        if C is not None and d is not None:
            m.addConstr(C[:, :M] @ p + C[:, M:] @ beta == d)
    modeling_time = time.time() - starting_time
    
    # sometimes we specify the LP solving method
    m.Params.LogToConsole = 0
    if LP_solve_method == "primal simplex":
        m.Params.Method = 0
    else: 
        pass  # use default method

    # if we have a warm start solution from previous iteration, we might want to use it
    if warm_start_primal is not None:
        m_vars = m.getVars()
        for i in range(len(m_vars)):
            m_vars[i].PStart = warm_start_primal[i]
    
    if model_prev_iter is not None or warm_start_primal is not None:
        m.update()
    m.optimize()
    
    # retrieve the number of iterations from lastest log file
    with open("temp_gfw.log", "r") as f:
        lines = f.readlines()
        for line in lines[::-1]:
            if "Solved in " in line:
                num_pivots = int(line.split("Solved in ")[1].split(" iterations")[0])
                break
    
    solving_time = time.time() - starting_time - modeling_time

    # return the optimal value and solution
    p_beta = m.getVars()
    p_beta_value = np.array([p_beta[i].X for i in range(len(p_beta))])
    p_value, beta_value = p_beta_value[:A.shape[1] - len(c)], p_beta_value[-len(c):]
    
    if return_dual and return_model:
        return p_value, beta_value, np.array(m.Pi), m, (modeling_time, solving_time, num_pivots)
    elif return_dual:
        return p_value, beta_value, np.array(m.Pi), (modeling_time, solving_time, num_pivots)
    elif return_model:
        return p_value, beta_value, m, (modeling_time, solving_time, num_pivots)
    else:
        return p_value, beta_value, (modeling_time, solving_time, num_pivots)


'''
Greedy Frank Wolfe - Main Algorithm
    - The algorithm ternimates when an exact CE is found, or the maximum number of iterations is reached
    - At each iteration, we check whether an approximate CE or exact CE is found
    - Running time does not include the time for evaluation
'''
def GFW(N, M, D, B, print_eq=False, max_iter=100, warm_start=True, LP_solve_method="default", report_total_pivots=False, return_eq=False):  

    # create the dual polyhedron
    A, b, C, d = polytope_dual(N, M, D, B)

    # initialize
    MAX_NUM_ITER = max_iter
    num_LMO_a1, num_LMO_e = MAX_NUM_ITER, MAX_NUM_ITER
    solved_a1, solved_e = False, False
    running_time_a1, running_time_e = None, None
    avg_modeling_time, avg_solving_time = 0, 0
    if report_total_pivots:
        total_pivots = 0

    num_LMO = 0
    running_time = 0
    solve_LP_time = 0

    p, beta = feasible_point(M, N, D, B)
    beta_ = beta  # 'beta_' keeps the last beta values

    m = None  # the model from the last iteration, used for warm start

    for k in range(MAX_NUM_ITER):

        LP_start = GFW_start = time.time()

        if warm_start:
            p, beta, gamma, m, time_count = LMO(B / beta, A, b, C, d,
                                            warm_start_primal=np.concatenate([p, beta]), 
                                            model_prev_iter=m, 
                                            LP_solve_method=LP_solve_method,
                                            return_dual=True, 
                                            return_model=True)
        else:
            p, beta, gamma, time_count = LMO(B / beta, A, b, C, d, 
                                             LP_solve_method=LP_solve_method,
                                             return_dual=True)
        
        # [option] when we want to observe the average modeling and solving time
        avg_modeling_time = (avg_modeling_time * num_LMO + time_count[0]) / (num_LMO + 1)
        avg_solving_time = (avg_solving_time * num_LMO + time_count[1]) / (num_LMO + 1)

        if report_total_pivots:
            total_pivots += time_count[2]

        solve_LP_time += time.time() - LP_start
        num_LMO += 1

        # we count the time to update x and beta because we need x as an output - the time cost of this step is minor
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
            if eps <= APPROXIMATE_THR and num_LMO_a1 == MAX_NUM_ITER:  # the second condition ensures that 'num_LMO_1' has not been updated, except we have run out of iteration budget
                solved_a1 = True
                num_LMO_a1 = num_LMO
                running_time_a1 = running_time

            if eps <= EXACT_THR:
                solved_e = True
                num_LMO_e = num_LMO
                running_time_e = running_time
                break  # reach exact CE, terminate

    # print("Average LP time cost percentage:", solve_LP_time / running_time * 100, "%")
    # print("Average modeling time:", avg_modeling_time, "Average solving time:", avg_solving_time)
    if print_eq:
        if solved_e:
            print("p:\n", np.round(p, 3))
            print("x:\n", np.round(x, 3))
            print("u:\n", np.round(B / beta, 3))

    with open("temp_gfw.log", "w") as f:
        f.write("")  # clear the log file

    if report_total_pivots:
        if return_eq and solved_e:
            return num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e, total_pivots, (p, x, B / beta)
        elif return_eq:
            return num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e, total_pivots, None
        return num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e, total_pivots
    else: 
        if return_eq and solved_e:
            return num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e, (p, x, B / beta)
        elif return_eq:
            return num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e, None
        return num_LMO_a1, num_LMO_e, solved_a1, solved_e, running_time_a1, running_time_e