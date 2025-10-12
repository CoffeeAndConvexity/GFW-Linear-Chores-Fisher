import numpy as np
import time
import gurobipy as gp

from utils import APPROXIMATE_THR, EXACT_THR, E2TOL, E3TOL, eps_approx_eq

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
            m_vars[i].PStart = warm_start_primal[i]

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