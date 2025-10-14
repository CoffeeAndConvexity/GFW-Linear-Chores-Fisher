import numpy as np
import pandas as pd
import time
import cvxpy as cp
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
        QP_solve_method="default", 
        warm_start_primal=None,
        model_prev_iter=None, 
        return_model=False):

    if solver == 'OSQP':  # Warm start has not been implemented for OSQP

        if warm_start_primal is not None:
            print("Note: Warm start has not been implemented for OSQP - ignore warm_start_primal.")

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
        
        # sometimes we specify the QP solving method
        m.Params.LogToConsole = 0
        if QP_solve_method == "barrier":
            m.Params.Method = 2
        else: 
            pass  # use default method

        if warm_start_primal is not None:
            m_vars = m.getVars()
            for i in range(len(m_vars)):
                m_vars[i].PStart = warm_start_primal[i]

        if model_prev_iter is not None or warm_start_primal is not None:
            m.update()
        
        # [option] we can set appropriate parameters here to ask for a higher accuracy - the following settings have been tuned
        m.Params.BarConvTol = 0
        m.Params.FeasibilityTol = 1e-9
        m.Params.OptimalityTol = 1e-9
        m.Params.BarCorrectors = 1000
        m.optimize()

        obj = m.getObjective()

        u_, x = m.getVars()[:N], m.getVars()[N:]
        u__value, x_value = np.array([u_[i].X for i in range(len(u_))]), np.array([x[i].X for i in range(len(x))])

        if return_model:
            return np.concatenate([u__value, x_value]), obj.getValue(), m
        else:
            return np.concatenate([u__value, x_value]), obj.getValue()

def find_x(N, M, u, A, b, C=None, d=None):
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
    x = find_x(N, M, u, A, b, C, d)

    if type(x) is np.ndarray:
        return True, x
    else:
        return False, None

def EPM(N, M, D, B, 
        QMO_solver='best', 
        QP_solve_method="default",
        print_quality=False, 
        print_progress=False, 
        ignore_print=False, 
        print_eq=False, 
        warm_start=True):

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
            if warm_start: 
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
                                QP_solve_method=QP_solve_method,
                                warm_start_primal=np.concatenate([u, x_ws.flatten()]), 
                                model_prev_iter=m,
                                return_model=True
                )
            else:
                ux_, min_dist = QMO(np.concatenate([u, np.zeros(M * N)]),  # used to construct the objective
                                N,
                                A, b,
                                np.concatenate([-np.identity(N), np.zeros((N, M * N))], axis=1), -u, 
                                C, d,
                                solver=QMO_solver, 
                                QP_solve_method=QP_solve_method
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