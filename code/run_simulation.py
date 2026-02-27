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

from utils import APPROXIMATE_THR, EXACT_THR, E2TOL, E3TOL, eps_approx_eq
from gfw import *
from epm import *

print("- cvxpy version:", cp.__version__, "-")
print("- cvxpy installed solvers:", cp.installed_solvers())

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

        res_GFW = GFW(N, M, D, B, print_eq=print_eq, 
                      warm_start=True, LP_solve_method="primal simplex")
        res_EPM = EPM(N, M, D, B, QMO_solver='GUROBI', ignore_print=True, print_eq=print_eq, 
                      warm_start=True, QP_solve_method="barrier")

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

def run_and_save(size_list=[2, 50, 100], random_generating_method='uniform', num_seeds=10, download_csv=False, save_dir="./data"):

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
        if type(size) is int: 
            N = M = size
            dict_['size'].append(size)
        else: 
            N, M = size
            dict_['size'].append(size[1])  # we set the larger dimension in the second position
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
    df.to_csv(f'{save_dir}/{random_generating_method}.csv')

    return dict_

if __name__ == "__main__": 

    size_list = [5, 100, 200, 300, 400, 500]
    rgm_list = ['uniform', 'lognormal', 'truncnormal', 'exponential', 'randint']

    for rgm in rgm_list:
        print(f"================== {rgm} ==================")
        data = run_and_save(size_list=size_list, random_generating_method=rgm, num_seeds=50, save_dir="./data")