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
import math 
import os

from utils import APPROXIMATE_THR, EXACT_THR, E2TOL, E3TOL, eps_approx_eq
from gfw import *
from epm import *




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

def run_GFW_vs_EPM_on_bidding_data(N, M, D, distance_matrix, with_noise=True, num_seeds=10, num_iter_max_cap=80, running_time_max_cap=100, print_eq=False):

    seeds = range(num_seeds)  # set how many instances we want to try for one size
    num_LMO_list = [[], []]
    running_time_GFW_list = [[], []]
    full_num_LMO_list = [[], []]
    full_running_time_GFW_list = [[], []]
    num_ins_GFW_solved = [0, 0]
    data_GFW = {
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
        'num-iter-a1': None,
        'num-iter-e': None,
        'solved-a1': None,
        'solved-e': None,
        'running-time-a1': None,
        'running-time-e': None,
    }

    print(f"-*-*-*- SIZE: {N}*{M} -*-*-*-")
    print("PROGRESS: ", end ="")

    """
    Collect data for each seed 
    """
    for s in seeds:
        np.random.seed(s)
        while True:  # loop utill we obtain a valid instance
            # randomly pick a paper 
            j = np.random.randint(D.shape[1])

            # find M-nearest papers
            M_nearest_neighbors = np.argsort(distance_matrix[j])[:M]

            # select reviewers with most responses 
            reviewer_num_response = np.ones(D.shape[0], dtype=int)
            for i in range(D.shape[0]):
                for j in M_nearest_neighbors: 
                    if D[i][j] < 5 - 1e3: 
                        reviewer_num_response[i] += 1 
            M_top_response_reviewers = np.flip(np.argsort(reviewer_num_response))[:M] 

            # attain sampled D
            D_sampled = D[np.ix_(M_top_response_reviewers, M_nearest_neighbors)] 

            if max(np.amin(D_sampled, axis=1)) < 6: 
                break

        if with_noise:
            np.random.seed(s)
            noise = 0.2 * np.random.normal(size=(N, M)) 
            D_sampled = np.maximum(D_sampled + noise, 1e-3)

        # all budgets are set to 1 
        B = np.ones(shape=N)

        res_GFW = GFW(N, M, D_sampled, B, print_eq=print_eq, 
                      warm_start=True, LP_solve_method="primal simplex")
        res_EPM = EPM(N, M, D_sampled, B, QMO_solver='GUROBI', ignore_print=True, print_eq=print_eq, 
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
            """
            We only count the iterations and running time for instances that both algorithms can solve
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


def run_and_save_bidding_data(D, distance_matrix, size_list=[2, 50, 100], num_seeds=10, with_noise=True, save_dir='./data'):

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
        res_GFW, res_EPM = run_GFW_vs_EPM_on_bidding_data(N, M, D, distance_matrix, with_noise=with_noise, num_seeds=num_seeds)

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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if with_noise:
        df.to_csv(f'{save_dir}/bidding_data_with_noise.csv')
    else:
        df.to_csv(f'{save_dir}/bidding_data.csv')

    return dict_

def average_and_std(lst):
    nonNone_list = list()

    for e in lst:
        if (e is not None) and (e is not np.nan) and (e is not pd.NA) and (not math.isnan(e)):
            nonNone_list.append(e)

    if len(nonNone_list) > 0:
        return np.mean(nonNone_list), np.std(nonNone_list)
    else:
        return None, None


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

    D = dict_pref_value['no response'] * np.ones(shape=(N, M))
    for row in list(df.itertuples(index=False, name=None)):
        D[dict_bidder_index[row[0]]][row[1] - 1] = dict_pref_value[row[2]]

    distance_matrix = distance_matrix_among_papers(D)

    size_list = [5, 50, 100, 150, 200, 250, 300]
    run_and_save_bidding_data(D, distance_matrix, size_list=size_list, num_seeds=100, with_noise=False, save_dir='./data')
    run_and_save_bidding_data(D, distance_matrix, size_list=size_list, num_seeds=100, with_noise=True, save_dir='./data')