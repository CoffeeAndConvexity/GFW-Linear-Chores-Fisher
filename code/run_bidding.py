# -*- coding: utf-8 -*-
"""
Chores CEEI Experiments
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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

    def average_std(lst):  # return the average of a list
        num_nonNone = 0
        sum = 0
        sum_sq = 0

        for e in lst:
            if e is not None:
                num_nonNone += 1
                sum += e
                sum_sq += e * e

        if num_nonNone > 0:
            mean = sum / num_nonNone
            variance = (sum_sq - num_nonNone * mean * mean) / (num_nonNone - 1) if num_nonNone > 1 else 0
            return mean, np.sqrt(variance)
        else:
            return np.inf, np.inf

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
        data_GFW_num_iter_a1 = average_std(num_LMO_list[0])
        data_GFW['num-iter-a1'] = min(data_GFW_num_iter_a1[0], num_iter_max_cap)
        data_GFW['num-iter-a1-std'] = data_GFW_num_iter_a1[1]
        data_GFW_running_time_a1 = average_std(running_time_GFW_list[0])
        data_GFW['running-time-a1'] = min(data_GFW_running_time_a1[0], running_time_max_cap)
        data_GFW['running-time-a1-std'] = data_GFW_running_time_a1[1]
        data_EPM_num_iter_a1 = average_std(num_QMO_list[0])
        data_EPM['num-iter-a1'] = min(data_EPM_num_iter_a1[0], num_iter_max_cap)
        data_EPM['num-iter-a1-std'] = data_EPM_num_iter_a1[1]
        data_EPM_running_time_a1 = average_std(running_time_EPM_list[0])
        data_EPM['running-time-a1'] = min(data_EPM_running_time_a1[0], running_time_max_cap)
        data_EPM['running-time-a1-std'] = data_EPM_running_time_a1[1]
    else:  # otherwise, use all instances that GFW can solve to compute the stats
        data_GFW_num_iter_a1 = average_std(full_num_LMO_list[0])
        data_GFW['num-iter-a1'] = min(data_GFW_num_iter_a1[0], num_iter_max_cap)
        data_GFW['num-iter-a1-std'] = data_GFW_num_iter_a1[1]
        data_GFW_running_time_a1 = average_std(full_running_time_GFW_list[0])
        data_GFW['running-time-a1'] = min(data_GFW_running_time_a1[0], running_time_max_cap)
        data_GFW['running-time-a1-std'] = data_GFW_running_time_a1[1]
        data_EPM['num-iter-a1'] = None
        data_EPM['num-iter-a1-std'] = None
        data_EPM['running-time-a1'] = None
        data_EPM['running-time-a1-std'] = None

    if data_EPM['solved-e'] > 0.05 * num_seeds:  # if EPM can solve at least 5% of the instances to exact CE, use the instances that both algorithms can solve to compute the stats
        data_GFW_num_iter_e = average_std(num_LMO_list[1])
        data_GFW['num-iter-e'] = min(data_GFW_num_iter_e[0], num_iter_max_cap)
        data_GFW['num-iter-e-std'] = data_GFW_num_iter_e[1]
        data_GFW_running_time_e = average_std(running_time_GFW_list[1])
        data_GFW['running-time-e'] = min(data_GFW_running_time_e[0], running_time_max_cap)
        data_GFW['running-time-e-std'] = data_GFW_running_time_e[1]
        data_EPM_num_iter_e = average_std(num_QMO_list[1])
        data_EPM['num-iter-e'] = min(data_EPM_num_iter_e[0], num_iter_max_cap)
        data_EPM['num-iter-e-std'] = data_EPM_num_iter_e[1]
        data_EPM_running_time_e = average_std(running_time_EPM_list[1])
        data_EPM['running-time-e'] = min(data_EPM_running_time_e[0], running_time_max_cap)
        data_EPM['running-time-e-std'] = data_EPM_running_time_e[1]
    else:  # otherwise, use all instances that GFW can solve to compute the stats
        data_GFW_num_iter_e = average_std(full_num_LMO_list[1])
        data_GFW['num-iter-e'] = min(data_GFW_num_iter_e[0], num_iter_max_cap)
        data_GFW['num-iter-e-std'] = data_GFW_num_iter_e[1]
        data_GFW_running_time_e = average_std(full_running_time_GFW_list[1])
        data_GFW['running-time-e'] = min(data_GFW_running_time_e[0], running_time_max_cap)
        data_GFW['running-time-e-std'] = data_GFW_running_time_e[1]
        data_EPM['num-iter-e'] = None
        data_EPM['num-iter-e-std'] = None
        data_EPM['running-time-e'] = None
        data_EPM['running-time-e-std'] = None

    print(
        f"STATS: {data_GFW['num-iter-e']}/{data_GFW['solved-e']}/{data_GFW['running-time-e']} "
        f"vs {data_EPM['num-iter-e']}/{data_EPM['solved-e']}/{data_EPM['running-time-e']}"
    )

    return data_GFW, data_EPM


def run_and_save_bidding_data(D, distance_matrix, size_list=[2, 50, 100], num_seeds=10, with_noise=True, save_dir='../data'):

    dict_ = {
        'size': list(),
        'iteration_GFW_a1': list(),
        'iteration_GFW_a1-std': list(),
        'iteration_GFW_e': list(),
        'iteration_GFW_e-std': list(),
        'iteration_EPM_a1': list(),
        'iteration_EPM_a1-std': list(),
        'iteration_EPM_e': list(),
        'iteration_EPM_e-std': list(),
        'solved_GFW_a1': list(),
        'solved_GFW_e': list(),
        'solved_EPM_a1': list(),
        'solved_EPM_e': list(),
        'runningtime_GFW_a1': list(),
        'runningtime_GFW_a1-std': list(),
        'runningtime_GFW_e': list(),
        'runningtime_GFW_e-std': list(),
        'runningtime_EPM_a1': list(),
        'runningtime_EPM_a1-std': list(),
        'runningtime_EPM_e': list(),
        'runningtime_EPM_e-std': list(),
    }

    size_string = ""
    for idx, size in enumerate(size_list):
        N = size
        M = size
        dict_['size'].append(f"{N}x{M}")
        res_GFW, res_EPM = run_GFW_vs_EPM_on_bidding_data(N, M, D, distance_matrix, with_noise=with_noise, num_seeds=num_seeds)

        dict_['iteration_GFW_a1'].append(res_GFW['num-iter-a1'])
        dict_['iteration_GFW_a1-std'].append(res_GFW['num-iter-a1-std'])
        dict_['solved_GFW_a1'].append(res_GFW['solved-a1'])
        dict_['runningtime_GFW_a1'].append(res_GFW['running-time-a1'])
        dict_['runningtime_GFW_a1-std'].append(res_GFW['running-time-a1-std'])
        dict_['iteration_GFW_e'].append(res_GFW['num-iter-e'])
        dict_['iteration_GFW_e-std'].append(res_GFW['num-iter-e-std'])
        dict_['solved_GFW_e'].append(res_GFW['solved-e'])
        dict_['runningtime_GFW_e'].append(res_GFW['running-time-e'])
        dict_['runningtime_GFW_e-std'].append(res_GFW['running-time-e-std'])

        dict_['iteration_EPM_a1'].append(res_EPM['num-iter-a1'])
        dict_['iteration_EPM_a1-std'].append(res_EPM['num-iter-a1-std'])
        dict_['solved_EPM_a1'].append(res_EPM['solved-a1'])
        dict_['runningtime_EPM_a1'].append(res_EPM['running-time-a1'])
        dict_['runningtime_EPM_a1-std'].append(res_EPM['running-time-a1-std'])
        dict_['iteration_EPM_e'].append(res_EPM['num-iter-e'])
        dict_['iteration_EPM_e-std'].append(res_EPM['num-iter-e-std'])
        dict_['solved_EPM_e'].append(res_EPM['solved-e'])
        dict_['runningtime_EPM_e'].append(res_EPM['running-time-e'])
        dict_['runningtime_EPM_e-std'].append(res_EPM['running-time-e-std'])

        if idx == 0:
            size_string += f"{N}x{M}"
        elif idx == 1:
            size_string += f".{N}x{M}"
            if len(size_list) > 3:
                size_string += "..."
        elif idx == len(size_list) - 1:
            size_string += f".{N}x{M}"

    # print(dict_)

    df = pd.DataFrame.from_dict(dict_)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if with_noise:
        df.to_csv(save_dir / f'biddingwithnoise_{size_string}_{num_seeds}.csv')
    else:
        df.to_csv(save_dir / f'bidding_{size_string}_{num_seeds}.csv')

    return dict_


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the AAMAS bidding-data experiments.")
    parser.add_argument("--data", type=Path, default=script_dir.parent / "data" / "bidding-data.csv")
    parser.add_argument("--sizes", nargs="+", type=int, default=[5, 50, 100, 150, 200, 250, 300])
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--variant", choices=["plain", "noise", "both"], default="both")
    parser.add_argument("--save-dir", type=Path, default=script_dir.parent / "data")
    args = parser.parse_args()

    print("Resolved path:", args.data.resolve())
    print("Exists:", args.data.exists())
    df = pd.read_csv(args.data)

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

    if args.variant in {"plain", "both"}:
        run_and_save_bidding_data(
            D,
            distance_matrix,
            size_list=args.sizes,
            num_seeds=args.num_seeds,
            with_noise=False,
            save_dir=args.save_dir,
        )
    if args.variant in {"noise", "both"}:
        run_and_save_bidding_data(
            D,
            distance_matrix,
            size_list=args.sizes,
            num_seeds=args.num_seeds,
            with_noise=True,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
