"""
Chores CEEI Experiments
"""

import argparse
from pathlib import Path

import numpy as np
np.seterr(divide='ignore')
from scipy.stats import truncnorm
import pandas as pd
import cvxpy as cp

from utils import APPROXIMATE_THR, EXACT_THR, E2TOL, E3TOL, eps_approx_eq
from gfw import *
from epm import *
from combinatorial import combinatorial_metrics

ALGORITHMS = ["GFW", "EPM", "COMB"]
NUM_SEEDS = 10
RGM = ["uniform", "randint", "lognormal", "truncnormal", "exponential", "randint10"]
VALID_RGM = RGM + ["uniformha"]


def parse_size(value):
    try:
        n_text, m_text = value.lower().split("x", 1)
        n, m = int(n_text), int(m_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("sizes must use NxM format, for example 30x30") from exc
    if n <= 0 or m <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return n, m

def _normalize_algorithms(algorithms=None):
    if algorithms is None:
        return {"GFW", "EPM", "COMB"}

    normalized = {str(a).upper() for a in algorithms}
    valid = {"GFW", "EPM", "COMB"}
    invalid = normalized - valid
    if invalid:
        raise ValueError(f"Unknown algorithm(s): {sorted(invalid)}. Valid options are {sorted(valid)}")
    if not normalized:
        raise ValueError("algorithms must contain at least one algorithm name.")
    return normalized


def _compute_welfare_from_utilities(u):
    if u is None:
        return None, None, None
    u_arr = np.asarray(u, dtype=float)
    if u_arr.size == 0:
        return None, None, None
    utilitarian = float(np.sum(u_arr))
    egalitarian = float(np.max(u_arr))
    if np.any(u_arr <= 0):
        nash = None
    else:
        nash = float(np.prod(u_arr))
    return utilitarian, egalitarian, nash

print("- cvxpy version:", cp.__version__, "-")
print("- cvxpy installed solvers:", cp.installed_solvers())

"""
run, save, and plot
"""

def run_GFW_vs_EPM(N, M, random_generating_method='uniform', num_seeds=10, num_iter_max_cap=80, running_time_max_cap=100, print_eq=False, comb_solver='scs_strict', algorithms=None):

    algorithms = _normalize_algorithms(algorithms)

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
    data_GFW = {'size': f'N*M = {N}*{M}',}
    num_QMO_list = [[], []]
    running_time_EPM_list = [[], []]
    num_ins_EPM_solved = [0, 0]
    data_EPM = {'size': f'N*M = {N}*{M}',}
    num_COMB_list = [[], []]
    running_time_COMB_list = [[], []]
    num_ins_COMB_solved = [0, 0]
    data_COMB = {'size': f'N*M = {N}*{M}',}

    welfare_data = {
        'size': f'N*M = {N}*{M}',
        'utilitarian_GFW_e': [],
        'egalitarian_GFW_e': [],
        'nash_GFW_e': [],
        'utilitarian_EPM_e': [],
        'egalitarian_EPM_e': [],
        'nash_EPM_e': [],
        'utilitarian_COMB_e': [],
        'egalitarian_COMB_e': [],
        'nash_COMB_e': [],
        'distant_prices': [],
    }

    print(f"-*-*-*- SIZE: {N}*{M} -*-*-*-")
    print("PROGRESS: ", end ="")

    for s in seeds:
        np.random.seed(s)
        if random_generating_method == 'randint':
            D = np.random.randint(low=1, high=1001, size=(N, M))
        elif random_generating_method in {'uniform', 'uniformha'}:
            D = np.random.uniform(size=(N, M))
        elif random_generating_method == 'lognormal':
            D = np.random.lognormal(size=(N, M))
        elif random_generating_method == 'truncnormal':
            D = truncnorm.rvs(a=1e-3, b=10, size=(N, M), random_state=s)
        elif random_generating_method == 'exponential':
            D = np.random.exponential(size=(N, M))
        elif random_generating_method == 'randint10':
            D = np.random.randint(low=1, high=11, size=(N, M))
        else:
            raise ValueError(
                f"Unknown distribution '{random_generating_method}'. Valid options are {VALID_RGM}."
            )

        B = np.ones(shape=N)

        if "GFW" in algorithms:
            res_GFW = GFW(N, M, D, B, print_eq=print_eq, 
                          warm_start=True, LP_solve_method="primal simplex", return_eq=True)
        else:
            res_GFW = (None, None, False, False, None, None, None)

        if "EPM" in algorithms:
            res_EPM = EPM(N, M, D, B, QMO_solver='GUROBI', ignore_print=True, print_eq=print_eq, 
                          warm_start=True, QP_solve_method="barrier",
                          high_accuracy=random_generating_method == 'uniformha', return_eq=True)
        else:
            res_EPM = (None, None, False, False, None, None, None)

        if "COMB" in algorithms:
            try:
                res_COMB = combinatorial_metrics(N, M, D, B, max_iter=800, print_eq=print_eq, solver=comb_solver, return_eq=True)
            except Exception:
                res_COMB = (800, 800, False, False, None, None, None)
        else:
            res_COMB = (None, None, False, False, None, None, None)

        if res_GFW[3] and res_GFW[6] is not None:
            if res_EPM[3] and res_EPM[6] is not None:
                if res_COMB[3] and res_COMB[6] is not None:
                    p_gfw, x_gfw, u_gfw = res_GFW[6]
                    util, egal, nash = _compute_welfare_from_utilities(u_gfw)
                    welfare_data['utilitarian_GFW_e'].append(util)
                    welfare_data['egalitarian_GFW_e'].append(egal)
                    welfare_data['nash_GFW_e'].append(nash)
                
                    p_epm, x_epm, u_epm = res_EPM[6]
                    util, egal, nash = _compute_welfare_from_utilities(u_epm)
                    welfare_data['utilitarian_EPM_e'].append(util)
                    welfare_data['egalitarian_EPM_e'].append(egal)
                    welfare_data['nash_EPM_e'].append(nash)
                
                    p_comb, x_comb, u_comb = res_COMB[6]
                    util, egal, nash = _compute_welfare_from_utilities(u_comb)
                    welfare_data['utilitarian_COMB_e'].append(util)
                    welfare_data['egalitarian_COMB_e'].append(egal)
                    welfare_data['nash_COMB_e'].append(nash)

                    welfare_data['distant_prices'].append(np.linalg.norm(p_gfw - p_epm) + np.linalg.norm(p_gfw - p_comb) + np.linalg.norm(p_epm - p_comb))

        progress_res = res_EPM if "EPM" in algorithms else res_GFW if "GFW" in algorithms else res_COMB
        if progress_res[3]:  
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
            num_ins_COMB_solved[i] += res_COMB[2 + i]
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
            if res_COMB[2 + i]:
                num_COMB_list[i].append(res_COMB[i])
                running_time_COMB_list[i].append(res_COMB[4 + i])

    data_GFW['solved-a1'] = num_ins_GFW_solved[0]
    data_GFW['solved-e'] = num_ins_GFW_solved[1]
    data_EPM['solved-a1'] = num_ins_EPM_solved[0]
    data_EPM['solved-e'] = num_ins_EPM_solved[1]
    data_COMB['solved-a1'] = num_ins_COMB_solved[0]
    data_COMB['solved-e'] = num_ins_COMB_solved[1]

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

    if data_COMB['solved-a1'] > 0:
        data_COMB_num_iter_a1 = average_std(num_COMB_list[0])
        data_COMB['num-iter-a1'] = min(data_COMB_num_iter_a1[0], num_iter_max_cap)
        data_COMB['num-iter-a1-std'] = data_COMB_num_iter_a1[1]
        data_COMB_running_time_a1 = average_std(running_time_COMB_list[0])
        data_COMB['running-time-a1'] = min(data_COMB_running_time_a1[0], running_time_max_cap)
        data_COMB['running-time-a1-std'] = data_COMB_running_time_a1[1]
    else:
        data_COMB['num-iter-a1'] = None
        data_COMB['num-iter-a1-std'] = None
        data_COMB['running-time-a1'] = None
        data_COMB['running-time-a1-std'] = None

    if data_COMB['solved-e'] > 0:
        data_COMB_num_iter_e = average_std(num_COMB_list[1])
        data_COMB['num-iter-e'] = min(data_COMB_num_iter_e[0], num_iter_max_cap)
        data_COMB['num-iter-e-std'] = data_COMB_num_iter_e[1]
        data_COMB_running_time_e = average_std(running_time_COMB_list[1])
        data_COMB['running-time-e'] = min(data_COMB_running_time_e[0], running_time_max_cap)
        data_COMB['running-time-e-std'] = data_COMB_running_time_e[1]
    else:
        data_COMB['num-iter-e'] = None
        data_COMB['num-iter-e-std'] = None
        data_COMB['running-time-e'] = None
        data_COMB['running-time-e-std'] = None

    print(
        f"STATS: {data_GFW['num-iter-e']}/{data_GFW['solved-e']}/{data_GFW['running-time-e']} "
        f"vs {data_EPM['num-iter-e']}/{data_EPM['solved-e']}/{data_EPM['running-time-e']} "
        f"vs {data_COMB['num-iter-e']}/{data_COMB['solved-e']}/{data_COMB['running-time-e']}"
    )

    welfare_summary = {
        'size': f'{N}x{M}',
        'utilitarian_GFW_e': average_std(welfare_data['utilitarian_GFW_e'])[0] if len(welfare_data['utilitarian_GFW_e']) > 0 else None,
        'egalitarian_GFW_e': average_std(welfare_data['egalitarian_GFW_e'])[0] if len(welfare_data['egalitarian_GFW_e']) > 0 else None,
        'nash_GFW_e': average_std(welfare_data['nash_GFW_e'])[0] if len(welfare_data['nash_GFW_e']) > 0 else None,
        'utilitarian_EPM_e': average_std(welfare_data['utilitarian_EPM_e'])[0] if len(welfare_data['utilitarian_EPM_e']) > 0 else None,
        'egalitarian_EPM_e': average_std(welfare_data['egalitarian_EPM_e'])[0] if len(welfare_data['egalitarian_EPM_e']) > 0 else None,
        'nash_EPM_e': average_std(welfare_data['nash_EPM_e'])[0] if len(welfare_data['nash_EPM_e']) > 0 else None,
        'utilitarian_COMB_e': average_std(welfare_data['utilitarian_COMB_e'])[0] if len(welfare_data['utilitarian_COMB_e']) > 0 else None,
        'egalitarian_COMB_e': average_std(welfare_data['egalitarian_COMB_e'])[0] if len(welfare_data['egalitarian_COMB_e']) > 0 else None,
        'nash_COMB_e': average_std(welfare_data['nash_COMB_e'])[0] if len(welfare_data['nash_COMB_e']) > 0 else None,
        'distant_prices': average_std(welfare_data['distant_prices'])[0] if len(welfare_data['distant_prices']) > 0 else None,
    }

    return data_GFW, data_EPM, data_COMB, welfare_summary

def run_and_save(size_list=[(2, 2), (50, 50), (100, 100)], random_generating_method='uniform', num_seeds=10, download_csv=False, save_dir="../data", comb_solver='scs_strict', algorithms=None):

    algorithms = _normalize_algorithms(algorithms)

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
        'iteration_COMB_a1': list(),
        'iteration_COMB_a1-std': list(),
        'iteration_COMB_e': list(),
        'iteration_COMB_e-std': list(),
        'solved_GFW_a1': list(),
        'solved_GFW_e': list(),
        'solved_EPM_a1': list(),
        'solved_EPM_e': list(),
        'solved_COMB_a1': list(),
        'solved_COMB_e': list(),
        'runningtime_GFW_a1': list(),
        'runningtime_GFW_a1-std': list(),
        'runningtime_GFW_e': list(),
        'runningtime_GFW_e-std': list(),
        'runningtime_EPM_a1': list(),
        'runningtime_EPM_a1-std': list(),
        'runningtime_EPM_e': list(),
        'runningtime_EPM_e-std': list(),
        'runningtime_COMB_a1': list(),
        'runningtime_COMB_a1-std': list(),
        'runningtime_COMB_e': list(),
        'runningtime_COMB_e-std': list(),
    }

    welfare_dict = {
        'size': list(),
        'utilitarian_GFW_e': list(),
        'egalitarian_GFW_e': list(),
        'nash_GFW_e': list(),
        'utilitarian_EPM_e': list(),
        'egalitarian_EPM_e': list(),
        'nash_EPM_e': list(),
        'utilitarian_COMB_e': list(),
        'egalitarian_COMB_e': list(),
        'nash_COMB_e': list(),
        'distant_prices': list(),
    }

    size_string = ""
    for idx, size in enumerate(size_list):
        N, M = size
        dict_['size'].append(f"{N}x{M}")
        res_GFW, res_EPM, res_COMB, welfare_res = run_GFW_vs_EPM(
            N,
            M,
            random_generating_method,
            num_seeds,
            num_iter_max_cap=800,
            running_time_max_cap=1000,
            comb_solver=comb_solver,
            algorithms=algorithms,
        )

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

        dict_['iteration_COMB_a1'].append(res_COMB['num-iter-a1'])
        dict_['iteration_COMB_a1-std'].append(res_COMB['num-iter-a1-std'])
        dict_['solved_COMB_a1'].append(res_COMB['solved-a1'])
        dict_['runningtime_COMB_a1'].append(res_COMB['running-time-a1'])
        dict_['runningtime_COMB_a1-std'].append(res_COMB['running-time-a1-std'])
        dict_['iteration_COMB_e-std'].append(res_COMB['num-iter-e-std'])
        dict_['iteration_COMB_e'].append(res_COMB['num-iter-e'])
        dict_['solved_COMB_e'].append(res_COMB['solved-e'])
        dict_['runningtime_COMB_e'].append(res_COMB['running-time-e'])
        dict_['runningtime_COMB_e-std'].append(res_COMB['running-time-e-std'])

        welfare_dict['size'].append(f"{N}x{M}")
        welfare_dict['utilitarian_GFW_e'].append(welfare_res['utilitarian_GFW_e'])
        welfare_dict['egalitarian_GFW_e'].append(welfare_res['egalitarian_GFW_e'])
        welfare_dict['nash_GFW_e'].append(welfare_res['nash_GFW_e'])
        welfare_dict['utilitarian_EPM_e'].append(welfare_res['utilitarian_EPM_e'])
        welfare_dict['egalitarian_EPM_e'].append(welfare_res['egalitarian_EPM_e'])
        welfare_dict['nash_EPM_e'].append(welfare_res['nash_EPM_e'])
        welfare_dict['utilitarian_COMB_e'].append(welfare_res['utilitarian_COMB_e'])
        welfare_dict['egalitarian_COMB_e'].append(welfare_res['egalitarian_COMB_e'])
        welfare_dict['nash_COMB_e'].append(welfare_res['nash_COMB_e'])
        welfare_dict['distant_prices'].append(welfare_res['distant_prices'])

        if idx == 0:
            size_string += f"{N}x{M}"
        elif idx == 1:
            size_string += f".{N}x{M}"
            if len(size_list) > 3:
                size_string += "..."
        elif idx == len(size_list) - 1:
            size_string += f".{N}x{M}"

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_dict(dict_)
    df.to_csv(save_dir / f'{random_generating_method}_{size_string}_{num_seeds}.csv')

    welfare_df = pd.DataFrame.from_dict(welfare_dict)
    welfare_df.to_csv(save_dir / f'{random_generating_method}_{size_string}_welfare_{num_seeds}.csv')

    return dict_

def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run synthetic Chores-CEEI experiments.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=parse_size,
        default=[(i, i) for i in range(5, 51, 5)],
        metavar="NxM",
    )
    parser.add_argument("--distributions", nargs="+", choices=VALID_RGM, default=RGM)
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument("--comb-solver", default="scs_strict")
    parser.add_argument("--save-dir", type=Path, default=script_dir.parent / "data")
    args = parser.parse_args()

    for distribution in args.distributions:
        print(f"================== {distribution} ==================")
        run_and_save(
            size_list=args.sizes,
            random_generating_method=distribution,
            num_seeds=args.num_seeds,
            save_dir=args.save_dir,
            comb_solver=args.comb_solver,
            algorithms=args.algorithms,
        )


if __name__ == "__main__":
    main()
