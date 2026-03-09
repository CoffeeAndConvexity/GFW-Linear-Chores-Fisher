# -*- coding: utf-8 -*-
"""
Chores CEEI Experiments
"""

import numpy as np
np.seterr(divide='ignore')
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
import time
import os



def plot_and_save(data, random_generating_method, num_seeds=10, download_fig=False, save_dir='./figures/'):

    # Create the folder if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure()
    plt.plot(data['size'], data['runningtime_GFW_a1'], label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['size'], data['runningtime_GFW_e'], label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)

    plt.xticks(data['size'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("Running time in seconds", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"{save_dir}/rt_{random_generating_method}_large.png")


    plt.figure()
    plt.plot(data['size'], data['iteration_GFW_a1'], label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['size'], data['iteration_GFW_e'], label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)

    plt.xticks(data['size'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("# iterations to reach CE", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"{save_dir}/ni_{random_generating_method}_large.png")

    plt.figure()
    plt.plot(data['size'], np.array(data['solved_GFW_a1']) / num_seeds, label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    plt.plot(data['size'], np.array(data['solved_GFW_e']) / num_seeds, label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)

    plt.xticks(data['size'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 1.05)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("Ratio of solved instances", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"{save_dir}/sr_{random_generating_method}_large.png")


if __name__ == "__main__": 

    rgm_list = ['uniform', 'lognormal', 'truncnormal', 'exponential', 'randint'] 

    # the location where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current Directory: {current_dir}")

    for rgm in rgm_list:
        data = pd.read_csv(f'{current_dir}/data/{rgm}_large.csv')
        plot_and_save(data, random_generating_method=rgm, num_seeds=100, save_dir=f'{current_dir}/figures/')