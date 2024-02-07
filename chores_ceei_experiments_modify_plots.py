import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

import numpy as np
import pandas as pd 

APPROXIMATE_THR = 0.01
EXACT_THR = r'1e-6'


def plot_and_save(data, random_generating_method, num_seeds=10, download_fig=False):

    plt.figure()
    plt.plot(data['x'], data['r_y1'], label=f'GFW: Approximate ({APPROXIMATE_THR})', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=20)
    plt.plot(data['x'], data['r_y2'], label=f'GFW: Exact ({EXACT_THR})', marker='^', color='g', linewidth=2.5, markersize=20)
    plt.plot(data['x'], data['r_z1'], label=f'EPM: Approximate ({APPROXIMATE_THR})', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=20)
    plt.plot(data['x'], data['r_z2'], label=f'EPM: Exact ({EXACT_THR})', marker='*', color='orange', linewidth=2.5, markersize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("Running time in seconds", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"rt_{random_generating_method}.png")


    plt.figure()
    plt.plot(data['x'], data['i_y1'], label=f'GFW: Approximate ({APPROXIMATE_THR})', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=20)
    plt.plot(data['x'], data['i_y2'], label=f'GFW: Exact ({EXACT_THR})', marker='^', color='g', linewidth=2.5, markersize=20)
    plt.plot(data['x'], data['i_z1'], label=f'EPM: Approximate ({APPROXIMATE_THR})', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=20)
    plt.plot(data['x'], data['i_z2'], label=f'EPM: Exact ({EXACT_THR})', marker='*', color='orange', linewidth=2.5, markersize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("# iterations to reach CE", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"ni_{random_generating_method}.png")

    plt.figure()
    plt.plot(data['x'], np.array(data['s_y1']) / num_seeds, label=f'GFW: Approximate ({APPROXIMATE_THR})', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=20)
    plt.plot(data['x'], np.array(data['s_y2']) / num_seeds, label=f'GFW: Exact ({EXACT_THR})', marker='^', color='g', linewidth=2.5, markersize=20)
    plt.plot(data['x'], np.array(data['s_z1']) / num_seeds, label=f'EPM: Approximate ({APPROXIMATE_THR})', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=20)
    plt.plot(data['x'], np.array(data['s_z2']) / num_seeds, label=f'EPM: Exact ({EXACT_THR})', marker='*', color='orange', linewidth=2.5, markersize=20)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Size of instances", fontsize=20)
    plt.ylabel("Ratio of solved instances", fontsize=20)
    plt.legend(fontsize=16)
    plt.tight_layout()

    plt.savefig(f"sr_{random_generating_method}.png")


if __name__ == "__main__": 

    size_list = [2, 50, 100, 150, 200, 250, 300]
    rgm_list = ['uniform', 'lognormal', 'truncnormal', 'exponential', 'randint']

    for rgm in rgm_list:
        data = pd.read_csv(f"{rgm}.csv").to_dict(orient='list')
        print(data)
        plot_and_save(data, random_generating_method=rgm, num_seeds=10)
