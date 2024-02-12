import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

import numpy as np
import pandas as pd 
import math 

def average_and_std(lst):
    nonNone_list = list()

    for e in lst:
        if (e is not None) and (e is not np.nan) and (e is not pd.NA) and (not math.isnan(e)):
            nonNone_list.append(e)

    if len(nonNone_list) > 0:
        return np.mean(nonNone_list), np.std(nonNone_list)
    else:
        return None, None

def plot_and_save(data, name):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(5)
    fig.set_figwidth(20)

    # ax1.plot(data['x'], data['i_y1'], label=f'GFW: Approximate', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=12)
    # ax1.plot(data['x'], data['i_y2'], label=f'GFW: Exact', marker='^', color='g', linewidth=2.5, markersize=12)
    # ax1.plot(data['x'], data['i_z1'], label=f'EPM: Approximate', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=12)
    # ax1.plot(data['x'], data['i_z2'], label=f'EPM: Exact', marker='*', color='orange', linewidth=2.5, markersize=12)

    ax1.errorbar(data['x'], data['i_y1'], label=f'GFW: Approximate', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=12)
    ax1.plot(data['x'], data['i_y2'], label=f'GFW: Exact', marker='^', color='g', linewidth=2.5, markersize=12)
    ax1.plot(data['x'], data['i_z1'], label=f'EPM: Approximate', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=12)
    ax1.plot(data['x'], data['i_z2'], label=f'EPM: Exact', marker='*', color='orange', linewidth=2.5, markersize=12)

    ax1.set_xticks(data['x'])
    # ax1.set_yticks(fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set(xlabel='')
    ax1.set_ylabel("# iterations to reach CE", fontsize=18)
    ax1.legend(fontsize=16)
    # plt.tight_layout() 

    ax2.plot(data['x'], data['r_y1'], label=f'GFW: Approximate', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=12)
    ax2.plot(data['x'], data['r_y2'], label=f'GFW: Exact', marker='^', color='g', linewidth=2.5, markersize=12)
    ax2.plot(data['x'], data['r_z1'], label=f'EPM: Approximate', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=12)
    ax2.plot(data['x'], data['r_z2'], label=f'EPM: Exact', marker='*', color='orange', linewidth=2.5, markersize=12)

    ax2.set_xticks(data['x'])
    # ax2.set_ticks('y', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set(xlabel='')
    ax2.set_ylabel("Running time in seconds", fontsize=18)
    ax2.legend(fontsize=16)
    # plt.tight_layout()

    ax3.plot(data['x'], np.array(data['s_y1']), label=f'GFW: Approximate', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=12)
    ax3.plot(data['x'], np.array(data['s_y2']), label=f'GFW: Exact', marker='^', color='g', linewidth=2.5, markersize=12)
    ax3.plot(data['x'], np.array(data['s_z1']), label=f'EPM: Approximate', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=12)
    ax3.plot(data['x'], np.array(data['s_z2']), label=f'EPM: Exact', marker='*', color='orange', linewidth=2.5, markersize=12)

    ax3.set_xticks(data['x'])
    # ax3.set_yticks(fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=18)
    ax3.set(xlabel='')
    ax3.set_ylabel("Ratio of solved instances", fontsize=18)
    ax3.legend(fontsize=16)
    # plt.tight_layout()

    fig.tight_layout()
    fig.savefig(f"{name}_int.png")

if __name__ == "__main__": 

    size_list = [2, 50, 100, 150, 200, 250]
    name_list = ['bidding_data', 'bidding_data_with_noise', ]
    name_list = ['uniform', 'lognormal', 'truncnormal', 'exponential', 'randint']

    for name in name_list:
        data = {
            'x': list(),
            'i_y1': list(),
            'i_y2': list(),
            'i_z1': list(),
            'i_z2': list(),
            's_y1': list(),
            's_y2': list(),
            's_z1': list(),
            's_z2': list(),
            'r_y1': list(),
            'r_y2': list(),
            'r_z1': list(),
            'r_z2': list(), 
            'e_i_y1': list(),
            'e_i_y2': list(),
            'e_i_z1': list(),
            'e_i_z2': list(),
            'e_s_y1': list(),
            'e_s_y2': list(),
            'e_s_z1': list(),
            'e_s_z2': list(),
            'e_r_y1': list(),
            'e_r_y2': list(),
            'e_r_z1': list(),
            'e_r_z2': list(), 
        }
        for size in size_list: 
            data_s = pd.read_csv(f"{name}_{size}.csv")
            
            # process data for one size 
            data['x'].append(size)
            
            for col in data_s.columns: 
                if col != 'Unnamed: 0':
                    mean, std = average_and_std(data_s[col])
                    data[col].append(mean)
                    data[f'e_{col}'].append(std)
            
        plot_and_save(data, name=name)