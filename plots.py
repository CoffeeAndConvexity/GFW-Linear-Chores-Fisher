import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

import numpy as np
import pandas as pd 

def average_and_std(lst):
    nonNone_list = list()

    for e in lst:
        if e is not None:
            nonNone_list.append(e)

    if len(nonNone_list) > 0:
        return np.mean(nonNone_list), np.std(nonNone_list)
    else:
        return None, None

def plot_and_save(data, name):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(5)
    fig.set_figwidth(18)

    ax1.plot(data['x'], data['i_y1'], label=f'GFW: Approximate', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=12)
    ax1.plot(data['x'], data['i_y2'], label=f'GFW: Exact', marker='^', color='g', linewidth=2.5, markersize=12)
    ax1.plot(data['x'], data['i_z1'], label=f'EPM: Approximate', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=12)
    ax1.plot(data['x'], data['i_z2'], label=f'EPM: Exact', marker='*', color='orange', linewidth=2.5, markersize=12)

    # ax1.set_xticks(fontsize=16)
    # ax1.set_yticks(fontsize=16)
    ax1.set(xlabel='', ylabel='# iterations to reach CE')
    # plt.legend(fontsize=16)
    # plt.tight_layout() 

    ax2.plot(data['x'], data['r_y1'], label=f'GFW: Approximate', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=12)
    ax2.plot(data['x'], data['r_y2'], label=f'GFW: Exact', marker='^', color='g', linewidth=2.5, markersize=12)
    ax2.plot(data['x'], data['r_z1'], label=f'EPM: Approximate', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=12)
    ax2.plot(data['x'], data['r_z2'], label=f'EPM: Exact', marker='*', color='orange', linewidth=2.5, markersize=12)

    # ax2.set_ticks('x', fontsize=16)
    # ax2.set_ticks('y', fontsize=16)
    ax2.set(xlabel='', ylabel="Running time in seconds")
    # plt.legend(fontsize=16)
    # plt.tight_layout()

    ax3.plot(data['x'], np.array(data['s_y1']), label=f'GFW: Approximate', marker='^', color='g', linestyle='dashed', linewidth=2.5, markersize=12)
    ax3.plot(data['x'], np.array(data['s_y2']), label=f'GFW: Exact', marker='^', color='g', linewidth=2.5, markersize=12)
    ax3.plot(data['x'], np.array(data['s_z1']), label=f'EPM: Approximate', marker='*', color='orange', linestyle='dashed', linewidth=2.5, markersize=12)
    ax3.plot(data['x'], np.array(data['s_z2']), label=f'EPM: Exact', marker='*', color='orange', linewidth=2.5, markersize=12)

    # ax3.set_xticks(fontsize=16)
    # ax3.set_yticks(fontsize=16)
    ax3.set(xlabel='', ylabel="Ratio of solved instances")
    # plt.legend(fontsize=16)
    # plt.tight_layout()

    plt.show()

if __name__ == "__main__": 

    size_list = [2, 50, 100, 150, 200, 250]
    name_list = ['bidding_data', ]

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