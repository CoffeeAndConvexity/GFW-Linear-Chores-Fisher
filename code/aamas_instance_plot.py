def plot_and_save(data, name):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_figheight(5)
    fig.set_figwidth(20)

    ax1.plot(data['x'], data['i_y1'], label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    ax1.plot(data['x'], data['i_y2'], label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)
    ax1.plot(data['x'], data['i_z1'], label=f'EPM: Approximate', marker='*', color='darkorange', linestyle='dashed', linewidth=2.5, markersize=18)
    ax1.plot(data['x'], data['i_z2'], label=f'EPM: Exact', marker='*', color='darkorange', linewidth=2.5, markersize=18)

    ax1.set_xticks(data['x'])
    # ax1.set_yticks(fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_xlabel("Size of instances", fontsize=20)
    ax1.set_ylabel("# iterations to reach CE", fontsize=20)
    ax1.legend(fontsize=16)
    # plt.tight_layout() 

    ax2.plot(data['x'], data['r_y1'], label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    ax2.plot(data['x'], data['r_y2'], label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)
    ax2.plot(data['x'], data['r_z1'], label=f'EPM: Approximate', marker='*', color='darkorange', linestyle='dashed', linewidth=2.5, markersize=18)
    ax2.plot(data['x'], data['r_z2'], label=f'EPM: Exact', marker='*', color='darkorange', linewidth=2.5, markersize=18)

    ax2.set_xticks(data['x'])
    # ax2.set_ticks('y', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.set_xlabel("Size of instances", fontsize=20)
    ax2.set_ylabel("Running time in seconds", fontsize=20)
    ax2.legend(fontsize=16)
    # plt.tight_layout()

    ax3.plot(data['x'], np.array(data['s_y1']), label=f'GFW: Approximate', marker='^', color='darkgreen', linestyle='dashed', linewidth=2.5, markersize=18)
    ax3.plot(data['x'], np.array(data['s_y2']), label=f'GFW: Exact', marker='^', color='darkgreen', linewidth=2.5, markersize=18)
    ax3.plot(data['x'], np.array(data['s_z1']), label=f'EPM: Approximate', marker='*', color='darkorange', linestyle='dashed', linewidth=2.5, markersize=18)
    ax3.plot(data['x'], np.array(data['s_z2']), label=f'EPM: Exact', marker='*', color='darkorange', linewidth=2.5, markersize=18)

    ax3.set_xticks(data['x'])
    # ax3.set_yticks(fontsize=16)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.set_xlabel("Size of instances", fontsize=20)
    ax3.set_ylabel("Ratio of solved instances", fontsize=20)
    ax3.legend(fontsize=16)
    # plt.tight_layout()

    fig.tight_layout()
    fig.savefig(f"{name}_int.png")