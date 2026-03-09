import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['text.usetex'] = True

from pypoman import compute_polytope_vertices
from scipy.spatial import ConvexHull

import gurobipy as gp
params = {
    "WLSACCESSID": '0f33ff11-ef92-485a-97f0-af4a28654d6b',
    "WLSSECRET": '880ecd7e-6fe7-4566-bb77-2590cc321d04',
    "LICENSEID": 2546016,
}
env = gp.Env(params=params)

N, M = 2, 8
np.random.seed(6)
D = np.random.uniform(size=(N, M))
B = np.array([1, 1])

print(D)
         


def polytope_dual(N, M, D, B):
    A = list()
    # IMPORTANT: use Gurobi, not include the nonnegative constraints
    for i in range(N):
        D_i_T = np.zeros(shape=(M, N))
        D_i_T[:, i] = - D[i]
        A_i = np.concatenate([np.identity(M), D_i_T], axis=1)
        A.append(A_i)
    A = np.concatenate(A, axis=0)
    b = np.zeros(shape=(M * N))
    C = np.concatenate([np.ones(M), np.zeros(N)]).reshape(1, -1)
    d = np.array(sum(B)).reshape(1, )

    return A, b, C, d

def feasible_point(M, N, D, B, random=False, random_seed=2024):  ###
    if random:
        np.random.seed(random_seed)
        p_0 = np.random.uniform(size=M)
        p_0 = p_0 / sum(p_0) * sum(B)
        beta_0 = np.amax(p_0 / D, axis=1) + np.random.uniform(size=N)
    else:
        # p_0 = (sum(B) / M) * np.ones(shape=M)
        # beta_0 = np.amax(p_0 / D, axis=1)
        # M = 1, N = 2
        beta_0 = np.array([2.8, 0.6])
        p_0 = np.amax(beta_0.reshape(-1, 1) * D, axis=0)

    return p_0, beta_0

def LMO(c, A, b, C=None, d=None, return_dual=False):
    m = gp.Model(env=env)
    N = len(c)

    M = A.shape[1] - N
    p = m.addMVar(M)
    beta = m.addMVar(N)

    m.setObjective(c @ beta)
    m.addConstr(A[:, :M] @ p + A[:, M:] @ beta <= b)
    if C is not None and d is not None:
        m.addConstr(C[:, :M] @ p + C[:, M:] @ beta == d)
    # m.PStart = x0

    m.Params.LogToConsole = 0
    # m.Params.Method = 1
    m.optimize()

    if return_dual:
        return p.X, beta.X, np.array(m.Pi)
    else:
        return p.X, beta.X

def GFW(N, M, D, B):  # Greedy Frank Wolfe

    # beta in algorithm
    beta_1_list = list()
    beta_2_list = list()

    # create the dual polyhedron
    A, b, C, d = polytope_dual(N, M, D, B)

    p, beta = feasible_point(M, N, D, B, random=False)
    beta_1_list.append(beta[0])
    beta_2_list.append(beta[1])
    beta_ = beta  # 'beta_' keeps the last beta values

    for k in range(10):

        # main
        p, beta, gamma = LMO(B / beta, A, b, C, d, return_dual=True)

        if sum((beta - beta_) ** 2) <= 1e-6:
            break
        else:
            beta_1_list.append(beta[0])
            beta_2_list.append(beta[1])

            x = -gamma[: N * M].reshape(N, M)
            x = sum(B) / sum(B * beta / beta_) * x
            beta_ = beta

    return np.array(beta_1_list), np.array(beta_2_list)

def polytope_dual_A_b_form(N, M, D, B):
    A = list()
    # IMPORTANT: Here, we need the nonnegative constraints
    for i in range(N):
        D_i_T = np.zeros(shape=(M, N))
        D_i_T[:, i] = - D[i]
        A_i = np.concatenate([np.identity(M), D_i_T], axis=1)
        A.append(A_i)
    A.append(np.concatenate([np.zeros(shape=(N, M)), np.identity(N)], axis=1))
    A.append(np.concatenate([-np.identity(M), np.zeros(shape=(M, N))], axis=1))
    A = np.concatenate(A, axis=0)
    b = np.concatenate([np.zeros(shape=(M * N)), 1e6 * np.ones(N), np.zeros(shape=M)])
    C = np.concatenate([np.ones(M), np.zeros(N)]).reshape(1, -1)
    d = np.array(sum(B)).reshape(1, )

    A = np.concatenate([A, C, -C], axis=0)
    b = np.concatenate([b, d, -d])

    return A, b



beta_1_in_algorithm, beta_2_in_algorithm = GFW(N, M, D, B)

A, b = polytope_dual_A_b_form(N, M, D, B)
vertices = compute_polytope_vertices(A, b)

vertices_considered = {
    "beta_1": list(), 
    "beta_2": list()
}
for vertex in vertices:
    if np.amax(vertex[-2:]) < 99 and np.amax(vertex[:M]) < sum(B) - 0.01 and np.amin(vertex[:M]) > 0.01:
        vertices_considered["beta_1"].append(vertex[-2])
        vertices_considered["beta_2"].append(vertex[-1])



min_v_beta_1, max_v_beta_1 = min(vertices_considered["beta_1"]), max(vertices_considered["beta_1"])
min_v_beta_2, max_v_beta_2 = min(vertices_considered["beta_2"]), max(vertices_considered["beta_2"])
vertices_considered["beta_1"].insert(0, min_v_beta_1)
vertices_considered["beta_2"].insert(0, 1.2 * max_v_beta_2)
vertices_considered["beta_1"].append(1.2 * max_v_beta_1)
vertices_considered["beta_2"].append(min_v_beta_2)


idx_sort_1 = np.argsort(vertices_considered["beta_1"])
vertices_considered["beta_1"] = np.array(vertices_considered["beta_1"])[idx_sort_1]
vertices_considered["beta_2"] = np.array(vertices_considered["beta_2"])[idx_sort_1]

fig, ax = plt.subplots(1, figsize=(10, 6))


# Plot the polytope
# Plot the feasible region and fill it
# ------------------------------------------------
plt.plot(vertices_considered["beta_1"], vertices_considered["beta_2"], lw=3, linestyle='-', c='green', marker='o', markersize=10)
plt.fill_between(vertices_considered["beta_1"], vertices_considered["beta_2"], 3, color='green', alpha=0.3)



# Plot the path of the algorithm
# ------------------------------------------------
plt.plot(beta_1_in_algorithm, beta_2_in_algorithm, lw=2, c='blue', marker='s', markersize=10, markerfacecolor='darkorange', markeredgewidth=2, zorder=20)
for i in range(len(beta_1_in_algorithm) - 1):
    segment_length = np.sqrt((beta_1_in_algorithm[i + 1] - beta_1_in_algorithm[i]) ** 2 + (beta_2_in_algorithm[i + 1] - beta_2_in_algorithm[i]) ** 2)
    plt.arrow(beta_1_in_algorithm[i], beta_2_in_algorithm[i], 0.6 * (beta_1_in_algorithm[i + 1] - beta_1_in_algorithm[i]), 0.6 * (beta_2_in_algorithm[i + 1] - beta_2_in_algorithm[i]), shape='full', lw=1, length_includes_head=True, head_length=0.06, head_width=0.05, overhang=0.2, color='blue', capstyle='round', zorder=16)

# Plot the descent direction and tangent plane

# ax = plt.gca()
def draw_descent_direction(beta_1, beta_2, with_point=False, color=['blue', 'orange']):
    plt.arrow(beta_1, beta_2, 0.1 * (- 1 / beta_1), 0.1 * (- 1 / beta_2), shape='full', lw=2, length_includes_head=False, head_length=0.06, head_width=0.02, overhang=0.2, color=color[0], capstyle='round')
    # arrow = mpatches.FancyArrowPatch((beta_1, beta_2), (beta_1 + 0.2 * (- 1 / beta_1), beta_2 + 0.2 * (- 1 / beta_2)), mutation_scale=100, arrowstyle=']->', color=color[0])
    # ax.add_patch(arrow)

    if with_point:
        plt.scatter(beta_1, beta_2, marker='o', color=color[1], zorder=10)

def draw_tangent_plane(beta_1, beta_2, color='darkorange'): 
    l = 0.2
    plt.plot([beta_1 - l * beta_1, beta_1 + l * beta_1], [beta_2 + l * beta_2, beta_2 - l * beta_2], lw=1, ls='--', c=color, zorder=6)

for i in range(len(beta_1_in_algorithm)):
    print(np.log(beta_1_in_algorithm[i]) + np.log(beta_2_in_algorithm[i]))
    draw_descent_direction(beta_1_in_algorithm[i], beta_2_in_algorithm[i], with_point=False, color=['orange', 'orange'])
    draw_tangent_plane(beta_1_in_algorithm[i], beta_2_in_algorithm[i], color='darkorange')



             

plt.axis('scaled')

plt.xlabel(r"$\beta_1$", fontsize=28, weight='bold')
plt.ylabel(r"$\beta_2$", fontsize=28, weight='bold')

ax = plt.gca()
# ax.spines[:].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_visible(False)
plt.arrow(-0.1, 0, 3.2, 0, shape='full', lw=3, length_includes_head=False, head_length=0.1, head_width=0.05, overhang=0.3, color='black')
plt.arrow(0, -0.1, 0, 1.5, shape='full', lw=3, length_includes_head=False, head_length=0.1, head_width=0.05, overhang=0.3, color='black')

# eq1 = (r"\begin{eqnarray*} & p_j \leq d_{1j} \beta_1 \;\forall\, j \in [1, \ldots, 8] \\ & p_j \leq d_{2j} \beta_2 \;\forall\, j \in [1, \ldots, 8] \\ & \sum_j p_j = 2 \\ & p, \beta \geq 0 \end{eqnarray*}")
# ax.text(1, 0.9, eq1, color='k', fontsize=18, horizontalalignment="right", verticalalignment="top")

plt.xticks(fontsize=20, weight='bold')
plt.yticks(fontsize=20, weight='bold')
plt.xlim(-0.1, 3.2)
plt.ylim(-0.1, 1.5)
plt.tight_layout()

import os
current_directory = os.getcwd()
plt.savefig(f"{current_directory}/GFW.png")