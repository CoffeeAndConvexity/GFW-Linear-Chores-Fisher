import numpy as np
import matplotlib.pyplot as plt

from pypoman import compute_polytope_vertices
from scipy.spatial import ConvexHull

import gurobipy as gp

N, M = 2, 8
np.random.seed(6)
D = np.random.uniform(size=(N, M))
B = np.array([1, 1])
         


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
        p_0 = (sum(B) / M) * np.ones(shape=M)
        beta_0 = np.amax(p_0 / D, axis=1)

    return p_0, beta_0

def LMO(c, A, b, C=None, d=None, return_dual=False):
    m = gp.Model()
    N = len(c)

    M = A.shape[1] - len(c)
    p = m.addMVar(M)
    beta = m.addMVar(len(c))

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

fig, ax = plt.subplots(1, figsize=(30, 24))

plt.plot(vertices_considered["beta_1"], vertices_considered["beta_2"], c='green', marker='o', linestyle='-')
plt.xlim(0.5 * min_v_beta_1, 1.1 * max_v_beta_1)
plt.ylim(0.5 * min_v_beta_2, 1.1 * max_v_beta_2)

plt.plot(beta_1_in_algorithm, beta_2_in_algorithm, c='blue', marker='H', zorder=10)
for i in range(len(beta_1_in_algorithm) - 1):
    segment_length = np.sqrt((beta_1_in_algorithm[i + 1] - beta_1_in_algorithm[i]) ** 2 + (beta_2_in_algorithm[i + 1] - beta_2_in_algorithm[i]) ** 2)
    plt.arrow(beta_1_in_algorithm[i], beta_2_in_algorithm[i], 0.7 * (beta_1_in_algorithm[i + 1] - beta_1_in_algorithm[i]), 0.7 * (beta_2_in_algorithm[i + 1] - beta_2_in_algorithm[i]), shape='full', lw=0, length_includes_head=True, head_length=min(0.4 * segment_length, 0.2), head_width=min(.3 * segment_length, 0.15), overhang=0.5, color='blue')

ax.spines[['right', 'top']].set_visible(False)
plt.arrow(0.5 * min_v_beta_1, 0.5 * min_v_beta_2, (1.1 * max_v_beta_1 - 0.5 * min_v_beta_1), 0, shape='full', lw=0, length_includes_head=True, head_length=0.1, head_width=0.05, overhang=0.3, color='black')
plt.arrow(0.5 * min_v_beta_1, 0.5 * min_v_beta_2, 0, (1.1 * max_v_beta_2 - 0.5 * min_v_beta_2), shape='full', lw=0, length_includes_head=True, head_length=0.05, head_width=0.2, overhang=0.3, color='black')

plt.xlabel(r"$\beta_1$")
plt.ylabel(r"$\beta_2$")

plt.savefig("./an-instance-feasible-region-boundary-and-algorithm.png")