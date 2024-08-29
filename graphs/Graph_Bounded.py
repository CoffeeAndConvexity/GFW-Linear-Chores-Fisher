from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import cvxpy as cp

A = np.array([[0, 0, 0], [1.5, 3, 3], [1.5, 3, 0], [1.5, 0, 0], [0, 3, 0]])
B = np.array([[1, 2, 2], [1.5, 2, 2], [1, 3, 2], [1.5, 3, 2]])
hull = ConvexHull(A)
hull.simplices

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

lightgray = '#D3D3D3'

hull_simplices = np.array([[3, 1, 0], 
                           [3, 2, 1], 
                           [4, 1, 0], 
                           [4, 2, 1], 
                           [4, 3, 2], 
                           [4, 3, 0]])

for s in hull_simplices:
    
    tri = Poly3DCollection([A[s]])

    if (s == hull_simplices[0]).all() or (s == hull_simplices[2]).all():
        tri.set_color('gray') 
    else:
        tri.set_color(lightgray) 
    tri.set_alpha(0.15)
    tri.set_edgecolor('none')
    ax.add_collection3d(tri)
    
showed_edges = [(0, 1), (0, 3), (0, 4)]
for v0, v1 in showed_edges:
    ax.plot(xs=A[[v0, v1], 0], ys=A[[v0, v1], 1], zs=A[[v0, v1], 2], color='black')

plain_simplices = np.array([[2, 1, 0], 
                           [3, 2, 1]])

for s in plain_simplices:
    
    tri = Poly3DCollection([B[s]])

    tri.set_color('blue')
    tri.set_alpha(0.5)
    tri.set_edgecolor('none')
    ax.add_collection3d(tri)

showed_edges = [(0, 1), (0, 2)]
for v0, v1 in showed_edges:
    ax.plot(xs=B[[v0, v1], 0], ys=B[[v0, v1], 1], zs=B[[v0, v1], 2], color='black')



# ax.scatter(A[0], A[0], A[0], marker='o', color='green')

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def draw_descent_direction(beta_1, beta_2, p, with_point=False, color=['blue', 'orange'], target=None):

    if target is None:
        a = Arrow3D([beta_1, beta_1 + 0.5 * (- 1 / beta_1)], [beta_2, beta_2 + 0.5 * (- 1 / beta_2)], [p, p + 0.5 * 1], mutation_scale=20, lw=3, arrowstyle="-|>", color=color[0])
    else:
        a = Arrow3D([beta_1, target[0]], [beta_2, target[1]], [p, target[2]], mutation_scale=8, lw=2, arrowstyle="-|>", color=color[0])
    ax.add_artist(a)

    if with_point:
        if target is None:
            ax.scatter(beta_1, beta_2, p, marker='o', color=color[1], zorder=10)
        else:
            ax.scatter(beta_1, beta_2, p, marker='o', s=5, color=color[1], zorder=10)

    

draw_descent_direction(1, 2, 2, with_point=True, color=['blue', 'orange'])


def polytope_dual_A_b_form_with_red_beta_p(N, M, D, B):
    A = list()
    # IMPORTANT: Here, we need the nonnegative constraints
    for i in range(N):
        D_i_T = np.zeros(shape=(M, N))
        D_i_T[:, i] = - D[i]
        A_i = np.concatenate([D_i_T, np.identity(M)], axis=1)
        A.append(A_i)
    A.append(-np.identity(M + N))
    A = np.concatenate(A, axis=0)
    b = np.concatenate([np.zeros(shape=(M * N)), np.zeros(shape=M + N)])

    return A, b

A, b = polytope_dual_A_b_form_with_red_beta_p(2, 1, np.array([[2], [1]]), np.array([1, 1, 1]))


ax.set_xticks(np.arange(0, 1.5, step=0.5))
ax.set_yticks(np.arange(0, 3, step=0.5))
ax.xaxis.line.set_color("gray")
ax.yaxis.line.set_color("gray")
ax.zaxis.line.set_color("gray")
ax.tick_params(axis='x', which='major', labelsize=23, pad=-5, color="gray")
ax.tick_params(axis='y', which='major', labelsize=23, pad=-5, color="gray")
ax.tick_params(axis='z', which='major', labelsize=23, pad=-5, color="gray")
ax.set_ylim([0, 3.2])
ax.set_xlim([0, 1.7])
ax.set_zlim([0, 3.2])
ax.set_ybound([0, 3])
ax.set_xbound([0, 1.5])
ax.set_zbound([0, 3])
ax.set_box_aspect([1.5, 3, 3])
# ax.set_aspect('equal', adjustable='box')

ax.set_xlabel(r"$\beta_1$", size=25, weight='bold')
ax.set_ylabel(r"$\beta_2$", size=25, weight='bold')
ax.set_zlabel(r"$p$", size=25, weight='bold')

plt.draw()
plt.tight_layout()
ax.view_init(elev=10., azim=-50.)
plt.savefig("./Graph_Bounded.png", bbox_inches='tight')