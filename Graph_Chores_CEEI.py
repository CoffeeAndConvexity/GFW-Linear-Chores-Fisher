import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3, 30)
y = np.linspace(0, 3, 30)
x, y = np.meshgrid(x, y)
# z = np.zeros(shape=x.shape)
z = np.minimum(x, 2*y)

len_x, len_y = x.shape
x_lst = list()
y_lst = list()
z_lst = list()
for r in range(len_x):
    for c in range(len_y):
        h = 0
        while h < np.minimum(x[r][c], 2*y[r][c]):
            x_lst.append(x[r][c])
            y_lst.append(y[r][c])
            z_lst.append(h)
            h += 0.02


y_label = r'x'
x_label = r'y'
z_label = r'z'

x_ = np.linspace(2, 3, 100)
y_ = np.linspace(1, 3, 200)
x_, y_ = np.meshgrid(x_, y_)
z_ = 2 

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
inner = ax.scatter(x_lst, y_lst, z_lst, color='green', s=0.01, alpha=0.05)
surf = ax.scatter(x, y, z, color='teal', s=0.01, alpha=0.6)
new_plane = ax.scatter(x_, y_, z_, color='y', s=0.01)
kkt_point = ax.scatter([2, ], [1, ], [2, ], color='blue', s=3)

ax.xaxis._axinfo['juggled'] = (0,0,0)
ax.yaxis._axinfo['juggled'] = (1,1,1)
ax.zaxis._axinfo['juggled'] = (2,2,2) 

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_zlabel(z_label)
# fig.show()

fig.savefig('Graph')