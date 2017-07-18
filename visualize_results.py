import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold


def visualizer(data, cluster_labels, name, labels = [], n = 2, real_data = False):
    if (n == 2) and (real_data == False):
        mds = manifold.MDS(2, max_iter=100, n_init=1, random_state=1)
        trans_data = mds.fit_transform(data).T

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(trans_data[0], trans_data[1], s=100,
                   c=np.array([x for x in range(20, 40)])[cluster_labels].tolist())
        plt.title(name)
    elif(n == 3) and (real_data == False):
        mds = manifold.MDS(3, max_iter=100, n_init=1, random_state=1)
        trans_data = mds.fit_transform(data).T

        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        ax.scatter(trans_data[0], trans_data[1], trans_data[2], s=100,
                    c=np.array([x for x in range(20, 40)])[cluster_labels].tolist())
        ax.set_title(name)
    elif (n == 3) and (real_data == True):
        mds = manifold.MDS(3, max_iter=100, n_init=1, random_state=1)
        trans_data = mds.fit_transform(data).T

        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        ax.scatter(trans_data[0], trans_data[1], trans_data[2], s=100,
                   c=np.array([x for x in range(20, 40)])[cluster_labels].tolist())
        for x, y, z, k in zip(trans_data[0], trans_data[1], trans_data[2], labels):
            ax.text(x + 0.1, y + 0.1, z + 0.1, '%s' % k, size=9, zorder=1, color='k')
        ax.set_title(name)


    plt.show()
