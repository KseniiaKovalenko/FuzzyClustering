import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold


def visualizer(data, cluster_labels, n = 2):
    if n == 2:
        mds = manifold.MDS(2, max_iter=100, n_init=1, random_state=1)
        trans_data = mds.fit_transform(data).T

        fig = plt.figure(figsize=(10, 10))
        #ax = Axes3D(fig)
        plt.scatter(trans_data[0], trans_data[1], s=100,
                   c=np.array([x for x in range(20, 40)])[cluster_labels].tolist())
    elif n ==3:
        mds = manifold.MDS(3, max_iter=100, n_init=1, random_state=1)
        trans_data = mds.fit_transform(data).T

        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        ax.scatter(trans_data[0], trans_data[1], trans_data[2], s=100,
                    c=np.array([x for x in range(20, 40)])[cluster_labels].tolist())

    plt.show()

