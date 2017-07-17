from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from visualize_results import visualizer
from clustering_estimation import estimations
from fuzzyrel import FuzzyRel

import time

N_ATTRS = 2
N_CLUSTERS = 2

if __name__ == '__main__':

    #X, y = make_blobs(n_samples=100, centers=N_CLUSTERS, n_features=N_ATTRS,
                       #random_state=0, cluster_std=1.5)

    #X, y = make_circles(noise = 0.03, factor = 0.6)
    X, y = make_moons(noise=.06)
    X = StandardScaler().fit_transform(X)

    visualizer(X, y)
    estimations('Etalon', (1 - 0), X, y, y)


    brc = Birch(n_clusters=N_CLUSTERS)
    t0 = time.time()
    brc.fit(X)
    cluster_labels = brc.predict(X)
    t1 = time.time()

    estimations('BIRCH', (t1 - t0), X, y, cluster_labels)
    visualizer(X, cluster_labels)


    k_means = MiniBatchKMeans(n_clusters=N_CLUSTERS)
    t0 = time.time()
    k_means.fit(X)
    cluster_labels = k_means.predict(X)
    t1 = time.time()

    estimations('K-Means++',(t1 - t0), X, y, cluster_labels)
    visualizer(X, cluster_labels)

    sc = SpectralClustering(n_clusters=N_CLUSTERS)
    t0 = time.time()
    cluster_labels = sc.fit_predict(X)
    t1 = time.time()

    estimations('Spectral Clustering', (t1 - t0), X, y, cluster_labels)
    visualizer(X, cluster_labels)

    fr = FuzzyRel(n_clusters = N_CLUSTERS, t_norm = 'min_max')#, D = 'euclidean')
    t0 = time.time()
    cluster_labels = fr.fit_predict(X)
    t1 = time.time()

    estimations('FRC', (t1 - t0), X, y, cluster_labels)
    visualizer(X, cluster_labels)



