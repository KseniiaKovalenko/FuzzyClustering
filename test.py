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
    estimations('Model', 0, X, y, y)

    brc = Birch(n_clusters=N_CLUSTERS)
    k_means = MiniBatchKMeans(n_clusters=N_CLUSTERS)
    sc = SpectralClustering(n_clusters=N_CLUSTERS)
    fr = FuzzyRel(n_clusters=N_CLUSTERS, t_norm='min_max', D = 'euclidean')

    clustering_algorithms = [brc, k_means, sc, fr]
    clustering_names = ['BIRCH', 'K-means++', 'Spectral Clustering', 'FRC']

    for name, algorithm in zip(clustering_names, clustering_algorithms):
        if name ==  'BIRCH' or name == 'K-means++':
            t0 = time.time()
            algorithm.fit(X)
            cluster_labels = algorithm.predict(X)
            t1 = time.time()
        elif name == 'Spectral Clustering' or name == 'FRC':
            t0 = time.time()
            cluster_labels = algorithm.fit_predict(X)
            t1 = time.time()

        estimations(name, (t1 - t0), X, y, cluster_labels)
        visualizer(X, cluster_labels)
