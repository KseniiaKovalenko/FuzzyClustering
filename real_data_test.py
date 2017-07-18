from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from fuzzyrel import FuzzyRel
from sklearn.preprocessing import StandardScaler
from visualize_results import visualizer
from data_utils import *
import time

N_ATTRS = 6
N_CLUSTERS = 4

if __name__ == '__main__':
    data = parse_data(N = N_ATTRS)
    label = []
    for k, v in data.items(): label.append(k)

    X = to_nparray(data)
    X = StandardScaler().fit_transform(X)

    brc = Birch(n_clusters=N_CLUSTERS)
    k_means = MiniBatchKMeans(n_clusters=N_CLUSTERS)
    sc = SpectralClustering(n_clusters=N_CLUSTERS)
    fr = FuzzyRel(n_clusters=N_CLUSTERS, t_norm='min_max')

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


        visualizer(X, cluster_labels, name, real_data = True, n = 3 , labels = label)