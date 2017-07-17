from sklearn import metrics


def estimations(name, time, X, y, cluster_labels):

    print("%s %.2g sec" % (name, time))
    print("Adjusted Rand Index: %0.3f"% metrics.adjusted_rand_score(y, cluster_labels))
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, cluster_labels))
    print("Completeness: %0.3f" % metrics.completeness_score(y, cluster_labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(y, cluster_labels))
    print("Calinski-Harabaz Index: %0.3f" % metrics.calinski_harabaz_score(X, cluster_labels))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, cluster_labels, metric='sqeuclidean'))
    print()