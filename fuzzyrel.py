import numpy as np
from time import time
from clustering_estimation import estimations
from collections import OrderedDict
from numpy import array
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from functools import reduce

class FuzzyRel:

    def __init__(self, n_clusters = 4, D='manhattan', t_norm='min_max', p = 5, r = 10):
        self.n_clusters = n_clusters
        self.t_norm = self.t_conorm = t_norm

        if D == 'euclidean':
            self.D = lambda data, x, y: np.sqrt(sum((x[i] - y[i]) ** 2 for i in range(len(data[0]))))
        elif D == 'power':
            self.p = p
            self.r = r
            self.D = lambda data, x, y: sum(abs(x[i] - y[i]) ** self.p
                                            for i in range(len(data[0]))) ** (1/self.r)
        else:
            self.D = lambda data, x, y: sum(abs(x[i] - y[i]) for i in range(len(data[0])))

        if t_norm == 'min_pow':
            self.p = p
            self.S = lambda x, y: min(1, (x ** self.p + y ** self.p) ** (1/self.p))
            self.T = lambda x, y: 1 - min(1, ((1 - x) ** self.p + (1 - y) ** self.p) ** (1/self.p))
        elif t_norm == 'lukash':
            self.S = lambda x, y: min(1, x + y)
            self.T = lambda x, y: max(0, x + y - 1)
        else:
            self.S = max
            self.T = min

    def to_nparray(self, ordered_data):
        return array([v for _, v in ordered_data.items()])


    def distances_mx(self, data):
        distance_matrix = []
        for k1 in range(len(data)):
            matrix_line = []
            for k2 in range(len(data)):
                if k1 >= k2:
                    matrix_line.append(0.0)
                elif k1 < k2:
                    matrix_line.append(self.D(data, data[k1], data[k2]))
            distance_matrix.append(matrix_line)
        return np.array(distance_matrix) + np.array(distance_matrix).T


    def norm_coefficient_of_relationship(self, d_mx):
        c_o_r_matrix = []
        for k in range(len(d_mx)):
            max_d = max(d_mx[k])
            c_o_r_matrix.append([ 1 - d_mx[k][i] / max_d for i in range(len(d_mx[k]))])
        return np.array(c_o_r_matrix)

    def relative_c_o_r(self, c_o_r_mx):
        relative_c_o_r_matrix = []
        for ki in range(len(c_o_r_mx)):
            f_line = []
            for kj in range(len(c_o_r_mx)):
                f_line.append(list(1 - abs(c_o_r_mx[ki][m] - c_o_r_mx[kj][m]) for m in range(len(c_o_r_mx[ki]))))
            relative_c_o_r_matrix.append(f_line)
        return np.array(relative_c_o_r_matrix)

    def ksi_ab(self, rel_c_o_r_mx):
        ksi_ab_matrix = []
        for k in range(len(rel_c_o_r_mx)):
            ksi_ab_matrix.append([])
            for i in range(len(rel_c_o_r_mx[k])):
                ksi_ab_matrix[k].append(reduce(self.T, rel_c_o_r_mx[k][i]))
        return np.array(ksi_ab_matrix)

    def equal_ksi(self, mx1, mx2):
        for i in range(len(mx1)):
            for j in range(i + 1, len(mx1)):
                if mx1[i][j] != mx2[i][j]:
                    return False
        return True

    def pow_ksi(self, ksi1, ksi2):
        powered_ksi = []
        for k1 in range(len(ksi1)):
            powered_ksi.append([])
            for i in range(k1): powered_ksi[k1].append(0.0)
            for k2 in range(k1, len(ksi2)):
                powered_ksi[k1].append(reduce(self.S, (self.T(ksi1[k1][k3], ksi2[k3][k2]) for k3 in range(len(ksi1)))))
        return np.array(powered_ksi) + np.array(powered_ksi).T - np.diag(np.array(powered_ksi).diagonal())

    def unite_ksi(self, ksi1, ksi2):
        united_ksi = []
        for k1 in range(len(ksi1)):
            united_ksi.append([])
            for i in range(k1): united_ksi[k1].append(0.0)
            for k2 in range(k1, len(ksi2)):
                united_ksi[k1].append(self.S(ksi1[k1][k2], ksi2[k1][k2]))
        return np.array(united_ksi) + np.array(united_ksi).T - np.diag(np.array(united_ksi).diagonal())

    def transitive_closure(self, data):
        prev_data = united_data = data
        powered_ksi = self.pow_ksi(data, data)
        while not self.equal_ksi(powered_ksi, prev_data):
            united_data = self.unite_ksi(united_data, powered_ksi)
            prev_data = powered_ksi
            powered_ksi = self.pow_ksi(powered_ksi, data)
        return united_data

    def make_clusters_matrix(self, un_d_mx, keys):
        alpha_inf_set = sorted(set(un_d_mx[i][j] for j in range(len(un_d_mx)) for i in range(len(un_d_mx))))
        clusters_matrix = OrderedDict()
        for a_inf in alpha_inf_set:
            clusters_matrix[a_inf] = []
            for obj, i in zip(keys, range(len(keys))):
                if clusters_matrix[a_inf] == []:
                    clusters_matrix[a_inf].append({obj})
                else:
                    clustered = False
                    for cluster in clusters_matrix[a_inf]:
                        for cluster_obj in cluster:
                            j = 0
                            for obj_s, i_s in zip(keys, range(len(keys))):
                                if cluster_obj == obj_s:
                                    j = i_s
                            if un_d_mx[i][j] < a_inf:
                                break
                        else:
                            cluster.add(obj)
                            clustered = True
                        if clustered:
                            break
                    if not clustered:
                        clusters_matrix[a_inf].append({obj})
        return clusters_matrix

    def fit_predict(self, X):

        data = {}
        for data_key in range(len(X)):
            data[data_key] = X[data_key]

        #t0 = time()
        np_data = self.to_nparray(data)
        #print(np_data)
        #t1 = time()
        #print("Data %.2g sec" % (t1-t0))

        #t0 = time()
        distance_matrix = self.distances_mx(np_data)
        #print(distance_matrix)
        #print()
        #t1 = time()
        #print("Dist %.2g sec" % (t1-t0))

        #t0 = time()
        c_o_r_matrix = self.norm_coefficient_of_relationship(distance_matrix)
        #print(c_o_r_matrix)
        #print()
        #t1 = time()
        #print("mu matrix %.2g sec" % (t1-t0))

        #t0 = time()
        relative_c_o_r_matrix = self.relative_c_o_r(c_o_r_matrix)
        #print()
        #t1 = time()
        #print("ksi rel %.2g sec" % (t1-t0))

        #t0 = time()
        ksi_ab_matrix = self.ksi_ab(relative_c_o_r_matrix)
        #print(ksi_ab_matrix)
        #print()
        #t1 = time()
        #print("ksi ab %.2g sec" % (t1-t0))

        #t0 = time()
        united_ksi = self.transitive_closure(ksi_ab_matrix)
        #print(united_ksi)
        #print()
        #t1 = time()
        #print("Transitive %.2g sec" % (t1-t0))

        #t0 = time()
        clusters_matrix = self.make_clusters_matrix(united_ksi, data.keys())
        #t1 = time()
        #print("Res %.2g sec" % (t1-t0))
        
        for k, v in clusters_matrix.items():
            cluster_pred = []
            for keys in data.keys():
                for label, num in zip(clusters_matrix.get(k), range(len(clusters_matrix.get(k)))):
                    if keys in label:
                        cluster_pred.append(num)
            #print(k, cluster_pred, len(clusters_matrix.get(k)))
            #print()
            
            if len(clusters_matrix.get(k)) == self.n_clusters:
                return cluster_pred
            

if __name__ == '__main__':

    X, y = make_blobs(n_samples=100, centers=4, n_features=6,
                      random_state=0, cluster_std=2)
    X = StandardScaler().fit_transform(X)

    data = {}
    for data_key in range(len(X)):
        data[data_key] = X[data_key]

    fr = FuzzyRel(n_clusters=4, t_norm='min_max')
    t0 = time()
    cluster_labels = fr.fit_predict(X)
    t1 = time()
    estimations('frc', t1-t0, X, y, cluster_labels)
    


