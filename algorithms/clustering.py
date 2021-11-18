"""
Author: Ben E
Date: 16/11/21
"""
import numpy as np
from numpy.random import uniform
from numpy.linalg import norm

class kmeans():
    def __init__(self, k_value: int, seed=1, iter_value=15, n_init=10):
        self.k_value = k_value
        self.seed = seed
        self.iter_value = iter_value
        self.fitted = False
        self.n_init = n_init
        np.random.seed(seed)

    def _initialise(self):
        centroids = []
        for k in range(0, self.k_value):
            centroids.append([uniform(self.data[:,dim].min(), self.data[:,dim].max()) \
                     for dim in range(0, self.data.shape[1])])
        return np.array(centroids)

    def _update_centroids(self):
        new_centroids = []
        for k in range(0, self.k_value):
            if sum(self.new_assignments == k) > 0:
                new_centroids.append(self.data[self.new_assignments == k, :].mean(axis=0))
            else:
                new_centroids.append(self.new_centroids[k])
        return np.array(new_centroids)

    def _calc_assignment(self, data_points):
        return norm(data_points - self.new_centroids, ord=2, axis=1).argmin()

    def _cluster_assignment(self):
        new_assignments = []
        for row in self.data:
            new_assignments.append(self._calc_assignment(row))
        return np.array(new_assignments)

    def _calc_inertia(self):
        inertia_p_k = []
        for kv in range(0, self.k_value):
            inertia_p_k.append(np.linalg.norm(self.new_centroids[kv] - self.data[self.new_assignments == kv]\
                                              , ord=2, axis=1).sum())
        return np.sum(inertia_p_k)

    def fit(self, data):
        self.dim = data.ndim
        self.data = data
        for init in range(0, self.n_init):
            self.new_centroids = self._initialise()
            for itr in range(0, self.iter_value):
                self.new_assignments = self._cluster_assignment()
                self.new_centroids = self._update_centroids()

            if self.fitted:
                new_inertia = self._calc_inertia()
                #print('%d - Inertia: %f - New Inertia: %f' % (init, self.inertia, new_inertia,))
                if new_inertia < self.inertia:
                    self.centroids = self.new_centroids
                    self.assignments = self.new_assignments
                    self.inertia = new_inertia
            else:
                self.inertia = self._calc_inertia()
                self.centroids = self.new_centroids
                self.assignments = self.new_assignments
            self.fitted = True

    def predict(self, new_data):
        if self.fitted:
            assignments = []
            if new_data.shape[1] == self.centroids.shape[1]:
                if new_data.shape[0] > 0:
                    for row in new_data:
                        assignments.append(self._calc_assignment(row))
                    return assignments
        else:
            print("Must obtain centroids prior to fitting")

if __name__=='__main__':
    FP_PATH='./algorithms/data/iris.csv'
    data = np.loadtxt(FP_PATH, delimiter=',', skiprows=1)
    data_points = data[:, :4]
    data_labels = data[:, 4]
    km = kmeans(k_value=3, seed=5, iter_value=15)
    km.fit(data_points)
    print(km.centroids)
