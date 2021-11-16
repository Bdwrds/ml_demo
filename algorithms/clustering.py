"""
Author: Ben E
Date: 16/11/21
"""

from numpy.random import uniform
from numpy.linalg import norm

class kmeans():
    def __init__(self, k_value: int, seed: int, iter_value):
        self.k_value = k_value
        self.seed = seed
        self.iter_value = iter_value
        self.fitted = False
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
            if sum(self.assignments == k)>0:
                new_centroids.append(self.data[self.assignments == k, :].mean(axis=0))
            else:
                new_centroids.append(self.centroids[k])
        return np.array(new_centroids)

    def _calc_assignment(self, data_points):
        return norm(data_points - self.centroids, ord=2, axis=1).argmin()

    def _cluster_assignment(self, data):
        new_assignments = []
        for row in data:
            new_assignments.append(self._calc_assignment(row))
        return np.array(new_assignments)

    def fit(self, data):
        self.dim = data.ndim
        self.data = data
        self.centroids = self._initialise()
        #self.assignments = np.zeros(self.data.shape[0])
        for itr in range(0, self.iter_value):
            self.assignments = self._cluster_assignment(data)
            self.centroids = self._update_centroids()
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
    import numpy as np
    from algorithms.clustering import kmeans
    data = np.loadtxt('./algorithms/iris.csv', delimiter=',')
    km = kmeans(k_value=3, seed=5, iter_value=15)
    km.fit(data)

    dt = data[:5,]

    km.predict(dt)

