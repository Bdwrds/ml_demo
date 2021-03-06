"""
Basic implementation of clustering algorithms using numpy
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
        """
        Initialise the centroids randomly between the min/max of that dimension.
        :return: Centroid values for all dimensions and number of clusters/k
        """
        centroids = []
        for k in range(0, self.k_value):
            centroids.append([uniform(self.data[:,dim].min(), self.data[:,dim].max()) \
                     for dim in range(0, self.data.shape[1])])
        return np.array(centroids)

    def _update_centroids(self):
        """
        Given the latest assignments - calculate the mean of the new clusters
        :return: Centroids values of said clusters
        """
        new_centroids = []
        for k in range(0, self.k_value):
            if sum(self.new_assignments == k) > 0:
                new_centroids.append(self.data[self.new_assignments == k, :].mean(axis=0))
            else:
                new_centroids.append(self.new_centroids[k])
        return np.array(new_centroids)

    def _calc_assignment(self, new_data, kval):
        """
        Given a new data point, how far is it from the a specific centroid.
        :param new_data: The new data
        :param kval: The specific centroid
        :return: The distance to that centroid.
        """
        return norm(new_data - self.new_centroids[kval], ord=2, axis=1)

    def _cluster_assignment(self):
        """
        Calculate the distance to centroids and give new assignments
        :return: New assignments for each data point
        """
        distances = []
        for k_v in range(0, self.k_value):
            distances.append(self._calc_assignment(self.data, k_v))
        new_assignments = np.array(distances).argmin(axis=0)
        return new_assignments

    def _calc_inertia(self):
        """
        Calculate the inertia/ total within sum of squared distances
        :return: Calculated inertia value
        """
        inertia_p_k = []
        for kv in range(0, self.k_value):
            inertia_p_k.append(np.sum((self.data[self.new_assignments == kv] - self.new_centroids[kv])**2))
        return np.sum(inertia_p_k)

    def fit(self, data):
        """
        Primary function to fit clusters to the data
        :param data: Primary data being fitted
        """
        self.dim = data.ndim
        self.data = data
        for init in range(0, self.n_init):
            self.new_centroids = self._initialise()
            for itr in range(0, self.iter_value):
                self.new_assignments = self._cluster_assignment()
                self.new_centroids = self._update_centroids()

            if self.fitted:
                new_inertia = self._calc_inertia()
                print('%d - Inertia: %f - New Inertia: %f' % (init, self.inertia, new_inertia,))
                if new_inertia < self.inertia*0.975:
                    self.centroids = self.new_centroids
                    self.assignments = self.new_assignments
                    self.inertia = new_inertia
            else:
                self.inertia = self._calc_inertia()
                self.centroids = self.new_centroids
                self.assignments = self.new_assignments
            self.fitted = True
        self.new_centroids=self.centroids

    def predict(self, new_data):
        """
        Predict the cluster assignment of the new data.
        :param new_data: New data being classified.
        :return: Assignments of the new data points.
        """
        if self.fitted:
            distance_new = []
            if new_data.shape[1] == self.centroids.shape[1]:
                if new_data.shape[0] > 0:
                    for k_v in range(0, self.k_value):
                        distance_new.append(self._calc_assignment(new_data, k_v))
                    assignments = np.array(distance_new).argmin(axis=0)
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