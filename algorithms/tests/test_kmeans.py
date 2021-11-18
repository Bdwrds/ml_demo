"""
Testing the results of my implementations
author: Ben E
Date: 17/11/21
"""
import numpy as np
import os
import logging
import pytest
from algorithms.clustering import kmeans

logging.basicConfig(
    filename='./logging/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)-15s - %(name)s - %(level)s - %(message)s'
)

FP_IRIS='../data/iris.csv'

@pytest.fixture(scope="session")
def data():
    try:
        assert os.path.isfile(FP_IRIS)
        logging.info("Filepath exists - loading iris")
        iris = np.loadtxt(FP_IRIS, delimiter=',', skiprows=1)
        return iris
    except OSError:
        logging.info("File doesn't exist - save in correct path")

@pytest.fixture(scope="session")
def data_points(data):
    assert data.shape[0] == 150
    assert data.shape[1] == 5
    logging.info("Loaded data - returning data points")
    data_points = data[:, :4]
    return data_points

@pytest.fixture(scope="session")
def data_labels(data):
    assert data.shape[0] == 150
    assert data.shape[1] == 5
    logging.info("Loaded data - returning data labels")
    data_labels = data[:, 4]
    return data_labels

def calc_centroids(X, y):
    cent=[]
    for kv in np.unique(y):
        cent.append(X[y == kv, :4].mean(axis=0))
    cent_adj = np.array(cent).sum(axis=1)
    cent_adj.sort()
    return cent_adj

def test_fitting(data_points, data_labels):
    km = kmeans(k_value=3, seed=5, iter_value=3)
    km.fit(data_points)
    centroids_est = km.centroids
    centroids_est_adj = centroids_est.sum(axis=1)
    centroids_est_adj.sort()
    centroids_real = calc_centroids(data_points, data_labels)
    assert round(centroids_est_adj[0], 0) == round(centroids_real[0], 0)
