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

FP_IRIS='./algorithms/data/iris.csv'

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

@pytest.fixture(scope="session")
def km_model(data_points):
    km = kmeans(k_value=3, seed=1, iter_value=15, n_init=5)
    km.fit(data_points)
    return km

def calc_centroids(X, y):
    cent=[]
    for kv in np.unique(y):
        cent.append(X[y == kv, :4].mean(axis=0))
    return np.array(cent)

def test_fitting(km_model, data_points, data_labels):
    centroids_est = km_model.centroids
    centroids_real = calc_centroids(data_points, data_labels)
    centroid_gap = np.linalg.norm(centroids_est - centroids_real, ord=2, axis=1)
    assert centroid_gap[0] < 0.5
    assert centroid_gap[1] < 0.5
    assert centroid_gap[2] < 0.5

def test_predict(km_model, data_points):
    dt = data_points[:5, ]
    pred_top_5 = km_model.predict(dt)
    assert (pred_top_5 == km_model.assignments[:5]).all()