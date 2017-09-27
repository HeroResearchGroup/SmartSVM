# -*- coding: utf-8 -*-

"""Functions for computing the Henze-Penrose estimate of the Bayes Error Rate

This module contains functions for computing the Henze-Penrose estimate of the 
Bayes Error Rate (BER).

"""

from __future__ import division

import numpy as np

from itertools import combinations
from math import sqrt
from multiprocessing import Pool
from networkx import Graph

from .divergence import divergence, ovr_divergence


def _worker(Xyijn):
    """Worker code for pairwise error estimation

    Calculates the error estimate for a pair of classes. The input Xyij is a 
    tuple with (X, y, i, j). The error is estimated between class i and class 
    j, where the data for classes i and j is gathered using ``X[y == i, :]``.

    Parameters
    ----------
    Xyijn : tuple
        Tuple with ``(X, y, i, j, normalize)`` where ``X`` is the dataset 
        containing at least the data for classes ``i`` and ``j``, ``y`` is the 
        vector of class labels, and ``normalize`` is a boolean variable 
        indicating if the normalized Bayes error estimates need to be 
        calculated.

    Returns
    -------
    tuple
        Tuple with (i, j, error)
    """
    X, y, i, j, norm = Xyijn
    Xi = X[y == i, :]
    Xj = X[y == j, :]
    error = hp_estimate(Xi, Xj, normalize=norm)
    return (i, j, error)


def compute_error_graph(X, y, n_jobs=1, normalize=True):
    """Compute the complete weighted graph of pairwise BER estimates

    Computes the estimate of the BER for each pair of classes and returns this 
    in a complete weighted graph. If desired, computing the error can be done 
    in parallel.

    Parameters
    ----------
    X : numpy.ndarray
        Numpy array of the data matrix

    y : numpy.ndarray
        Numpy array with the class labels

    n_jobs : int
        Number of parallel jobs to use

    normalize : bool
        Whether or not to use normalized BER estimation

    Returns
    -------
    G : networkx.Graph
        Weighted complete graph where the class labels are the nodes and the 
        edge weights are given by the BER estimate

    """
    classes = np.unique(y)

    # When n_jobs = 1 we don't want to use a pool of workers. This is useful 
    # when training using GridSearchCV from sklearn, since there it is not 
    # allowed for a child of GridSearchCV to have children of its own.
    args = [(X, y, i, j, normalize) for i, j in combinations(classes, 2)]
    if n_jobs == 1:
        results = [_worker(arg) for arg in args]
    else:
        pool = Pool(n_jobs)
        results = pool.map(_worker, args)
        pool.close()

    G = Graph()
    for i, j, weight in results:
        G.add_edge(i, j, weight=weight)

    return G


def compute_ovr_error(X, y, normalize=True):
    """Compute OvR-BER for each class

    The One-vs-Rest Bayes error rate is the error rate for a single class 
    compared to all other classes combined. This function computes this error 
    rate for each class in the dataset, using the 
    :func:`.divergence.ovr_divergence` function. By default the OvR-BER is 
    normalized with the estimates of the class prior probabilities.

    Parameters
    ----------
    X : numpy.ndarray
        The data matrix

    y : numpy.ndarray
        A vector of class labels

    normalize : bool
        Whether or not to normalize the OvR-BER

    Returns
    -------
    estimates : dict
        Dictionary with a mapping from the class label to a float representing 
        the OvR-BER estimate.

    """
    divergences = ovr_divergence(X, y)
    estimates = {k : 1.0/2.0 - 1.0/4.0*sqrt(dp) - 1.0/4.0*dp for k, dp in 
            divergences.items()}
    if normalize:
        for k in estimates:
            phat = X[y == k, :].shape[0] / (X.shape[0])
            qhat = X[y != k, :].shape[0] / (X.shape[0])
            ratio = min(phat, qhat)
            estimates[k] = estimates[k] / ratio
    return estimates


def hp_estimate(X1, X2, normalize=True):
    """Henze-Penrose estimation of the Bayes Error Rate

    Estimate the (normalized) Bayes error rate using the Henze-Penrose 
    estimator. The estimate is formed by averaging the upper and lower bounds 
    on the Bayes error. By default, the estimate is normalized with the 
    empirical estimates of the class prior probability.

    Parameters
    ----------
    X1 : numpy.ndarray
        Data matrix for one class

    X2 : numpy.ndarray
        Data matrix for the other class

    normalize : bool
        Whether or not to normalize the error using empirical estimates of 
        prior probabilities

    Returns
    -------
    estimate : float
        The Henze-Penrose estimate of the Bayes error rate

    """

    Dptilde = divergence(X1, X2)
    estimate = 1.0/2.0 - 1.0/4.0*sqrt(Dptilde) - 1.0/4.0*Dptilde
    if normalize:
        phat = X1.shape[0] / (X1.shape[0] + X2.shape[0])
        qhat = X2.shape[0] / (X1.shape[0] + X2.shape[0])
        ratio = min(phat, qhat)
        estimate = estimate / ratio
    return estimate
