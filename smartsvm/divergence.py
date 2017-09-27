# -*- coding: utf-8 -*-

"""Functions for computing the Henze-Penrose divergence

This module contains functions for computing the Henze-Penrose divergence 
between data from two classes (:func:`.divergence`) or between one class and 
the collection of all other classes (:func:`.ovr_divergence`).

"""

from __future__ import division

import numpy as np

from math import sqrt
from sklearn.preprocessing import LabelEncoder

from .cross_connections import (binary_cross_connections, 
        ovr_cross_connections)


def merge_and_label(X1, X2):
    """Merge two datasets

    Merge the datasets into one array and create a vector of labels to preserve 
    class membership.

    Parameters
    ----------
    X1 : numpy.ndarray
        Data matrix for one of the classes

    X2 : numpy.ndarray
        Data matrix for the other class

    Returns
    -------
    data : numpy.ndarray
        Data matrix which vertically stacks X1 and X2

    labels : numpy.ndarray
        Labels vector of -1 and 1 for X1 and X2 data respectively

    """
    # Number of instances in both datasets
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Combine data
    data = np.vstack((X1, X2))

    # Preserve class information
    labels1 = np.ones(n1, dtype=np.int_) * -1
    labels2 = np.ones(n2, dtype=np.int_)
    labels = np.hstack((labels1, labels2))

    return data, labels


def divergence(X1, X2, nTrees=3, bias_correction=True):
    """Compute the Henze-Penrose divergence

    Compute the Henze-Penrose divergence between data from two classes using 
    the Friedman-Rafsky statistic. This is based on the Euclidean minimal 
    spanning tree between two classes. To reduce variance in the estimate, a 
    number of orthogonal MSTs are used that can be set with a function 
    parameter. A bias correction to the divergence is applied, unless this is 
    disabled by the user (if disabled, the estimate is still corrected to 
    ensure non-negativity).

    Parameters
    ----------
    X1 : numpy.ndarray
        Data matrix for one of the classes

    X2 : numpy.ndarray
        Data matrix for the other class

    nTrees : int
        Number of orthogonal minimal spanning trees to use

    bias_correction : bool
        Whether or not to apply bias correction to the estimator

    Returns
    -------
    divergence : float
        The Henze-Penrose divergence between the classes

    """

    # Get array dimensions
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    N = n1 + n2

    # Create a merged dataset and keep the labels
    data, labels = merge_and_label(X1, X2)

    # Compute the cross connections
    C = binary_cross_connections(data, labels, nTrees=nTrees)

    # Calculate the divergence measure
    if bias_correction:
        # Calculate the bias-correction factor
        phat = n1/N
        qhat = n2/N
        gamma = (2.0*N*min(phat, qhat) - 3.0/4.0 * N + 1.0/4.0*N*sqrt(9.0 - 
            16.0*min(phat, qhat)))
        Dptilde = 1.0 - 2.0*min(gamma, C)/N
    else:
        Dptilde = 1.0 - 2.0*min(N/2, C)/N

    return Dptilde


def ovr_divergence(X, labels, nTrees=3, bias_correction=True):
    """Compute the One-vs-Rest Henze-Penrose Divergence for all classes

    This function is similar to :func:`.divergence` with the difference that it 
    computes the One-vs-Rest divergence. This is the divergence between one 
    class and the collection of all the other classes together. This divergence 
    is computed for each class simultaneously.

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix

    labels : numpy.ndarray
        Class labels for the instances

    nTrees : int
        The number of orthogonal MSTs to construct

    bias_correction : bool
        Whether or not to apply bias correction

    Returns
    -------
    ovr_bers : dict
        Dictionary with a mapping from the class label to the OvR-BER estimate

    """

    # Encode the classes
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    nClass = len(encoder.classes_)
    N = X.shape[0]

    Cs = ovr_cross_connections(X, y, nClass, nTrees=nTrees)

    Dptildes = np.zeros((nClass, ))
    for i in range(nClass):
        if bias_correction:
            phat = X[y == i, :].shape[0] / N
            qhat = X[y != i, :].shape[0] / N
            gamma = (2.0*N*min(phat, qhat) - 3.0/4.0 * N + 1.0/4.0*N*sqrt(9.0 - 
                16.0*min(phat, qhat)))
            Dptildes[i] = 1.0 - 2.0*min(gamma, Cs[i])/N
        else:
            Dptildes[i] = 1.0 - 2.0*min(N/2, Cs[i])/N

    ovr_bers = {}
    for i, dp in zip(encoder.classes_, Dptildes):
        ovr_bers[i] = dp
    return ovr_bers
