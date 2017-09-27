# -*- coding: utf-8 -*-

"""Code for computing the number of cross connections in an MST.

The functions in this module are used to compute the number of 
cross-connections between data points of different classes in the Euclidean 
Minimal Spanning Tree. It is the interface between the Cython code talking to 
the C implementations, and the higher level functions that compute the 
Henze-Penrose divergence.

"""

from __future__ import division

from scipy.spatial.distance import pdist, squareform

from .ortho_mst import ortho_mst_count
from .multiclass_mst_count import multiclass_mst_count


def binary_cross_connections(X, y, nTrees=3):
    """Compute the number of cross connections for two classes

    This function computes the average number of non-trivial cross-connections 
    in the orthogonal MSTs between points from different classes. For each MST 
    the number of times a point from one class is connected to a point from a 
    different class is recorded. This is reduced by 1 to adjust for the trivial 
    connection that will always exist, even in the case of large separation.  
    The result is averaged for each of the orthogonal MSTs.

    A warning can occur when the requested number of orthogonal MSTs can't be 
    computed on the data. This happens when there are either too few datapoints 
    to construct this many MSTs, or in the extreme case where each edge to a 
    data point is used in previous MSTs.

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix

    y : np.ndarray
        Class labels, assumed to be only +1 and -1

    nTrees : int
        The number of orthogonal MSTs to compute

    Returns
    -------
    C : float
        The (average) number of non-trivial connections between instances of 
        different classes in the MSTs

    """

    # Calculate the Euclidean distance matrix
    weights = squareform(pdist(X, 'euclidean'))

    # Number of edges connecting nodes from different classes
    S, actual_trees = ortho_mst_count(weights, y, nTrees)
    if actual_trees < nTrees:
        print("Warning: instead of the requested %i orthogonal MSTs, %i "
            "orthogonal MSTs were used. Usually this is nothing to worry "
            "about." % (nTrees, actual_trees))
    nTrees = actual_trees

    # Average this number and correct it slightly
    C = (S - nTrees)/nTrees

    return C


def ovr_cross_connections(X, y, nClass, nTrees=3):
    """Compute the One-vs-Rest cross-connection counts

    This function computes the cross-connection counts in the One-vs-Rest 
    setting by constructing orthogonal MSTs on the entire dataset, and 
    collecting cross-connection counts for each class.

    Parameters
    ----------

    X : numpy.ndarray
        Numpy array of the data matrix

    y : numpy.ndarray
        Numpy array of the class labels, assumed to be in 0..nClass-1

    nClass : int
        The number of classes in y for the full problem. This needs to be 
        supplied in case y is a subset of the full dataset where not all 
        classes are present. This ensures that the outcome matrix has the 
        appropriate size.

    nTrees : int
        The number of orthogonal MSTs to compute

    Returns
    -------
    Cs : numpy.ndarray
        Vector of class cross-connection counts, ordered by class label (nClass 
        x nClass)

    """
    full_set = set(range(nClass))
    if not set(y).issubset(full_set):
        raise ValueError("Class labels should be in {0, ..., nClass-1}")

    # Calculate the Euclidean distance matrix
    weights = squareform(pdist(X, 'euclidean'))

    # Compute the cross-connection counts for each class
    # (matrix of nClass x nClass)
    C, actual_trees = multiclass_mst_count(weights, y, nClass, nTrees)

    # Throw a warning when not enough MSTs could be computed
    if actual_trees < nTrees:
        print("Warning: instead of the requested %i orthogonal MSTs, %i "
            "orthogonal MSTs were used. Usually this is nothing to worry "
            "about." % (nTrees, actual_trees))
    nTrees = actual_trees

    # Sum the matrix to obtain the class cross-connection counts and correct 
    # for trivial connection
    sums = C.sum(0)
    sums[sums > 0] -= nTrees

    # Average over the number of MSTs computed
    Cs = sums/nTrees
    return Cs
