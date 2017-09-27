"""
Orthogonal MST Counter

Compute the first `k` orthogonal minimal spanning trees and return the number 
of edges that come from different samples.

"""

import cython

import numpy as np
cimport numpy as np

cdef extern void c_ortho_mst_count(double *weights, long *labels, long nTrees, 
        long nObs, long *count_ret, long *n_tree_ret)

@cython.boundscheck(False)
@cython.wraparound(False)
def ortho_mst_count(np.ndarray[double, ndim=2, mode="c"] weights not None,
        np.ndarray[long, ndim=1, mode="c"] labels not None, long nTrees):
    """
    ortho_mst_count(weights, labels, nTrees)

    Takes the weight matrix and constructs nTree minimal spanning trees. For 
    each minimal spanning tree the number of edges that connect points with 
    different labels is calculated. The total number of edges for which this 
    occurs is returned. For the MST algorithm the algorithm of Whitney is used 
    (see the C file).

    param: weights -- a 2-d symmetric numpy array of np.float64.
    param: labels -- a 1-d numpy array of np.intc
    param: nTrees -- the number of minimal spanning trees to construct

    """

    cdef long nObs
    cdef long count
    cdef long actual_trees

    nObs = weights.shape[0]

    c_ortho_mst_count(&weights[0, 0], &labels[0], nTrees, nObs, &count, 
            &actual_trees)

    return count, actual_trees
