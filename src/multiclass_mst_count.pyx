"""

"""

import ctypes
import cython

import numpy as np
cimport numpy as np

cdef extern void c_multiclass_mst_count(double *weights, long *labels, 
        long nTrees, long nObs, long nClass, long *count_ret, long *n_tree_ret)


@cython.boundscheck(False)
@cython.wraparound(False)
def multiclass_mst_count(
        np.ndarray[double, ndim=2, mode="c"] weights not None,
        np.ndarray[long, ndim=1, mode="c"] labels not None,
        long nClass,
        long nTrees):
    """
    """

    cdef long nObs
    cdef long actual_trees

    nObs = weights.shape[0]

    cdef np.ndarray[long, ndim=2, mode='c'] C
    C = np.zeros((nClass, nClass), dtype=ctypes.c_long)

    c_multiclass_mst_count(&weights[0, 0], &labels[0], nTrees, nObs, nClass, 
            &C[0, 0], &actual_trees)

    return C, actual_trees
