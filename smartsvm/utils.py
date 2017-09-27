# -*- coding: utf-8 -*-

"""Useful numerical utilities.
"""

from __future__ import division

import numpy as np

def indices_from_classes(classes, y):
    """Create index vector for all labels in classes

    Create an index vector with the same dimensions as the vector y, with 
    indices equal to True if the label is in the list of classes.

    Parameters
    ----------
    classes : list or numpy array
        Classes to check membership of

    y : numpy.ndarray
        Label vector

    Returns
    -------
    indices : numpy.ndarray (bool)
        Boolean vector with True for the elements of ``y`` which occur in 
        ``classes`` and False otherwise.

    """
    indices = np.zeros(y.shape, dtype=bool)
    for c in classes:
        indices = np.logical_or(indices, y == c)
    return indices
