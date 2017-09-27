# -*- coding: utf-8 -*-

"""Base object for hierarchical classifiers

This module contains the definition of a general hierarchical classifier, which 
forms the basis for the :class:`.SmartSVM` classifier. The 
:class:`.HierarchicalClassifier` class is an abstract base class, which leaves 
the ``fit`` and ``_split`` methods to be defined in the subclass.

"""

from __future__ import division

import numpy as np
import six

from abc import ABCMeta, abstractmethod, abstractproperty

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from .utils import indices_from_classes


@six.add_metaclass(ABCMeta)
class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """Base class for a hierarchical classifier

    This class is a base class for a hierarchical classifier which contains a 
    hierarchy of binary classification problems. It forms the basis of the 
    :class:`.SmartSVM` class and can be used to implement other hierarchical 
    classifiers. Note that this class is also the node class for the binary 
    tree: it has children which are in turn also 
    :class:`.HierarchicalClassifier` instances.

    Parameters
    ----------

    binary_clf : classifier
        The type of the binary classifier to be used for each binary 
        subproblem. This will typically be a scikit-learn classifier such as 
        LinearSVC, DecisionTreeClassifier, etc.

    clf_params : dict
        Parameters to pass on to the constructor of the ``binary_clf`` type 
        classifier. It must be a dict with a mapping from parameter to value.

    n_jobs : int
        Number of parallel jobs to use where applicable.


    Attributes
    ----------

    classifier_ : classifier
        The binary classifier for this node in the tree, if it exists. This can 
        also be obtained with the ``_get_binary()`` method.

    negative_child_ : HierarchicalClassifier
        The "left" child of the binary tree below this node.

    positive_child_ : HierarchicalClassifier
        The "right" child of the binary tree below this node.

    negative_ : set
        Set of labels of the binary subproblem that are on the "left" part of 
        the tree.

    positive_ : set
        Set of labels on the binary subproblem that are on the "right" part of 
        the tree.

    Methods
    -------

    predict(X)
        Perform classification on dataset X

    """

    def __init__(self, binary_clf=LinearSVC, clf_params=None, n_jobs=1):
        self.binary_clf = binary_clf
        self.clf_params = {} if clf_params is None else clf_params
        self.n_jobs = n_jobs

    @abstractmethod # pragma: no cover
    def fit(self, X, y):
        """ Fit model. """

    @abstractproperty # pragma: no cover
    def elements(self):
        """ Elements in the graph or list of nodes """

    @abstractmethod # pragma: no cover
    def _split(self):
        """ Method for splitting up the classes. """

    def predict(self, X):
        """ Predict the class labels using the hierarchical classifier """
        if X.shape[0] == 0:
            return np.tile(float("nan"), (X.shape[0], ))
        if not self.is_splitable:
            return np.tile(self.elements[0], (X.shape[0], ))
        check_is_fitted(self, "classifier_")
        check_is_fitted(self, "negative_child_")
        check_is_fitted(self, "positive_child_")
        pred = self.classifier_.predict(X)
        idx_neg = pred == -1
        idx_pos = pred == 1
        dtype = np.array(self.elements).dtype
        predictions = np.zeros((X.shape[0], ), dtype=dtype)
        predictions[idx_neg, ] = self.negative_child_.predict(X[idx_neg, :])
        predictions[idx_pos, ] = self.positive_child_.predict(X[idx_pos, :])
        return predictions

    @property
    def is_splitable(self):
        """ Check if the elements of this node can be split further """
        return len(self.elements) > 1

    @property
    def is_splitted(self):
        """ Check if this node is already split """
        return hasattr(self, "positive_") and hasattr(self, "negative_")

    def _stack(self):
        """ Construct a flat list of all the nodes in the hierarchy """
        stack = [self]
        if self.is_splitted:
            stack.extend(self.negative_child_._stack())
            stack.extend(self.positive_child_._stack())
        return stack

    def _split_all(self):
        """ Recursively split all nodes """
        self._split()
        if self.is_splitted:
            self.negative_child_._split_all()
            self.positive_child_._split_all()

    def _set_binary(self, clf=None):
        """ Initialize the binary classifier of this node """
        if not self.is_splitable:
            self.classifier_ = None
            return
        if not clf is None:
            self.classifier_ = clf
            return
        self.classifier_ = self.binary_clf(**self.clf_params)

    def _get_binary(self):
        """ Return the classifier of the node, initializing if needed """
        if not hasattr(self, "classifier_"):
            self._set_binary()
        return self.classifier_

    def _binary_data(self, X, y):
        """ Return data for the binary classification problem of the node """
        idx_negative = indices_from_classes(self.negative_, y)
        idx_positive = indices_from_classes(self.positive_, y)
        idx_all = np.logical_or(idx_negative, idx_positive)

        y_np = np.zeros((y.shape[0], ))
        y_np[idx_negative, ] = -1
        y_np[idx_positive, ] = 1

        X_needed = X[idx_all, ]
        y_needed = y_np[idx_all, ]

        return X_needed, y_needed

    def _fit_binary(self, X, y):
        """ Fit the classifier of the node """
        X_needed, y_needed = self._binary_data(X, y)
        self._set_binary()
        self.classifier_.fit(X_needed, y_needed)

    def __str__(self): # pragma: no cover
        s = repr(self) + "\n"
        s += "Tree:" + "\n"
        s += self.print_tree()
        return s

    def print_tree(self, prefix="", is_tail=None): # pragma: no cover
        """ Print the tree """
        s = prefix
        if is_tail is None:
            s += ""
        elif is_tail:
            s += "└── R "
        else:
            s += "├── L "

        s += repr(self.negative_.union(self.positive_)) + "\n"
        if is_tail is None or is_tail:
            buf = "%s    " % prefix
        else:
            buf = "%s|   " % prefix

        if self.negative_child_ and len(self.negative_) > 1:
            s += self.negative_child_.print_tree(prefix=buf, is_tail=False)
        else:
            s += buf + "├── L " + repr(self.negative_) + "\n"

        if self.positive_child_ and len(self.positive_) > 1:
            s += self.positive_child_.print_tree(prefix=buf, is_tail=True)
        else:
            s += buf + "└── R " + repr(self.positive_) + "\n"

        return s
