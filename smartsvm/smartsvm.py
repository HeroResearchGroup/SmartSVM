# -*- coding: utf-8 -*-

from __future__ import division

import networkx as nx
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets

from .base import HierarchicalClassifier
from .cut import graph_cut
from .error_estimate import compute_error_graph


class SmartSVM(HierarchicalClassifier):
    """SmartSVM classifier for multiclass classification

    This is the SmartSVM classifier. This classifier splits the classes into a 
    hierarchy based on the weighted complete graph of pairwise estimates of the 
    Bayes Error Rate between classes.

    Note that this class is used in a binary tree. It therefore has children 
    which are also elements of the :class:`.SmartSVM` class.

    This class inherits from the :class:`.HierarchicalClassifier` class, more 
    documentation on the basic methods of the class can be found there.

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

    graph : networkx.Graph
        Complete weighted graph with the pairwise Bayes error estimates. If it 
        is not supplied, it will be computed when the :meth:`.fit` method is 
        called.  This parameter exists for situations when the graph is 
        precomputed.

    cut_algorithm : str
        The algorithm to use for the graph cut. Several algorithms are 
        currently implemented, see :func:`.cut.graph_cut` for more details.

    normalize_error : bool
        Whether or not to normalize the Bayes error estimates with the 
        empirical estimate of the prior probability. 


    """

    def __init__(self, binary_clf=LinearSVC, clf_params=None, n_jobs=1, 
            graph=None, cut_algorithm='stoer_wagner', normalize_error=True):
        super(SmartSVM, self).__init__(binary_clf=binary_clf, 
                clf_params=clf_params, n_jobs=n_jobs)
        self.graph = graph
        self.cut_algorithm = cut_algorithm
        self.normalize_error = normalize_error

    @property
    def elements(self):
        """ The labels of this node in the hierarchy """
        if not isinstance(self.graph, nx.Graph):
            raise ValueError("graph should be a networkx Graph object.")
        return list(self.graph.nodes())

    def fit(self, X, y):
        """Fit the SmartSVM classifier

        This method fits the SmartSVM classifier on the given data. If the 
        ``graph`` attribute of the class is not defined, it will be computed 
        with :func:`.error_estimate.compute_error_graph`. If this node in the 
        hierarchy can not be split further, no classifier will be fit.  
        Otherwise, the node will be split to form a binary classification 
        problem. Next, the binary classifier will be trained on this problem.  
        Finally, the left and right children of the classifier will be trained 
        as a recursive step.

        **Important:** This method constructs the ``graph`` attribute of the 
        class if it does not exist yet. However, if the graph does exist, it is 
        not recomputed. This means that a problem can occur if you construct an 
        instance of the :class:`.SmartSVM` classifier, fit it, then fit it 
        again with different data. Namely, the graph for the first data may not 
        be appropriate for the second dataset. The solution is however simple: 
        when fitting with a different dataset, reset the graph using 
        ``clf.graph = None``.

        Parameters
        ----------
        X : numpy.ndarray, accept csr sparse
            Training data, feature matrix
        y : numpy.ndarray
            Training data, label vector

        Returns
        -------
        self : object
            Returns self.

        """

        check_classification_targets(y)
        self.classes_ = np.unique(y)

        if not self.graph:
            X, y = check_X_y(X, y, accept_sparse='csr', dtype=np.float64,
                             order="C")
            self.graph = compute_error_graph(X, y, n_jobs=self.n_jobs, 
                    normalize=self.normalize_error)

        if not self.is_splitable:
            return

        self._split()
        self._fit_binary(X, y)

        self.negative_child_.fit(X, y)
        self.positive_child_.fit(X, y)

        return self

    def _split(self):
        """Split a leaf node in the binary tree if possible

        This method splits the current node into two leaf nodes of the binary 
        tree if possible. This is done by cutting the weighted error graph and 
        initializing the children using the obtained subgraphs.

        """
        if not self.is_splitable:
            return
        G_neg, G_pos = graph_cut(self.graph, algorithm=self.cut_algorithm)

        self.negative_ = set(G_neg.nodes())
        self.positive_ = set(G_pos.nodes())

        self.negative_child_ = SmartSVM(cut_algorithm=self.cut_algorithm, 
                binary_clf=self.binary_clf, clf_params=self.clf_params, 
                n_jobs=self.n_jobs, graph=G_neg, 
                normalize_error=self.normalize_error)

        self.positive_child_ = SmartSVM(cut_algorithm=self.cut_algorithm, 
                binary_clf=self.binary_clf, clf_params=self.clf_params, 
                n_jobs=self.n_jobs, graph=G_pos, 
                normalize_error=self.normalize_error)

    def __repr__(self):
        name = self.__class__.__name__
        if self.is_splitted:
            return ("%s(negative=%r, positive=%r)" % (name, self.negative_, 
                self.positive_))
        elif self.graph is None:
            return "%s()" % (name)
        return "%s(%r)" % (name, set(self.elements))
