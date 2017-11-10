SmartSVM
========

SmartSVM is a Python package which implements the methods from `Fast 
Meta-Learning for Adaptive Hierarchical Classifier Design 
<https://arxiv.org/abs/1711.03512>`_ by `Gerrit J.J. van den Burg 
<https://gertjanvandenburg.com/research>`_ and `Alfred O. Hero 
<https://web.eecs.umich.edu/~hero/>`_. The package contains functions for 
estimating the Bayes error rate (BER) using the Henze-Penrose divergence and a 
hierarchical classifier called SmartSVM. See the Usage documentation below for 
more details.

Installation
============

SmartSVM is available on PyPI and can be installed easily with:

.. code:: bash

    pip install smartsvm


Usage
=====

In the paper the main focus is on the accurate Bayes error estimates and the 
hierarchical classifier SmartSVM. These will therefore be of most interest to 
users of the SmartSVM package. Below we briefly explain how to use these 
functions.

Citing
------

If you use this package in your research, please cite the paper using the 
following BibTex entry::

    @article{van2017fast,
      title={Fast Meta-Learning for Adaptive Hierarchical Classifier Design},
      author={Gerrit J.J. van den Burg and Alfred O. Hero},
      journal={arXiv preprint arXiv:1711.03512},
      archiveprefix={arXiv},
      year={2017},
      eprint={1711.03512},
      url={https://arxiv.org/abs/1711.03512},
      primaryclass={cs.LG}
    }

Bayes error estimates
---------------------

Error estimation is implemented three functions:

* ``hp_estimate`` for the Henze-Penrose estimator of the Bayes error rate.  
  This can be used as:

  .. code:: python

    >>> import numpy as np
    >>> from smartsvm import hp_estimate
    >>> X1 = np.random.multivariate_normal([-1, 0], [[1, 0], [0, 1]], 100)
    >>> X2 = np.random.multivariate_normal([1, 0], [[1, 0], [0, 1]], 100)
    >>> hp_estimate(X1, X2) # with normalization
    >>> hp_estimate(X1, X2, normalize=False) # without normalization

* ``compute_error_graph`` and ``compute_ovr_error`` respectively compute the 
  complete weighted graph of pairwise BER estimates or the One-vs-Rest BER for 
  each class. They have a similar interface:

  .. code:: python

    >>> import numpy as np
    >>> from smartsvm import compute_error_graph, compute_ovr_error
    >>> from sklearn.datasets import load_digits
    >>> digits = load_digits(5)
    >>> n_samples = len(digits.images)
    >>> X = digits.images.reshape((n_samples, -1))
    >>> y = digits.target
    >>> G = compute_error_graph(X, y, n_jobs=2, normalize=True)
    >>> d = compute_ovr_error(X, y, normalize=True)


SmartSVM Classifier
-------------------

SmartSVM is an adaptive hierarchical classifier which constructs a 
classification hierarchy based on the Henze-Penrose estimates of the Bayes 
error between each pair of classes. The classifier is build on top of 
Scikit-Learn and can be used in the exact same way as other sklearn 
classifiers:

.. code:: python

    >>> import numpy as np
    >>> from smartsvm import SmartSVM
    >>> from sklearn.datasets import load_digits
    >>> digits = load_digits(10)
    >>> n_samples = len(digits.images)
    >>> X = digits.images.reshape((n_samples, -1))
    >>> y = digits.target
    >>> clf = SmartSVM()
    >>> clf.fit(X, y)
    >>> clf.predict(X)

By default, the SmartSVM classifier uses the Linear Support Vector Machine 
(``LinearSVC``) as the underlying binary classifier for each binary subproblem 
in the hierarchy.  This can easily be changed with the ``binary_clf`` 
parameter to the class constructor, for instance:

.. code:: python

    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = SmartSVM(binary_clf=DecisionTreeClassifier)
    >>> clf.fit(X, y)
    >>> clf._get_binary()
    DecisionTreeClassifier(class_weight=None, criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

You may optionally add parameters for the classifier through the 
``clf_params`` parameter. This should be a dict with the parameters to the 
binary classifier, as follows:

.. code:: python

    >>> clf = SmartSVM(binary_clf=DecisionTreeClassifier, clf_params={'criterion': 'entropy'})
    >>> clf.fit(X, y)
    >>> clf._get_binary()
    DecisionTreeClassifier(class_weight=None, criterion='entropy',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

For more information about parameters to SmartSVM, see the API documentation 
`here <http://smartsvm.readthedocs.io/>`_.

Known Limitations
=================

The Henze-Penrose estimator of the Bayes error rate is based on construction 
of the Euclidean minimal spanning tree. The current algorithm for this in the 
SmartSVM package uses an adaptation of `Whitney's algorithm 
<https://dl.acm.org/citation.cfm?id=361299>`_. This is not the fastest way to 
construct a minimal spanning tree. The `Fast Euclidean Minimal Spanning Tree 
algorithm by March et al. <http://www.mlpack.org/papers/emst.pdf>`_, would be 
a faster option but this makes it more difficult to construct orthogonal MSTs.  
Incorporating this algorithm into the SmartSVM package is considered a topic 
for future work.

References
==========

The main reference for this package is:

* `G.J.J. van den Burg and A.O. Hero - Fast Meta-Learning for Adaptive 
  Hierarchical Classifier Design (2017) <https://arxiv.org/abs/1711.03512>`_.

The theory of the Henze-Penrose estimator is developed in:

* `V. Berisha, A. Wisler, A.O. Hero, A. Spanias - Empirically Estimable 
  Classification Bounds Based on a Nonparametric Divergence Measure (2016) 
  <http://ieeexplore.ieee.org/abstract/document/7254229/>`_.
* `V. Berisha, A.O. Hero -  Empirical Non-Parametric Estimation of the Fisher 
  Information (2015) 
  <http://ieeexplore.ieee.org/abstract/document/6975144/>`_.

