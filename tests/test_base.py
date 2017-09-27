"""
    Unit tests for the base classes of the Hierarchical Classifiers.

    Note that since the base classes are abstract classes, we mock the abstract 
    methods and test only the ones that remain.

"""

from __future__ import division

import math
import networkx as nx
import numpy as np
import six
import unittest

from sklearn.svm import LinearSVC
from sklearn.utils.validation import NotFittedError
from smartsvm import base

if six.PY2:
    from mock import patch, PropertyMock
else:
    from unittest.mock import patch, PropertyMock


class MockClassifier(object):

    def __init__(self):
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = X
        self._y = y
        self.coef_ = None

    def predict(self, X):
        return np.tile(1, (X.shape[0], ))

class MockHierarchicalElements(base.HierarchicalClassifier):

    def __init__(self, elements=None):
        self._elements = elements

    @property
    def elements(self):
        return self._elements


class HierarchicalClassifierTestCase(unittest.TestCase):

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set())
    def test_init_1(self):
        """ BASE: HierarchicalClassifier test init defaults """
        inst = base.HierarchicalClassifier()
        self.assertEqual(inst.binary_clf, LinearSVC)
        self.assertEqual(inst.n_jobs, 1)
        self.assertEqual(inst.clf_params, {})

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set())
    def test_init_2(self):
        """ BASE: HierarchicalClassifier test init specified """
        clf = MockClassifier()
        inst = base.HierarchicalClassifier(binary_clf=clf, n_jobs=2,
                clf_params={'C': 10.0})
        self.assertEqual(inst.binary_clf, clf)
        self.assertEqual(inst.n_jobs, 2)
        self.assertEqual(inst.clf_params, {'C': 10.0})

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            elements=PropertyMock(return_value=[10]))
    def test_is_splitable_1(self):
        """ BASE: HierarchicalClassifier test is_splitable (1) """
        inst = base.HierarchicalClassifier()
        self.assertFalse(inst.is_splitable)

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            elements=PropertyMock(return_value=[10, 10]))
    def test_is_splitable_2(self):
        """ BASE: HierarchicalClassifier test is_splitable (2) """
        inst = base.HierarchicalClassifier()
        self.assertTrue(inst.is_splitable)

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set())
    def test_is_splitted(self):
        """ BASE: HierarchicalClassifier test is_splitted """
        inst = base.HierarchicalClassifier()
        self.assertFalse(inst.is_splitted)
        inst.positive_ = None
        self.assertFalse(inst.is_splitted)
        inst.negative_ = None
        self.assertTrue(inst.is_splitted)

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            is_splitable=PropertyMock(return_value=False),
            elements=PropertyMock(return_value=[10]))
    def test_predict_1(self):
        """ BASE: HierarchicalClassifier test predict not splitable """
        inst = base.HierarchicalClassifier()
        X = np.array([[1, 1], [2, 2]])
        res = inst.predict(X)
        exp = np.array([10, 10])
        self.assertEqual(exp[0], res[0])
        self.assertEqual(exp[1], res[1])

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            is_splitable=PropertyMock(return_value=True))
    def test_predict_2(self):
        """ BASE: HierarchicalClassifier test predict empty data """
        inst = base.HierarchicalClassifier()
        X = np.array(())
        res = inst.predict(X)
        exp = np.array(())
        self.assertTrue(np.array_equal(exp, res))

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            is_splitable=PropertyMock(return_value=True))
    def test_predict_3(self):
        """ BASE: HierarchicalClassifier test predict not fitted (1) """
        inst = base.HierarchicalClassifier()
        X = np.array([[1, 1], [2, 2]])
        with self.assertRaises(NotFittedError) as cm:
            res = inst.predict(X)
            del res

        the_exception = cm.exception
        self.assertEqual(the_exception.args,
                ("This HierarchicalClassifier instance is not fitted yet. "
                    "Call 'fit' with appropriate arguments before using "
                    "this method.", ))

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            elements=PropertyMock(return_value=[10, 5]))
    @patch.multiple(MockHierarchicalElements, __abstractmethods__=set())
    def test_predict_5(self):
        """ BASE: HierarchicalClassifier test predict """
        inst = base.HierarchicalClassifier()
        X = np.array([[1, 1], [2, 2]])
        inst.classifier_ = MockClassifier()
        inst.classifier_.fit(X, None)
        inst.negative_child_ = MockHierarchicalElements(elements=[10])
        inst.positive_child_ = MockHierarchicalElements(elements=[5])
        res = inst.predict(X)
        exp = np.array([5, 5])
        self.assertTrue(np.array_equal(res, exp))

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            is_splitable=PropertyMock(return_value=False))
    def test_set_binary_1(self):
        """ BASE: HierarchicalClassifier test _set_binary not splitable """
        inst = base.HierarchicalClassifier()
        inst._set_binary()
        self.assertEqual(inst.classifier_, None)

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            is_splitable=PropertyMock(return_value=True))
    def test_set_binary_2(self):
        """ BASE: HierarchicalClassifier test _set_binary with clf"""
        inst = base.HierarchicalClassifier()
        clf = MockClassifier()
        inst._set_binary(clf=clf)
        self.assertEqual(inst.classifier_, clf)

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            is_splitable=PropertyMock(return_value=True))
    def test_set_binary_3(self):
        """ BASE: HierarchicalClassifier test _set_binary without clf (1)"""
        inst = base.HierarchicalClassifier()
        inst._set_binary()
        self.assertTrue(isinstance(inst.classifier_, LinearSVC))

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            is_splitable=PropertyMock(return_value=True))
    def test_set_binary_4(self):
        """ BASE: HierarchicalClassifier test _set_binary without clf (2)"""
        inst = base.HierarchicalClassifier(binary_clf=MockClassifier)
        inst._set_binary()
        self.assertTrue(isinstance(inst.classifier_, MockClassifier))

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            is_splitable=PropertyMock(return_value=True))
    def test_get_binary_1(self):
        """ BASE: HierarchicalClassifier test _get_binary (1) """
        inst = base.HierarchicalClassifier()
        clf = inst._get_binary()
        self.assertTrue(isinstance(clf, LinearSVC))

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            is_splitable=PropertyMock(return_value=True))
    def test_get_binary_2(self):
        """ BASE: HierarchicalClassifier test _get_binary (2) """
        inst = base.HierarchicalClassifier()
        inst.classifier_ = MockClassifier()
        clf = inst._get_binary()
        self.assertTrue(isinstance(clf, MockClassifier))

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set())
    def test_binary_data_1(self):
        """ BASE: HierarchicalClassifier test _binary_data (1)"""
        inst = base.HierarchicalClassifier()
        inst.negative_ = [4]
        inst.positive_ = [2, 3]
        X = np.array([
            [1.0, 1.0, 1.0],
            [1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0],
            [2.5, 2.5, 2.5],
            [3.0, 3.0, 3.0],
            [3.5, 3.5, 3.5],
            [4.0, 4.0, 4.0],
            [4.5, 4.5, 4.5]
            ])
        y = np.array([1, 1, 2, 2, 3, 3, 4, 4])

        res_X, res_y = inst._binary_data(X, y)
        exp_X = np.array([
            [2.0, 2.0, 2.0],
            [2.5, 2.5, 2.5],
            [3.0, 3.0, 3.0],
            [3.5, 3.5, 3.5],
            [4.0, 4.0, 4.0],
            [4.5, 4.5, 4.5]
            ])
        exp_y = np.array([1, 1, 1, 1, -1, -1])
        self.assertTrue(np.array_equal(res_X, exp_X))
        self.assertTrue(np.array_equal(res_y, exp_y))

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set())
    def test_binary_data_2(self):
        """ BASE: HierarchicalClassifier test _binary_data (2)"""
        inst = base.HierarchicalClassifier()
        inst.negative_ = [8]
        inst.positive_ = [2, 3]
        X = np.array([
            [1.0, 1.0, 1.0],
            [1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0],
            [2.5, 2.5, 2.5],
            [3.0, 3.0, 3.0],
            [3.5, 3.5, 3.5],
            [4.0, 4.0, 4.0],
            [4.5, 4.5, 4.5]
            ])
        y = np.array([1, 1, 2, 2, 3, 3, 4, 4])

        res_X, res_y = inst._binary_data(X, y)
        exp_X = np.array([
            [2.0, 2.0, 2.0],
            [2.5, 2.5, 2.5],
            [3.0, 3.0, 3.0],
            [3.5, 3.5, 3.5],
            ])
        exp_y = np.array([1, 1, 1, 1])
        self.assertTrue(np.array_equal(res_X, exp_X))
        self.assertTrue(np.array_equal(res_y, exp_y))

    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set())
    def test_binary_data_3(self):
        """ BASE: HierarchicalClassifier test _binary_data (3)"""
        inst = base.HierarchicalClassifier()
        inst.negative_ = ['orange']
        inst.positive_ = ['blue', 'green']
        X = np.array([
            [1.0, 1.0, 1.0],
            [1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0],
            [2.5, 2.5, 2.5],
            [3.0, 3.0, 3.0],
            [3.5, 3.5, 3.5],
            [4.0, 4.0, 4.0],
            [4.5, 4.5, 4.5]
            ])
        y = np.array(['red', 'red', 'blue', 'blue', 'green', 'green', 'purple', 
            'purple'])

        res_X, res_y = inst._binary_data(X, y)
        exp_X = np.array([
            [2.0, 2.0, 2.0],
            [2.5, 2.5, 2.5],
            [3.0, 3.0, 3.0],
            [3.5, 3.5, 3.5],
            ])
        exp_y = np.array([1, 1, 1, 1])
        self.assertTrue(np.array_equal(res_X, exp_X))
        self.assertTrue(np.array_equal(res_y, exp_y))


    @patch.multiple(base.HierarchicalClassifier, __abstractmethods__=set(),
            is_splitable=PropertyMock(return_value=True))
    def test_fit_binary_1(self):
        """ BASE: HierarchicalClassifier test fit_binary """
        inst = base.HierarchicalClassifier(binary_clf=MockClassifier)
        inst.negative_ = [4]
        inst.positive_ = [2, 3]
        X = np.array([
            [1.0, 1.0, 1.0],
            [1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0],
            [2.5, 2.5, 2.5],
            [3.0, 3.0, 3.0],
            [3.5, 3.5, 3.5],
            [4.0, 4.0, 4.0],
            [4.5, 4.5, 4.5]
            ])
        y = np.array([1, 1, 2, 2, 3, 3, 4, 4])

        exp_X = np.array([
            [2.0, 2.0, 2.0],
            [2.5, 2.5, 2.5],
            [3.0, 3.0, 3.0],
            [3.5, 3.5, 3.5],
            [4.0, 4.0, 4.0],
            [4.5, 4.5, 4.5]
            ])
        exp_y = np.array([1, 1, 1, 1, -1, -1])

        inst._fit_binary(X, y)
        self.assertTrue(hasattr(inst, "classifier_"))
        self.assertTrue(hasattr(inst.classifier_, "coef_"))
        self.assertTrue(np.array_equal(inst.classifier_._X, exp_X))
        self.assertTrue(np.array_equal(inst.classifier_._y, exp_y))


def mock_split(obj):
    print("spliting on obj: %r" % id(obj))
    exp_weight = 1.0/2.0 - 1.0/4.0 * math.sqrt(1.0/3.0) - 1.0/4.0*1.0/3.0
    if len(obj.elements) == 3:
        obj.negative_ = set([1, 2])
        obj.positive_ = set([3])
        G_neg = nx.Graph()
        G_neg.add_edge(1, 2, weight=exp_weight)
        G_pos = nx.Graph()
        G_pos.add_node(3)
    elif len(obj.elements) == 2:
        obj.negative_ = set([1])
        obj.positive_ = set([2])
        G_neg = nx.Graph()
        G_neg.add_node(1)
        G_pos = nx.Graph()
        G_pos.add_node(2)
    obj.negative_child_ = base.GraphClassifier(binary_clf=MockClassifier, 
            graph=G_neg)
    obj.positive_child_ = base.GraphClassifier(binary_clf=MockClassifier, 
            graph=G_pos)

if __name__ == '__main__':
    unittest.main()
