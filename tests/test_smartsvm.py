"""

    Unit tests for the SmartSVM class

"""

from __future__ import division

import math
import networkx as nx
import numpy as np
import six
import unittest

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from smartsvm import SmartSVM

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


class SmartSVMTestCase(unittest.TestCase):

    def test_init_1(self):
        """ SMARTSVM: Test init defaults """
        inst = SmartSVM()
        self.assertEqual(inst.cut_algorithm, 'stoer_wagner')
        self.assertEqual(inst.binary_clf, LinearSVC)
        self.assertEqual(inst.n_jobs, 1)
        self.assertEqual(inst.graph, None)

    def test_init_2(self):
        """ SMARTSVM: Test init supplied """
        inst = SmartSVM(cut_algorithm='other', binary_clf='test', n_jobs=4, 
                graph='graph')
        self.assertEqual(inst.cut_algorithm, 'other')
        self.assertEqual(inst.binary_clf, 'test')
        self.assertEqual(inst.n_jobs, 4)
        self.assertEqual(inst.graph, 'graph')

    def test_elements_1(self):
        """ SMARTSVM: Test elements without graph """
        inst = SmartSVM()
        with self.assertRaises(ValueError) as cm:
            x = inst.elements
            del x

        the_exception = cm.exception
        self.assertEqual(the_exception.args,
                ("graph should be a networkx Graph object.", ))

    def test_elements_2(self):
        """ SMARTSVM: Test elements with graph """
        G = nx.Graph()
        G.add_edge(1, 2, weight=1.1)
        G.add_edge(1, 3, weight=1.2)
        G.add_edge(1, 4, weight=1.3)
        G.add_edge(2, 3, weight=1.4)
        inst = SmartSVM(graph=G)
        self.assertEqual(sorted(inst.elements), [1, 2, 3, 4])

    @patch.multiple(SmartSVM, __abstractmethods__=set(), 
            is_splitable=PropertyMock(return_value=False))
    def test_fit_1(self):
        """ SMARTSVM: Test fit no graph not splitable """
        inst = SmartSVM(binary_clf=MockClassifier, normalize_error=False)
        # Uses the same data as in test_build_error_graph
        X = np.array([
            [-0.05, 3],
            [0.05, 3],
            [0, 3.00],
            [3, 0.05],
            [3, -0.05],
            [3.00, 0],
            [-0.05, -3],
            [0.05, -3],
            [0, -3.00],
            [-3, 0.05],
            [-3, -0.05],
            [-3.00, 0],
            ])
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        exp_weight = 1.0/2.0 - 1.0/4.0 * math.sqrt(1.0/3.0) - 1.0/4.0*1.0/3.0
        exp_G = nx.Graph()
        exp_G.add_edge(1, 2, weight=exp_weight)
        exp_G.add_edge(1, 3, weight=exp_weight)
        exp_G.add_edge(1, 4, weight=exp_weight)
        exp_G.add_edge(2, 3, weight=exp_weight)
        exp_G.add_edge(2, 4, weight=exp_weight)
        exp_G.add_edge(3, 4, weight=exp_weight)

        inst.fit(X, y)
        em = nx.algorithms.isomorphism.numerical_edge_match('weight', 1)
        self.assertTrue(nx.is_isomorphic(inst.graph, exp_G, edge_match=em))
        self.assertEqual(sorted(inst.graph.nodes()), sorted(exp_G.nodes()))

    def test_fit_2(self):
        """ SMARTSVM: Test fit no graph splitable """
        inst = SmartSVM(binary_clf=MockClassifier, normalize_error=False)
        X = np.array([
            [-0.05, 3],
            [0.05, 3],
            [0, 3.00],
            [3, 0.05],
            [3, -0.05],
            [3.00, 0],
            [-0.05, -3],
            [0.05, -3],
            [0, -3.00],
            ])
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        inst.fit(X, y)

        inst_y = np.array([-1, -1, -1, -1, -1, -1, 1, 1, 1])
        self.assertTrue(np.array_equal(inst.classifier_._X, X))
        # xor because labelling could be inverted
        self.assertTrue(np.array_equal(inst.classifier_._y, inst_y) ^ 
                np.array_equal(inst.classifier_._y, -1 * inst_y))

        ninst_X = np.array([
            [-0.05, 3],
            [0.05, 3],
            [0, 3.00],
            [3, 0.05],
            [3, -0.05],
            [3.00, 0],
            ])
        ninst_y = np.array([-1, -1, -1, 1, 1, 1])

        # a bit verbose because of symmetry
        if hasattr(inst.negative_child_, "classifier_"):
            self.assertTrue(np.array_equal(inst.negative_child_.classifier_._X, 
                ninst_X))
            self.assertTrue(
                    np.array_equal(
                        inst.negative_child_.classifier_._y,
                        ninst_y
                        )
                    ^
                    np.array_equal(
                        inst.negative_child_.classifier_._y,
                        -1 * ninst_y
                        )
                    )
        else:
            self.assertTrue(np.array_equal(inst.positive_child_.classifier_._X, 
                ninst_X))
            self.assertTrue(
                    np.array_equal(
                        inst.positive_child_.classifier_._y,
                        ninst_y
                        )
                    ^
                    np.array_equal(
                        inst.positive_child_.classifier_._y,
                        -1 * ninst_y
                        )
                    )

    def test_fit_predict_strings(self):
        """ SMARTSVM: Test fit and predict with string targets """
        digits = load_digits(10)
        n_samples = len(digits.images)
        X = digits.images.reshape(n_samples, -1)
        y = digits.target
        yy = np.zeros((len(y), ), dtype='a10')
        yy[y == 0] = 'red'
        yy[y == 1] = 'blue'
        yy[y == 2] = 'green'
        yy[y == 3] = 'purple'
        yy[y == 4] = 'white'
        yy[y == 5] = 'grey'
        yy[y == 6] = 'black'
        yy[y == 7] = 'yellow'
        yy[y == 8] = 'brown'
        yy[y == 9] = 'pink'

        X_train, X_test, y_train, y_test = train_test_split(X, yy, 
                test_size=0.33)
        clf = SmartSVM()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        self.assertTrue(np.array_equal(np.unique(y_train), np.unique(y_pred)))

    def test_repr(self):
        """ SMARTSVM: Test repr """
        inst = SmartSVM()
        self.assertEqual(repr(inst), "SmartSVM()")

        inst = SmartSVM()
        inst.negative_ = set([1, 2, 3])
        inst.positive_ = set([4, 5])
        if six.PY2:
            self.assertEqual(repr(inst), ("SmartSVM(negative="
                "set([1, 2, 3]), positive=set([4, 5]))"))
        else:
            self.assertEqual(repr(inst), ("SmartSVM(negative="
                "{1, 2, 3}, positive={4, 5})"))

        inst = SmartSVM()
        inst.graph = nx.Graph()
        inst.graph.add_node(1)
        if six.PY2:
            self.assertEqual(repr(inst), "SmartSVM(set([1]))")
        else:
            self.assertEqual(repr(inst), "SmartSVM({1})")

    def test_split_1(self):
        """ SMARTSVM: test split not splitable """
        graph = nx.Graph()
        graph.add_node(1)
        inst = SmartSVM(graph=graph)
        inst._split()
        self.assertFalse(hasattr(inst, "negative_"))
        self.assertFalse(hasattr(inst, "positive_"))

    def test_split_2(self):
        """ SMARTSVM: test split """
        # Uses the same data as in test_graph_cut_sw
        G = nx.Graph()
        G.add_edge('x','a', weight=3)
        G.add_edge('x','b', weight=2)
        G.add_edge('a','c', weight=3)
        G.add_edge('b','c', weight=5)
        G.add_edge('b','d', weight=4)
        G.add_edge('d','e', weight=2)
        G.add_edge('c','y', weight=2)
        G.add_edge('e','y', weight=3)
        G1 = nx.Graph()
        G1.add_edge('x','a', weight=3)
        G1.add_edge('x','b', weight=2)
        G1.add_edge('a','c', weight=3)
        G1.add_edge('b','c', weight=5)
        G1.add_edge('b','d', weight=4)
        G2 = nx.Graph()
        G2.add_edge('e','y', weight=3)

        inst = SmartSVM(graph=G)
        inst._split()

        self.assertTrue(isinstance(inst.negative_child_, SmartSVM))
        self.assertTrue(isinstance(inst.positive_child_, SmartSVM))

        em = nx.algorithms.isomorphism.numerical_edge_match('weight', 1)

        # graphs can be switched
        exp_set_1 = set(['a', 'b', 'c', 'd', 'x'])
        exp_set_2 = set(['e', 'y'])
        if inst.negative_ == exp_set_1:
            self.assertEqual(inst.negative_, exp_set_1)
            self.assertEqual(inst.positive_, exp_set_2)
            self.assertTrue(nx.is_isomorphic(inst.negative_child_.graph, G1, 
                edge_match=em))
            self.assertTrue(nx.is_isomorphic(inst.positive_child_.graph, G2, 
                edge_match=em))
        else:
            self.assertEqual(inst.positive_, exp_set_1)
            self.assertEqual(inst.negative_, exp_set_2)
            self.assertTrue(nx.is_isomorphic(inst.negative_child_.graph, G2, 
                edge_match=em))
            self.assertTrue(nx.is_isomorphic(inst.positive_child_.graph, G1, 
                edge_match=em))

if __name__ == '__main__':
    unittest.main()
