"""
    Unit tests for the error estimate module of the SmartSVM package.
"""

from __future__ import division

import math
import unittest

import numpy as np
import networkx as nx

from smartsvm import error_estimate


class ErrorEstimateTestCase(unittest.TestCase):

    def test_worker_1(self):
        """ ERROR_ESTIMATE: Test worker """
        # This uses the same dataset in the test_divergence code, so we know 
        # that the divergence is 1/3, so we can calculate what the error 
        # estimate should be.
        X = np.array([
            [-1, -1],
            [-2, 4],
            [-5, -5],
            [4, 1],
            [3, -3],
            [5, 4]
            ])
        y = np.array([1, 1, 1, 2, 2, 2])
        i = 1
        j = 2
        normalize = False
        Xyijn = (X, y, i, j, normalize)
        exp = (i, j, 1/2 - 1/4 * math.sqrt(1/3) - 1/4*1/3)
        res = error_estimate._worker(Xyijn)
        self.assertEqual(exp[0], res[0])
        self.assertEqual(exp[1], res[1])
        self.assertAlmostEqual(exp[2], res[2])

    def test_compute_error_graph(self):
        """ ERROR ESTIMATE: Test build error graph """
        # Test using a very simple dataset for which all pairwise errors are 
        # the same, equal to 1/2 - 1/4*sqrt(1/3) - 1/4*1/3
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
        # divide by 0.5 since we use the normalized error
        exp_weight = ((1.0/2.0 - 1.0/4.0 * math.sqrt(1.0/3.0) - 
            1.0/4.0*1.0/3.0) / 0.5)
        exp_G = nx.Graph()
        exp_G.add_edge(1, 2, weight=exp_weight)
        exp_G.add_edge(1, 3, weight=exp_weight)
        exp_G.add_edge(1, 4, weight=exp_weight)
        exp_G.add_edge(2, 3, weight=exp_weight)
        exp_G.add_edge(2, 4, weight=exp_weight)
        exp_G.add_edge(3, 4, weight=exp_weight)

        res_G_1 = error_estimate.compute_error_graph(X, y, n_jobs=1)
        em = nx.algorithms.isomorphism.numerical_edge_match('weight', 1)
        self.assertTrue(nx.is_isomorphic(res_G_1, exp_G, edge_match=em))
        self.assertEqual(sorted(res_G_1.nodes()), sorted(exp_G.nodes()))

        res_G_2 = error_estimate.compute_error_graph(X, y, n_jobs=2)
        em = nx.algorithms.isomorphism.numerical_edge_match('weight', 1)
        self.assertTrue(nx.is_isomorphic(res_G_2, exp_G, edge_match=em))
        self.assertEqual(sorted(res_G_2.nodes()), sorted(exp_G.nodes()))

    def test_compute_ovr_error_normalize(self):
        """ ERROR ESTIMATE: Test build error graph OvR (normalized)"""
        # Test using a very simple dataset. Error rates are checked using 
        # Matlab.
        X = np.array([
            [-0.05, 3],
            [ 0.05, 3],
            [ 0.00, 3],
            [3.1,  0.05],
            [3.1, -0.05],
            [3.1,  0.00],
            [-0.05, -3.3],
            [ 0.05, -3.3],
            [ 0.00, -3.3],
            [-3.7,  0.05],
            [-3.7, -0.05],
            [-3.7,  0.00],
            ])
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        N = X.shape[0]

        v1 = (1.0/2.0 - 1.0/4.0*math.sqrt(0.381966011250105) - 1.0/4.0 * 
                0.381966011250105) / min(3/N, (N-3)/N)
        v2 = (1.0/2.0 - 1.0/4.0*math.sqrt(0.388888888888889) - 1.0/4.0 * 
                0.388888888888889) / min(3/N, (N-3)/N)
        v3 = (1.0/2.0 - 1.0/4.0*math.sqrt(0.833333333333333) - 1.0/4.0 * 
                0.833333333333333) / min(3/N, (N-3)/N)
        v4 = (1.0/2.0 - 1.0/4.0*math.sqrt(0.777777777777778) - 1.0/4.0 * 
                0.777777777777778) / min(3/N, (N-3)/N)

        exp = {1: v1, 2: v2, 3: v3, 4: v4}

        res_1 = error_estimate.compute_ovr_error(X, y)
        res_2 = error_estimate.compute_ovr_error(X, y)
        for i in range(1, 5):
            self.assertAlmostEqual(exp[i], res_1[i])
            self.assertAlmostEqual(exp[i], res_2[i])

    def test_compute_ovr_error_no_normalize(self):
        """ ERROR ESTIMATE: Test build error graph OvR (not normalized)"""
        # Test using a very simple dataset. Error rates are checked using 
        # Matlab.
        X = np.array([
            [-0.05, 3],
            [ 0.05, 3],
            [ 0.00, 3],
            [3.1,  0.05],
            [3.1, -0.05],
            [3.1,  0.00],
            [-0.05, -3.3],
            [ 0.05, -3.3],
            [ 0.00, -3.3],
            [-3.7,  0.05],
            [-3.7, -0.05],
            [-3.7,  0.00],
            ])
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

        exp = {
                1: (1.0/2.0 - 1.0/4.0*math.sqrt(0.381966011250105) - 1.0/4.0 * 
                    0.381966011250105),
                2: (1.0/2.0 - 1.0/4.0*math.sqrt(0.388888888888889) - 1.0/4.0 * 
                    0.388888888888889),
                3: (1.0/2.0 - 1.0/4.0*math.sqrt(0.833333333333333) - 1.0/4.0 * 
                    0.833333333333333),
                4: (1.0/2.0 - 1.0/4.0*math.sqrt(0.777777777777778) - 1.0/4.0 * 
                    0.777777777777778)
        }

        res_1 = error_estimate.compute_ovr_error(X, y, normalize=False)
        res_2 = error_estimate.compute_ovr_error(X, y, normalize=False)
        for i in range(1, 5):
            self.assertAlmostEqual(exp[i], res_1[i])
            self.assertAlmostEqual(exp[i], res_2[i])



    def test_hp_estimate_1(self):
        """ ERROR ESTIMATE: Test error estimate even unnormalized """
        X = np.array([
            [-1, -1],
            [-2, 4],
            [-5, -5],
            ])
        Y = np.array([
            [4, 1],
            [3, -3],
            [5, 4]
            ])

        exp = 5.0/12.0 - 1.0/(4.0*math.sqrt(3.0))
        res = error_estimate.hp_estimate(X, Y, normalize=False)
        self.assertEqual(res, exp)

    def test_hp_estimate_2(self):
        """ ERROR ESTIMATE: Test error estimate uneven unnormalized """
        X = np.array([
            [-1, -1],
            [-2, 4],
            [-5, -5],
            ])
        Y = np.array([
            [4, 1],
            [3, -3],
            [5, 4],
            [3, 3]
            ])

        exp = 0.22919372518014852
        res = error_estimate.hp_estimate(X, Y, normalize=False)
        self.assertEqual(res, exp)

    def test_hp_estimate_3(self):
        """ ERROR ESTIMATE: Test error estimate even normalized """
        X = np.array([
            [-1, -1],
            [-2, 4],
            [-5, -5],
            ])
        Y = np.array([
            [4, 1],
            [3, -3],
            [5, 4],
            ])

        exp = 5.0/12.0 - 1.0/(4.0*math.sqrt(3.0))
        exp = exp * 2.0
        res = error_estimate.hp_estimate(X, Y, normalize=True)
        self.assertEqual(res, exp)

    def test_hp_estimate_4(self):
        """ ERROR ESTIMATE: Test error estimate uneven normalized """
        X = np.array([
            [-1, -1],
            [-2, 4],
            [-5, -5],
            ])
        Y = np.array([
            [4, 1],
            [3, -3],
            [5, 4],
            [3, 3]
            ])

        exp = 0.22919372518014852
        exp = exp / (3.0 / 7.0)
        res = error_estimate.hp_estimate(X, Y, normalize=True)
        self.assertEqual(res, exp)


if __name__ == '__main__':
    unittest.main()
