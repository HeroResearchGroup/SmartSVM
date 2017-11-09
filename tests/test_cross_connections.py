# -*- coding: utf-8 -*-

"""
"""

from __future__ import division

import unittest
import numpy as np

from smartsvm.cross_connections import (binary_cross_connections, 
        ovr_cross_connections)

class CrossConnectionsTestCase(unittest.TestCase):

    def test_binary_cross_connections_1(self):
        """ CROSS-CONNECTIONS: Test binary cross connections (1) """
        X = np.array([
            [-1, -1],
            [-2, 4],
            [-5, -5],
            [4, 1],
            [3, -3],
            [5, 4]
            ])
        y = np.array([-1, -1, -1, 1, 1, 1])

        C = binary_cross_connections(X, y, nTrees=3)
        self.assertAlmostEqual(C, 2.0)

    def test_binary_cross_connections_2(self):
        """ CROSS-CONNECTIONS: Test binary cross connections (2) """
        X = np.array([
            [-1, -1],
            [-2, 4],
            [-5, -5],
            [4, 1],
            [3, -3],
            [5, 4]
            ])
        y = np.array([-1, -1, -1, 1, 1, 1])

        C = binary_cross_connections(X, y, nTrees=4)
        self.assertAlmostEqual(C, 2.0)

    def test_ovr_cross_connections_1(self):
        """ CROSS-CONNECTIONS: Test OvR cross connections (1) """
        X = np.array([
            [-1, -1],
            [-2, 4],
            [-5, -5],
            [4, 1],
            [3, -3],
            [5, 4]
            ])
        y = np.array([0, 0, 0, 1, 1, 2])

        C = ovr_cross_connections(X, y, 3, nTrees=1)
        self.assertTrue(np.array_equal([0, 1, 0], C))

    def test_ovr_cross_connections_2(self):
        """ CROSS-CONNECTIONS: Test OvR cross connections (2) """
        X = np.array([
            [-1, -1],
            [-2, 4],
            [-5, -5],
            [4, 1],
            [3, -3],
            [5, 4]
            ])
        y = np.array([0, 0, 0, 1, 1, 2])

        C = ovr_cross_connections(X, y, 3, nTrees=4)
        self.assertTrue(np.array_equal([2, 5/3, 2/3], C))

    def test_ovr_cross_connections_error(self):
        """ CROSS-CONNNECTIONS: Test OvR cross connections error """
        X = np.array([
            [-1, -1],
            [-2, 4],
            [-5, -5],
            [4, 1],
            [3, -3],
            [5, 4]
            ])
        y = np.array([0, 0, 0, 1, 1, 2])
        y += 1

        with self.assertRaises(ValueError) as err:
            C = ovr_cross_connections(X, y, 3, nTrees=4)
            del C

        the_exception = err.exception
        self.assertEqual(the_exception.args,
                ("Class labels should be in {0, ..., nClass-1}",))



if __name__ == '__main__':
    unittest.main()
