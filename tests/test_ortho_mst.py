"""
    Unit tests for the ortho_mst C extension of the SmartSVM package.
"""

from __future__ import division

import unittest
import numpy as np

from smartsvm import ortho_mst


class MstTestCase(unittest.TestCase):

    def test_empty(self):
        """ ORTHO_MST: Empty matrices should be okay. """
        weights = np.ones((0, 2), dtype=np.float_)
        labels = np.array((0, ), dtype=np.int_)
        count, _ = ortho_mst.ortho_mst_count(weights, labels, 3)
        self.assertEqual(count, 0)

    def test_orthogonal_mst_1(self):
        """ ORTHO_MST: Making 3 orthogonal MST returns all edges"""
        # We need at least six nodes to make 3 orthogonal MSTs
        weights = np.array([
            [0, 1, 3, 2, 2, 3],
            [1, 0, 2, 2, 1, 3],
            [3, 2, 0, 1, 3, 2],
            [2, 2, 1, 0, 1, 3],
            [2, 1, 3, 1, 0, 1],
            [3, 3, 2, 3, 1, 0],
            ], dtype=np.float_)
        labels = np.array([-1, -1, -1, 1, 1, 1], dtype=np.int_)

        count, trees = ortho_mst.ortho_mst_count(weights, labels, 3)
        self.assertEqual(count, 9)
        self.assertEqual(trees, 3)

    def test_orthogonal_mst_2(self):
        """ ORTHO_MST: Making 3 orthogonal MST returns all edges"""
        # We need at least six nodes to make 3 orthogonal MSTs
        weights = np.array([
            [0, 1, 3, 2, 2, 3],
            [1, 0, 2, 2, 1, 3],
            [3, 2, 0, 1, 3, 2],
            [2, 2, 1, 0, 1, 3],
            [2, 1, 3, 1, 0, 1],
            [3, 3, 2, 3, 1, 0],
            ], dtype=np.float_)
        labels = np.array([-1, -1, -1, -1, 1, 1], dtype=np.int_)

        count, trees = ortho_mst.ortho_mst_count(weights, labels, 3)
        self.assertEqual(count, 8)
        self.assertEqual(trees, 3)

    def test_orthogonal_mst_3(self):
        """ ORTHO_MST: Making 3 orthogonal MST returns all edges"""
        # We need at least six nodes to make 3 orthogonal MSTs
        weights = np.array([
            [0, 1, 3, 2, 2, 3],
            [1, 0, 2, 2, 1, 3],
            [3, 2, 0, 1, 3, 2],
            [2, 2, 1, 0, 1, 3],
            [2, 1, 3, 1, 0, 1],
            [3, 3, 2, 3, 1, 0],
            ], dtype=np.float_)
        labels = np.array([-1, 1, -1, 1, -1, 1], dtype=np.int_)

        count, trees = ortho_mst.ortho_mst_count(weights, labels, 3)
        self.assertEqual(count, 9)
        self.assertEqual(trees, 3)

if __name__ == '__main__':
    unittest.main()
