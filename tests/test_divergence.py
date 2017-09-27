# -*- coding: utf-8 -*-

"""
    Unit tests for the divergence module of the SmartSVM package.
"""

from __future__ import division

import unittest
import numpy as np

from smartsvm import divergence


class MstTestCase(unittest.TestCase):

    def test_merge_and_label(self):
        """ DIVERGENCE: Test merging and labelling of datasets """
        X = np.array([
            [0, 1, 2],
            [3, 2, 1],
            ])
        Y = np.array([
            [9, 8, 7],
            [6, 5, 4],
            [0, 1, 0]
            ])
        data, labels = divergence.merge_and_label(X, Y)
        self.assertTrue(np.array_equal(data, np.array([
            [0, 1, 2],
            [3, 2, 1],
            [9, 8, 7],
            [6, 5, 4],
            [0, 1, 0]
            ])))
        self.assertTrue(np.array_equal(labels,
                        np.array([-1., -1., 1., 1., 1.])))

    def test_divergence_correction_1(self):
        """ DIVERGENCE: Testing divergence measure on small dataset. """
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
        div = divergence.divergence(X, Y)
        self.assertAlmostEqual(div, 1.0/3.0)

    def test_divergence_no_correction_1(self):
        """ DIVERGENCE: Testing divergence measure on small dataset. """
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
        div = divergence.divergence(X, Y, bias_correction=False)
        self.assertAlmostEqual(div, 1.0/3.0)




if __name__ == '__main__':
    unittest.main()
