"""
    Unit tests for the utilities of the SmartSVM package
"""

from __future__ import division

import numpy as np
import unittest

from smartsvm import utils

class UtilsTestCase(unittest.TestCase):

    def test_indices_from_classes(self):
        """ UTILS: Test creating index vector from y and classes set """
        classes = set([1, 2, 3])
        y = np.array([1, 1, 2, 4, 5, 3, 2, 3, 5, 4, 1, 6])
        exp = np.array([True, True, True, False, False, True ,True ,True, 
            False, False, True, False])
        res = utils.indices_from_classes(classes, y)
        for e, r in zip(exp, res):
            self.assertEqual(e, r)


if __name__ == '__main__':
    unittest.main()
