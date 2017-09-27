"""
    Unit tests for the cut module of the SmartSVM package.
"""

from __future__ import division

import unittest
import networkx as nx

from smartsvm import cut


class CutTestCase(unittest.TestCase):

    def test_graph_cut_1(self):
        """ CUT: Test main graph_cut function - default args"""
        # This function simply tests if dispatching is done correctly.
        G = nx.Graph()
        G.add_edge('x','y', weight=3)
        G1_exp = nx.Graph()
        G1_exp.add_node('x')
        G2_exp = nx.Graph()
        G2_exp.add_node('y')

        G1_res, G2_res = cut.graph_cut(G)
        self.assertTrue((sorted(G1_res.nodes()) == sorted(G1_exp.nodes())) ^
                (sorted(G1_res.nodes()) == sorted(G2_exp.nodes())))
        self.assertTrue((sorted(G2_res.nodes()) == sorted(G1_exp.nodes())) ^
                (sorted(G2_res.nodes()) == sorted(G2_exp.nodes())))

    def test_graph_cut_2(self):
        """ CUT: Test main graph_cut function - Stoer-Wagner"""
        G = nx.Graph()
        G.add_edge('x', 'y', weight=3)
        G1_exp = nx.Graph()
        G1_exp.add_node('x')
        G2_exp = nx.Graph()
        G2_exp.add_node('y')

        G1_res, G2_res = cut.graph_cut(G, algorithm='stoer_wagner')
        self.assertTrue((sorted(G1_res.nodes()) == sorted(G1_exp.nodes())) ^
                (sorted(G1_res.nodes()) == sorted(G2_exp.nodes())))
        self.assertTrue((sorted(G2_res.nodes()) == sorted(G1_exp.nodes())) ^
                (sorted(G2_res.nodes()) == sorted(G2_exp.nodes())))

    def test_graph_cut_3(self):
        """ CUT: Test main graph_cut function - Spectral Clustering (1) """
        G = nx.Graph()
        G.add_edge('x', 'y', weight=3)
        G.add_edge('x', 'z', weight=5)
        G1_exp = nx.Graph()
        G1_exp.add_node('x')
        G1_exp.add_node('y')
        G2_exp = nx.Graph()
        G2_exp.add_node('z')

        G1_res, G2_res = cut.graph_cut(G, algorithm='spectral_clustering')
        self.assertTrue((sorted(G1_res.nodes()) == sorted(G1_exp.nodes())) ^
                (sorted(G1_res.nodes()) == sorted(G2_exp.nodes())))
        self.assertTrue((sorted(G2_res.nodes()) == sorted(G1_exp.nodes())) ^
                (sorted(G2_res.nodes()) == sorted(G2_exp.nodes())))

    def test_graph_cut_4(self):
        """ CUT: Test main graph_cut function - Spectral Clustering (2) """
        G = nx.Graph()
        G.add_edge('x', 'y', weight=3)
        G.add_edge('x', 'z', weight=5)
        G1_exp = nx.Graph()
        G1_exp.add_node('x')
        G1_exp.add_node('y')
        G2_exp = nx.Graph()
        G2_exp.add_node('z')

        G1_res, G2_res = cut.graph_cut(G, algorithm='normalized_cut')
        self.assertTrue((sorted(G1_res.nodes()) == sorted(G1_exp.nodes())) ^
                (sorted(G1_res.nodes()) == sorted(G2_exp.nodes())))
        self.assertTrue((sorted(G2_res.nodes()) == sorted(G1_exp.nodes())) ^
                (sorted(G2_res.nodes()) == sorted(G2_exp.nodes())))

    def test_graph_cut_5(self):
        """ CUT: Test main graph_cut function - Unknown algorithm"""
        # This function simply tests if dispatching is done correctly.
        G = nx.Graph()
        G.add_edge('x','y', weight=3)
        with self.assertRaises(ValueError) as cm:
            cut.graph_cut(G, algorithm='test')

        the_exception = cm.exception
        self.assertEqual(the_exception.args,
                ("Unknown graph cut algorithm supplied: test",))

    def test_graph_cut_sw(self):
        """ CUT: Test Stoer-Wagner graph cut. """
        # This test verifies if the returned graphs are the expected subgraphs.
        # Uses the example of Stoer-Wagner from the networkx docs, slightly 
        # modified to make sure the cut is unique.
        G = nx.Graph()
        G.add_edge('x','a', weight=3)
        G.add_edge('x','b', weight=2)
        G.add_edge('a','c', weight=3)
        G.add_edge('b','c', weight=5)
        G.add_edge('b','d', weight=4)
        G.add_edge('d','e', weight=2)
        G.add_edge('c','y', weight=2)
        G.add_edge('e','y', weight=3)
        G1_exp = nx.Graph()
        G1_exp.add_edge('x','a', weight=3)
        G1_exp.add_edge('x','b', weight=2)
        G1_exp.add_edge('a','c', weight=3)
        G1_exp.add_edge('b','c', weight=5)
        G1_exp.add_edge('b','d', weight=4)
        G2_exp = nx.Graph()
        G2_exp.add_edge('e','y', weight=3)

        G1_res, G2_res = cut.graph_cut_sw(G)
        em = nx.algorithms.isomorphism.numerical_edge_match('weight', 1)
        self.assertTrue(nx.is_isomorphic(G1_res, G1_exp, edge_match=em) ^
                nx.is_isomorphic(G1_res, G2_exp, edge_match=em))
        self.assertTrue(nx.is_isomorphic(G2_res, G1_exp, edge_match=em) ^
                nx.is_isomorphic(G2_res, G2_exp, edge_match=em))
        self.assertTrue((sorted(G1_res.nodes()) == sorted(G1_exp.nodes())) ^
                (sorted(G1_res.nodes()) == sorted(G2_exp.nodes())))
        self.assertTrue((sorted(G2_res.nodes()) == sorted(G1_exp.nodes())) ^
                (sorted(G2_res.nodes()) == sorted(G2_exp.nodes())))


if __name__ == '__main__':
    unittest.main()
