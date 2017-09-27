# -*- coding: utf-8 -*-

"""Functions for graph cutting

This module contains functions for cutting the weighted error graph. The 
default algorithm for cutting the graph is the Stoer-Wagner algorithm, but this 
module also contains code for experimenting with the Spectral Clustering 
algorithm to create graph cuts.

"""

from __future__ import division

import networkx as nx
import numpy as np

from collections import Counter

from sklearn.cluster import spectral_clustering
from scipy.sparse.csgraph import connected_components


def graph_cut(G, algorithm='stoer_wagner'):
    """Cut a connected graph into two disjoint graphs

    This is a wrapper function to make it easy to use different graph cut 
    algorithms.

    Parameters
    ----------
    G : networkx.Graph
        The graph that needs to be cut

    algorithm : str
        The graph cut algorithm to use. Available values are: 
        ``'stoer_wagner'``, ``'spectral_clustering'``, and 
        ``'normalized_cut'``.

    Returns
    -------
    G1 : networkx.Graph
        One of the subgraphs

    G2 : networkx.Graph
        The other subgraph

    Raises
    ------
    ValueError
        When an unknown graph cut algorithm is supplied.

    """
    if algorithm == 'stoer_wagner':
        return graph_cut_sw(G)
    elif algorithm == 'spectral_clustering':
        return graph_cut_sc(G, assign_labels='kmeans')
    elif algorithm == 'normalized_cut':
        return graph_cut_sc(G, assign_labels='discretize')
    else:
        raise ValueError("Unknown graph cut algorithm supplied: %s" %
                         algorithm)


def graph_cut_sw(G):
    """Apply the Stoer-Wagner graph cut

    Use the Stoer-Wagner algorithm for cutting the graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph that needs to be cut

    Returns
    -------
    G1 : networkx.Graph
        One of the subgraphs

    G2 : networkx.Graph
        The other subgraph


    """
    cut_value, partition = nx.stoer_wagner(G)
    G1 = nx.subgraph(G, partition[0])
    G2 = nx.subgraph(G, partition[1])
    return G1, G2


def graph_cut_sc(G, assign_labels=None):
    """Apply the Spectral Clustering algorithm to cut the graph

    Create two subgraphs using a Spectral Clustering of the adjacency matrix of 
    the graph.

    Important: Since the matrix of weights is a dissimilarity matrix (high 
    numbers correspond to difficult to separate classes, we turn it into a 
    similarity matrix for the Spectral Clustering algorithm by using the 
    normalized exponent of the weight matrix. This is also done in the examples 
    of Scikit-Learn for Spectral Clustering. Weights that were zero in the 
    weight matrix are set to zero in the dissimilarity matrix as well.

    Parameters
    ----------
    G : networkx.Graph
        The graph that needs to be cut

    assign_labels : str
        Parameter for the Scikit-Learn ``spectral_clustering`` function.  
        Available values are: ``kmeans`` and ``discretize``.


    Returns
    -------
    G1 : networkx.Graph
        One of the subgraphs

    G2 : networkx.Graph
        The other subgraph


    """

    nodes = np.array(G.nodes())
    weights = np.array(nx.to_numpy_matrix(G))
    if weights.std() == 0.:
        data = weights.copy()
    else:
        data = np.exp(-weights/weights.std())
        data[weights == 0] = 0

    n_components, component_idx = connected_components(data)
    if n_components == 1:
        if len(nodes) > 2:
            labels = spectral_clustering(data, n_clusters=2, 
                    assign_labels=assign_labels)
        else:
            labels = np.array([0, 1])

        partition_0 = nodes[labels == 0]
        partition_1 = nodes[labels == 1]
    elif n_components == 2:
        partition_0 = nodes[component_idx == 0]
        partition_1 = nodes[component_idx == 1]
    else:
        most_common = Counter(component_idx).most_common(1)[0][0]
        partition_0 = nodes[component_idx == most_common]
        partition_1 = nodes[component_idx != most_common]

    G1 = nx.subgraph(G, partition_0)
    G2 = nx.subgraph(G, partition_1)

    return G1, G2
