/**
 * @file c_multiclass_mst_count.c
 * @author G.J.J. van den Burg
 * @date 2017-02-28
 * @brief Code for constructing the MST once and counting the connections 
 * efficiently.
 *
 * @copyright
 * Copyright (C) G.J.J. van den Burg

 This file is part of SmartSVM.

 SmartSVM is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 SmartSVM is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with SmartSVM; if not, see <http://www.gnu.org/licenses/>.

 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define Malloc(type, size) \
	mymalloc(__FILE__, __LINE__, (size)*sizeof(type))

// RowMajor order. i = row, j = column
#define matrix_set(M, cols, i, j, val) M[(i)*(cols)+j] = val
#define matrix_add(M, cols, i, j, val) M[(i)*(cols)+j] += val
#define matrix_get(M, cols, i, j) M[(i)*(cols)+j]

/**
 * @brief Wrapper for malloc() which warns when allocation fails
 *
 * @details
 * This is a wrapper function around malloc from <stdlib.h>. It tries to
 * allocate the requested memory and checks if the memory was correctly
 * allocated. If not, an error is printed to stderr, which describes the
 * file and linenumber and size failed to allocate. After this, the program
 * exits.
 *
 * @note
 * This function should not be used directly. Malloc() should be used.
 *
 * @param[in] 	file 		filename used for error printing
 * @param[in] 	line 		line number used for error printing
 * @param[in] 	size 		the size to allocate
 * @return 			the pointer to the memory allocated
 */
void *mymalloc(const char *file, int line, unsigned long size)
{
	void *ptr = malloc(size);
	if (!ptr) {
		fprintf(stderr, "Could not allocate memory: "
				"%ld bytes (%s:%d)\n", size, file, line);
		exit(EXIT_FAILURE);
	}
	return ptr;
}

/**
 * @brief Calculates a matrix of cross-connection counts in orthogonal MSTs
 *
 * @details
 * This function computes `nTrees` orthogonal MSTs on the given dataset and 
 * returns a vector of length nClass where each ith entry denotes the number 
 * of times a vertex of class i is connected to a vertex of a different class.
 *
 * This function is adapted from an implementation of Whitney's algorithm 
 * (Communications of the ACM (15) 273, April 1972) by Visar Berisha 
 * (http://www.public.asu.edu/~visar/).
 *
 * @param[in] 	weights 	pointer to weight matrix
 * @param[in] 	labels 		labels of the nodes
 * @param[in] 	nTrees 		number of orthogonal MSTs to construct
 * @param[in] 	N 		size of the weight matrix and label array
 * @param[in] 	K 		number of classes
 * @param[out] 	result_counts 	vector of cross connection counts
 * @param[out] 	n_tree_ret 	number of orthogonal MSTs constructed
 *
 */
void c_multiclass_mst_count(double *weights, long *labels, long nTrees, long N,
		long K, long *result_counts, long *n_tree_ret)
{
	bool breakout = false;
	int i, j, k, idx_add, v_add, nRest, success_tree = 0,
	    *closest = NULL,
	    *v_remain = NULL;

	long l1, l2, **tree_counts = NULL;
	double temp, e_cheap, maxWeight, *cheap_edges = NULL;

	// set the maximum weight to a very high number.
	maxWeight = 0;
	for (i=0; i<N*N; i++)
		maxWeight = (maxWeight < weights[i]) ? weights[i] : maxWeight;
	maxWeight *= 10.0;

	// malloc space
	closest = Malloc(int, N);
	v_remain = Malloc(int, N);
	cheap_edges = Malloc(double, N);
	tree_counts = Malloc(long *, nTrees);
	for (i=0; i<nTrees; i++) {
		tree_counts[i] = Malloc(long, K*K);
		for (j=0; j<K*K; j++) tree_counts[i][j] = 0;
	}
	for (i=0; i<K*K; i++) result_counts[i] = 0;

	for (k=0; k<nTrees; k++) {
		// initialize variables
		// node to be added to the tree - start with last node
		v_add = N - 1;
		// number of nodes not yet in tree - should end at nRest = 0
		nRest = N - 1;

		for (i=0; i<nRest; i++) {
			// array of nodes not yet in tree
			v_remain[i] = i;
			// node of partial tree closest to node v_remain(i)
			closest[i] = v_add;
			// edge weight from v_remain(i) to closest(i)
			cheap_edges[i] = weights[N*i + v_add];
		}

		// Form MST one node at a time, keep adding nodes until all
		// are on MST
		while (nRest > 0) {
			// update labels of nodes not yet in tree
			for (i=0; i<nRest; i++) {
				temp = weights[N*v_remain[i] + v_add];
				if (cheap_edges[i] > temp) {
					closest[i] = v_add;
					cheap_edges[i] = temp;
				}
			}

			// find node outside tree nearest to tree
			// initialize lowest edge weight e_cheap to first edge
			e_cheap = cheap_edges[0];
			idx_add = 0;
			for (i=0; i<nRest; i++) {
				if (cheap_edges[i] <= e_cheap) {
					// current best edge weight to add
					e_cheap = cheap_edges[i];
					// current best node to add
					idx_add = i;
				}
			}

			// get node to add
			v_add = v_remain[idx_add];

			// Stop when the maxWeight is used. This
			// indicates that the number of nodes is too small to
			// create the desired number of orthogonal MSTs. This 
			// can also occur in the extreme case where one of the 
			// previous MSTs uses all edges to a vertex.
			if (fabs(e_cheap - maxWeight) < 1e-10) {
				breakout = true;
				break;
			}

			// set edge weight to very high number, this is meant
			// to create orthogonality.
			matrix_set(weights, N, v_add, closest[idx_add],
				       	maxWeight);
			matrix_set(weights, N, closest[idx_add], v_add,
					maxWeight);

			// increment the result for every edge that connects
			// data with different labels.
			if (labels[v_add] != labels[closest[idx_add]]) {
				l1 = labels[v_add];
				l2 = labels[closest[idx_add]];
				matrix_add(tree_counts[k], K, l1, l2, 1);
				matrix_add(tree_counts[k], K, l2, l1, 1);
			}


			// Remove newly found MST node from array v_remain by
			// replacing it with the last node in v_remain, then
			// decrement the number of nodes not yet in the tree.
			cheap_edges[idx_add] = cheap_edges[nRest - 1];
			v_remain[idx_add] = v_remain[nRest - 1];
			closest[idx_add] = closest[nRest - 1];
			nRest--;
		}
		if (breakout)
			break;
		success_tree++;
	}

	// compute the total number of cross connections in all trees
	for (i=0; i<success_tree; i++) {
		for (j=0; j<K*K; j++) {
			result_counts[j] += tree_counts[i][j];
		}
	}

	*n_tree_ret = success_tree;

	// free memory
	free(closest);
	free(v_remain);
	free(cheap_edges);
	for (i=0; i<nTrees; i++) {
		free(tree_counts[i]);
	}
	free(tree_counts);
}
