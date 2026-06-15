"""Precomputation of per-leg cost matrices.

For every ordered pair of nodes (u, v) we precompute two quantities along the
shortest path between them:

    dist[u, v]    = sum of edge lengths            (the static distance term)
    penalty[u, v] = sum of (edge length ** beta)   (the weight-penalty term)

With these, a leg cost collapses to O(1):

    leg(u, v, load) = dist[u, v] + (alpha * load) ** beta * penalty[u, v]

which is algebraically identical to summing `d + (alpha * d * load) ** beta`
over the leg's edges, because the load is constant along a leg. This is the same
decomposition the original solver used, and it removes the per-query shortest-
path walk that made the naive backend slow.
"""

from __future__ import annotations

import numpy as np


def build_cost_matrices(instance):
    """Return (dist_matrix, penalty_matrix) as dense n x n float arrays."""
    graph = instance.graph
    paths = instance.paths
    beta = instance.beta
    n = graph.number_of_nodes()

    # Per-edge length and length**beta, cached so each edge is touched once.
    edge_dist = {}
    edge_pen = {}
    for a, b, d in graph.edges(data="dist"):
        edge_dist[(a, b)] = edge_dist[(b, a)] = d
        p = d ** beta
        edge_pen[(a, b)] = edge_pen[(b, a)] = p

    dist = np.zeros((n, n), dtype=float)
    penalty = np.zeros((n, n), dtype=float)

    for u, targets in paths.items():
        du = dist[u]
        pu = penalty[u]
        for v, path in targets.items():
            if u == v:
                continue
            d_sum = 0.0
            p_sum = 0.0
            prev = path[0]
            for node in path[1:]:
                d_sum += edge_dist[(prev, node)]
                p_sum += edge_pen[(prev, node)]
                prev = node
            du[v] = d_sum
            pu[v] = p_sum

    return dist, penalty
