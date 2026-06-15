"""Cost backends.

A cost backend answers one question: *what does it cost to travel from `u` to
`v` (along the shortest path) carrying a constant `load`?* under the edge-summed
model

    leg(u, v, load) = sum over shortest-path edges of  d + (alpha * d * load) ** beta

The solver depends only on the `leg_cost` interface, so a faster backend can be
dropped in later without touching the algorithm.

`EdgeWalkCost` is the naive backend: it re-walks the shortest path every call.
Correct, but it does O(path length) work per query.
"""

from __future__ import annotations

from typing import Protocol


class CostModel(Protocol):
    def leg_cost(self, u: int, v: int, load: float) -> float: ...


class EdgeWalkCost:
    """Naive backend: recompute the leg by summing shortest-path edge costs."""

    def __init__(self, instance):
        self.graph = instance.graph
        self.alpha = instance.alpha
        self.beta = instance.beta
        self.paths = instance.paths

    def leg_cost(self, u: int, v: int, load: float) -> float:
        total = 0.0
        alpha, beta = self.alpha, self.beta
        path = self.paths[u][v]
        for x, y in zip(path, path[1:]):
            d = self.graph[x][y]["dist"]
            total += d + (alpha * d * load) ** beta
        return total


class MatrixCost:
    """Fast backend: O(1) leg cost from precomputed dist / penalty matrices.

        leg(u, v, load) = dist[u, v] + (alpha * load) ** beta * penalty[u, v]

    Matrices are stored as lists-of-lists, which give noticeably faster scalar
    indexing than numpy arrays in this Python-level hot loop.
    """

    def __init__(self, instance):
        from .preprocessing import build_cost_matrices

        self.alpha = instance.alpha
        self.beta = instance.beta
        dist, penalty = build_cost_matrices(instance)
        self.dist = dist.tolist()
        self.penalty = penalty.tolist()

    def leg_cost(self, u: int, v: int, load: float) -> float:
        return self.dist[u][v] + (self.alpha * load) ** self.beta * self.penalty[u][v]
