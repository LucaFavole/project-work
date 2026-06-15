r"""Beta optimization: analytical trip-splitting for super-linear penalties.

When ``beta > 1`` the cost of carrying load ``w`` over distance grows faster than
linearly, so hauling all of a tour's gold in one pass is wasteful. Splitting the
same route into ``N`` identical passes, each carrying ``1/N`` of the gold, turns
the tour's weighted cost from ``W`` into ``W * N**(1-beta)`` while multiplying the
static distance cost by ``N``. Minimising

    f(N) = N * S + alpha**beta * W * N**(1 - beta)

over ``N`` (with ``S`` the static distance sum and ``W = sum (d*w)**beta``) gives a
closed form for the optimal number of passes:

    N_opt = ( (beta - 1) * W * alpha**beta / S ) ** (1 / beta)

The actual split is delegated to the project's `src/beta_optimizer.path_optimizer`
(kept verbatim), applied per tour. This module only handles tour splitting,
expansion and scoring around it. It is a post-processing step on a finished
solution; it does not change the search. The relaxation it exploits is that a
city's gold may be collected across several passes (fractional pickups), which
the assignment's cost model permits.
"""

from __future__ import annotations

from ..beta_optimizer import path_optimizer
from .domain import DEPOT


def split_tours(stops: list[int]) -> list[list[int]]:
    """Split a depot-anchored stop sequence into individual depot->...->depot tours."""
    tours = []
    current = [DEPOT]
    for node in stops[1:]:
        current.append(node)
        if node == DEPOT:
            if len(current) > 2:  # skip empty depot->depot
                tours.append(current)
            current = [DEPOT]
    return tours


def expand_tour(tour: list[int], instance) -> list[tuple[int, float]]:
    """Expand collection stops into an adjacent-node walk annotated with gold pickups."""
    paths, gold = instance.paths, instance.gold
    walk = [(DEPOT, 0.0)]
    current = DEPOT
    for stop in tour[1:]:
        segment = paths[current][stop]
        for node in segment[1:]:
            walk.append((node, gold[stop] if node == stop else 0.0))
        current = stop
    return walk


def path_cost(walk, instance) -> float:
    """Authoritative edge-summed cost of a full (node, gold) walk with depot resets."""
    graph, alpha, beta = instance.graph, instance.alpha, instance.beta
    total = 0.0
    load = 0.0
    for (u, gold_u), (v, _gold_v) in zip(walk, walk[1:]):
        if u == DEPOT:
            load = 0.0
        load += gold_u
        d = graph[u][v]["dist"]
        total += d + (alpha * d * load) ** beta
    return total


def expand_solution(stops, instance):
    """Expand a stop sequence into a full (node, gold) walk, without splitting."""
    walk = [(DEPOT, 0.0)]
    for tour in split_tours(stops):
        walk.extend(expand_tour(tour, instance)[1:])  # skip the leading depot
    return walk


def optimize_solution(stops, instance):
    """Apply the project's per-tour beta optimizer to a solution. Returns (walk, cost)."""
    walk = [(DEPOT, 0.0)]
    for tour in split_tours(stops):
        expanded = expand_tour(tour, instance)
        optimized = path_optimizer(expanded, instance.problem)
        walk.extend(optimized[1:])  # skip the leading depot; we already end at one
    return walk, instance.problem.path_cost(walk)
