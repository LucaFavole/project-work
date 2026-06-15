"""Core domain types.

`Instance` is the solver-facing view of a problem: it pulls everything the GA
needs out of the assignment's `Problem` object exactly once (graph, gold,
all-pairs shortest paths, the list of collectable cities) so the hot loop never
touches the `Problem` API again.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

DEPOT = 0


@dataclass
class Instance:
    graph: nx.Graph
    alpha: float
    beta: float
    gold: dict[int, float]
    paths: dict[int, dict[int, list[int]]]
    cities: list[int]
    problem: object = None  # source Problem, used by the beta optimizer / scorer

    @classmethod
    def from_problem(cls, problem) -> "Instance":
        graph = problem.graph
        gold = nx.get_node_attributes(graph, "gold")
        paths = dict(nx.all_pairs_dijkstra_path(graph, weight="dist"))
        cities = [c for c in graph.nodes if c != DEPOT]
        return cls(
            graph=graph,
            alpha=problem.alpha,
            beta=problem.beta,
            gold=gold,
            paths=paths,
            cities=cities,
            problem=problem,
        )
