"""Feasibility checking for solutions.

A walk is feasible if it starts and ends at the depot, only uses graph edges,
and collects exactly each city's gold (summed across visits, since the beta
optimization may collect a city's gold over several fractional passes).
"""

from __future__ import annotations

from collections import defaultdict

from .domain import DEPOT


class InfeasibleSolution(ValueError):
    pass


def check_feasibility(walk: list[tuple[int, float]], instance, *, tol: float = 1e-6) -> bool:
    """Return True if `walk` is feasible, else raise InfeasibleSolution."""
    if not walk or walk[0][0] != DEPOT or walk[-1][0] != DEPOT:
        raise InfeasibleSolution("walk must start and end at the depot")

    graph = instance.graph
    collected: dict[int, float] = defaultdict(float)
    for (u, _gu), (v, gv) in zip(walk, walk[1:]):
        if u != v and not graph.has_edge(u, v):
            raise InfeasibleSolution(f"no edge between {u} and {v}")
        collected[v] += gv

    for city in instance.cities:
        want = instance.gold[city]
        got = collected.get(city, 0.0)
        if abs(got - want) > tol * max(1.0, want):
            raise InfeasibleSolution(
                f"city {city}: collected {got:.6f}, expected {want:.6f}"
            )
    return True
