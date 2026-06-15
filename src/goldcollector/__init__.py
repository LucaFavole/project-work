"""Gold Collector: a genetic-algorithm solver for the weight-dependent TSP.

Public API:

    from goldcollector import Instance, GeneticSolver, GAConfig

    instance = Instance.from_problem(problem)
    solution = GeneticSolver(instance, GAConfig()).solve()
    print(solution.cost, solution.stops)
"""

from .domain import Instance, DEPOT
from .config import GAConfig
from .solver import GeneticSolver, Solution

__all__ = ["Instance", "DEPOT", "GAConfig", "GeneticSolver", "Solution"]
