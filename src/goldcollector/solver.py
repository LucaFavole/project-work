"""Genetic-algorithm solver, composed from the cost / decoder / operator pieces.

Same algorithm as the naive baseline (so results are bit-for-bit reproducible),
but now assembled from swappable components: the cost backend and operators are
injected rather than hard-coded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import GAConfig
from .cost import MatrixCost
from .decoder import decode, total_cost
from .postprocess import expand_solution, optimize_solution


@dataclass
class Solution:
    stops: list[int]
    cost: float
    walk: list[tuple[int, float]] | None = None  # full path, set when post-processed
    raw_cost: float | None = None  # GA cost before beta optimization

    @property
    def num_tours(self) -> int:
        return self.stops.count(0) - 1


class GeneticSolver:
    def __init__(self, instance, config: GAConfig | None = None, cost_model=None,
                 optimize: bool = True):
        self.instance = instance
        self.config = config or GAConfig()
        self.cost = cost_model or MatrixCost(instance)
        # Beta optimization only ever helps (or is a no-op) when beta > 1.
        self.optimize = optimize and instance.beta > 1

    def solve(self) -> Solution:
        cfg = self.config
        rng = random.Random(cfg.seed)
        gold = self.instance.gold
        cost = self.cost
        select, cross, mutate = cfg.selection, cfg.crossover, cfg.mutation

        def fitness(order):
            return total_cost(decode(order, cost, gold), cost, gold)

        population = []
        for _ in range(cfg.pop_size):
            ind = self.instance.cities[:]
            rng.shuffle(ind)
            population.append(ind)

        best_order, best_fit = None, float("inf")
        for _ in range(cfg.generations):
            scored = [(ind, fitness(ind)) for ind in population]
            scored.sort(key=lambda x: x[1])

            if scored[0][1] < best_fit:
                best_order, best_fit = scored[0][0][:], scored[0][1]

            new_population = [ind for ind, _ in scored[: cfg.elite_size]]
            while len(new_population) < cfg.pop_size:
                parent1 = select.select(rng, scored)
                parent2 = select.select(rng, scored)
                child = cross.cross(rng, parent1, parent2)
                mutate.mutate(rng, child)
                new_population.append(child)
            population = new_population

        stops = decode(best_order, cost, gold)
        if self.optimize:
            walk, opt_cost = optimize_solution(stops, self.instance)
            return Solution(stops=stops, cost=opt_cost, walk=walk, raw_cost=best_fit)
        walk = expand_solution(stops, self.instance)
        return Solution(stops=stops, cost=best_fit, walk=walk, raw_cost=best_fit)
