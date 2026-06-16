"""Typed configuration for the genetic algorithm.

The config fully describes a run, including which evolutionary operators to use,
so experiments are reproducible from a single object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .operators import (
    Crossover,
    Mutation,
    OrderedCrossover,
    Selection,
    TournamentSelection,
    default_mutation,
)


@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 1000
    elite_frac: float = 0.05
    seed: int = 0

    selection: Selection = field(default_factory=lambda: TournamentSelection(k=3))
    crossover: Crossover = field(default_factory=OrderedCrossover)
    mutation: Mutation = field(default_factory=default_mutation)

    @property
    def elite_size(self) -> int:
        return max(2, int(self.pop_size * self.elite_frac))
