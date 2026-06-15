"""Operator interfaces (strategy pattern).

Each evolutionary operator is an object with a single method, so alternatives
can be swapped in via configuration without touching the solver. The RNG is
always passed in to keep runs reproducible and the draw order explicit.
"""

from __future__ import annotations

from typing import Protocol


class Selection(Protocol):
    def select(self, rng, scored): ...


class Crossover(Protocol):
    def cross(self, rng, p1, p2): ...


class Mutation(Protocol):
    def mutate(self, rng, order) -> None: ...
