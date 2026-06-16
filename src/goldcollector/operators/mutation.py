"""Mutation strategies."""

from __future__ import annotations

from dataclasses import dataclass


class InsertionMutation:
    """Relocate one city to a new position."""

    def mutate(self, rng, order):
        i, j = rng.sample(range(len(order)), 2)
        order.insert(j, order.pop(i))


class InversionMutation:
    """Reverse a random subsequence."""

    def mutate(self, rng, order):
        i, j = sorted(rng.sample(range(len(order)), 2))
        order[i:j] = order[i:j][::-1]


@dataclass
class ProbabilisticMutation:
    """Draw one uniform and apply the first operator whose band covers it.

    `choices` is a list of (probability, operator). With
    [(0.2, Insertion), (0.2, Inversion)] there is a 20% chance of insertion, a
    20% chance of inversion, and a 60% chance of no mutation -- in one RNG draw.
    """

    choices: list

    def mutate(self, rng, order):
        r = rng.random()
        cumulative = 0.0
        for prob, op in self.choices:
            cumulative += prob
            if r < cumulative:
                op.mutate(rng, order)
                return
