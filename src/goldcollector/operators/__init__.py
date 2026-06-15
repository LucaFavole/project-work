"""Pluggable evolutionary operators."""

from .base import Crossover, Mutation, Selection
from .crossover import OrderedCrossover
from .mutation import InsertionMutation, InversionMutation, ProbabilisticMutation
from .selection import TournamentSelection

__all__ = [
    "Selection",
    "Crossover",
    "Mutation",
    "TournamentSelection",
    "OrderedCrossover",
    "InsertionMutation",
    "InversionMutation",
    "ProbabilisticMutation",
]


def default_mutation() -> ProbabilisticMutation:
    """The original mutation policy: 20% insertion, 20% inversion."""
    return ProbabilisticMutation([(0.2, InsertionMutation()), (0.2, InversionMutation())])
