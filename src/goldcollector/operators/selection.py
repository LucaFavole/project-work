"""Selection strategies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TournamentSelection:
    """Pick the fittest of `k` random (genome, fitness) candidates."""

    k: int = 3

    def select(self, rng, scored):
        return min(rng.sample(scored, self.k), key=lambda x: x[1])[0]
