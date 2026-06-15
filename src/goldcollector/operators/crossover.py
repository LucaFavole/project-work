"""Crossover strategies."""

from __future__ import annotations


class OrderedCrossover:
    """Ordered crossover (OX): keep a slice of p1, fill the rest in p2's order."""

    def cross(self, rng, p1, p2):
        size = len(p1)
        start, end = sorted(rng.sample(range(size), 2))
        child = [None] * size
        child[start:end] = p1[start:end]
        taken = set(child[start:end])
        j = 0
        for i in range(size):
            if child[i] is None:
                while p2[j] in taken:
                    j += 1
                child[i] = p2[j]
                j += 1
        return child
