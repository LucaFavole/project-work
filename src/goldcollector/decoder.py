"""Permutation -> tour decoder.

`decode` turns a permutation of cities into a depot-anchored stop sequence using
a greedy split rule: for each city, compare the marginal cost of *extending* the
current tour against the cost of *closing* it and serving the city on a fresh
trip, and take the cheaper. `total_cost` then scores a finished stop sequence
authoritatively under the same edge-summed model.
"""

from __future__ import annotations

from .domain import DEPOT


def decode(order, cost, gold) -> list[int]:
    stops = [DEPOT]
    load = 0.0
    current = DEPOT
    for city in order:
        g = gold[city]
        # Marginal cost of extending the current tour to include `city`...
        extend = (cost.leg_cost(current, city, load)
                  + cost.leg_cost(city, DEPOT, load + g)
                  - cost.leg_cost(current, DEPOT, load))
        # ...versus serving `city` on a dedicated depot->city->depot trip.
        # (The cost of closing the current tour is common to both branches, so it
        # is excluded here to keep the comparison consistent -- matching the
        # original solver's decoder.)
        reset = (cost.leg_cost(DEPOT, city, 0.0)
                 + cost.leg_cost(city, DEPOT, g))
        if extend <= reset:
            stops.append(city)
            load += g
        else:
            stops.append(DEPOT)
            stops.append(city)
            load = g
        current = city
    stops.append(DEPOT)
    return stops


def total_cost(stops, cost, gold) -> float:
    total = 0.0
    load = 0.0
    for u, v in zip(stops, stops[1:]):
        total += cost.leg_cost(u, v, load)
        load = 0.0 if v == DEPOT else load + gold[v]
    return total
