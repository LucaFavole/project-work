"""Unit and parity tests for the goldcollector package."""

import random

import pytest

from src.goldcollector import GAConfig, GeneticSolver, Instance
from src.goldcollector.cost import EdgeWalkCost, MatrixCost
from src.goldcollector.decoder import decode, total_cost
from src.goldcollector.postprocess import expand_solution, path_cost
from src.goldcollector.validate import InfeasibleSolution, check_feasibility
from Problem import Problem


def make_instance(beta=1, n=30, density=0.5, alpha=1, seed=42):
    return Instance.from_problem(
        Problem(num_cities=n, density=density, alpha=alpha, beta=beta, seed=seed)
    )


def solve(instance, *, seed=0, optimize=True, gens=30, pop=40):
    return GeneticSolver(
        instance, GAConfig(pop_size=pop, generations=gens, seed=seed), optimize=optimize
    ).solve()


def test_instance_has_every_city():
    inst = make_instance(n=30)
    assert len(inst.cities) == 29
    assert 0 not in inst.cities


@pytest.mark.parametrize("beta", [1, 2])
def test_cost_backends_agree(beta):
    inst = make_instance(beta=beta)
    ew, mc = EdgeWalkCost(inst), MatrixCost(inst)
    rng = random.Random(0)
    for _ in range(2000):
        u, v = rng.randrange(30), rng.randrange(30)
        if u == v:
            continue
        load = rng.random() * 4000
        a, b = ew.leg_cost(u, v, load), mc.leg_cost(u, v, load)
        assert a == pytest.approx(b, rel=1e-9)


def test_seed_is_reproducible():
    inst = make_instance(beta=2)
    assert solve(inst, seed=3).cost == solve(inst, seed=3).cost


def test_decode_matches_expanded_path_cost():
    inst = make_instance(beta=2)
    cost = MatrixCost(inst)
    order = inst.cities[:]
    random.Random(1).shuffle(order)
    stops = decode(order, cost, inst.gold)
    walk = expand_solution(stops, inst)
    assert total_cost(stops, cost, inst.gold) == pytest.approx(path_cost(walk, inst), rel=1e-9)


def test_beta_opt_is_noop_for_beta1():
    inst = make_instance(beta=1)
    sol = solve(inst, optimize=True)
    assert sol.cost == sol.raw_cost  # nothing to optimize


def test_beta_opt_improves_beta2():
    inst = make_instance(beta=2)
    sol = solve(inst, optimize=True)
    assert sol.cost < sol.raw_cost
    assert sol.cost / sol.raw_cost < 0.5  # a large, not marginal, gain


@pytest.mark.parametrize("beta,optimize", [(1, False), (2, False), (2, True)])
def test_solution_is_feasible(beta, optimize):
    inst = make_instance(beta=beta)
    sol = solve(inst, optimize=optimize)
    assert check_feasibility(sol.walk, inst)


def test_infeasible_walk_is_rejected():
    inst = make_instance(beta=1)
    bad = [(0, 0.0), (1, inst.gold[1]), (0, 0.0)]  # most cities never collected
    with pytest.raises(InfeasibleSolution):
        check_feasibility(bad, inst)
