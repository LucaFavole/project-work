"""Command-line interface: ``python -m goldcollector``.

Generates a problem instance with the assignment's provided generator, solves
it, and prints the baseline cost, the solver cost, the improvement and the time.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import GAConfig, GeneticSolver, Instance
from .validate import check_feasibility


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="goldcollector", description=__doc__)
    p.add_argument("-n", "--num-cities", type=int, default=100)
    p.add_argument("-d", "--density", type=float, default=0.5)
    p.add_argument("-a", "--alpha", type=float, default=1.0)
    p.add_argument("-b", "--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42, help="instance generation seed")
    p.add_argument("--ga-seed", type=int, default=0, help="genetic algorithm seed")
    p.add_argument("--pop", type=int, default=100)
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--no-optimize", action="store_true", help="disable beta optimization")
    p.add_argument("--check", action="store_true", help="verify solution feasibility")
    return p


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    args = build_parser().parse_args(argv)

    try:
        from Problem import Problem
    except ImportError:
        sys.exit("could not import `Problem`; run from the repository root")

    p = Problem(num_cities=args.num_cities, density=args.density,
                alpha=args.alpha, beta=args.beta, seed=args.seed)
    baseline = p.baseline()

    t0 = time.perf_counter()
    instance = Instance.from_problem(p)
    solution = GeneticSolver(
        instance,
        GAConfig(pop_size=args.pop, generations=args.generations, seed=args.ga_seed),
        optimize=not args.no_optimize,
    ).solve()
    elapsed = time.perf_counter() - t0

    improvement = (baseline - solution.cost) / baseline * 100
    print(f"instance    : n={args.num_cities} density={args.density} "
          f"alpha={args.alpha} beta={args.beta} seed={args.seed}")
    print(f"baseline    : {baseline:16.4f}")
    print(f"solver cost : {solution.cost:16.4f}")
    if solution.raw_cost != solution.cost:
        print(f"  (raw GA   : {solution.raw_cost:14.4f}, beta-opt applied)")
    print(f"improvement : {improvement:15.2f}%")
    print(f"time        : {elapsed:15.3f}s")
    print(f"tours       : {solution.num_tours}")

    if args.check:
        check_feasibility(solution.walk, instance)
        print("feasibility : OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
