"""Seeded project-work GA runner (for parity verification)."""

import json
import os
import random
import sys


def main():
    target = sys.argv[1]
    cfg = json.loads(sys.argv[2])
    sys.path.insert(0, target)
    os.chdir(target)

    from Problem import Problem
    from src.ga import GeneticAlgorithm
    from src.utils import compute_distance_matrices

    p = Problem(num_cities=cfg["n"], density=cfg["d"], alpha=cfg["a"], beta=cfg["b"],
                seed=cfg.get("pseed", 42))
    dm, bm, _ = compute_distance_matrices(p)
    ga = GeneticAlgorithm(p, dm, bm, pop_size=cfg["pop"], generations=cfg["gen"])

    random.seed(cfg["gaseed"])  # seed the global RNG the GA uses, right before run()
    _path, cost = ga.run()
    print(json.dumps({"cost": cost}))


if __name__ == "__main__":
    main()
