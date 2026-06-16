"""Seeded project-work GA runner (for parity verification). Reports cost and time."""

import json
import os
import random
import sys
import time


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

    random.seed(cfg["gaseed"])  # nothing below consumes the global RNG before run()
    t0 = time.perf_counter()
    dm, bm, _ = compute_distance_matrices(p)
    ga = GeneticAlgorithm(p, dm, bm, pop_size=cfg["pop"], generations=cfg["gen"])
    _path, cost = ga.run()
    elapsed = time.perf_counter() - t0

    print(json.dumps({"cost": cost, "time": elapsed}))


if __name__ == "__main__":
    main()
