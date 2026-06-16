"""Seeded mio GA runner (for parity verification). Reports raw cost, beta-opt cost, time.

One solve yields both the raw GA cost (beta optimizer OFF -- must match
project-work bit-for-bit) and the beta-optimized cost (what mio delivers); they
share the same GA search, hence one time.
"""

import json
import os
import sys
import time


def main():
    target = sys.argv[1]
    cfg = json.loads(sys.argv[2])
    sys.path.insert(0, target)
    os.chdir(target)

    from Problem import Problem
    from src.goldcollector import GAConfig, GeneticSolver, Instance

    p = Problem(num_cities=cfg["n"], density=cfg["d"], alpha=cfg["a"], beta=cfg["b"],
                seed=cfg.get("pseed", 42))

    t0 = time.perf_counter()
    instance = Instance.from_problem(p)
    sol = GeneticSolver(
        instance,
        GAConfig(pop_size=cfg["pop"], generations=cfg["gen"], seed=cfg["gaseed"]),
        optimize=True,
    ).solve()
    elapsed = time.perf_counter() - t0

    print(json.dumps({"raw_cost": sol.raw_cost, "opt_cost": sol.cost, "time": elapsed}))


if __name__ == "__main__":
    main()
