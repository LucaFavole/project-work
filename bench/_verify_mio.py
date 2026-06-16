"""Seeded mio GA runner (for parity verification).

One solve reports both the raw GA cost (beta optimizer OFF -- this is what must
match project-work bit-for-bit) and the beta-optimized cost (what mio actually
delivers). Beta-opt is post-processing, so it shares the same GA search.
"""

import json
import os
import sys


def main():
    target = sys.argv[1]
    cfg = json.loads(sys.argv[2])
    sys.path.insert(0, target)
    os.chdir(target)

    from Problem import Problem
    from src.goldcollector import GAConfig, GeneticSolver, Instance

    p = Problem(num_cities=cfg["n"], density=cfg["d"], alpha=cfg["a"], beta=cfg["b"],
                seed=cfg.get("pseed", 42))
    instance = Instance.from_problem(p)
    sol = GeneticSolver(
        instance,
        GAConfig(pop_size=cfg["pop"], generations=cfg["gen"], seed=cfg["gaseed"]),
        optimize=True,
    ).solve()
    print(json.dumps({"raw_cost": sol.raw_cost, "opt_cost": sol.cost}))


if __name__ == "__main__":
    main()
