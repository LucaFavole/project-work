"""Seeded mio GA runner, beta optimizer off (for parity verification)."""

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
        optimize=False,
    ).solve()
    print(json.dumps({"cost": sol.cost}))


if __name__ == "__main__":
    main()
