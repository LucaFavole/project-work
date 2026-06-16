"""Merge approach runner on a given instance (for comparison). Needs a main worktree."""

import json
import os
import sys


def main():
    target = sys.argv[1]
    cfg = json.loads(sys.argv[2])
    sys.path.insert(0, target)
    os.chdir(target)

    from Problem import Problem
    from src.solver_framework import merge_solver

    p = Problem(num_cities=cfg["n"], density=cfg["d"], alpha=cfg["a"], beta=cfg["b"],
                seed=cfg.get("pseed", 42))
    _path, cost = merge_solver(p)
    print(json.dumps({"cost": cost}))


if __name__ == "__main__":
    main()
