"""Merge approach runner (for comparison). Needs a main worktree. Reports cost and time."""

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
    from src.solver_framework import merge_solver

    p = Problem(num_cities=cfg["n"], density=cfg["d"], alpha=cfg["a"], beta=cfg["b"],
                seed=cfg.get("pseed", 42))

    t0 = time.perf_counter()
    _path, cost = merge_solver(p)
    elapsed = time.perf_counter() - t0

    print(json.dumps({"cost": cost, "time": elapsed}))


if __name__ == "__main__":
    main()
