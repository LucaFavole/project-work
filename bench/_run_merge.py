"""Subprocess runner for the original MERGE approach (constructive heuristic).

Run against a worktree of `main`, where the merge solver still lives.

    python _run_merge.py <main_worktree_dir> '<json config>'  ->  JSON {baseline, cost, time}
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
    from src.solver_framework import merge_solver

    p = Problem(num_cities=cfg["n"], density=cfg["d"], alpha=cfg["a"], beta=cfg["b"], seed=42)
    baseline = p.baseline()

    t0 = time.perf_counter()
    _path, cost = merge_solver(p)
    elapsed = time.perf_counter() - t0

    print(json.dumps({"baseline": baseline, "cost": cost, "time": elapsed}))


if __name__ == "__main__":
    main()
