"""Subprocess runner for THIS project (project-work-mio) solver.

    python _run_mio.py <project_work_mio_dir> '<json config>'  ->  JSON {baseline, cost, time}
"""

import json
import multiprocessing
import os
import sys
import time


def main():
    target = sys.argv[1]
    cfg = json.loads(sys.argv[2])
    sys.path.insert(0, target)
    os.chdir(target)

    from Problem import Problem
    from src.solver_framework import problem_solver

    p = Problem(num_cities=cfg["n"], density=cfg["d"], alpha=cfg["a"], beta=cfg["b"], seed=42)
    baseline = p.baseline()

    t0 = time.perf_counter()
    _path, cost = problem_solver(p, optimize=cfg.get("optimize", True))
    elapsed = time.perf_counter() - t0

    print(json.dumps({"baseline": baseline, "cost": cost, "time": elapsed}))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
