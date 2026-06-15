"""Reusable solver benchmark runner.

Drives whatever `problem_solver` is currently wired into
`src.solver_framework`, so the SAME script measures the legacy framework
(before) and the rewrite (after). Emits a JSON list on stdout.

Usage:
    python bench/_bench.py '[[100,0.2,1,2],[100,1.0,1,1]]'
"""

import json
import multiprocessing
import os
import sys
import time

# Make the repo root importable (Problem.py, src/) regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Problem import Problem
from src.solver_framework import problem_solver


def main():
    configs = json.loads(sys.argv[1])
    out = []
    for n, d, a, b in configs:
        p = Problem(num_cities=n, density=d, alpha=a, beta=b, seed=42)
        baseline = p.baseline()
        t0 = time.perf_counter()
        path, cost = problem_solver(p)
        elapsed = time.perf_counter() - t0
        out.append({
            "n": n, "d": d, "alpha": a, "beta": b,
            "baseline": baseline, "cost": cost, "time": elapsed,
            "improvement": (baseline - cost) / baseline * 100,
            "feasible_len": len(path),
        })
    print(json.dumps(out))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
