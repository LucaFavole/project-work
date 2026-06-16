"""Subprocess runner for THIS project (project-work-mio) solver.

Runs a single GA search and reports BOTH the beta-optimized and the
non-optimized cost (beta-opt is only post-processing, so one search yields
both). Optional cfg keys: "gen"/"pop" override the GA budget; otherwise the
size-scaled default is used.

    python _run_mio.py <dir> '{"n":1000,"d":0.2,"a":1,"b":2,"seed":42,"gen":1000}'
      -> JSON {baseline, noopt_cost, opt_cost, time}
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
    from src.goldcollector import GAConfig, GeneticSolver, Instance
    from src.solver_framework import _baseline_walk, _config_for

    p = Problem(num_cities=cfg["n"], density=cfg["d"], alpha=cfg["a"], beta=cfg["b"], seed=42)
    baseline = p.baseline()

    gen = cfg.get("gen")
    config = (GAConfig(pop_size=cfg.get("pop", 100), generations=gen)
              if gen else _config_for(p.graph.number_of_nodes()))

    t0 = time.perf_counter()
    instance = Instance.from_problem(p)
    sol = GeneticSolver(instance, config, optimize=True).solve()
    elapsed = time.perf_counter() - t0

    baseline_cost = p.path_cost(_baseline_walk(p))  # guard: never worse than baseline
    print(json.dumps({
        "baseline": baseline,
        "noopt_cost": min(sol.raw_cost, baseline_cost),
        "opt_cost": min(sol.cost, baseline_cost),
        "time": elapsed,
    }))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
