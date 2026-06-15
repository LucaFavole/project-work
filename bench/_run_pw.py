"""Subprocess runner for the ORIGINAL project-work solver.

    python _run_pw.py <project_work_dir> '<json config>'   ->  JSON {baseline, cost, time}
"""

import json
import os
import sys
import time


def edge_cost(graph, path, alpha, beta):
    """Edge-summed cost of a (node, gold) walk with depot resets."""
    if path[0][0] != 0:
        path = [(0, 0.0)] + path
    total = 0.0
    load = 0.0
    for (u, gold_u), (v, _gv) in zip(path, path[1:]):
        if u == 0:
            load = 0.0
        load += gold_u
        d = graph[u][v]["dist"]
        total += d + (alpha * d * load) ** beta
    return total


def main():
    target = sys.argv[1]
    cfg = json.loads(sys.argv[2])
    sys.path.insert(0, target)
    os.chdir(target)

    from Problem import Problem
    from s339239 import solution

    p = Problem(num_cities=cfg["n"], density=cfg["d"], alpha=cfg["a"], beta=cfg["b"], seed=42)
    baseline = p.baseline()

    t0 = time.perf_counter()
    path = solution(p)
    elapsed = time.perf_counter() - t0

    cost = edge_cost(p.graph, path, p.alpha, p.beta)
    print(json.dumps({"baseline": baseline, "cost": cost, "time": elapsed}))


if __name__ == "__main__":
    main()
