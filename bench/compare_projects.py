"""Compare project-work-mio against the original project-work and the Merge approach.

Runs each solver in its own subprocess on identical instances and reports cost,
improvement over baseline, and wall-clock time.

  * PW          - original project-work GA (gen=1000)
  * Merge       - original constructive Merge heuristic (+ beta optimizer), run
                  from a worktree of `main` at ../_merge_wt
  * mio noβopt  - this project's GA, beta optimizer disabled
  * mio βopt    - this project's full pipeline

    python bench/compare_projects.py
    python bench/compare_projects.py --configs '[[1000,0.2,1,2]]' --tag _n1000
"""

import argparse
import json
import os
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
MIO = os.path.dirname(HERE)
PARENT = os.path.dirname(MIO)
PW = os.path.join(PARENT, "project-work")
MERGE_WT = os.path.join(PARENT, "_merge_wt")

DEFAULT_CONFIGS = [[100, 0.2, 1, 1], [100, 0.2, 1, 2], [100, 1.0, 1, 1], [100, 1.0, 1, 2]]


def run(runner, target, cfg):
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, runner), target, json.dumps(cfg)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{runner} failed:\n{proc.stderr[-1500:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def cell(base, res):
    if res is None:
        return "n/a"
    return f"{(base - res['cost']) / base * 100:.2f}% / {res['time']:.2f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()
    configs = json.loads(args.configs) if args.configs else DEFAULT_CONFIGS

    if not os.path.isdir(PW):
        sys.exit(f"original project-work not found at {PW}")
    have_merge = os.path.isdir(MERGE_WT)
    if not have_merge:
        print(f"warning: merge worktree not found at {MERGE_WT}; skipping Merge column")

    rows = []
    for n, d, a, b in configs:
        cfg = {"n": n, "d": d, "a": a, "b": b, "seed": 42}
        print(f"  n={n} d={d} α={a} β={b} ...", flush=True)
        pw = run("_run_pw.py", PW, cfg)
        merge = run("_run_merge.py", MERGE_WT, cfg) if have_merge else None
        mio_off = run("_run_mio.py", MIO, {**cfg, "optimize": False})
        mio_on = run("_run_mio.py", MIO, {**cfg, "optimize": True})
        base = pw["baseline"]
        rows.append({"n": n, "d": d, "alpha": a, "beta": b, "baseline": base,
                     "pw": pw, "merge": merge, "mio_off": mio_off, "mio_on": mio_on})

    header = (
        "| n | d | α | β | baseline | PW: Δ / t | Merge: Δ / t | mio noβopt: Δ / t | mio βopt: Δ / t |\n"
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    lines = []
    for r in rows:
        base = r["baseline"]
        lines.append(
            f"| {r['n']} | {r['d']} | {r['alpha']} | {r['beta']} | {base:.0f} "
            f"| {cell(base, r['pw'])} | {cell(base, r['merge'])} "
            f"| {cell(base, r['mio_off'])} | {cell(base, r['mio_on'])} |"
        )
    md = (
        "# project-work-mio vs. original project-work vs. Merge approach\n\n"
        "Same instances and seeds. **PW** = original project-work GA (gen=1000); "
        "**Merge** = the original constructive Merge heuristic + beta optimizer; "
        "**mio noβopt** = this project's GA with the beta optimizer off; "
        "**mio βopt** = this project's full pipeline. Δ = improvement over the "
        "per-city baseline; t = wall-clock solve time.\n\n" + header + "\n".join(lines) + "\n"
    )
    out_md = os.path.join(HERE, f"PROJECTS_COMPARISON{args.tag}.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(HERE, f"projects_comparison{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("\n" + md)
    print(f"written to {out_md}")


if __name__ == "__main__":
    main()
