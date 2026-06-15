"""Compare project-work-mio against the original project-work.

Runs both solvers (each in its own subprocess, in its own directory) on identical
instances and reports cost, improvement over baseline, and wall-clock time.

    python bench/compare_projects.py            # default 4 configs (n=100)
    python bench/compare_projects.py --configs '[[100,0.2,1,2]]'
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
PW = os.path.join(os.path.dirname(MIO), "project-work")

DEFAULT_CONFIGS = [[100, 0.2, 1, 1], [100, 0.2, 1, 2], [100, 1.0, 1, 1], [100, 1.0, 1, 2]]


def run(runner, target, cfg):
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, runner), target, json.dumps(cfg)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{runner} failed:\n{proc.stderr[-1500:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=str, default=None)
    args = ap.parse_args()
    configs = json.loads(args.configs) if args.configs else DEFAULT_CONFIGS

    if not os.path.isdir(PW):
        sys.exit(f"original project-work not found at {PW}")

    rows = []
    for n, d, a, b in configs:
        cfg = {"n": n, "d": d, "a": a, "b": b, "seed": 42}
        print(f"  n={n} d={d} α={a} β={b} ...", flush=True)
        pw = run("_run_pw.py", PW, cfg)
        mio_off = run("_run_mio.py", MIO, {**cfg, "optimize": False})
        mio_on = run("_run_mio.py", MIO, {**cfg, "optimize": True})
        base = pw["baseline"]
        same = (abs(base - mio_off["baseline"]) <= 1e-6 * max(1.0, abs(base))
                and abs(base - mio_on["baseline"]) <= 1e-6 * max(1.0, abs(base)))
        rows.append({
            "n": n, "d": d, "alpha": a, "beta": b, "baseline": base, "same_instance": same,
            "pw_cost": pw["cost"], "pw_time": pw["time"], "pw_impr": (base - pw["cost"]) / base * 100,
            "off_cost": mio_off["cost"], "off_time": mio_off["time"], "off_impr": (base - mio_off["cost"]) / base * 100,
            "on_cost": mio_on["cost"], "on_time": mio_on["time"], "on_impr": (base - mio_on["cost"]) / base * 100,
            "speedup": pw["time"] / mio_on["time"] if mio_on["time"] else float("nan"),
        })

    header = (
        "| n | d | α | β | baseline | PW: Δ / t | mio noβopt: Δ / t | mio βopt: Δ / t | PW→mio speed |\n"
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    lines = []
    for r in rows:
        flag = "" if r["same_instance"] else " ⚠"
        lines.append(
            f"| {r['n']} | {r['d']} | {r['alpha']} | {r['beta']}{flag} | {r['baseline']:.0f} "
            f"| {r['pw_impr']:.2f}% / {r['pw_time']:.1f}s "
            f"| {r['off_impr']:.2f}% / {r['off_time']:.1f}s "
            f"| {r['on_impr']:.2f}% / {r['on_time']:.1f}s "
            f"| {r['speedup']:.0f}× |"
        )
    md = (
        "# project-work-mio vs. original project-work\n\n"
        "Same instances and seeds. *PW* = original flat solver (no beta optimizer); "
        "*mio noβopt* = this project's GA with the beta optimizer disabled; "
        "*mio βopt* = full pipeline (GA + beta optimizer). Δ is improvement over the "
        "per-city baseline; t is wall-clock solve time. The first two columns are the "
        "apples-to-apples architecture comparison (neither uses the beta optimizer); "
        "the third shows what the beta optimizer adds.\n\n" + header + "\n".join(lines) + "\n"
    )
    out_md = os.path.join(HERE, "PROJECTS_COMPARISON.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(HERE, "projects_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("\n" + md)
    print(f"instance parity: {'OK' if all(r['same_instance'] for r in rows) else 'MISMATCH'}")


if __name__ == "__main__":
    main()
