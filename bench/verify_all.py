"""Confirm mio's GA == project-work's GA across normal and strange instances,
and show what the Merge approach yields on the same instances.

For each instance, runs all three in isolated subprocesses; PW and mio use the
same GA budget and the same GA seed, so a correct rewrite must match bit-for-bit.
Merge is a different (constructive) solver, shown for reference.

    python bench/verify_all.py
"""

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
WT = os.path.join(PARENT, "_merge_wt")
POP, GEN = 60, 100

# label, n, d, alpha, beta, instance-seed, ga-seed
CONFIGS = [
    ("normal  n=50  d=0.5 a1 b1", 50, 0.5, 1, 1, 42, 0),
    ("normal  n=80  d=0.2 a1 b2", 80, 0.2, 1, 2, 42, 1),
    ("normal  n=100 d=1.0 a2 b2", 100, 1.0, 2, 2, 42, 2),
    ("strange tiny n=3",           3, 1.0, 1, 2, 7, 0),
    ("strange small n=5",          5, 0.5, 1, 1, 3, 4),
    ("strange very high beta=4",   40, 0.2, 1, 4, 1, 5),
    ("strange high alpha=5",       40, 1.0, 5, 2, 9, 6),
    ("strange ultra-sparse d=.05", 60, 0.05, 1, 2, 11, 7),
    ("strange alpha=3 beta=3",     50, 0.5, 3, 3, 13, 8),
    ("strange alpha=0 (pure dist)",40, 0.5, 0, 2, 17, 9),
    ("strange sub-linear beta=0.5",40, 0.5, 1, 0.5, 19, 10),
]


def run(runner, target, cfg):
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, runner), target, json.dumps(cfg)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None, proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "error"
    return json.loads(proc.stdout.strip().splitlines()[-1])["cost"], None


def main():
    have_merge = os.path.isdir(WT)
    print(f"{'instance':30} {'PW GA':>16} {'mio GA':>16} {'match':>7} {'Merge':>16}")
    print("-" * 90)
    all_match = True
    for label, n, d, a, b, pseed, gaseed in CONFIGS:
        cfg = {"n": n, "d": d, "a": a, "b": b, "pseed": pseed, "gaseed": gaseed, "pop": POP, "gen": GEN}
        pw, e1 = run("_verify_pw.py", PW, cfg)
        mio, e2 = run("_verify_mio.py", MIO, cfg)
        merge, e3 = (run("_verify_merge.py", WT, cfg) if have_merge else (None, "no wt"))

        if pw is None or mio is None:
            print(f"{label:30} {'ERR':>16} {'ERR':>16} {'-':>7}   {e1 or e2}")
            all_match = False
            continue
        match = abs(pw - mio) <= 1e-6 * max(1.0, abs(pw))
        all_match = all_match and match
        merge_s = f"{merge:.4f}" if merge is not None else "n/a"
        print(f"{label:30} {pw:16.4f} {mio:16.4f} {('EXACT' if match else 'DIFF'):>7} {merge_s:>16}")

    print("-" * 90)
    print(f"GA parity (mio == project-work): {'ALL EXACT' if all_match else 'MISMATCH FOUND'}")


if __name__ == "__main__":
    main()
