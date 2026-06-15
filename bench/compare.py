"""Render a before-vs-after comparison from before.json and after.json.

    python bench/compare.py     # writes bench/COMPARISON.md and prints it
"""

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return {(r["n"], r["d"], r["alpha"], r["beta"]): r for r in json.load(f)}


def main():
    before = load("before.json")
    after = load("after.json")

    header = (
        "| n | d | α | β | baseline | legacy cost | legacy Δ | legacy t | "
        "new cost | new Δ | new t | speedup |\n"
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    rows = []
    for key in sorted(before):
        n, d, a, b = key
        lo, ne = before[key], after[key]
        speed = lo["time"] / ne["time"] if ne["time"] else float("nan")
        rows.append(
            f"| {n} | {d} | {a} | {b} | {lo['baseline']:.0f} "
            f"| {lo['cost']:.0f} | {lo['improvement']:.2f}% | {lo['time']:.1f}s "
            f"| {ne['cost']:.0f} | {ne['improvement']:.2f}% | {ne['time']:.1f}s "
            f"| {speed:.1f}× |"
        )

    md = (
        "# Before vs. after: legacy framework → goldcollector rewrite\n\n"
        "Same instances and seeds. *legacy* = the previous parallel "
        "Genetic/Merge/ILS framework; *new* = the goldcollector GA + the kept "
        "beta optimizer, with a baseline guard. Δ is improvement over the "
        "per-city baseline.\n\n" + header + "\n".join(rows) + "\n"
    )
    out = os.path.join(HERE, "COMPARISON.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
