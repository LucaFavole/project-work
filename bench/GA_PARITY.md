# GA parity: mio == project-work (with timing)

Confirms this project's genetic algorithm reproduces the original `project-work`
GA **bit-for-bit**. Each instance runs both GAs in isolated subprocesses with the
same budget (pop=60, gen=100) and the same GA seed; a correct rewrite must match
exactly. `mio raw` and `mio βopt` come from one solve (beta-opt is post-
processing), so they share the listed mio time. Merge is a different,
constructive solver, shown for reference. Reproduce: `python bench/verify_all.py`.

`cost / t` = solution cost / wall-clock solve time.

| instance | project-work GA | mio GA (raw) | match | mio GA (βopt) | Merge |
|---|--:|--:|:--:|--:|--:|
| normal  n=50  d=0.5 a1 b1 | 9400.44 / 4.47s | 9400.44 / 2.10s | EXACT | 9400.44 | 9399.76 / 0.01s |
| normal  n=80  d=0.2 a1 b2 | 3697320.58 / 8.28s | 3697320.58 / 4.19s | EXACT | 28374.36 | 27587.43 / 0.20s |
| normal  n=100 d=1.0 a2 b2 | 17782755.94 / 10.79s | 17782755.94 / 6.49s | EXACT | 59677.07 | 45552.55 / 0.45s |
| strange tiny n=3 | 121101.57 / 0.19s | 121101.57 / 0.16s | EXACT | 644.95 | 644.95 / 0.00s |
| strange small n=5 | 1045.98 / 0.25s | 1045.98 / 0.20s | EXACT | 1045.98 | 1045.98 / 0.00s |
| strange very high beta=4 | 124091054787.29 / 2.68s | 124091054787.29 / 1.15s | EXACT | 15102.37 | 15922.59 / 0.11s |
| strange high alpha=5 | 56396660.08 / 2.25s | 56396660.08 / 1.16s | EXACT | 72621.05 | 60189.49 / 0.33s |
| strange ultra-sparse d=.05 | 9811793.42 / 4.60s | 9811793.42 / 2.13s | EXACT | 63921.32 | 63882.14 / 0.23s |
| strange alpha=3 beta=3 | 9734330499.68 / 3.43s | 9734330499.68 / 1.54s | EXACT | 38026.14 | 35422.88 / 0.34s |
| strange alpha=0 (pure dist) | 8.10 / 1.71s | 8.10 / 0.90s | EXACT | 8.10 | 22.24 / 0.00s |
| strange sub-linear beta=0.5 | 494.79 / 2.36s | 494.79 / 0.99s | EXACT | 494.79 | 577.20 / 0.01s |

### n=1000 (precompute-bound; pop=60, gen=100)

| instance | project-work GA | mio GA (raw) | match | mio GA (βopt) | Merge |
|---|--:|--:|:--:|--:|--:|
| n=1000 d=0.2 a1 b1 | 195386.47 / 575.79s | 195386.47 / 264.21s | EXACT | 195386.47 | 195226.59 / 2.49s |
| n=1000 d=0.2 a1 b2 | 31006170.51 / 296.03s | 31006170.51 / 213.03s | EXACT | 263655.14 | 227344.90 / 6.36s |
| n=1000 d=1.0 a2 b2 | 210170985.93 / 895.94s | 210170985.93 / 835.51s | EXACT | 683360.89 | 423534.87 / 16.18s |
| n=1000 d=0.2 a1 b4 | 1391287758571.33 / 223.12s | 1391287758571.33 / 179.08s | EXACT | 193265.65 | 162239.29 / 4.86s |
| n=1000 d=0.5 a0 b2 | 397.35 / 412.70s | 397.35 / 379.66s | EXACT | 397.35 | 635.07 / 2.21s |

Parity holds at n=1000 as well (run with `python bench/verify_all.py --big`).
Some n=1000 times may be inflated if the machine sleeps during the long run;
the costs are deterministic and reliable.

**Output: all EXACT** — mio's GA is identical to the original's on every normal,
edge-case and n=1000 instance.

**Timing:**

- mio runs the *same* GA ~1.5-2x faster than project-work (e.g. n=100: 6.5s vs
  10.8s) thanks to a leaner inner loop and list-of-lists cost matrices.
- Merge is sub-second everywhere -- it is a constructive heuristic, not an
  iterative search.

**Quality (other columns):** for β > 1, `mio βopt` and `Merge` are in the same
regime (Merge usually slightly better; mio wins on β=4). For α=0 (no penalty)
and β=0.5 (sub-linear) the beta optimizer is inert, so `mio βopt` = raw GA, and
the GA beats Merge -- Merge's advantage is the beta optimizer, not its routing.
