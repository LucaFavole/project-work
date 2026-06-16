# GA parity: mio == project-work

Confirms this project's genetic algorithm reproduces the original
`project-work` GA **bit-for-bit**. Each instance runs both GAs in isolated
subprocesses with the same budget (pop=60, gen=100) and the same GA seed; a
correct rewrite must match exactly. `mio GA (βopt)` is what mio actually
delivers (same GA search + beta optimizer). Merge is shown for reference (a
different, constructive solver). Reproduce with `python bench/verify_all.py`.

| instance | project-work GA | mio GA (raw) | match | mio GA (βopt) | Merge |
|---|--:|--:|:--:|--:|--:|
| normal  n=50  d=0.5 a1 b1 | 9400.4365 | 9400.4365 | EXACT | 9400.4365 | 9399.7591 |
| normal  n=80  d=0.2 a1 b2 | 3697320.5806 | 3697320.5806 | EXACT | 28374.3587 | 27587.4263 |
| normal  n=100 d=1.0 a2 b2 | 17782755.9361 | 17782755.9361 | EXACT | 59677.0676 | 45552.5512 |
| strange tiny n=3 | 121101.5653 | 121101.5653 | EXACT | 644.9453 | 644.9453 |
| strange small n=5 | 1045.9820 | 1045.9820 | EXACT | 1045.9820 | 1045.9820 |
| strange very high beta=4 | 124091054787.29 | 124091054787.29 | EXACT | 15102.3715 | 15922.5868 |
| strange high alpha=5 | 56396660.0816 | 56396660.0816 | EXACT | 72621.0488 | 60189.4855 |
| strange ultra-sparse d=.05 | 9811793.4155 | 9811793.4155 | EXACT | 63921.3209 | 63882.1361 |
| strange alpha=3 beta=3 | 9734330499.69 | 9734330499.69 | EXACT | 38026.1386 | 35422.8754 |
| strange alpha=0 (pure dist) | 8.1009 | 8.1009 | EXACT | 8.1009 | 22.2429 |
| strange sub-linear beta=0.5 | 494.7873 | 494.7873 | EXACT | 494.7873 | 577.2021 |

**Raw GA: all EXACT** — the rewrite's GA is identical to the original's across
normal and edge-case instances.

Reading the other columns:

- **mio GA (βopt)** vs **Merge** for β > 1: same regime (both run the beta
  optimizer). Merge usually edges it out (better base routes from the
  constructive heuristic), but mio's GA route wins on some instances (e.g.
  β=4: 15102 vs 15923).
- **α=0 (no weight penalty)** and **β=0.5 (sub-linear)**: the beta optimizer is a
  no-op, so `mio GA (βopt)` equals the raw GA, and the GA's tour search beats
  Merge (8.10 vs 22.24; 494.79 vs 577.20). Merge's advantage is the beta
  optimizer, not the route construction.
