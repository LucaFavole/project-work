# GA parity: mio == project-work

Confirms this project's genetic algorithm reproduces the original
`project-work` GA **bit-for-bit**. Each instance runs both GAs in isolated
subprocesses with the same budget (pop=60, gen=100) and the same GA seed; a
correct rewrite must match exactly. The Merge approach is shown for reference
(it is a different, constructive solver). Reproduce with `python bench/verify_all.py`.

| instance | project-work GA | mio GA | match | Merge |
|---|--:|--:|:--:|--:|
| normal  n=50  d=0.5 a1 b1 | 9400.4365 | 9400.4365 | EXACT | 9399.7591 |
| normal  n=80  d=0.2 a1 b2 | 3697320.5806 | 3697320.5806 | EXACT | 27587.4263 |
| normal  n=100 d=1.0 a2 b2 | 17782755.9361 | 17782755.9361 | EXACT | 45552.5512 |
| strange tiny n=3 | 121101.5653 | 121101.5653 | EXACT | 644.9453 |
| strange small n=5 | 1045.9820 | 1045.9820 | EXACT | 1045.9820 |
| strange very high beta=4 | 124091054787.29 | 124091054787.29 | EXACT | 15922.5868 |
| strange high alpha=5 | 56396660.0816 | 56396660.0816 | EXACT | 60189.4855 |
| strange ultra-sparse d=.05 | 9811793.4155 | 9811793.4155 | EXACT | 63882.1361 |
| strange alpha=3 beta=3 | 9734330499.69 | 9734330499.69 | EXACT | 35422.8754 |
| strange alpha=0 (pure dist) | 8.1009 | 8.1009 | EXACT | 22.2429 |
| strange sub-linear beta=0.5 | 494.7873 | 494.7873 | EXACT | 577.2021 |

**Result: all EXACT** — the rewrite's GA is identical to the original's across
normal and edge-case instances.

Notes on Merge: it dominates whenever the beta optimizer applies (β > 1), often
by orders of magnitude. But with no weight penalty (α=0) or a sub-linear penalty
(β=0.5) the beta optimizer is a no-op, and the GA's tour search wins — so Merge's
advantage is the beta optimizer, not the route construction per se.
