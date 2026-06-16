# project-work-mio vs. original project-work vs. Merge approach

Same instances and seeds. **PW** = original project-work GA (gen=1000); **Merge** = the original constructive Merge heuristic + beta optimizer; **mio noβopt** = this project's GA with the beta optimizer off; **mio βopt** = this project's full pipeline. Δ = improvement over the per-city baseline; t = wall-clock solve time. mio GA budget: gen=1000, pop=100.

| n | d | α | β | baseline | PW: Δ / t | Merge: Δ / t | mio noβopt: Δ / t | mio βopt: Δ / t |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 100 | 0.2 | 1 | 1 | 25266 | 0.13% / 226.27s | 0.12% / 0.11s | 0.13% / 129.35s | 0.13% / 129.35s |
| 100 | 0.2 | 1 | 2 | 5334402 | 7.79% / 213.82s | 99.29% / 0.85s | 6.99% / 114.73s | 99.28% / 114.73s |
| 100 | 1.0 | 1 | 1 | 18266 | 0.07% / 110.27s | -0.00% / 0.03s | 0.07% / 40.84s | 0.07% / 40.84s |
| 100 | 1.0 | 1 | 2 | 5404978 | 22.01% / 72.22s | 99.58% / 0.18s | 22.02% / 43.84s | 99.47% / 43.84s |
