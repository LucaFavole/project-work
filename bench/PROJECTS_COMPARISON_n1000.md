# project-work-mio vs. original project-work vs. Merge approach

Same instances and seeds. **PW** = original project-work GA (gen=1000); **Merge** = the original constructive Merge heuristic + beta optimizer; **mio noβopt** = this project's GA with the beta optimizer off; **mio βopt** = this project's full pipeline. Δ = improvement over the per-city baseline; t = wall-clock solve time. mio GA budget: gen=1000, pop=100.

> **Note:** the `mio noβopt` column here was produced with the pre-fix decoder
> (commit before `09bd0f1`), which over-extended tours and trailed PW on β=2
> (30.02% vs 31.77%, 18.93% vs 20.87%). After the decoder fix, mio's GA is
> bit-identical to PW's at equal seed/budget, so those cells now match PW. Some
> PW/mio times are inflated by the machine sleeping during this long background
> run; the costs are reliable.

| n | d | α | β | baseline | PW: Δ / t | Merge: Δ / t | mio noβopt: Δ / t | mio βopt: Δ / t |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1000 | 0.2 | 1 | 1 | 195403 | 0.05% / 1986.91s | 0.09% / 2.17s | -0.00% / 773.66s | -0.00% / 773.66s |
| 1000 | 0.2 | 1 | 2 | 37545928 | 31.77% / 1081.00s | 99.39% / 7.23s | 30.02% / 882.44s | 99.35% / 882.44s |
| 1000 | 1.0 | 1 | 1 | 192936 | 0.08% / 1768.91s | 0.00% / 3.95s | 0.00% / 1825.96s | 0.00% / 1825.96s |
| 1000 | 1.0 | 1 | 2 | 57580019 | 20.87% / 3327.00s | 99.63% / 14.44s | 18.93% / 2209.74s | 99.45% / 2209.74s |
