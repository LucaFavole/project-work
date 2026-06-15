# Before vs. after: legacy framework → goldcollector rewrite

Same instances and seeds. *legacy* = the previous parallel Genetic/Merge/ILS framework; *new* = the goldcollector GA + the kept beta optimizer, with a baseline guard. Δ is improvement over the per-city baseline.

| n | d | α | β | baseline | legacy cost | legacy Δ | legacy t | new cost | new Δ | new t | speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 100 | 0.2 | 1 | 1 | 25266 | 25235 | 0.12% | 49.0s | 25266 | -0.00% | 2.8s | 17.5× |
| 100 | 0.2 | 1 | 2 | 5334402 | 37802 | 99.29% | 44.0s | 38307 | 99.28% | 3.1s | 14.1× |
| 100 | 1.0 | 1 | 1 | 18266 | 18262 | 0.02% | 52.0s | 18266 | -0.00% | 3.2s | 16.3× |
| 100 | 1.0 | 1 | 2 | 5404978 | 22776 | 99.58% | 46.2s | 29848 | 99.45% | 3.4s | 13.7× |
