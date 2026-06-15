# project-work-mio vs. original project-work

Same instances and seeds. *PW* = original flat solver (no beta optimizer); *mio noβopt* = this project's GA with the beta optimizer disabled; *mio βopt* = full pipeline (GA + beta optimizer). Δ is improvement over the per-city baseline; t is wall-clock solve time. The first two columns are the apples-to-apples architecture comparison (neither uses the beta optimizer); the third shows what the beta optimizer adds.

| n | d | α | β | baseline | PW: Δ / t | mio noβopt: Δ / t | mio βopt: Δ / t | PW→mio speed |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 100 | 0.2 | 1 | 1 | 25266 | 0.13% / 189.8s | -0.00% / 2.9s | -0.00% / 2.8s | 68× |
| 100 | 0.2 | 1 | 2 | 5334402 | 7.82% / 116.1s | 6.67% / 3.0s | 99.28% / 3.3s | 35× |
| 100 | 1.0 | 1 | 1 | 18266 | 0.07% / 105.8s | -0.00% / 6.2s | -0.00% / 6.3s | 17× |
| 100 | 1.0 | 1 | 2 | 5404978 | 21.42% / 180.7s | 16.87% / 6.0s | 99.45% / 9.1s | 20× |
