# bench/

Before/after comparison for replacing the legacy multi-solver framework with the
`src/goldcollector` rewrite (which keeps `src/beta_optimizer.py`).

```bash
# run the currently-wired solver over a config set -> JSON
python bench/_bench.py '[[100,0.2,1,1],[100,0.2,1,2],[100,1.0,1,1],[100,1.0,1,2]]' > bench/after.json
# render before.json vs after.json
python bench/compare.py
```

`before.json` was captured from the legacy framework before it was removed;
`after.json` from the rewrite. See `COMPARISON.md` for the rendered table.

## Summary

- **Speed:** the rewrite is ~14–17× faster (single GA + beta optimizer vs the
  parallel Genetic/Merge/ILS ensemble).
- **β = 2:** matched on the sparse graph (99.28% vs 99.29%); slightly behind on
  the dense graph (99.45% vs 99.58%), where the removed Merge heuristic had an
  edge.
- **β = 1:** the baseline guard keeps the solver from ever finishing worse than
  the per-city baseline (it lands exactly at baseline here).
