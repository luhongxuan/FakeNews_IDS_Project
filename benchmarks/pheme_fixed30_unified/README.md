# PHEME Unified Fixed-30 Benchmark

This benchmark compares fixed-30-minute PHEME rankers on one immutable,
rumour-only identity universe and one corrected target lineage.

Protocol v1:

- identity universe: the 2,373 rumour threads shared by both Graph7 artifacts
  and the corrected-v5 mapping;
- events in the dataset/training universe: all 9 retained PHEME events;
- primary macro evaluation: the 7 events with at least 100 candidates;
- cutoff: 30 minutes after the source timestamp;
- target: `log1p(corrected_preventable_impact)`;
- reported budgets: 1, 3, 5, 10, 20, 50, and 100;
- feature/model selection: outer-train inner validation only;
- deterministic ranking tie-break: descending score, then ascending thread ID.

The two small events (`ebola-essien`, `gurlitt`) remain part of the shared
training universe but are not included in the primary seven-event macro.
Removing them entirely would define a different 2,298-thread protocol.

Existing baseline artifacts and historical results are never overwritten.

## Canonical evaluation workflow

Models are trained in their own canonical folders. This benchmark validates the
immutable 2,373-thread contract and recomputes tables from frozen OOF scores.

```powershell
.\venv\Scripts\python.exe .\benchmarks\pheme_fixed30_unified\run.py --smoke
```

Recompute the full seven-event table:

```powershell
.\venv\Scripts\python.exe .\benchmarks\pheme_fixed30_unified\run.py
```

The authoritative original nested-training result remains under
`reference_result/`; this evaluation runner does not retrain or retune models.
