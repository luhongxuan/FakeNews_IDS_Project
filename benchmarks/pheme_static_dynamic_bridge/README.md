# Bridge A: unified static--dynamic Budget-50 benchmark

This directory contains a new benchmark and does not replace any historical
fixed-30 or multi-checkpoint result.

## Frozen comparison contract

- Population: the 2,373 rumour threads in the fixed-30 intersection contract.
- Events: the same nine-event training universe; the same seven events with at
  least 100 threads form the primary event-macro evaluation.
- Target lineage: corrected all-roots preventable impact.
- Total intervention budget: 50 threads per event.
- Static arms: rank once at 30 minutes and select 50.
- Dynamic arms: rank at 10, 20, 30, 40, 50, and 60 minutes with quotas
  `9/9/8/8/8/8`; a thread may be selected only once.
- Common denominator: total future growth still remaining at 10 minutes in the
  event.
- Primary metric: `unified_horizon_capture_at_budget50`.

The primary table contains exact random expectation, current size, the four
fixed-30 RF arms, Balanced Cumulative RF, and Hawkes-inspired Cumulative RF.
The dynamic RFs must be re-fit after filtering to the frozen 2,373 identities;
filtering their old 2,402-thread OOF predictions would not be a strict result.

Bridge A intentionally compares policies that act at different times. It is an
end-to-end intervention comparison, not an isolated ranking-accuracy test. The
output therefore reports mean action minute alongside the common capture
metric.

## Run sequence

Short bounded replay:

```powershell
python .\benchmarks\pheme_static_dynamic_bridge\run.py --smoke
```

Full seven-event replay from immutable OOF predictions:

```powershell
python .\benchmarks\pheme_static_dynamic_bridge\run.py
```

Every run creates a new timestamped directory under `experiments/` and saves a
complete `run_record.json`. Model fitting remains inside each model folder.

## Separate native-metric tables

Historical separate-family metric experiments remain under ignored
`analysis/benchmarks/pheme_static_dynamic_bridge/`.

This writes two deliberately separate within-family evaluations:

- fixed-30 `ImpactCapture@K = sum selected PI(30) / sum candidate PI(30)`;
- dynamic checkpoint diagnostic `DIC(T) = sum newly-selected PI(T) / sum
  remaining-candidate PI(T)` before that checkpoint decision;
- dynamic overall capture `sum selected PI(T_i) / sum candidate PI(10)`, where
  the initial PI opportunity is counted exactly once.

Previously selected threads are removed from later dynamic risk sets. Both native
metrics use PI rather than mixing PI and FG. The per-checkpoint diagnostic matches T
on both sides; the overall dynamic score uses one fixed starting denominator to avoid
counting the same remaining threads six times. This run does **not** define or claim a
common fixed-versus-dynamic score. In the current materialization PI and FG happen to
be identical at every row, so the older `FG(10)` HR has the same numeric value.
