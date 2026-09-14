# Removed files

The following legacy cumulative-pipeline files were removed only after the new
shared implementation passed syntax checks, 4/4 unit tests, and a real
frozen-artifact smoke run on 2026-09-10.

| Removed file | Previous purpose | Canonical replacement / retained provenance |
|---|---|---|
| `training/run_multicheckpoint_models_full.py` | Full cumulative and window/delta LOEO runner | `training/run_full.py` + `training/common.py` |
| `training/run_multicheckpoint_models_smoke.py` | Separate smoke implementation | `training/run_smoke.py`, which calls the same `training/common.py` path as full |
| `training/multicheckpoint_model_common.py` | Earlier model fitting and evaluation helpers | `training/common.py`, `features/cumulative_features.py`, and `evaluation/sequential_policy.py` |
| `training/materialize_multicheckpoint_dataset.py` | Built the historical multi-checkpoint feature table | The hash-locked table, schema, source-code snapshot, and provenance remain in `reference_result/20260903_160025_balanced_cumulative_sequential_policy/` |
| `training/materialize_multicheckpoint_smoke.py` | Thin smoke wrapper for the old materializer | Canonical model smoke validates the frozen materialization directly via `training/run_smoke.py` |
| `training/multicheckpoint_common.py` | Earlier materialization and snapshot feature helpers | Frozen reference input/schema plus the public, cutoff-guarded live builder in `features/cumulative_features.py`; the historical implementation remains in Git history |
| `training/test_multicheckpoint_common.py` | Tests for the retired materializer | `tests/test_pipeline.py` validates the retained model contract and sequential policy |

Hawkes-inspired files are intentionally not removed here. They remain temporarily
until the separate `pheme_hawkes_multicheckpoint_rf` folder is canonicalized and
validated, so its only surviving implementation is not lost during reorganization.
