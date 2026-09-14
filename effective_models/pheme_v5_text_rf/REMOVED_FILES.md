# Removed files

Removal occurred only after the canonical replacement passed syntax checks, 4/4 unit
tests, and a real-data smoke run using the protected graph and raw-node artifacts.

| Removed path | Former purpose | Why removed | Canonical replacement |
|---|---|---|---|
| `training/run_nested_text_impact_ranker_rf.py` | Full nested LOEO selection between structural baseline and text PCA64 | Monolithic runner duplicated all loading, fitting, selection, and reporting logic and did not share a smoke path or complete run-record lifecycle | `training/common.py`, `training/run_smoke.py`, `training/run_full.py` |
| `training/run_fold_safe_temporal_rf.py` | Fold-safe structural feature reconstruction and standalone LOEO baseline | Its useful feature logic is now part of the one canonical text-model workflow; retaining a second runnable model path would make the folder ambiguous | `features/v5_features.py` and `training/common.py` |
| `training/run_temporal_tabular_baseline.py` | Earlier temporal RF baseline and metric implementation | Read numeric values from globally prepared `graph.x`; superseded by fold-safe reconstruction from protected raw node rows | `features/v5_features.py` and `evaluation/metrics.py` |
| `training/text_feature_common.py` | Source plus observed-reply centroid embedding extraction | Duplicate helper copy | `features/v5_features.py::text_matrix` |
| `features/source_reply_text_features.py` | Independent copy of the same source/reply text extraction | Duplicate definition created two potential sources of truth | `features/v5_features.py::text_matrix` |
| `features/temporal_structure_features.py` | Relocated copy of structural features plus a full runnable LOEO script | Mixed feature definition with training and duplicated `run_fold_safe_temporal_rf.py` | `features/v5_features.py` |
| `evaluation/intervention_metrics.py` | Relocated copy of the earlier baseline, training loop, and intervention metrics | Evaluation folder contained another complete model runner and used the superseded `graph.x` numeric path | `evaluation/metrics.py` |

All deleted source files remain recoverable from Git history. Historical experiment and
reference-result directories were not modified or removed.
