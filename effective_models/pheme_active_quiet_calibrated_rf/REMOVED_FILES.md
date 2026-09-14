# Removed files

This cleanup occurred only after syntax checks, 4/4 unit tests, and a real-data smoke
run completed successfully. Earlier exploratory branches remain documented in the
repository handoff and preserved under the ignored `analysis/` tree.

| Removed path | Former purpose | Why removed | Canonical replacement |
|---|---|---|---|
| `training/run_paired_oof_pheme_quiet_expert_calibration.py` | Monolithic full outer-LOEO experiment with inner OOF calibration | Did not share implementation with a smoke path and mixed orchestration, metrics, persistence, and configuration | `training/common.py`, `training/run_smoke.py`, `training/run_full.py`, `evaluation/metrics.py` |
| `training/pheme_account_age_v5_common.py` | Safe temporal/text/account-age feature construction and RF fitting | Byte-identical duplicate of the feature copy | `features/account_age_text_temporal_features.py` |
| `training/pheme_quiet_expert_common.py` | Recency gate and two-expert prediction | Byte-identical duplicate of the feature copy | `features/quiet_active_gate_features.py` |
| `training/pheme_quiet_expert_calibration_common.py` | Inner OOF Ridge calibration | Byte-identical duplicate of the feature copy | `features/expert_score_calibration.py` |
| `training/text_feature_common.py` | Snapshot source/reply embedding extraction | Duplicate helper | `features/text_feature_common.py` |
| `training/run_fold_safe_temporal_rf.py` | Older single-RF fold-safe baseline | Not the retained calibrated two-expert model and duplicated v5 structural logic | Canonical v5 model folder for the single text RF; this folder retains only Active/Quiet |
| `training/run_temporal_tabular_baseline.py` | Earlier temporal baseline reading prepared `graph.x` numeric fields | Superseded and not part of the best calibrated Active/Quiet workflow | `features/account_age_text_temporal_features.py` reconstructs allowed fields from protected raw rows |

All removed source files remain recoverable through Git history. Protected inputs,
historical reference results, and experiment outputs were not deleted.
