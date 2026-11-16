# Removed files

The self-contained replacement passed syntax checks, 3/3 unit tests, and a real
paired artifact smoke before these legacy files were removed.

| Removed file | Previous purpose | Canonical replacement / retained evidence |
|---|---|---|
| `../pheme_multicheckpoint_rf/training/hawkes_intensity_candidate_common.py` | Cross-model Hawkes feature/model helpers | `features/hawkes_features.py`, `evaluation/sequential_policy.py`, `training/common.py` |
| `../pheme_multicheckpoint_rf/training/run_hawkes_intensity_feature_smoke.py` | Separate legacy smoke | `training/run_smoke.py`, sharing `training/common.py` with full |
| `../pheme_multicheckpoint_rf/training/run_hawkes_intensity_feature_full.py` | Seed-42 paired full run | `training/run_full.py` |
| `../pheme_multicheckpoint_rf/training/analyze_hawkes_intensity_stability.py` | Post-hoc event/bootstrap stability audit | Historical output preserved in `reference_result/20260904_022203_738733_hawkes_intensity_stability_audit/` |
| `../pheme_multicheckpoint_rf/training/run_hawkes_intensity_multiseed_full.py` | Five-seed paired stability run | Historical output preserved in `reference_result/20260904_023222_030707_hawkes_intensity_multiseed_full/` |
| `training/compare_dynamic_random_size_baselines.py` | One-off random/current-size comparison | Results remain documented in `RESULTS_20260904_ZH.md`; it is not part of model training/inference |
| `training/README.md` | Commands pointing to the old cross-model paths | Root `README.md` now documents canonical commands |
