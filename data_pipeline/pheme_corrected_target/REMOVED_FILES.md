# Removed files

| Removed file | Previous purpose | Replacement / retained evidence |
|---|---|---|
| `training/run_corrected_v5_smoke.py` | Mixed target correction with model smoke | `run_smoke.py` validates target construction only |
| `training/run_corrected_v5_nested_full.py` | Duplicate corrected-v5 training | Canonical model is `effective_models/pheme_v5_text_rf`; historical output remains under `reference_result/` |
| `training/corrected_target_common.py` | Old location | Moved to `target_builder.py` |
