# PHEME corrected preventable-impact target lineage

This is a deterministic data/label pipeline, not a model. It traverses every
independently reachable root and preserves the protected strict-30 assets.

```powershell
.\venv\Scripts\python.exe .\data_pipeline\pheme_corrected_target\run_smoke.py
```

Model training belongs to canonical folders under `effective_models/`.
Historical corrected-v5 output remains under `reference_result/` as provenance
and must not silently replace the current v5 baseline.
