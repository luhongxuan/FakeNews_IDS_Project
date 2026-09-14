# PHEME Graph7 Author-aware RF

- Result: model CRR@10 0.1192; early-size 0.1059; random 0.0390.
- Protocol: strict 30-minute schema, fixed LOEO configuration.
- Model: RandomForestRegressor over 15 activity, time, structure, author
  history/profile-summary inputs approved by the frozen schema.

This folder has its own copies of the author-aware snapshot and schema
finalization feature scripts; it does not share them with the discourse model.
The protected completed artifact and reference result are stored locally under
`artifacts/` and `reference_result/`; exact paths and hashes are documented in
`ARTIFACTS.md`.

```powershell
cd C:\FakeNews_IDS_Project
.\venv\Scripts\python.exe .\effective_models\pheme_graph7_author_aware_rf\training\run_smoke.py
```

Only after the smoke succeeds, run the complete LOEO reproduction manually:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_graph7_author_aware_rf\training\run_full.py
```

Both entry points use `training/common.py`; smoke changes only the number of
outer folds and trees and is not a research result. See `REMOVED_FILES.md` for
the cleanup history.

Original source is preserved under
`research_scratch/legacy_full/graphsage_intervention_7/`.
