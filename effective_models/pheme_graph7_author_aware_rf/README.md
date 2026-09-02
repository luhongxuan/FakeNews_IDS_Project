# PHEME Graph7 Author-aware RF

- Result: model CRR@10 0.1192; early-size 0.1059; random 0.0390.
- Protocol: strict 30-minute schema, fixed LOEO configuration.
- Model: RandomForestRegressor over 15 activity, time, structure, author
  history/profile-summary inputs approved by the frozen schema.

This folder has its own copies of the author-aware snapshot and schema
finalization feature scripts; it does not share them with the discourse model.
The protected completed artifact and reference result are stored locally under
`artifacts/` and `reference_result/`.

```powershell
cd C:\FakeNews_IDS_Project\effective_models\pheme_graph7_author_aware_rf\training
..\..\..\venv\Scripts\python.exe .\run_fold_safe_author_aware_rf.py
```

Original source is preserved under
`research_scratch/legacy_full/graphsage_intervention_7/`.
