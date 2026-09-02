# PHEME Graph7 Discourse/Context RF

- Result: nested eligible-event macro CRR@10 = **0.1385**.
- Protocol: schema-locked PHEME 30-minute artifact, rumour-only evaluation,
  nested LOEO.
- Target: unchanged `preventable_y`.
- Model: RandomForestRegressor with inner selection among role base,
  strict-past event context, and early discourse bundles.

The four scripts under `features/` are private copies of this model's artifact
construction chain. The protected completed artifact and reference result are
stored locally under `artifacts/` and `reference_result/`.

```powershell
cd C:\FakeNews_IDS_Project\effective_models\pheme_graph7_discourse_rf\training
..\..\..\venv\Scripts\python.exe .\run_nested_discourse_context_selection_rf.py
```

Original source is preserved under
`research_scratch/legacy_full/graphsage_intervention_7/`.
