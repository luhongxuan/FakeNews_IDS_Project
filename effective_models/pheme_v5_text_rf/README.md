# PHEME v5 Text RF

- Result: eligible-event macro reduction@10 = **0.1467**.
- Protocol: PHEME rumour-only, strict 30 minutes, nested LOEO.
- Target: `log1p(preventable_impact)`.
- Model: RandomForestRegressor, 300 trees, depth 8, minimum leaf 3.
- Inner selection: temporal/structure baseline versus baseline + train-fold
  source/reply text PCA64.

`training/` contains the original fitting entry point and its local helper
copies. `features/` independently preserves the temporal/structure and
source/reply embedding definitions. `evaluation/` contains the intervention
metric implementation.

Protected inputs are stored under
`data/protected_research_assets/pheme_v5_strict30/` with verified SHA-256
copies of the original graph and raw observable-node files.

```powershell
cd C:\FakeNews_IDS_Project\effective_models\pheme_v5_text_rf\training
..\..\..\venv\Scripts\python.exe .\run_nested_text_impact_ranker_rf.py
```

Original source: `graphsage_intervention_5/run_nested_text_impact_ranker_rf.py`.
The retained copy no longer imports code from that legacy directory.
