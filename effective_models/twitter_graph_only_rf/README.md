# Twitter15/16 Graph-only RF

- Twitter15 test CRR@10: **0.1288**; Oracle 0.2383.
- Twitter16 test CRR@10: **0.2598**; Oracle 0.4465.
- Protocol: fixed source-level train/validation/test split, strict 30 minutes.
- Inputs: 15 early graph/activity features only; no text, user profile,
  original label, mutable counts, future nodes, or future edges.
- Selection: RF depth chosen on validation, test evaluated once.

The model is self-contained. Both smoke and full entry points call the same
implementation in `training/common.py`; smoke mode changes only corpus scope,
test evaluation, and tree count.

```powershell
cd C:\FakeNews_IDS_Project
.\venv\Scripts\python.exe .\effective_models\twitter_graph_only_rf\training\run_smoke.py
```

Only after smoke succeeds, the canonical full reproduction command is:

```powershell
.\venv\Scripts\python.exe .\effective_models\twitter_graph_only_rf\training\run_full.py
```

The protected test split has already been evaluated and must not be reused for
tuning. See `ARTIFACTS.md` for local-only inputs and hashes, and
`REMOVED_FILES.md` for cleanup history.

Original source is preserved under
`research_scratch/legacy_full/graphsage_intervention_9/`.
