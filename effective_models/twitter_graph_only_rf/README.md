# Twitter15/16 Graph-only RF

- Twitter15 test CRR@10: **0.1288**; Oracle 0.2383.
- Twitter16 test CRR@10: **0.2598**; Oracle 0.4465.
- Protocol: fixed source-level train/validation/test split, strict 30 minutes.
- Inputs: 15 early graph/activity features only; no text, user profile,
  original label, mutable counts, future nodes, or future edges.
- Selection: RF depth chosen on validation, test evaluated once.

`features/twitter_graph_only_ranker_common.py` is a model-private copy of the
feature list and protected loader.

```powershell
cd C:\FakeNews_IDS_Project\effective_models\twitter_graph_only_rf\training
..\..\..\venv\Scripts\python.exe .\run_twitter15_16_graph_only_ranker.py
```

Original source is preserved under
`research_scratch/legacy_full/graphsage_intervention_9/`.
