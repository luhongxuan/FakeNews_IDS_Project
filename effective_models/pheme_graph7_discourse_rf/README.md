# PHEME Graph7 Discourse/Context RF

Canonical fixed-30 nested RF workflow. It selects one of three feature bundles
using inner event folds only:

1. role/author-aware base features;
2. base plus strict-past event-context features;
3. base, context, and early discourse/semantic-response features.

The historical reference result is nested eligible-event macro CRR@10 =
**0.1385**. It is preserved under `reference_result/` and is not overwritten.

## Canonical workflow

Run the short smoke first:

```powershell
cd C:\FakeNews_IDS_Project\effective_models\pheme_graph7_discourse_rf
..\..\venv\Scripts\python.exe .\training\run_smoke.py
```

After smoke and artifact validation succeed, the manual full entry point is:

```powershell
..\..\venv\Scripts\python.exe .\training\run_full.py
```

The full nested LOEO run is long-running and must be started manually. Both
entry points share `training/common.py`; smoke only uses five trees, one outer
fold, and two eligible inner folds. See `ARTIFACTS.md` for local artifact hashes
and `REMOVED_FILES.md` for cleanup provenance.
