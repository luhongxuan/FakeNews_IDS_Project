# PHEME v5 Text RF

Canonical strict-30 nested RF workflow. Inner event folds select between:

1. cutoff-safe temporal and structural features reconstructed from observable nodes;
2. the same baseline plus source and observed-early-reply RoBERTa embeddings reduced
   to PCA64 using training rows only.

The historical eligible-event macro Impact Capture@10 is **0.1467**. Historical
outputs remain under `reference_result/` and are never overwritten.

## Canonical workflow

```powershell
cd C:\FakeNews_IDS_Project\effective_models\pheme_v5_text_rf
..\..\venv\Scripts\python.exe .\training\run_smoke.py
```

After smoke succeeds, manually start the long-running full nested LOEO workflow:

```powershell
..\..\venv\Scripts\python.exe .\training\run_full.py
```

Both entry points share `training/common.py`. Smoke uses a four-event subset, one
outer fold, two inner folds, and five trees; it is not a research result. See
`ARTIFACTS.md` for protected input hashes and `REMOVED_FILES.md` for cleanup history.
