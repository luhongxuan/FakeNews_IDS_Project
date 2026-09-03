# Corrected-target active/quiet + quiet-tier hybrid

This candidate replays the original strict-30 active/quiet and quiet-tier
pipeline on the versioned corrected all-traversable-roots target artifact.
Protected datasets and feature materializations remain read-only.

The runner verifies that the protected and corrected graph artifacts differ
only in `preventable_impact` and `preventable_y`. The legacy tier outcome column
is replaced only in an in-memory derived frame. Features, event identities,
snapshot nodes and edges, gate, PCA, model families, calibration, and selection
logic are unchanged.

Run the bounded smoke first:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_active_quiet_corrected_target\training\run_corrected_quiet_tier_hybrid_smoke.py
```

After smoke and leakage checks pass, run the full 63-fold-call experiment in
the foreground:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_active_quiet_corrected_target\training\run_corrected_quiet_tier_hybrid_full.py
```

Both modes save outer scores and reusable per-thread nested inner OOF scores in
new timestamped experiment directories.

After a completed full run, produce the same-target Top-50 comparison against
the corrected active/quiet baseline, frozen legacy hybrid, and corrected v5:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_active_quiet_corrected_target\training\analyze_corrected_top50_comparison.py
```

Diagnose whether existing strict-30 protected feature families and predeclared
real-world interaction hypotheses can rescue the quiet Oracle Top-50 threads
missed by the corrected hybrid (bounded LOEO diagnostic; no full training):

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_active_quiet_corrected_target\training\analyze_quiet_false_negative_separability.py
```

Run the bounded nested quiet-rescue smoke, then the full seven-event foreground
evaluation. The rescue policy keeps the corrected hybrid Top-50 and reports
fixed additions of 5, 10, 15, and 20 quiet threads:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_active_quiet_corrected_target\training\run_quiet_rescue_expert_smoke.py
.\venv\Scripts\python.exe .\effective_models\pheme_active_quiet_corrected_target\training\run_quiet_rescue_expert_full.py
```
