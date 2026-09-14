# PHEME Calibrated Active/Quiet RF

- Paired global RF reduction@10: 0.1319.
- Uncalibrated two-expert reduction@10: 0.1421.
- Train-only calibrated two-expert reduction@10: **0.1461**.
- Protocol: protected v5 rumour-only artifact, strict 30 minutes, outer LOEO.
- Gate: outer-train 60th percentile of observed recency.
- Experts: separate RF utility regressors for active and quiet subsets.
- Calibration: positive-slope Ridge mappings learned from inner event-OOF
  predictions only.

The feature copies in this model are independent from the v5 model folder.
They include safe temporal/depth values, source/reply PCA, immutable observed
account age, the recency gate, and expert-score calibration. Mutable profile
and engagement counters are excluded.

Canonical smoke and full entry points share `training/common.py`:

```powershell
cd C:\FakeNews_IDS_Project\effective_models\pheme_active_quiet_calibrated_rf
..\..\venv\Scripts\python.exe .\training\run_smoke.py
..\..\venv\Scripts\python.exe .\training\run_full.py
```

Run smoke first. The full nested-calibration workflow is long-running and must be
started manually. See `ARTIFACTS.md` and `REMOVED_FILES.md` for local data and cleanup
provenance.

Original source: `graphsage_intervention_14/`. The retained training directory
now contains only the calibrated Active/Quiet model and its direct helpers.
Quiet-tier, quiet-head, change-point, stance/text, user-overlap, and policy
follow-up experiments are preserved locally under
`analysis/pheme_active_quiet_calibrated_rf/` and are excluded from Git.
