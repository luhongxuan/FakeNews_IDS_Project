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

```powershell
cd C:\FakeNews_IDS_Project\effective_models\pheme_active_quiet_calibrated_rf\training
..\..\..\venv\Scripts\python.exe .\run_paired_oof_pheme_quiet_expert_calibration.py
```

Original source: `graphsage_intervention_14/`. The retained training and feature
code now uses private helpers plus the verified protected inputs under `data/`;
the original directory remains untouched only because its exhaustive
feature-combination experiment is still running.
