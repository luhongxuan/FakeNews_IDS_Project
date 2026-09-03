# PHEME multi-checkpoint cumulative and window/delta RFs

This candidate evaluates intervention rankings every 10 minutes from minute 10
through minute 60 on the exact 2,402 corrected-v5 rumour-thread allowlist.

- `cumulative`: all observations available from source time through checkpoint T.
- `window_delta`: observations in `(T-10m, T]`, the preceding 10-minute window,
  and explicit current-minus-previous changes. Static source-text summaries are
  available to both families.

Both models are pooled elapsed-time-aware Random Forest regressors. Outer
evaluation leaves out an entire event, including all checkpoints from all of
that event's threads. The target at each checkpoint is
`log1p(corrected dynamic preventable impact)`.

The benchmark evaluates a fresh ranking at each checkpoint. It does not yet
simulate removing threads that were already intervened at an earlier checkpoint.

The completed fixed-Budget-50 sequential replay does remove previously selected
threads. Its consolidated inputs, fitted all-event models, OOF results, exact
code snapshot, and provenance are stored at:

`reference_result/20260903_160025_balanced_cumulative_sequential_policy`

The selected candidate is the balanced cumulative policy with checkpoint quotas
`[9, 9, 8, 8, 8, 8]`. It achieved 0.326085 mean horizon reduction across seven
eligible held-out events; see the reference bundle README for the comparison
boundary and research-safety caveats.

## Foreground commands

From the repository root, first materialize the full derived dataset:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\materialize_multicheckpoint_dataset.py
```

After that succeeds, run the full event-separated model comparison:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_multicheckpoint_models_full.py
```

After the saved multi-checkpoint OOF run exists, validate the nested adaptive
policy with the bounded smoke runner:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_adaptive_cumulative_policy_smoke.py
```

The full nested adaptive policy run is intentionally separate because it fits
56 inner-event RF models. Run it in the foreground only after the smoke,
boundary, and leakage checks pass:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_adaptive_cumulative_policy_full.py
```

The completed nested predictions can also be reused without model retraining to
evaluate the exploratory opportunity-reserve controller. This policy limits
cumulative budget release and can require a higher predicted impact early in
the horizon:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_saved_nested_opportunity_reserve_policy.py
```

Because this controller was proposed after inspecting the first adaptive outer
replay, its PHEME result is exploratory even though each fold still performs
inner-only policy selection.

The distinct timing-signal experiment derives next-10-minute wait loss as an
outcome-only target, then trains a second RF to decide whether delaying a
utility-qualified intervention is costly. Its bounded smoke command is:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_wait_loss_hazard_policy_smoke.py
```

After the smoke, boundary, and leakage checks pass, run the full 63-fit nested
experiment in the foreground:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_wait_loss_hazard_policy_full.py
```

The model runner automatically resolves the newest complete full
materialization and records its absolute path. Incomplete or failed runs are
ignored. Both scripts print bounded progress and save complete run records in
new timestamped directories under `experiments/`.

## Quiet next-10-minute acceleration experiment

This learnability diagnostic restricts prediction to fold-gated quiet threads
and changes the outcome to a true burst-onset target:

```text
next10_signed_acceleration = next10_reply_count - previous10_reply_count
target = log1p(max(next10_signed_acceleration, 0))
```

The quiet threshold is the per-checkpoint 60th percentile of cumulative seconds
since last activity and is fitted from model-training events only. Acceleration
and all next-window fields are saved as outcomes for audit and are never model
inputs. The model receives only the 88 cutoff-safe Window/Delta features.

The derived acceleration table has already been materialized. To reproduce it:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\materialize_quiet_acceleration.py
```

Run the bounded smoke diagnostic with:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_quiet_acceleration_rf_smoke.py
```

The full event-separated experiment fits 63 pooled RF models and is therefore
left for explicit foreground execution:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_quiet_acceleration_rf_full.py
```

This experiment measures whether burst onset is learnable among currently
quiet threads. It does not by itself claim an intervention-policy improvement.

After the full acceleration OOF run, the saved-prediction qualification
diagnostic selects fold-specific score thresholds from inner OOF only. It
compares sparse macro-recall and worst-event-robust frontiers at minutes 20 and
30, intervenes on every threshold-qualified thread without a Top-K ranking, and
never intervenes on the same thread twice:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_saved_quiet_acceleration_qualification.py
```

Oracle Top-50 membership, acceleration outcomes, and future preventable impact
are used only for held-out evaluation. They are not threshold inputs.

The two-threshold follow-up joins those acceleration scores to the matching
nested cumulative future-impact scores. Inner folds choose configurations that
minimize interventions at several quiet-impact capture targets; held-out replay
requires both thresholds to pass and still performs no Top-K ranking:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_saved_quiet_dual_threshold_qualification.py
```

This is a saved-OOF diagnostic and does not refit either Random Forest.

The fixed-Budget-50 challenger sensitivity reserves predeclared slots at minutes
20 and 30 for dual-qualified quiet threads, then fills every unused slot using
the frozen strict-30 quiet-tier hybrid score:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_saved_quiet_hybrid_challenger.py
```

The older hybrid score is retained only as an event-separated frozen ranker.
Its legacy outcome column is discarded, and all policy metrics use the corrected
multi-checkpoint target. Reserve curves are descriptive; outer results do not
select a winning reserve configuration.

A separate leakage-safe timing-only diagnostic commits Top-50 identities from
the minute-20 cumulative OOF score. Dual-qualified quiet members act immediately
and all other committed identities act at minute 30:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_saved_quiet_acceleration_timing_only.py
```

The strict-30 hybrid Top-50 cannot be used to authorize a minute-20 action,
because its identity set is not observable until minute 30. The timing-only
runner therefore uses no minute-30 prediction when committing identities.
