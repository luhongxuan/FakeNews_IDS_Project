# PHEME Balanced Cumulative Multi-checkpoint RF

The canonical model evaluates cumulative intervention utility every 10 minutes from
minute 10 through minute 60 on the exact 2,402 corrected-v5 rumour-thread allowlist.

- `cumulative`: all observations available from source time through checkpoint T.
Cumulative RF is a pooled elapsed-time-aware Random Forest regressor. Outer
evaluation leaves out an entire event, including all checkpoints from all of
that event's threads. The target at each checkpoint is
`log1p(corrected dynamic preventable impact)`.

The canonical fixed-Budget-50 sequential replay removes previously selected
threads before each later checkpoint. Its consolidated inputs, fitted all-event
models, OOF results, exact code snapshot, and provenance are stored at:

`reference_result/20260903_160025_balanced_cumulative_sequential_policy`

The selected candidate is the balanced cumulative policy with checkpoint quotas
`[9, 9, 8, 8, 8, 8]`. It achieved 0.326085 mean horizon reduction across seven
eligible held-out events; see the reference bundle README for the comparison
boundary and research-safety caveats.

## Canonical commands

The canonical workflow reads the hash-locked materialization in the reference bundle.
Run smoke first, then manually start the long full evaluation:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_smoke.py
.\venv\Scripts\python.exe .\effective_models\pheme_multicheckpoint_rf\training\run_full.py
```

The Hawkes-inspired candidate is maintained independently in
`effective_models/pheme_hawkes_multicheckpoint_rf`; this folder contains only the
canonical Balanced Cumulative implementation.

## Archived research branches

Adaptive cumulative, opportunity-reserve, wait-loss hazard,
quiet-acceleration, timing-only, dual-threshold, quiet-hybrid challenger,
Agent-vs-RF, and quota/cache experiments are not runtime dependencies or
canonical model entry points. Their local copies are preserved under:

```text
analysis/pheme_multicheckpoint_rf/
```

The repository ignores `analysis/`; these negative-result and exploratory
branches therefore remain available on the research workstation without being
included in the system Git history.
