# Effective Models

This directory contains only retained model families required by the running
system or a defensible reported result. Exploratory,
negative-result, diagnostic, and presentation-only branches are kept locally
under `analysis/` and are excluded from Git.

## Retained PHEME model families

- `pheme_multicheckpoint_rf`: deployed dynamic RF feature/model path used by
  backend intervention scoring.
- `pheme_v5_text_rf`: deployed fixed-30 text RF score path.
- `pheme_active_quiet_calibrated_rf`: retained Active/Quiet calibrated RF and
  dependencies used by the unified fixed-30 benchmark.
- `pheme_graph7_author_aware_rf`: retained fixed-30 author-aware benchmark model.
- `pheme_graph7_discourse_rf`: retained fixed-30 discourse/context benchmark model.
- `pheme_hawkes_multicheckpoint_rf`: retained dynamic candidate model.

## Retained Twitter model families

- `twitter_graph_only_rf`
- `twitter_weak_order_scalar_mlp`

Target construction is maintained under `data_pipeline/pheme_corrected_target`.
Cross-model comparisons are maintained under `benchmarks/`, not in this model
directory.

## Python environment and imports

Install the minimal tested model dependencies from `effective_models/requirements.txt`.
The recorded environment uses Python 3.13.5. Every retained model is an explicit
Python package and internal imports use its full `effective_models.<model>`
namespace, so all eight model families can be loaded safely in one process.

## Restoring ignored local artifacts

After cloning or reorganizing the repository, run the foreground-only helper:

```powershell
.\venv\Scripts\python.exe .\effective_models\prepare_local_artifacts.py
```

It copies (never moves) the protected sources into the three model-local
artifact contracts, verifies the recorded SHA-256 values, refuses to overwrite
different existing files, and then runs the three affected canonical smoke
entry points. Progress and final status are saved under
`effective_models/experiments/<timestamp>_local_artifact_copy/`.

## Storage policy

- Do not put exploratory model branches back in this directory merely to keep
  them in Git.
- Keep each model's protected inputs under that model's Git-ignored `artifacts/`
  contract documented in its `ARTIFACTS.md`; do not commit generated artifacts.
- New model families should enter this directory only after their protocol,
  dependencies, and role in the system or reported benchmark are documented.
