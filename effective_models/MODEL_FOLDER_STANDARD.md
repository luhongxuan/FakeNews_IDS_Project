# Effective model folder standard

Every retained model is an independently understandable and executable unit.
Model source code must not import source code from another model directory.
Large datasets, fitted weights, OOF predictions, and experiment outputs remain
local-only and are described by a tracked `ARTIFACTS.md` file.

```text
<model>/
  __init__.py
  README.md
  ARTIFACTS.md
  REMOVED_FILES.md
  config.py
  features/
    __init__.py
  training/
    __init__.py
    common.py
    run_smoke.py
    run_full.py
  evaluation/
    __init__.py
  tests/
    __init__.py
  artifacts/          # ignored local files
  reference_result/   # ignored local files
  experiments/        # ignored run outputs
```

Internal imports must use the complete `effective_models.<model>...` namespace.
Generic top-level imports such as `import config` or `from training.common`
are prohibited because they collide when multiple models share one process.

`run_smoke.py` and `run_full.py` must call the same implementation in
`training/common.py`. Smoke mode may reduce corpora, samples, folds, trees, or
epochs, but it must not replace the loader, features, model family, or metric
implementation with a separate pipeline.

Before obsolete files are deleted, their former path, purpose, deletion reason,
and canonical replacement must be recorded in that model's `REMOVED_FILES.md`.
