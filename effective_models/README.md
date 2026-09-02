# Effective Model Catalog

This directory contains the model families retained as defensible positive
results in `PROJECT_RESULTS_20260901.md`. Each model owns a separate copy of
its training and feature-definition code even when another model uses the same
feature implementation.

## Layout

```text
<model>/
  training/    model fitting and model-selection entry points
  features/    this model's private feature builders/definitions
  evaluation/  frozen or reusable evaluation code, when applicable
  README.md    protocol, result, inputs, and source provenance
```

Large protected datasets, derived artifacts, and completed experiment outputs
remain in their original locations. The catalog code points to those immutable
locations and writes new experiments inside its own model directory.

The active exhaustive quiet-feature experiment is not promoted here until its
nested LOEO run completes and its result is reviewed.
