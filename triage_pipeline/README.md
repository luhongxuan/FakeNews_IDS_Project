# Propagation thread triage pipeline

This offline research pipeline ranks threads by later **other-author replies**. It does not assess truth, harm, or intervention effects. `SPEC.md`, `SPEC_AMENDMENT_1.md`, `PREREG.md`, and its amendment fix the protocol for the development run.

## Stages

Run from the repository root with the existing virtual environment. Set `PLATFORM` to `bluesky` or `pheme` and run each stage in order:

```powershell
$env:PLATFORM='bluesky'
$env:RUN_STAGE='unit'
.\venv\Scripts\python.exe -m triage_pipeline.run_pipeline
$env:RUN_STAGE='consistency'
.\venv\Scripts\python.exe -m triage_pipeline.run_pipeline
$env:RUN_STAGE='profile'
.\venv\Scripts\python.exe -m triage_pipeline.run_pipeline
```

Repeat with `PLATFORM='pheme'`. Each invocation creates a new directory under `analysis/20261127_triage_pipeline_dev_run/{stage}_{timestamp}/{platform}/`. The consistency stage writes a hashed, text-free unified cache. Profile finds the latest completed cache for its platform; set `CACHE_RUN` to a particular completed consistency directory to select it explicitly. The profile stage reads no method scores and fits no model.

`RUN_STAGE='full'` is prepared for a later user decision. It fits the fixed random forest, selects from the selection portion, evaluates once on the test portion, and writes a recommendation and report. Set `PROFILE_RUN` to the completed profile directory to select a particular profile. Full work is expected to be long-running and is not started by unit, consistency, or profile.

The raw source data, split definitions, specification, preregistration, products, and model artifacts are read-only. Runs record input and code hashes, progress, final status, and output inventory. A failure keeps its output directory for inspection.
