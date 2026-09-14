# Removed files

This log records obsolete files removed while establishing the canonical model workflow.
The protected frozen artifact and historical reference result were not removed.

Removal occurred only after the replacement pipeline passed syntax checks, 4/4 unit
tests, and a real-data nested smoke run.

| Removed path | Former purpose | Why removed | Canonical replacement |
|---|---|---|---|
| `training/run_nested_discourse_context_selection_rf.py` | Monolithic full nested-LOEO training and evaluation runner | Duplicated configuration, loading, fitting, selection, and reporting in one file; lacked the required complete run record and a shared smoke path | `config.py`, `features/discourse_features.py`, `training/common.py`, `training/run_smoke.py`, and `training/run_full.py` |
| `features/finalize_partial_observation_schema_v2.py` | Copied an earlier partial-observation artifact into a schema-locked dataset | Its relocated paths resolved below `features/artifacts/`, so it could not reproduce the protected artifact from the model's current layout | The immutable, hash-documented artifact in `ARTIFACTS.md`; construction provenance remains in project history |
| `features/prepare_pheme_author_aware_snapshots.py` | Built author-role features from raw PHEME data | Relocated private copy imported a missing model-local `data_pipeline` module and wrote to obsolete `features/artifacts/` paths | Frozen feature artifact loader and validation in `features/discourse_features.py` |
| `features/prepare_event_context_features.py` | Added strict-past event-context percentile and z-score features | Relocated copy depended on missing model-local raw data/parser paths and no longer targeted the final artifact location | Frozen feature artifact loader and the `plus_event_context` bundle in `features/discourse_features.py` |
| `features/prepare_discourse_response_features.py` | Added early enquiry, correction, evidence, question, and semantic-response features | Relocated copy depended on missing model-local raw/text-semantic artifacts and could not recreate the protected final artifact independently | Frozen feature artifact loader and the `plus_event_context_discourse` bundle in `features/discourse_features.py` |

The removed builders are not represented as runnable canonical workflows. Their roles,
cutoff-safety intent, and replacement are documented here; their exact source remains
recoverable from Git history.
