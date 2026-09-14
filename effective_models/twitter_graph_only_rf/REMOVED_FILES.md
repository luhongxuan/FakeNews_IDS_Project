# Removed files

This log records files removed after the canonical shared smoke/full pipeline
was validated. No file should be deleted from this model without adding a row.

| Removed path | Original purpose | Removal reason | Canonical replacement |
|---|---|---|---|
| `features/twitter_graph_only_ranker_common.py` | Duplicate loader, feature list, fitting, and evaluation helpers. | It duplicated the training helper and split one workflow across two competing copies. | `config.py`, `features/graph_only_features.py`, and `training/common.py` |
| `training/twitter_graph_only_ranker_common.py` | Legacy loader, RF configurations, fitting, and evaluation helpers. | Replaced by the shared canonical implementation used by both smoke and full entry points. | `config.py`, `features/graph_only_features.py`, and `training/common.py` |
| `training/run_twitter15_16_graph_only_ranker.py` | Monolithic full runner for both Twitter corpora. | It had no matching smoke entry point and duplicated run-record orchestration now handled centrally. | `training/run_full.py` calling `training/common.py` |
