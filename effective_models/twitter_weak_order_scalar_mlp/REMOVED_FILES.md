# Removed files

The canonical shared implementation passed syntax checks, 3/3 unit tests, and a
real two-epoch smoke in which all 224 protected test rows were skipped before
feature/target parsing. The following files were then removed.

| Removed file | Previous purpose | Canonical replacement / retained provenance |
|---|---|---|
| `features/strict_feature_loader.py` | Strict train/validation loader | `features/strict30_features.py` |
| `training/head_focused_v2_strict_common.py` | Duplicate strict loader | `features/strict30_features.py` |
| `evaluation/head_focused_v2_strict_common.py` | Second duplicate strict loader | `features/strict30_features.py` |
| `training/run_validation_head_focused_scalar.py` | Original v1 model/loss implementation | `model/scalar_mlp.py`, `evaluation/metrics.py`, `training/common.py` |
| `evaluation/run_validation_head_focused_scalar.py` | Byte-for-byte-style duplicate of v1 trainer retained beside evaluation | Same canonical modules above |
| `training/run_validation_head_focused_scalar_v2_strict.py` | Validation grid that selected order weight 0.05 | Selection is frozen in `reference_result/20260901_135117_434080_validation_head_focused_scalar_v2_strict_weak_order/`; canonical runs no longer retune it |
| `evaluation/evaluate_test_head_focused_scalar_v2_strict_weak_order.py` | One-time frozen held-out test evaluation | `training/run_full.py` reproduces the frozen config only; immutable original result remains under `reference_result/` |
| `features/build_twitter_graph_head_features.py` | Historical strict-30 artifact builder | Frozen artifact and provenance remain under `artifacts/`; canonical model runs do not regenerate protected data |
| `features/raw_graph_feature_builder.py` | Exploratory raw-graph/oracle-gap analysis and feature helper | Not part of the selected model workflow; historical source remains in Git history |
| `features/twitter_graph_only_ranker_common.py` | Copied dependency from the separate graph-only RF model | Not needed by the frozen weak-order MLP; graph-only model has its own canonical folder |
