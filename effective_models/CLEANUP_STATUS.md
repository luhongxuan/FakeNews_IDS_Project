# Effective-model cleanup status

This checklist tracks the structural cleanup. A model is marked complete only
after its canonical smoke and full entry points share one implementation, the
smoke has actually passed, local artifact requirements are documented, and all
deleted legacy files are listed in that model's `REMOVED_FILES.md`.

| Component | Role | Status |
|---|---|---|
| `twitter_graph_only_rf` | Retained auxiliary Twitter graph-only RF | **Canonicalized and smoke-validated** |
| `twitter_weak_order_scalar_mlp` | Retained auxiliary Twitter weak-order MLP | **Canonicalized and smoke-validated** |
| `pheme_graph7_author_aware_rf` | Retained fixed-30 author-aware RF | **Canonicalized and smoke-validated** |
| `pheme_graph7_discourse_rf` | Retained fixed-30 discourse/context RF | **Canonicalized and smoke-validated** |
| `pheme_v5_text_rf` | Retained fixed-30 text RF | **Canonicalized and smoke-validated** |
| `pheme_active_quiet_calibrated_rf` | Retained fixed-30 two-expert RF | **Canonicalized and smoke-validated** |
| `pheme_multicheckpoint_rf` | Retained dynamic cumulative RF | **Canonicalized and smoke-validated** |
| `pheme_hawkes_multicheckpoint_rf` | Retained dynamic Hawkes-inspired RF | **Canonicalized and smoke-validated** |
| `data_pipeline/pheme_corrected_target` | Corrected-target data/label lineage utility | **Relocated and smoke-validated** |
| `benchmarks/pheme_fixed30_unified` | Unified fixed-30 evaluation suite | **Relocated and smoke-validated** |
| `benchmarks/pheme_static_dynamic_bridge` | Static/dynamic bridge evaluation suite | **Relocated and smoke-validated** |

Exploratory branches already moved to ignored `analysis/` retain their original
relative structure where practical. They are not canonical deployment models.

## Large local artifacts

Global ignore rules cover CSV, pickle/joblib, NumPy array, Parquet, and PyTorch
weight files, as well as `artifacts/`, `reference_result/`, and `experiments/`.
Each retained model must document the exact local files it needs in
`ARTIFACTS.md` so the repository stays reproducible without committing the
large files themselves.

Three root CSV files are already tracked by Git and therefore are not affected
by a new ignore rule until they are explicitly removed from the Git index:

- `pheme_features_full_1.csv`
- `pheme_features_full_2.csv`
- `pheme_reply_level.csv`

They remain untouched locally during this phase.
