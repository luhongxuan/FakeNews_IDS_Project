# Removed files

Files are added here only after their replacement pipeline passes its smoke.

| Removed path | Original purpose | Removal reason | Canonical replacement |
|---|---|---|---|
| `training/run_fold_safe_author_aware_rf.py` | Monolithic formal LOEO RF runner. | Replaced by a shared, run-record-complete implementation with matching smoke and full entry points. | `training/common.py`, `training/run_smoke.py`, `training/run_full.py` |
| `features/prepare_pheme_author_aware_snapshots.py` | Historical raw-PHEME snapshot builder copy. | After relocation its repository-root resolution and imported helper path were invalid, and it did not reproduce the exact frozen schema-locked artifact used by this model. Keeping it would falsely imply a supported canonical build path. | Frozen artifact contract in `ARTIFACTS.md`; canonical loader in `features/author_aware_features.py` |
| `features/finalize_partial_observation_schema_v2.py` | Historical script that copied an intermediate artifact into a schema-locked artifact. | Its source/output paths referred to the pre-relocation artifact layout and the required intermediate source is not part of this model's canonical local contract. | Frozen schema, manifest, and hashes in `ARTIFACTS.md`; loader validation in `features/author_aware_features.py` |
