# Research Code Reorganization — 2026-09-02

## Scope

This reorganization separates retained effective model code from exploratory,
smoke, diagnostic, and superseded experiment scripts. It does not change raw
data, derived datasets, labels, splits, cutoff definitions, artifacts, or saved
experiment results.

## Effective model catalog

The retained catalog is under `effective_models/`:

| Folder | Retained result | Original source |
|---|---|---|
| `pheme_v5_text_rf` | PHEME reduction@10 0.1467 | `graphsage_intervention_5` |
| `pheme_graph7_discourse_rf` | PHEME CRR@10 0.1385 | `graphsage_intervention_7` |
| `pheme_graph7_author_aware_rf` | PHEME CRR@10 0.1192 vs size 0.1059 | `graphsage_intervention_7` |
| `pheme_active_quiet_calibrated_rf` | PHEME reduction@10 0.1461 | `graphsage_intervention_14` |
| `twitter_graph_only_rf` | T15/T16 CRR@10 0.1288/0.2598 | `graphsage_intervention_9` |
| `twitter_weak_order_scalar_mlp` | T15/T16 reduction@10 0.1986/0.3292 | `graphsage_intervention_14` |

Each model has `training/`, `features/`, and `evaluation/` directories. Shared
feature implementations are intentionally duplicated so that every retained
model has an explicit private input definition.

## Archived code

All superseded version directories were removed from the repository root and
preserved in a recoverable local archive beneath:

```text
research_scratch/legacy_full/graphsage_intervention_2/
research_scratch/legacy_full/graphsage_intervention_2_temp/
research_scratch/legacy_full/graphsage_intervention_3/
research_scratch/legacy_full/graphsage_intervention_3_temp/
research_scratch/legacy_full/graphsage_intervention_4/
research_scratch/legacy_full/graphsage_intervention_5/
research_scratch/legacy_full/graphsage_intervention_6/
research_scratch/legacy_full/graphsage_intervention_7/
research_scratch/legacy_full/graphsage_intervention_8/
research_scratch/legacy_full/graphsage_intervention_9/
research_scratch/legacy_full/graphsage_intervention_10/
research_scratch/legacy_full/graphsage_intervention_11/
research_scratch/legacy_full/graphsage_intervention_12/
research_scratch/legacy_full/graphsage_intervention_13/
research_scratch/legacy_full/graphsage_intervention_14/
research_scratch/legacy_full/graphsage_intervention_15/
```

`research_scratch/` is ignored by Git. The archive contains the complete former
directories, including code, artifacts, experiment results, analysis outputs,
and non-Python research assets. Reference paths in the project results report
were updated, while results belonging to retained effective models were also
copied into those models' `reference_result/` directories.

The recovery archive currently contains 16 complete version directories, 1,415
files, and 9,870,377,274 bytes. These directories were removed from the project
root by moving them intact, not by permanently erasing their contents.

The protected v5 graph and raw observable-node inputs needed by retained models
were copied to `data/protected_research_assets/pheme_v5_strict30/`. Source and
saved copies were checked byte-for-byte through SHA-256 and all three matched:

- graph artifact: `DD5637B76ACEE343537A866272D4D488F31918802C6EB229FD642678A3D0019B`
- node feature CSV: `BF66708A642B0924C0FD964186972BA72A646F6B4ACFA72661D221340DF2849D`
- reply offset CSV: `573D284C2F6EAF63EF72E9A1227082D734E2A4DC58DC60C8F4E6FFCB56AA9B78`

Retained v5 and active/quiet model code was also detached from the legacy text
classifier helpers; each model now owns its unchanged snapshot-only source plus
observed-reply centroid implementation.

## Deferred run completed and archived

The exhaustive nested quiet-feature combination run completed successfully and
was reviewed before final archival. It improved quiet Oracle efficiency only at
budgets 1 and 3, while degrading budgets 5, 10, 20, 50, and 100 and reducing
macro Spearman correlation. It is therefore retained as an exploratory negative
result, not promoted into the effective-model catalog.

The complete result is preserved at:

```text
research_scratch/legacy_full/graphsage_intervention_14/experiments/
20260901_233016_536789_nested_pheme_quiet_feature_combinations
```

Versions 4, 5, and 14 were then moved into `research_scratch/legacy_full/`.
Graph14's archived inventory matches its pre-move inventory at 494 files and
28,017,551 bytes. The final 4,145,902-byte CSV initially held by Microsoft Excel
was removed from the root only after its archived copy matched the source
SHA-256 (`FDDD694505228321ECB3763CF5E132FE38A3D3ABC4F40F75CC5A24F98E1E858F`).
No numbered `graphsage_intervention_*` directory remains at the repository root.

## Validation performed

- All Python files copied into `effective_models/` passed `py_compile`.
- All six retained training modules passed import/path checks.
- Every checked protected input path resolved successfully.
- All three protected v5 input copies matched their source SHA-256 hashes.
- `research_scratch/legacy_full/` was confirmed ignored by Git.
- The exhaustive run completed before versions 4, 5, and 14 were archived.
