# Local artifacts

The canonical model uses its own local copy of the protected strict-30 inputs
under `artifacts/pheme_v5_strict30/`, including reply offsets:

| File | SHA-256 |
|---|---|
| `pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt` | `DD5637B76ACEE343537A866272D4D488F31918802C6EB229FD642678A3D0019B` |
| `pheme_node_features_roberta_replyv4.csv` | `BF66708A642B0924C0FD964186972BA72A646F6B4ACFA72661D221340DF2849D` |
| `pheme_reply_level_v4.csv` | `573D284C2F6EAF63EF72E9A1227082D734E2A4DC58DC60C8F4E6FFCB56AA9B78` |

Historical reference:

`reference_result/20260901_210513_260899_paired_oof_pheme_quiet_expert_calibration/`

- `run_record.json`: `2AB15C4EC86C0201C4E783159AE497413463B83E87F8B868FE78C4F8E7860CDC`
- `eligible_event_paired_budget_summary.csv`: `548ADAD1A150CEA36DBB1560386F1A276D7DB71525F709E6DC25B68ADEE2F45B`

Large protected inputs, references, and experiment outputs are intentionally ignored by Git.

For the current workstation migration, copy the hash-identical files from
`data/protected_research_assets/pheme_v5_strict30/` into this model-local
directory. Do not move or modify the protected source files.
