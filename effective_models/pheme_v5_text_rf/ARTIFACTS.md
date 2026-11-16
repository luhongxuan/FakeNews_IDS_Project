# Local artifacts

The canonical pipeline expects these protected local inputs under this model's
`artifacts/pheme_v5_strict30/` directory:

| File | Size | SHA-256 |
|---|---:|---|
| `pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt` | 71,982,127 bytes | `DD5637B76ACEE343537A866272D4D488F31918802C6EB229FD642678A3D0019B` |
| `pheme_node_features_roberta_replyv4.csv` | 1,529,893,774 bytes | `BF66708A642B0924C0FD964186972BA72A646F6B4ACFA72661D221340DF2849D` |

Historical reference:

`reference_result/20260831_223827_909201_v5_best_text_oof_min_interventions/result.json`

SHA-256: `D6C18C6C5FFB0940CED6A3272CB6E3AB0766A0E7C9B8127E5D0DD81665D7F277`

These large artifacts and generated experiment directories are intentionally ignored by Git.

For the current workstation migration, copy the hash-identical files from
`data/protected_research_assets/pheme_v5_strict30/` into this model-local
directory. Do not move or modify the protected source files.
