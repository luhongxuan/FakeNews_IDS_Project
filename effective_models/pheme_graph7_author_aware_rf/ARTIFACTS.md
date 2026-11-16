# Local artifact contract

Large data and frozen results are intentionally excluded from Git. Restore the
following exact local files before running the canonical pipeline.

| Local path | Purpose | SHA-256 |
|---|---|---|
| `artifacts/20260830_133000_pheme_author_aware_30min_schema_v2/thread_level_records.csv` | Frozen schema-locked model rows | `A268F89B00B250CEEB62093CB3E336CEFE0047B8005ADDED145B2E76203D8EB8` |
| `artifacts/20260830_133000_pheme_author_aware_30min_schema_v2/schema.json` | Feature/metadata/target boundary | `DB3266E76154D5257C1063BD4D65228E11B48B5DC7C9C691884DF5B80568947A` |
| `artifacts/20260830_133000_pheme_author_aware_30min_schema_v2/manifest.json` | Dataset construction provenance | `D9AA47D12D4793FDCC36C97333DD2B869C024581C899BFED0721B85219A1FBD0` |
| `reference_result/20260830_140800_517954_author_aware_rf_partial_formal/result.json` | Frozen historical formal result | `5B57A628377883B7DAAB21F15F37F661AF268CE30CD8687BE2A6AE2469F579CD` |

The canonical model begins from this immutable schema-locked artifact. The old
raw-data construction copies did not retain a self-contained, path-correct
reproduction of the exact frozen artifact and are therefore not canonical
entry points.

