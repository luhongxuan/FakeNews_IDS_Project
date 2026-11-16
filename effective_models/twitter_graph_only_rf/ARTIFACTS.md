# Required local artifacts

Large CSV/model artifacts are intentionally excluded from Git. Restore them at
the exact relative paths below before running smoke or full evaluation.

| Relative path from this model folder | Purpose | Required | SHA-256 |
|---|---|---:|---|
| `artifacts/20260830_024514_251_twitter15_16_30min_time_respecting_preventable_impact/thread_level_records.csv` | Immutable 30-minute records and outcome-only target | Yes | `0F92EAD327D6B8DE1B78E5499EF91552DDEA9F0DDAF83E7E486A1BEDC58A2CCC` |
| `artifacts/20260830_024515_294_twitter15_16_source_level_split_v1/assignments.csv` | Frozen source-level train/validation/test assignments | Yes | `B129C0AF12A04348FF5111C7B53F41DFF411EBD27987F699D85F1CE6A9BB5916` |
| `artifacts/20260830_024515_294_twitter15_16_source_level_split_v1/manifest.json` | Split provenance | Yes | `99EC7D9A43030F20A7922C8889FFF1BD9BFA6BD2F1D1D0653EFFEEAD625E63A4` |
| `effective_models/twitter_graph_only_rf/reference_result/20260901_013107_933844_twitter15_16_graph_only_ranker/run_record.json` | Frozen formal result record | For result audit | `5591A5BE0E9F37EDE1BAF72BD3CE63EFB6B7C9007087D5DF80803482BDC83380` |
| `effective_models/twitter_graph_only_rf/reference_result/20260901_013107_933844_twitter15_16_graph_only_ranker/test_model_selected_threads.csv` | Frozen held-out selections | For result audit | `316135FD8D08DCD78CBFFF6DADF9C570154FB69DBB268A0EDF68DD5BEAFEDA19` |

The full runner reproduces a test evaluation whose protected test split has
already been consumed. It must not be used for additional model selection.

For the current workstation migration, copy (do not delete or rewrite) the two
hash-locked source directories from `data/derived/` into this model's
`artifacts/` directory using the same directory names shown above.
