# Local artifacts

The canonical Hawkes-inspired workflow is self-contained at runtime and expects
these local, Git-ignored files under `artifacts/`:

| File | SHA-256 | Purpose |
|---|---|---|
| `pheme_multicheckpoint_features.csv` | `C9C19E019EDB0E899E17E212F1106D9C50391CC64862CA6A41C87154DF37EB49` | Frozen 2,402-thread, six-checkpoint cumulative feature/target rows |
| `feature_schema.json` | `62A601D22C9029283C94E21F5407A7547C3A2F49478CF6319A8DC3712C6CAE8A` | Feature allowlist and outcome exclusions |
| `pheme_reply_level.csv` | `0D7F6D39069D31B17868B30DDE0500787C74D972331C57C28C1BE248F74F30AA` | Reply offsets used to derive cutoff-safe decay summaries |

These files are deliberately not committed. Copy them locally before running the
smoke/full entry points; never replace missing rows with fabricated values.

