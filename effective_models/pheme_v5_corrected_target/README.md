# Corrected PHEME v5 target replay

This candidate preserves the original strict-30 v5 snapshots, features,
Random Forest settings, nested leave-one-event-out split, and validation-only
feature-set selection. It changes only the malformed multi-root target:
preventable future impact is traversed from every independently traversable
root instead of only the last root encountered in the reply table.

The protected v5 dataset is never modified. A full run writes a corrected,
versioned graph artifact and complete result record into a new timestamped
directory under `experiments/`.

## Validated smoke

The bounded event-disjoint smoke exercises both original v5 feature paths and
reproduces the known multi-root correction for thread `552808071387701248`
from 0 to 93 preventable future nodes.

## Full foreground run

From the repository root:

```powershell
.\venv\Scripts\python.exe .\effective_models\pheme_v5_corrected_target\training\run_corrected_v5_nested_full.py
```

This is a long nested LOEO run and should remain visible in the foreground.
Do not interpret its comparison against legacy v5 as a same-label model gain:
the purpose is target-correction sensitivity under an otherwise fixed protocol.
