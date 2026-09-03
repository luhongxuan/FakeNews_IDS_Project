"""Foreground entry point for the full raw strict-past user-history protocol."""
from __future__ import annotations

import materialize_and_diagnose_strict30_user_overlap as experiment


experiment.HISTORY_SCOPE_MODE = "global_raw_strict_past"
experiment.RUN_SUFFIX = "_strict30_global_raw_user_overlap_diagnostic"


if __name__ == "__main__":
    experiment.main()
