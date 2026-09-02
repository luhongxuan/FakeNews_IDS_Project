"""Bounded strict-30 ablation: baseline tier inputs plus relational feature block."""
from __future__ import annotations

from pathlib import Path

import run_strict30_two_expert_tier as base


HERE = Path(__file__).resolve().parent
MATERIALIZATION = HERE / "experiments" / "20260902_145009_008419_strict30_relational_feature_materialization"


def main() -> None:
    base.RUN_MODE = "smoke"
    base.EXPERIMENT_NAME = "strict30_relational_tier_ablation"
    base.EXTRA_TEMPORAL_PREFIXES = ("rel_",)
    base.MATERIALIZATION = MATERIALIZATION
    base.FEATURE_TABLE = MATERIALIZATION / "strict30_protected_relational_snapshot_features.csv"
    base.MATERIALIZATION_RECORD = MATERIALIZATION / "run_record.json"
    base.main()


if __name__ == "__main__":
    main()
