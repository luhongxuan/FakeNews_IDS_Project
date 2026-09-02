"""Bounded ablation: source/reply semantic interaction embedding for quiet tiers."""
from __future__ import annotations

import run_strict30_two_expert_tier as base


def main() -> None:
    base.RUN_MODE = "smoke"
    base.EXPERIMENT_NAME = "strict30_semantic_interaction_tier_ablation"
    base.EXTRA_TEMPORAL_PREFIXES = ()
    base.EMBEDDING_MODE = "source_reply_interactions"
    base.main()


if __name__ == "__main__":
    main()
