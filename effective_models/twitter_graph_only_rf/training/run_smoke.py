"""Bounded validation-only smoke for the canonical graph-only RF pipeline."""
from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from effective_models.twitter_graph_only_rf.training.common import run_experiment  # noqa: E402


if __name__ == "__main__":
    run_experiment("smoke")
