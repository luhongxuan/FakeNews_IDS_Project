"""Short preflight for the canonical v5 text RF pipeline."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from effective_models.pheme_v5_text_rf.training.common import run_experiment


if __name__ == "__main__":
    run_experiment("smoke")
