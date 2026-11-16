"""Canonical full nested calibration entry point; run manually after smoke."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from effective_models.pheme_active_quiet_calibrated_rf.training.common import run_experiment


if __name__ == "__main__":
    run_experiment("full")
