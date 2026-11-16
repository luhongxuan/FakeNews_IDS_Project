from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from effective_models.pheme_hawkes_multicheckpoint_rf.training.common import run

if __name__ == "__main__":
    run("full")

