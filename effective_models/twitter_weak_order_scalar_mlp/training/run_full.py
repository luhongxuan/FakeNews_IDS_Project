from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from effective_models.twitter_weak_order_scalar_mlp.training.common import run
if __name__ == "__main__": run("full")

