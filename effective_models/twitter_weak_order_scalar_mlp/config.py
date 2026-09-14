from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts" / "20260901_133740_887252_twitter15_16_strict30_graph_head_features_v1"
RECORDS = ARTIFACT / "thread_level_records.csv"
SCHEMA = ARTIFACT / "schema.json"
EXPERIMENTS = ROOT / "experiments"
REFERENCE_SELECTION = ROOT / "reference_result" / "20260901_135117_434080_validation_head_focused_scalar_v2_strict_weak_order"

CUTOFF_SECONDS = 1800
BUDGETS = (1, 3, 5, 10, 20, 40)
SEEDS = (13, 42, 71)
EPOCHS = 160
ORDER_LAMBDA = 0.05
SOFT_TOPK_LAMBDA = 0.0

