from pathlib import Path
ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "artifacts" / "20260906_unified_fixed30_intersection_v1"
ALLOWLIST = ARTIFACT / "unified_thread_allowlist.csv"
REFERENCE = ROOT / "reference_result" / "20260906_163045_109094_unified_fixed30_nested_full"
OOF = REFERENCE / "oof_thread_scores.csv"
EXPERIMENTS = ROOT / "experiments"
BUDGETS = (1, 3, 5, 10, 20, 50, 100)
ELIGIBLE_MIN_THREADS = 100
CUTOFF_SECONDS = 1800
