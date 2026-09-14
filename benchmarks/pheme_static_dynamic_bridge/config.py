from pathlib import Path
ROOT=Path(__file__).resolve().parent
REFERENCE=ROOT/"reference_result"/"20260907_154306_938376_bridge_a_full"
FIXED_OOF=REFERENCE/"fixed30_oof_scores.csv"
DYNAMIC_OOF=REFERENCE/"dynamic_oof_scores.csv"
EXPERIMENTS=ROOT/"experiments"
CHECKPOINTS=(600,1200,1800,2400,3000,3600)
QUOTAS=(9,9,8,8,8,8)
TOTAL_BUDGET=50
ELIGIBLE_MIN_THREADS=100
