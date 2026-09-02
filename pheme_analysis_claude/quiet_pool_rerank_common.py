"""Leakage-safe two-stage quiet pool + within-pool rerank policy.

Motivation (see PROJECT_HANDOFF_20260902_QUIET_POLICY.md): the existing
three-tier quiet classifier only enriches the "high" tier by roughly 1.76x
over random when a hard top-20% cut is taken, and misses about two-thirds of
the true high-impact quiet threads outside that cut. A hard tier cut is
therefore not reliable enough to use alone as the final quiet ranking.

This module instead builds a full ranking in two stages:

1. Pool: take the top ``pool_fraction`` of quiet threads by the existing
   tier model's train-fitted P(high) score. A wider pool (e.g. 40-60%)
   keeps most true high-impact threads reachable, unlike a hard 20% cut.
2. Rerank: within the pool, order threads by a rank fusion of the existing
   quiet utility RF score and the same tier P(high) score. Threads outside
   the pool keep the identical fusion score but are always ranked below
   every pool member, so large budgets still degrade gracefully instead of
   truncating the ranking.

``pool_fraction=1.0`` together with ``utility_weight=1.0`` is the identity
fallback: every thread is "in pool" and the fusion score is exactly the
existing quiet utility RF score, i.e. the unmodified current baseline.

No new feature construction is introduced. This module only reuses existing
leakage-audited fitting code:

- quiet utility score: ``pheme_quiet_expert_common.gated_predict`` (the
  quiet expert half of the deployed calibrated active/quiet RF).
- tier P(high): ``run_quiet_tier_hybrid_smoke.tier_probabilities`` (source
  PCA16 + temporal ExtraTrees fusion already used by the deployed hybrid
  policy). Its source/temporal fusion weight is held fixed at that policy's
  predeclared 0.50 blend and is not re-selected here, to keep this
  experiment to one clear change (pool_fraction, utility_weight) rather than
  combining unrelated architectural changes.

``pool_fraction`` and ``utility_weight`` are the only two knobs, and both
are selected purely from inner event-OOF quiet efficiency computed on
outer-train events -- never from the outer held-out event. Selection uses
the per-inner-fold MEDIAN (not the mean) at every budget, including K=1 and
K=3, specifically because K=3 sums only three threads' impact and a mean
can be pulled above baseline by a single lucky inner fold; see
``select_config`` for the concrete outer-event failure this was revised to
prevent.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACTIVE_QUIET_TRAINING = ROOT / "effective_models" / "pheme_active_quiet_calibrated_rf" / "training"
sys.path.insert(0, str(ACTIVE_QUIET_TRAINING))

from pheme_account_age_v5_common import (  # noqa: E402
    DATASET, ELIGIBLE_MIN_THREADS, SEED, load_graphs, load_node_offsets, load_safe_nodes,
)
from pheme_quiet_expert_common import gated_predict, recency_values  # noqa: E402
from quiet_tier_hybrid_score import percentile_rank  # noqa: E402
import run_quiet_tier_hybrid_smoke as hybrid_common  # noqa: E402


POOL_FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
UTILITY_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BUDGETS = (1, 3, 5, 10, 20, 50, 100)
GUARD_BUDGETS = BUDGETS  # every budget is guarded, including K=1 and K=3; see select_config
GUARD_TOLERANCE = 0.02
BASELINE_CONFIG = (1.0, 1.0)
FEATURE_TABLE = hybrid_common.FEATURE_TABLE
MATERIALIZATION_RECORD = hybrid_common.MATERIALIZATION_RECORD


def graph_inputs(graphs: list) -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray]]:
    """Delegate to the already-audited protected feature/embedding loader."""
    return hybrid_common.graph_inputs(graphs)


def quiet_split(
    train: list, test: list, depth_nodes, account_age_nodes, node_offsets,
) -> tuple[list, list, np.ndarray, dict]:
    """Fit the existing two-expert utility gate once; return quiet-only subsets and scores."""
    utility, gate, _ = gated_predict(train, test, depth_nodes, account_age_nodes, node_offsets)
    train_quiet = recency_values(train) >= gate["threshold"]
    test_quiet = recency_values(test) >= gate["threshold"]
    quiet_train = [graph for graph, keep in zip(train, train_quiet) if keep]
    quiet_test = [graph for graph, keep in zip(test, test_quiet) if keep]
    quiet_utility = np.asarray(utility, dtype=np.float32)[test_quiet]
    return quiet_train, quiet_test, quiet_utility, gate


def pool_rerank_scores(
    utility_score: np.ndarray, tier_prob_high: np.ndarray, thread_ids: list[str],
    pool_fraction: float, utility_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Full ranking key: every pool member outranks every non-pool member."""
    n = len(thread_ids)
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)
    if not 0.0 < pool_fraction <= 1.0:
        raise ValueError("pool_fraction must lie in (0, 1]")
    if not 0.0 <= utility_weight <= 1.0:
        raise ValueError("utility_weight must lie in [0, 1]")
    order = sorted(range(n), key=lambda i: (-float(tier_prob_high[i]), str(thread_ids[i])))
    pool_size = max(1, math.ceil(n * pool_fraction))
    pool_mask = np.zeros(n, dtype=bool)
    pool_mask[order[:pool_size]] = True
    utility_rank = percentile_rank(utility_score, thread_ids)
    tier_rank = percentile_rank(tier_prob_high, thread_ids)
    within_score = utility_weight * utility_rank + (1.0 - utility_weight) * tier_rank
    key = pool_mask.astype(np.float32) * 10.0 + within_score
    return key.astype(np.float32), pool_mask


def quiet_efficiency_curve(
    thread_ids: list[str], impact: np.ndarray, score: np.ndarray, budgets: tuple[int, ...] = BUDGETS,
) -> dict[int, float]:
    """Blocked-impact-by-model / blocked-impact-by-oracle at each budget, within one quiet pool."""
    frame = pd.DataFrame({"thread_id": thread_ids, "impact": impact, "score": score})
    ranked = frame.sort_values(["score", "thread_id"], ascending=[False, True], kind="stable")
    oracle = frame.sort_values(["impact", "thread_id"], ascending=[False, True], kind="stable")
    result: dict[int, float] = {}
    for budget in budgets:
        actual = min(budget, len(frame))
        denominator = float(oracle.impact.iloc[:actual].sum())
        numerator = float(ranked.impact.iloc[:actual].sum())
        result[int(budget)] = numerator / denominator if denominator else 0.0
    return result


def select_config(inner_rows: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Deterministic, guarded config selection from raw per-inner-fold efficiency rows.

    ``inner_rows`` columns: fold, pool_fraction, utility_weight, budget,
    efficiency -- one row per inner validation event, config, and budget,
    NOT pre-aggregated. Never reads outer-event information.

    Aggregation across inner folds uses the per-budget MEDIAN, not the mean,
    and every budget in BUDGETS (including K=1 and K=3) is guarded, not just
    K>=5. An earlier version of this rule aggregated by mean and only
    guarded K>=5; K=3 sums the impact of just three threads, so a single
    lucky inner fold could pull the *mean* K=1/K=3 gain above the baseline
    while most inner folds looked no better. That configuration then failed
    badly on the actual outer event (see run
    20260902_145025_840498_quiet_pool_rerank_full: putinmissing's held-out
    K=3 efficiency dropped from 0.605 to 0.105 even though its inner mean
    K=1/K=3 gain looked positive). The median requires at least half of the
    inner folds to actually be no worse than the baseline before a
    configuration can be preferred at any budget, which directly removes
    that single-fold-fluke failure mode.

    A configuration is only eligible if its median efficiency at every
    budget in GUARD_BUDGETS is not below the pool_fraction=1.0,
    utility_weight=1.0 baseline's median by more than GUARD_TOLERANCE.
    Among eligible configurations, the one with the largest median K=1/K=3
    gain over that baseline is selected; ties break toward the larger
    (safer) pool, then the smallest |utility_weight - 1.0| (closest to the
    utility_weight=1.0 baseline). If no configuration is eligible, selection
    falls back to the baseline itself.
    """
    median_by_config_budget = inner_rows.pivot_table(
        index=["pool_fraction", "utility_weight", "budget"], values="efficiency", aggfunc="median"
    ).reset_index()
    summary = median_by_config_budget.pivot_table(
        index=["pool_fraction", "utility_weight"], columns="budget", values="efficiency"
    ).reset_index()
    baseline_rows = summary[
        (summary.pool_fraction == BASELINE_CONFIG[0]) & (summary.utility_weight == BASELINE_CONFIG[1])
    ]
    if baseline_rows.empty:
        raise ValueError("Baseline configuration (pool_fraction=1.0, utility_weight=1.0) is missing from inner rows")
    baseline = baseline_rows.iloc[0]

    def guard_ok(row: pd.Series) -> bool:
        return all(row[budget] - baseline[budget] >= -GUARD_TOLERANCE for budget in GUARD_BUDGETS)

    summary["guard_ok"] = summary.apply(guard_ok, axis=1)
    summary["head_gain"] = ((summary[1] + summary[3]) / 2.0) - ((baseline[1] + baseline[3]) / 2.0)
    # Tie-break distance to the baseline itself, not raw utility_weight: a
    # naive ascending sort on utility_weight would prefer 0.0 (furthest from
    # the utility_weight=1.0 baseline) over 1.0 (closest) whenever several
    # configs tie on head_gain, which defeats the "prefer the smaller,
    # safer deviation from baseline" intent.
    summary["utility_weight_distance_from_baseline"] = (summary.utility_weight - BASELINE_CONFIG[1]).abs()
    candidates = summary[summary.guard_ok].copy()
    if candidates.empty:
        candidates = baseline_rows.copy()
        candidates["head_gain"] = 0.0
        candidates["utility_weight_distance_from_baseline"] = 0.0
    candidates = candidates.sort_values(
        ["head_gain", "pool_fraction", "utility_weight_distance_from_baseline"],
        ascending=[False, False, True], kind="stable",
    )
    chosen = candidates.iloc[0]
    return float(chosen.pool_fraction), float(chosen.utility_weight), summary


def safety_statement() -> str:
    return (
        "Pool membership and within-pool rank fusion use only the existing train-fold quiet "
        "utility RF and tier P(high) probability, both fit on outer-train (or inner-train) "
        "events only via unchanged, already-audited helpers. pool_fraction and utility_weight "
        "are selected solely from the per-inner-fold MEDIAN quiet efficiency computed on "
        "outer-train events; the outer held-out event contributes no score, target, or "
        "threshold to selection. The guard requires the median efficiency at every budget in "
        f"{GUARD_BUDGETS} (including K=1 and K=3, not only K>=5) not to regress by more than "
        f"{GUARD_TOLERANCE} versus the pool_fraction=1.0/utility_weight=1.0 identity baseline "
        "(plain quiet utility RF ranking); if no configuration satisfies the guard, selection "
        "falls back to that "
        "baseline unchanged."
    )
