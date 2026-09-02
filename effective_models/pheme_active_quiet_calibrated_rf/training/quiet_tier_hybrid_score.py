"""Outcome-free quiet-score fusion used by the active/quiet hybrid policy."""
from __future__ import annotations

import numpy as np


def percentile_rank(values: np.ndarray, thread_ids: list[str]) -> np.ndarray:
    """Deterministic within-pool percentile ranks; ties use thread ID only."""
    values = np.asarray(values, dtype=float)
    if len(values) != len(thread_ids) or not np.isfinite(values).all():
        raise ValueError("Invalid score or identifier input for quiet hybrid")
    order = sorted(range(len(values)), key=lambda i: (-values[i], str(thread_ids[i])))
    result = np.empty(len(values), dtype=np.float32)
    denominator = max(len(values) - 1, 1)
    for position, index in enumerate(order):
        result[index] = 1.0 - position / denominator
    return result


def tier_priority(probabilities: np.ndarray) -> np.ndarray:
    """Map high/middle/low probabilities to an outcome-free ordinal priority."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite [high, low, middle] tier probabilities")
    return 2.0 * probabilities[:, 0] + probabilities[:, 2]


def hybrid_quiet_score(utility_score: np.ndarray, tier_probabilities: np.ndarray, thread_ids: list[str], utility_weight: float) -> np.ndarray:
    """Fuse utility and tier signals without labels, impacts, or future inputs."""
    if not 0.0 <= utility_weight <= 1.0:
        raise ValueError("utility_weight must lie in [0, 1]")
    # This is an exact baseline candidate.  It must not be rank-transformed,
    # otherwise a supposedly "no tier" choice changes active/quiet calibration.
    if utility_weight == 1.0:
        return np.asarray(utility_score, dtype=np.float32).copy()
    utility_rank = percentile_rank(utility_score, thread_ids)
    tier_rank = percentile_rank(tier_priority(tier_probabilities), thread_ids)
    return utility_weight * utility_rank + (1.0 - utility_weight) * tier_rank


def soft_pool_boost_quiet_score(
    utility_score: np.ndarray, tier_probabilities: np.ndarray, thread_ids: list[str],
    pool_fraction: float, bonus: float,
) -> np.ndarray:
    """Keep every quiet thread and apply a bounded boost to the high-tier pool.

    ``bonus == 0`` is deliberately the exact utility baseline.  Any nonzero
    bonus operates on deterministic within-quiet utility ranks, so its scale is
    interpretable and cannot depend on the held-out event's raw-score range.
    """
    if not 0.0 < pool_fraction <= 1.0 or not 0.0 <= bonus <= 1.0:
        raise ValueError("pool_fraction must be in (0, 1] and bonus in [0, 1]")
    utility_score = np.asarray(utility_score, dtype=np.float32)
    if bonus == 0.0:
        return utility_score.copy()
    probabilities = np.asarray(tier_probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape != (len(utility_score), 3):
        raise ValueError("Expected aligned finite three-class tier probabilities")
    rank = percentile_rank(utility_score, thread_ids)
    pool_size = max(1, int(np.ceil(len(rank) * pool_fraction)))
    high_order = sorted(range(len(rank)), key=lambda i: (-probabilities[i, 0], str(thread_ids[i])))
    in_pool = np.zeros(len(rank), dtype=np.float32)
    in_pool[high_order[:pool_size]] = 1.0
    return rank + float(bonus) * in_pool
