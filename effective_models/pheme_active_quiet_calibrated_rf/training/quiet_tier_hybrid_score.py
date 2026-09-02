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
    utility_rank = percentile_rank(utility_score, thread_ids)
    tier_rank = percentile_rank(tier_priority(tier_probabilities), thread_ids)
    return utility_weight * utility_rank + (1.0 - utility_weight) * tier_rank
