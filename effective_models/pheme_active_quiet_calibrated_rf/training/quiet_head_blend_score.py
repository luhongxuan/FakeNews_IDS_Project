"""Outcome-free bounded quiet-score blend for the exhaustive head model."""
from __future__ import annotations

import numpy as np

from quiet_tier_hybrid_score import percentile_rank


def bounded_head_blend(baseline: np.ndarray, head: np.ndarray, thread_ids: list[str], weight: float, maximum_rank_shift: float) -> np.ndarray:
    """Blend quiet ranks while bounding each head-model correction.

    A zero weight is exactly the unmodified quiet-RF score, so inner selection
    can decline the head model without any hidden re-scaling.
    """
    baseline = np.asarray(baseline, dtype=np.float32)
    head = np.asarray(head, dtype=np.float32)
    if baseline.shape != head.shape or len(baseline) != len(thread_ids):
        raise ValueError("baseline, head, and thread IDs must align")
    if not 0.0 <= weight <= 1.0 or not 0.0 <= maximum_rank_shift <= 1.0:
        raise ValueError("weight and maximum_rank_shift must lie in [0, 1]")
    if not np.isfinite(baseline).all() or not np.isfinite(head).all():
        raise ValueError("blend scores must be finite")
    if weight == 0.0:
        return baseline.copy()
    base_rank = percentile_rank(baseline, thread_ids)
    head_rank = percentile_rank(head, thread_ids)
    correction = np.clip(head_rank - base_rank, -maximum_rank_shift, maximum_rank_shift)
    return base_rank + weight * correction
