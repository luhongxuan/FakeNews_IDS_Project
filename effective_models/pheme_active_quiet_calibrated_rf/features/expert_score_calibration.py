"""Train-only, monotonic event-balanced calibration for PHEME expert scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from effective_models.pheme_active_quiet_calibrated_rf import config
from effective_models.pheme_active_quiet_calibrated_rf.features.quiet_active_gate_features import (
    gated_predict, recency_values,
)


CALIBRATION_ALPHA = config.CALIBRATION_ALPHA


def inner_oof_predictions(outer_train: list, depth_nodes, account_age_nodes, node_offsets, model_params: dict | None = None, progress_prefix: str = "") -> pd.DataFrame:
    """Generate raw expert OOF scores using only events within the outer train fold."""
    events = sorted({str(graph.event_id) for graph in outer_train})
    rows = []
    for index, validation_event in enumerate(events, 1):
        inner_train = [graph for graph in outer_train if str(graph.event_id) != validation_event]
        validation = [graph for graph in outer_train if str(graph.event_id) == validation_event]
        print(f"{progress_prefix}calibration inner {index}/{len(events)}: validation={validation_event}", flush=True)
        raw_score, gate, _ = gated_predict(inner_train, validation, depth_nodes, account_age_nodes, node_offsets, model_params)
        quiet = recency_values(validation) >= gate["threshold"]
        rows.extend({
            "validation_event": validation_event,
            "thread_id": str(graph.thread_id),
            "expert": "quiet" if is_quiet else "active",
            "raw_score": float(score),
            "target": float(graph.preventable_y.item()),
        } for graph, score, is_quiet in zip(validation, raw_score, quiet))
    frame = pd.DataFrame(rows)
    if len(frame) != len(outer_train) or frame.thread_id.duplicated().any() or not np.isfinite(frame[["raw_score", "target"]].to_numpy(dtype=float)).all():
        raise ValueError("Invalid inner OOF calibration predictions")
    # Every event contributes equal total weight, then weights are rescaled to
    # mean 1 so Ridge's alpha keeps its documented scale as sample count varies.
    event_count = frame.validation_event.nunique()
    frame["event_weight"] = frame.groupby("validation_event")["thread_id"].transform(
        lambda value: len(frame) / (event_count * len(value))
    )
    return frame


def fit_calibrators(oof: pd.DataFrame) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Fit one non-rank-reversing calibration line per expert from train-only OOF scores."""
    models: dict[str, dict[str, float]] = {}
    details: list[dict] = []
    for expert in ("active", "quiet"):
        subset = oof[oof.expert == expert]
        if len(subset) < 20:
            raise ValueError(f"Too few train-only OOF calibration rows for {expert}: {len(subset)}")
        ridge = Ridge(alpha=CALIBRATION_ALPHA).fit(
            subset[["raw_score"]].to_numpy(dtype=float),
            subset.target.to_numpy(dtype=float),
            sample_weight=subset.event_weight.to_numpy(dtype=float),
        )
        raw_coefficient = float(ridge.coef_[0])
        # Calibration may change score scale but must never reverse an expert's
        # within-group order.  A non-positive OOF relationship is recorded and
        # uses the identity map rather than fabricating an inverted ranking.
        if raw_coefficient > 0.0:
            coefficient, intercept, method = raw_coefficient, float(ridge.intercept_), "event_balanced_ridge_positive_slope"
        else:
            coefficient, intercept, method = 1.0, 0.0, "identity_fallback_nonpositive_oof_slope"
        models[expert] = {"coefficient": coefficient, "intercept": intercept}
        details.append({
            "expert": expert,
            "calibration_method": method,
            "ridge_alpha": CALIBRATION_ALPHA,
            "oof_threads": len(subset),
            "oof_events": int(subset.validation_event.nunique()),
            "raw_ridge_coefficient": raw_coefficient,
            "coefficient": coefficient,
            "intercept": intercept,
        })
    return models, details


def calibrated_scores(raw_score: np.ndarray, test_graphs: list, gate: dict, calibrators: dict[str, dict[str, float]]) -> np.ndarray:
    """Apply expert-specific calibrators to a final held-out event without its labels."""
    quiet = recency_values(test_graphs) >= float(gate["threshold"])
    result = np.empty(len(test_graphs), dtype=np.float32)
    for expert, mask in (("quiet", quiet), ("active", ~quiet)):
        if mask.any():
            calibrator = calibrators[expert]
            result[mask] = calibrator["coefficient"] * np.asarray(raw_score)[mask] + calibrator["intercept"]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite calibrated held-out score")
    return result


def calibration_safety_statement() -> str:
    return (
        "For each outer fold, expert-specific Ridge calibration is fitted only on inner-LOEO OOF predictions from outer-train events. "
        "The held-out event contributes no target, score, threshold, or sample weight to calibration fitting. "
        "Ridge sample weights balance inner validation events rather than letting large events dominate; a non-positive OOF slope uses an explicit identity fallback so calibration cannot reverse within-expert ranking."
    )
