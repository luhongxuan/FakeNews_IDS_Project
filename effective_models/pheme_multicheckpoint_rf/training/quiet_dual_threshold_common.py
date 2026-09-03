"""Two-score, threshold-only qualification using saved nested OOF predictions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CHECKPOINTS = (1200, 1800)
ACCELERATION_QUANTILES = (0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
UTILITY_THRESHOLDS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0)
CAPTURE_TARGETS = (0.25, 0.40, 0.50, 0.60)


@dataclass(frozen=True)
class DualThresholdConfig:
    acceleration_quantile: float
    min_predicted_impact: float

    @property
    def config_id(self) -> str:
        q = str(self.acceleration_quantile).replace(".", "p")
        u = str(self.min_predicted_impact).replace(".", "p")
        return f"accq{q}_impact{u}"


def validate_scored(frame: pd.DataFrame, inner: bool) -> None:
    required = {
        "thread_id", "event_id", "checkpoint_sec", "acceleration_prediction",
        "utility_prediction", "predicted_positive_acceleration",
        "predicted_impact", "dynamic_preventable_impact", "future_growth",
    }
    if inner:
        required.add("outer_event")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing dual-score columns: {sorted(missing)}")
    keys = ["thread_id", "event_id", "checkpoint_sec"]
    if inner:
        keys.insert(0, "outer_event")
    if frame.duplicated(keys).any():
        raise ValueError("duplicate dual-score key")
    numeric = frame[[
        "acceleration_prediction", "utility_prediction",
        "predicted_positive_acceleration", "predicted_impact",
        "dynamic_preventable_impact", "future_growth",
    ]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric[:, 2:] < 0).any():
        raise ValueError("invalid dual-score value or outcome")
    if inner and (frame.outer_event == frame.event_id).any():
        raise ValueError("outer event leaked into nested inner scores")


def acceleration_thresholds(frame: pd.DataFrame, quantile: float) -> dict[int, float]:
    if not 0 <= quantile <= 1:
        raise ValueError("invalid acceleration quantile")
    result = {}
    for checkpoint in CHECKPOINTS:
        values = frame.loc[
            frame.checkpoint_sec.eq(checkpoint), "acceleration_prediction"
        ]
        if values.empty:
            raise ValueError(f"missing checkpoint {checkpoint}")
        result[checkpoint] = float(values.quantile(quantile))
    return result


def apply_config(
    event: pd.DataFrame,
    config: DualThresholdConfig,
    acceleration_cutoffs: dict[int, float],
    policy_name: str | None = None,
) -> pd.DataFrame:
    """Intervene on every dual-qualified row, with no Top-K and no repeats."""
    if event.event_id.nunique() != 1:
        raise ValueError("dual-threshold replay requires one event")
    selected_ids: set[str] = set()
    chunks = []
    for checkpoint in CHECKPOINTS:
        available = event.loc[
            event.checkpoint_sec.eq(checkpoint) & ~event.thread_id.isin(selected_ids)
        ].copy()
        chosen = available.loc[
            available.acceleration_prediction.ge(acceleration_cutoffs[checkpoint])
            & available.predicted_impact.ge(config.min_predicted_impact)
        ].copy()
        if len(chosen):
            chosen["action_checkpoint_sec"] = checkpoint
            chosen["acceleration_threshold"] = acceleration_cutoffs[checkpoint]
            chosen["min_predicted_impact"] = config.min_predicted_impact
            chosen["config_id"] = config.config_id
            chosen["policy_name"] = policy_name or config.config_id
            chunks.append(chosen)
            selected_ids.update(chosen.thread_id)
    if not chunks:
        return event.head(0).assign(
            action_checkpoint_sec=pd.Series(dtype=int),
            acceleration_threshold=pd.Series(dtype=float),
            min_predicted_impact=pd.Series(dtype=float),
            config_id=pd.Series(dtype=str), policy_name=pd.Series(dtype=str),
        )
    result = pd.concat(chunks, ignore_index=True)
    if result.thread_id.duplicated().any():
        raise ValueError("dual threshold selected a thread twice")
    return result


def evaluate_inner_grid(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed candidate configurations entirely within nested inner OOF."""
    rows = []
    for quantile in ACCELERATION_QUANTILES:
        cutoffs = acceleration_thresholds(frame, quantile)
        for utility in UTILITY_THRESHOLDS:
            config = DualThresholdConfig(quantile, utility)
            for event_id, event in frame.groupby("event_id", sort=True):
                selected = apply_config(event, config, cutoffs)
                reference = apply_config(
                    event, DualThresholdConfig(0.0, 0.0),
                    {checkpoint: -np.inf for checkpoint in CHECKPOINTS},
                )
                reference_impact = float(reference.dynamic_preventable_impact.sum())
                blocked = float(selected.dynamic_preventable_impact.sum())
                rows.append({
                    "event_id": event_id, "config_id": config.config_id,
                    "acceleration_quantile": quantile,
                    "min_predicted_impact": utility,
                    "acceleration_threshold_20m": cutoffs[1200],
                    "acceleration_threshold_30m": cutoffs[1800],
                    "interventions": len(selected),
                    "blocked_impact": blocked,
                    "all_quiet_reference_impact": reference_impact,
                    "quiet_impact_capture": (
                        blocked / reference_impact if reference_impact else 0.0
                    ),
                })
    per_event = pd.DataFrame(rows)
    macro = per_event.groupby(
        ["config_id", "acceleration_quantile", "min_predicted_impact",
         "acceleration_threshold_20m", "acceleration_threshold_30m"],
        as_index=False,
    ).agg(
        validation_events=("event_id", "nunique"),
        mean_interventions=("interventions", "mean"),
        max_interventions=("interventions", "max"),
        mean_blocked_impact=("blocked_impact", "mean"),
        mean_quiet_impact_capture=("quiet_impact_capture", "mean"),
        worst_event_quiet_impact_capture=("quiet_impact_capture", "min"),
    )
    return per_event, macro


def select_capture_policies(macro: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for target in CAPTURE_TARGETS:
        feasible = macro.loc[macro.mean_quiet_impact_capture.ge(target)].copy()
        if feasible.empty:
            raise ValueError(f"no dual-threshold config reaches capture target {target}")
        chosen = feasible.sort_values(
            ["mean_interventions", "worst_event_quiet_impact_capture",
             "mean_quiet_impact_capture", "min_predicted_impact",
             "acceleration_quantile", "config_id"],
            ascending=[True, False, False, False, False, True], kind="stable",
        ).iloc[0]
        row = chosen.to_dict()
        row["policy_name"] = f"inner_impact_capture_{int(target * 100)}"
        row["target_quiet_impact_capture"] = target
        selected.append(row)
    return pd.DataFrame(selected)


def pareto_frontier(macro: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for row in macro.itertuples(index=False):
        dominates = (
            macro.mean_interventions.le(float(row.mean_interventions))
            & macro.mean_quiet_impact_capture.ge(float(row.mean_quiet_impact_capture))
            & (
                macro.mean_interventions.lt(float(row.mean_interventions))
                | macro.mean_quiet_impact_capture.gt(float(row.mean_quiet_impact_capture))
            )
        )
        keep.append(not bool(dominates.any()))
    return macro.loc[keep].sort_values(
        ["mean_interventions", "mean_quiet_impact_capture"],
        ascending=[True, False], kind="stable",
    ).reset_index(drop=True)
