"""Opportunity-reserve policy over saved split-safe cumulative RF scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import adaptive_policy_common as adaptive


RELEASE_PROFILES = {
    "unrestricted": (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    "balanced": (0.18, 0.36, 0.52, 0.68, 0.84, 1.00),
    "backloaded": (0.10, 0.22, 0.38, 0.58, 0.78, 1.00),
    "strong_reserve": (0.05, 0.15, 0.30, 0.50, 0.75, 1.00),
}
EARLY_PREMIUM_PROFILES = {
    "flat": (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    "moderate": (2.00, 1.70, 1.40, 1.20, 1.10, 1.00),
    "strong": (4.00, 3.00, 2.20, 1.60, 1.20, 1.00),
}
MAX_BUDGETS = (20, 30, 40, 50)
BASE_IMPACT_THRESHOLDS = (0.0, 3.0, 5.0, 8.0, 13.0)


@dataclass(frozen=True)
class ReserveConfig:
    max_budget: int
    min_predicted_impact: float
    release_profile: str
    early_premium_profile: str

    @property
    def config_id(self) -> str:
        threshold = str(self.min_predicted_impact).replace(".", "p")
        return (
            f"cap{self.max_budget}_impact{threshold}_"
            f"release-{self.release_profile}_premium-{self.early_premium_profile}"
        )


def candidate_configs() -> list[ReserveConfig]:
    configs = []
    for budget in MAX_BUDGETS:
        for threshold in BASE_IMPACT_THRESHOLDS:
            for release in RELEASE_PROFILES:
                for premium in EARLY_PREMIUM_PROFILES:
                    # Multiplying a zero threshold gives identical policies.
                    if threshold == 0.0 and premium != "flat":
                        continue
                    configs.append(ReserveConfig(budget, threshold, release, premium))
    if len(configs) != 208 or len({config.config_id for config in configs}) != len(configs):
        raise ValueError("unexpected opportunity-reserve configuration grid")
    return configs


def reserve_select(event: pd.DataFrame, config: ReserveConfig) -> pd.DataFrame:
    if len(set(event.event_id)) != 1:
        raise ValueError("reserve_select requires exactly one event")
    release = RELEASE_PROFILES[config.release_profile]
    premiums = EARLY_PREMIUM_PROFILES[config.early_premium_profile]
    selected_ids: set[str] = set()
    chunks = []
    for index, checkpoint in enumerate(adaptive.CHECKPOINTS):
        released_cap = min(
            config.max_budget,
            int(np.ceil(config.max_budget * release[index])),
        )
        available_slots = released_cap - len(selected_ids)
        if available_slots <= 0:
            continue
        available = event.loc[
            event.checkpoint_sec.eq(checkpoint) & ~event.thread_id.isin(selected_ids)
        ].copy()
        available["predicted_impact"] = np.maximum(
            np.expm1(available.prediction.to_numpy(dtype=float)), 0.0
        )
        threshold = config.min_predicted_impact * premiums[index]
        qualified = available.loc[
            available.predicted_impact.ge(threshold)
        ].sort_values(
            ["prediction", "thread_id"], ascending=[False, True], kind="stable"
        )
        chosen = qualified.head(available_slots).copy()
        if len(chosen):
            chosen["action_checkpoint_sec"] = checkpoint
            chosen["released_cap"] = released_cap
            chosen["applied_impact_threshold"] = threshold
            chosen["config_id"] = config.config_id
            chunks.append(chosen)
            selected_ids.update(chosen.thread_id)
    if not chunks:
        empty = event.head(0).copy()
        for name, dtype in (
            ("predicted_impact", float), ("action_checkpoint_sec", int),
            ("released_cap", int), ("applied_impact_threshold", float),
            ("config_id", str),
        ):
            empty[name] = pd.Series(dtype=dtype)
        return empty
    selected = pd.concat(chunks, ignore_index=True)
    if selected.thread_id.duplicated().any() or len(selected) > config.max_budget:
        raise ValueError("opportunity-reserve selection violated identity or budget")
    return selected


def evaluate_configs(
    scored: pd.DataFrame,
    configs: Iterable[ReserveConfig] | None = None,
) -> pd.DataFrame:
    adaptive.validate_scored_frame(scored)
    rows = []
    for event_id, event in scored.groupby("event_id", sort=True):
        horizon = float(
            event.loc[event.checkpoint_sec.eq(adaptive.CHECKPOINTS[0]), "future_growth"].sum()
        )
        if horizon <= 0:
            raise ValueError(f"non-positive 10-minute horizon for {event_id}")
        for config in configs or candidate_configs():
            selected = reserve_select(event, config)
            blocked = float(selected.dynamic_preventable_impact.sum())
            rows.append({
                "event_id": event_id,
                "config_id": config.config_id,
                "max_budget": config.max_budget,
                "min_predicted_impact": config.min_predicted_impact,
                "release_profile": config.release_profile,
                "early_premium_profile": config.early_premium_profile,
                "interventions": len(selected),
                "positive_interventions": int(
                    selected.dynamic_preventable_impact.gt(0).sum()
                ),
                "blocked_impact": blocked,
                "horizon_future_at_10m": horizon,
                "horizon_reduction": blocked / horizon,
                "mean_action_minute": (
                    float(selected.action_checkpoint_sec.mean() / 60.0)
                    if len(selected) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def summarize_grid(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "config_id", "max_budget", "min_predicted_impact",
        "release_profile", "early_premium_profile",
    ]
    return metrics.groupby(keys, as_index=False).agg(
        validation_events=("event_id", "nunique"),
        mean_interventions=("interventions", "mean"),
        max_interventions=("interventions", "max"),
        mean_positive_interventions=("positive_interventions", "mean"),
        mean_blocked_impact=("blocked_impact", "mean"),
        mean_horizon_reduction=("horizon_reduction", "mean"),
        worst_event_horizon_reduction=("horizon_reduction", "min"),
        mean_action_minute=("mean_action_minute", "mean"),
    ).sort_values(
        ["mean_interventions", "mean_horizon_reduction", "config_id"],
        ascending=[True, False, True], kind="stable",
    ).reset_index(drop=True)


def apply_selected_policies(
    outer_scores: pd.DataFrame,
    selected_configs: pd.DataFrame,
    outer_event: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event = outer_scores.loc[outer_scores.event_id.eq(outer_event)].copy()
    if event.empty:
        raise ValueError(f"missing outer scores for {outer_event}")
    horizon = float(
        event.loc[event.checkpoint_sec.eq(adaptive.CHECKPOINTS[0]), "future_growth"].sum()
    )
    metric_rows = []
    selected_rows = []
    for row in selected_configs.itertuples(index=False):
        config = ReserveConfig(
            max_budget=int(row.max_budget),
            min_predicted_impact=float(row.min_predicted_impact),
            release_profile=str(row.release_profile),
            early_premium_profile=str(row.early_premium_profile),
        )
        chosen = reserve_select(event, config)
        blocked = float(chosen.dynamic_preventable_impact.sum())
        metric_rows.append({
            "outer_event": outer_event,
            "policy_name": row.policy_name,
            "config_id": config.config_id,
            "max_budget": config.max_budget,
            "min_predicted_impact": config.min_predicted_impact,
            "release_profile": config.release_profile,
            "early_premium_profile": config.early_premium_profile,
            "interventions": len(chosen),
            "positive_interventions": int(chosen.dynamic_preventable_impact.gt(0).sum()),
            "blocked_impact": blocked,
            "horizon_future_at_10m": horizon,
            "horizon_reduction": blocked / horizon if horizon else 0.0,
            "mean_action_minute": (
                float(chosen.action_checkpoint_sec.mean() / 60.0)
                if len(chosen) else np.nan
            ),
            "inner_mean_interventions": float(row.mean_interventions),
            "inner_mean_horizon_reduction": float(row.mean_horizon_reduction),
            "inner_worst_event_horizon_reduction": float(
                row.worst_event_horizon_reduction
            ),
            "used_fallback": bool(row.used_fallback),
        })
        if len(chosen):
            selected_rows.append(chosen.assign(
                outer_event=outer_event, policy_name=row.policy_name
            )[[
                "outer_event", "policy_name", "config_id", "thread_id", "event_id",
                "action_checkpoint_sec", "released_cap", "applied_impact_threshold",
                "prediction", "predicted_impact", "dynamic_preventable_impact",
                "future_growth",
            ]])
    return pd.DataFrame(metric_rows), (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows else pd.DataFrame()
    )
