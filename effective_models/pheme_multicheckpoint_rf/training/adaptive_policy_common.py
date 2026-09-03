"""Split-safe adaptive cumulative intervention policy utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

import multicheckpoint_model_common as model_common


CHECKPOINTS = (600, 1200, 1800, 2400, 3000, 3600)
MAX_BUDGETS = (10, 20, 30, 40, 50)
IMPACT_THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0)
TARGET_REDUCTIONS = (0.20, 0.25, 0.30)
COST_WEIGHTS = (0.05, 0.10, 0.20)
TARGET_WORST_EVENT_FRACTION = 0.50
BALANCED_QUOTAS_50 = (9, 9, 8, 8, 8, 8)


@dataclass(frozen=True)
class AdaptiveConfig:
    max_budget: int
    min_predicted_impact: float

    @property
    def config_id(self) -> str:
        threshold = str(self.min_predicted_impact).replace(".", "p")
        return f"cap{self.max_budget}_impact{threshold}"


def candidate_configs() -> list[AdaptiveConfig]:
    return [
        AdaptiveConfig(max_budget=budget, min_predicted_impact=threshold)
        for budget in MAX_BUDGETS
        for threshold in IMPACT_THRESHOLDS
    ]


def validate_scored_frame(frame: pd.DataFrame) -> None:
    required = {
        "thread_id", "event_id", "checkpoint_sec", "prediction",
        "dynamic_preventable_impact", "future_growth",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing scored columns: {sorted(missing)}")
    if frame.duplicated(["thread_id", "event_id", "checkpoint_sec"]).any():
        raise ValueError("duplicate event/thread/checkpoint score")
    if set(frame.checkpoint_sec.astype(int)) != set(CHECKPOINTS):
        raise ValueError("unexpected checkpoint coverage")
    numeric = frame[["prediction", "dynamic_preventable_impact", "future_growth"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("non-finite score or outcome")
    if (frame.dynamic_preventable_impact < 0).any() or (frame.future_growth < 0).any():
        raise ValueError("negative outcome")
    if (frame.dynamic_preventable_impact > frame.future_growth).any():
        raise ValueError("preventable impact exceeds observable future growth")


def fit_inner_oof(
    frame: pd.DataFrame,
    outer_event: str,
    inner_events: Iterable[str],
    columns: list[str],
    model_params: dict,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Predict inner events with models that exclude both outer and inner events."""
    if outer_event not in set(frame.event_id):
        raise ValueError(f"unknown outer event: {outer_event}")
    chunks = []
    inner_events = list(inner_events)
    for index, inner_event in enumerate(inner_events, 1):
        if inner_event == outer_event:
            raise ValueError("outer event cannot be an inner validation event")
        train = frame.loc[
            frame.event_id.ne(outer_event) & frame.event_id.ne(inner_event)
        ].copy()
        validation = frame.loc[frame.event_id.eq(inner_event)].copy()
        if not len(train) or not len(validation):
            raise ValueError(f"empty nested fold for {outer_event}/{inner_event}")
        if outer_event in set(train.event_id) or inner_event in set(train.event_id):
            raise ValueError("outer or inner event leaked into nested training rows")
        if progress:
            progress(
                f"    inner event {index}/{len(inner_events)}: {inner_event} "
                f"(train events={train.event_id.nunique()})"
            )
        _, prediction = model_common.fit_predict(
            train, validation, columns, params=model_params
        )
        scored = validation[[
            "thread_id", "event_id", "checkpoint_sec",
            "dynamic_preventable_impact", "future_growth",
        ]].copy()
        scored["prediction"] = prediction
        scored["outer_event"] = outer_event
        chunks.append(scored)
    result = pd.concat(chunks, ignore_index=True)
    validate_scored_frame(result.drop(columns="outer_event"))
    if outer_event in set(result.event_id):
        raise ValueError("outer event appeared in inner OOF predictions")
    return result


def adaptive_select(event: pd.DataFrame, config: AdaptiveConfig) -> pd.DataFrame:
    """Select only score-qualified threads, never forcing the cap to be filled."""
    events = set(event.event_id)
    if len(events) != 1:
        raise ValueError("adaptive_select requires exactly one event")
    selected_ids: set[str] = set()
    chunks = []
    for checkpoint in CHECKPOINTS:
        remaining = config.max_budget - len(selected_ids)
        if remaining <= 0:
            break
        available = event.loc[
            event.checkpoint_sec.eq(checkpoint) & ~event.thread_id.isin(selected_ids)
        ].copy()
        available["predicted_impact"] = np.maximum(
            np.expm1(available.prediction.to_numpy(dtype=float)), 0.0
        )
        qualified = available.loc[
            available.predicted_impact.ge(config.min_predicted_impact)
        ].sort_values(
            ["prediction", "thread_id"], ascending=[False, True], kind="stable"
        )
        chosen = qualified.head(remaining).copy()
        if len(chosen):
            chosen["action_checkpoint_sec"] = checkpoint
            chosen["config_id"] = config.config_id
            chunks.append(chosen)
            selected_ids.update(chosen.thread_id)
    if not chunks:
        empty = event.head(0).copy()
        empty["predicted_impact"] = pd.Series(dtype=float)
        empty["action_checkpoint_sec"] = pd.Series(dtype=int)
        empty["config_id"] = pd.Series(dtype=str)
        return empty
    selected = pd.concat(chunks, ignore_index=True)
    if selected.thread_id.duplicated().any() or len(selected) > config.max_budget:
        raise ValueError("adaptive selection violated identity or budget")
    return selected


def evaluate_configs(
    scored: pd.DataFrame,
    configs: Iterable[AdaptiveConfig] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a fixed policy grid on event-separated validation predictions."""
    validate_scored_frame(scored)
    metric_rows = []
    selected_rows = []
    for event_id, event in scored.groupby("event_id", sort=True):
        horizon = float(
            event.loc[event.checkpoint_sec.eq(CHECKPOINTS[0]), "future_growth"].sum()
        )
        if horizon <= 0:
            raise ValueError(f"non-positive 10-minute horizon for {event_id}")
        for config in configs or candidate_configs():
            selected = adaptive_select(event, config)
            blocked = float(selected.dynamic_preventable_impact.sum())
            metric_rows.append({
                "event_id": event_id,
                "config_id": config.config_id,
                "max_budget": config.max_budget,
                "min_predicted_impact": config.min_predicted_impact,
                "interventions": len(selected),
                "positive_interventions": int(
                    selected.dynamic_preventable_impact.gt(0).sum()
                ),
                "blocked_impact": blocked,
                "horizon_future_at_10m": horizon,
                "horizon_reduction": blocked / horizon,
            })
            if len(selected):
                selected_rows.append(selected.assign(policy_event=event_id)[[
                    "policy_event", "config_id", "thread_id", "event_id",
                    "action_checkpoint_sec", "prediction", "predicted_impact",
                    "dynamic_preventable_impact", "future_growth",
                ]])
    metrics = pd.DataFrame(metric_rows)
    selected = (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows else pd.DataFrame()
    )
    return metrics, selected


def summarize_grid(metrics: pd.DataFrame) -> pd.DataFrame:
    summary = metrics.groupby(
        ["config_id", "max_budget", "min_predicted_impact"], as_index=False
    ).agg(
        validation_events=("event_id", "nunique"),
        mean_interventions=("interventions", "mean"),
        max_interventions=("interventions", "max"),
        mean_positive_interventions=("positive_interventions", "mean"),
        mean_blocked_impact=("blocked_impact", "mean"),
        mean_horizon_reduction=("horizon_reduction", "mean"),
        worst_event_horizon_reduction=("horizon_reduction", "min"),
    )
    return summary.sort_values(
        ["mean_interventions", "mean_horizon_reduction", "config_id"],
        ascending=[True, False, True], kind="stable",
    ).reset_index(drop=True)


def pareto_frontier(macro: pd.DataFrame) -> pd.DataFrame:
    """Return policies not dominated on mean interventions and reduction."""
    required = {"policy_name", "mean_interventions", "mean_horizon_reduction"}
    missing = required - set(macro.columns)
    if missing:
        raise ValueError(f"missing Pareto columns: {sorted(missing)}")
    keep = []
    for row in macro.itertuples(index=False):
        no_more_actions = macro.mean_interventions.le(float(row.mean_interventions))
        no_less_reduction = macro.mean_horizon_reduction.ge(
            float(row.mean_horizon_reduction)
        )
        strictly_better = (
            macro.mean_interventions.lt(float(row.mean_interventions))
            | macro.mean_horizon_reduction.gt(float(row.mean_horizon_reduction))
        )
        dominated = bool((no_more_actions & no_less_reduction & strictly_better).any())
        keep.append(not dominated)
    return macro.loc[keep].sort_values(
        ["mean_interventions", "mean_horizon_reduction", "policy_name"],
        ascending=[True, False, True], kind="stable",
    ).reset_index(drop=True)


def select_policies(summary: pd.DataFrame) -> pd.DataFrame:
    """Select target-constrained and cost-penalized policies using inner results only."""
    selected = []
    for target in TARGET_REDUCTIONS:
        robust_floor = target * TARGET_WORST_EVENT_FRACTION
        feasible = summary.loc[
            summary.mean_horizon_reduction.ge(target)
            & summary.worst_event_horizon_reduction.ge(robust_floor)
        ].copy()
        used_fallback = feasible.empty
        if used_fallback:
            feasible = summary.copy()
            chosen = feasible.sort_values(
                ["mean_horizon_reduction", "worst_event_horizon_reduction",
                 "mean_interventions", "config_id"],
                ascending=[False, False, True, True], kind="stable",
            ).iloc[0]
        else:
            chosen = feasible.sort_values(
                ["mean_interventions", "worst_event_horizon_reduction",
                 "mean_horizon_reduction", "config_id"],
                ascending=[True, False, False, True], kind="stable",
            ).iloc[0]
        row = chosen.to_dict()
        row.update({
            "policy_name": f"target_{int(round(target * 100))}",
            "selection_type": "minimum interventions subject to target and worst-event floor",
            "target_reduction": target,
            "cost_weight": np.nan,
            "used_fallback": used_fallback,
        })
        selected.append(row)
    for cost in COST_WEIGHTS:
        candidates = summary.copy()
        candidates["cost_utility"] = (
            candidates.mean_horizon_reduction
            - cost * candidates.mean_interventions / max(MAX_BUDGETS)
        )
        chosen = candidates.sort_values(
            ["cost_utility", "worst_event_horizon_reduction",
             "mean_interventions", "config_id"],
            ascending=[False, False, True, True], kind="stable",
        ).iloc[0]
        row = chosen.to_dict()
        row.update({
            "policy_name": f"cost_{str(cost).replace('.', 'p')}",
            "selection_type": "maximize reduction minus normalized intervention cost",
            "target_reduction": np.nan,
            "cost_weight": cost,
            "used_fallback": False,
        })
        selected.append(row)
    result = pd.DataFrame(selected)
    if result.policy_name.duplicated().any():
        raise ValueError("duplicate selected policy name")
    return result


def apply_selected_policies(
    outer_scores: pd.DataFrame,
    selected_configs: pd.DataFrame,
    outer_event: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event = outer_scores.loc[outer_scores.event_id.eq(outer_event)].copy()
    if event.empty:
        raise ValueError(f"missing outer scores for {outer_event}")
    horizon = float(
        event.loc[event.checkpoint_sec.eq(CHECKPOINTS[0]), "future_growth"].sum()
    )
    metric_rows = []
    selected_rows = []
    for row in selected_configs.itertuples(index=False):
        config = AdaptiveConfig(
            max_budget=int(row.max_budget),
            min_predicted_impact=float(row.min_predicted_impact),
        )
        chosen = adaptive_select(event, config)
        blocked = float(chosen.dynamic_preventable_impact.sum())
        metric_rows.append({
            "outer_event": outer_event,
            "policy_name": row.policy_name,
            "config_id": config.config_id,
            "max_budget": config.max_budget,
            "min_predicted_impact": config.min_predicted_impact,
            "interventions": len(chosen),
            "positive_interventions": int(chosen.dynamic_preventable_impact.gt(0).sum()),
            "blocked_impact": blocked,
            "horizon_future_at_10m": horizon,
            "horizon_reduction": blocked / horizon if horizon else 0.0,
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
                "action_checkpoint_sec", "prediction", "predicted_impact",
                "dynamic_preventable_impact", "future_growth",
            ]])
    return pd.DataFrame(metric_rows), (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows else pd.DataFrame()
    )


def fixed_balanced_select(event: pd.DataFrame) -> pd.DataFrame:
    selected_ids: set[str] = set()
    chunks = []
    for checkpoint, quota in zip(CHECKPOINTS, BALANCED_QUOTAS_50):
        available = event.loc[
            event.checkpoint_sec.eq(checkpoint) & ~event.thread_id.isin(selected_ids)
        ].sort_values(
            ["prediction", "thread_id"], ascending=[False, True], kind="stable"
        )
        chosen = available.head(quota).copy()
        if len(chosen) != quota:
            raise ValueError("insufficient candidates for fixed balanced baseline")
        chosen["action_checkpoint_sec"] = checkpoint
        chunks.append(chosen)
        selected_ids.update(chosen.thread_id)
    result = pd.concat(chunks, ignore_index=True)
    if len(result) != 50 or result.thread_id.duplicated().any():
        raise ValueError("invalid fixed balanced baseline selection")
    return result


def load_outer_oof(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, dtype={"thread_id": str, "event_id": str})
    scores = scores.loc[scores.family.eq("cumulative")].drop(columns="family")
    validate_scored_frame(scores)
    return scores
