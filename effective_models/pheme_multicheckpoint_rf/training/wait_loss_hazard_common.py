"""Next-10-minute wait-loss RF and two-signal intervention policy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

import adaptive_policy_common as adaptive


HAZARD_MODEL_PARAMS = {
    "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 3,
    "max_features": 0.8, "n_jobs": 1, "random_state": 42,
}
MAX_BUDGETS = (20, 30, 40, 50)
UTILITY_THRESHOLDS = (3.0, 5.0, 8.0)
HAZARD_THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)


@dataclass(frozen=True)
class HazardConfig:
    max_budget: int
    min_predicted_impact: float
    min_predicted_wait_loss: float

    @property
    def config_id(self) -> str:
        utility = str(self.min_predicted_impact).replace(".", "p")
        hazard = str(self.min_predicted_wait_loss).replace(".", "p")
        return f"cap{self.max_budget}_utility{utility}_hazard{hazard}"


def candidate_configs() -> list[HazardConfig]:
    configs = [
        HazardConfig(budget, utility, hazard)
        for budget in MAX_BUDGETS
        for utility in UTILITY_THRESHOLDS
        for hazard in HAZARD_THRESHOLDS
    ]
    if len(configs) != 72:
        raise ValueError("unexpected hazard policy grid")
    return configs


def prediction_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "checkpoint_sec", "next10_wait_loss", "hazard_prediction"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"missing hazard metric columns: {sorted(missing)}")
    rows = []
    for (event_id, checkpoint), group in scored.groupby(
        ["event_id", "checkpoint_sec"], sort=True
    ):
        actual = group.next10_wait_loss.to_numpy(dtype=float)
        prediction = np.maximum(np.expm1(group.hazard_prediction.to_numpy(dtype=float)), 0.0)
        positive = actual > 0
        order = np.argsort(-prediction, kind="stable")[: min(20, len(group))]
        oracle = np.argsort(-actual, kind="stable")[: min(20, len(group))]
        oracle_loss = float(actual[oracle].sum())
        spearman = pd.Series(actual).corr(pd.Series(prediction), method="spearman")
        rows.append({
            "event_id": event_id, "checkpoint_sec": int(checkpoint), "rows": len(group),
            "positive_rows": int(positive.sum()),
            "mae_wait_loss": float(np.mean(np.abs(actual - prediction))),
            "spearman": float(spearman) if np.isfinite(spearman) else 0.0,
            "roc_auc_positive": (
                float(roc_auc_score(positive, prediction))
                if 0 < positive.sum() < len(positive) else np.nan
            ),
            "pr_auc_positive": (
                float(average_precision_score(positive, prediction))
                if 0 < positive.sum() < len(positive) else np.nan
            ),
            "top20_wait_loss_capture": (
                float(actual[order].sum() / oracle_loss) if oracle_loss else 0.0
            ),
        })
    return pd.DataFrame(rows)


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str], params: dict,
) -> tuple[RandomForestRegressor, np.ndarray]:
    if set(train.event_id) & set(test.event_id):
        raise ValueError("event overlap in hazard fit")
    x_train = train[columns].to_numpy(dtype=np.float32)
    x_test = test[columns].to_numpy(dtype=np.float32)
    y_train = train.next10_wait_loss_y.to_numpy(dtype=np.float32)
    if not all(np.isfinite(x).all() for x in (x_train, x_test, y_train)):
        raise ValueError("non-finite hazard array")
    model = RandomForestRegressor(**params).fit(x_train, y_train)
    prediction = model.predict(x_test)
    if prediction.shape != (len(test),) or not np.isfinite(prediction).all():
        raise ValueError("invalid hazard prediction")
    return model, prediction


def hazard_select(event: pd.DataFrame, config: HazardConfig) -> pd.DataFrame:
    if len(set(event.event_id)) != 1:
        raise ValueError("hazard_select requires one event")
    selected_ids: set[str] = set()
    chunks = []
    for checkpoint in adaptive.CHECKPOINTS:
        remaining = config.max_budget - len(selected_ids)
        if remaining <= 0:
            break
        available = event.loc[
            event.checkpoint_sec.eq(checkpoint) & ~event.thread_id.isin(selected_ids)
        ].copy()
        available["predicted_impact"] = np.maximum(np.expm1(available.prediction), 0.0)
        available["predicted_wait_loss"] = np.maximum(
            np.expm1(available.hazard_prediction), 0.0
        )
        qualified = available.predicted_impact.ge(config.min_predicted_impact)
        if checkpoint < 3600:
            qualified &= available.predicted_wait_loss.ge(config.min_predicted_wait_loss)
        chosen = available.loc[qualified].sort_values(
            ["prediction", "hazard_prediction", "thread_id"],
            ascending=[False, False, True], kind="stable",
        ).head(remaining).copy()
        if len(chosen):
            chosen["action_checkpoint_sec"] = checkpoint
            chosen["config_id"] = config.config_id
            chunks.append(chosen)
            selected_ids.update(chosen.thread_id)
    if not chunks:
        return event.head(0).assign(
            predicted_impact=pd.Series(dtype=float),
            predicted_wait_loss=pd.Series(dtype=float),
            action_checkpoint_sec=pd.Series(dtype=int),
            config_id=pd.Series(dtype=str),
        )
    result = pd.concat(chunks, ignore_index=True)
    if result.thread_id.duplicated().any() or len(result) > config.max_budget:
        raise ValueError("hazard policy identity or cap violation")
    return result


def evaluate_configs(scored: pd.DataFrame) -> pd.DataFrame:
    adaptive.validate_scored_frame(scored)
    if "hazard_prediction" not in scored or scored.hazard_prediction.isna().any():
        raise ValueError("missing hazard prediction")
    rows = []
    for event_id, event in scored.groupby("event_id", sort=True):
        horizon = float(
            event.loc[event.checkpoint_sec.eq(600), "future_growth"].sum()
        )
        for config in candidate_configs():
            selected = hazard_select(event, config)
            blocked = float(selected.dynamic_preventable_impact.sum())
            rows.append({
                "event_id": event_id, "config_id": config.config_id,
                "max_budget": config.max_budget,
                "min_predicted_impact": config.min_predicted_impact,
                "min_predicted_wait_loss": config.min_predicted_wait_loss,
                "interventions": len(selected),
                "positive_interventions": int(selected.dynamic_preventable_impact.gt(0).sum()),
                "blocked_impact": blocked,
                "horizon_future_at_10m": horizon,
                "horizon_reduction": blocked / horizon if horizon else 0.0,
                "mean_action_minute": (
                    float(selected.action_checkpoint_sec.mean() / 60.0)
                    if len(selected) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def summarize_grid(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby([
        "config_id", "max_budget", "min_predicted_impact",
        "min_predicted_wait_loss",
    ], as_index=False).agg(
        validation_events=("event_id", "nunique"),
        mean_interventions=("interventions", "mean"),
        max_interventions=("interventions", "max"),
        mean_positive_interventions=("positive_interventions", "mean"),
        mean_blocked_impact=("blocked_impact", "mean"),
        mean_horizon_reduction=("horizon_reduction", "mean"),
        worst_event_horizon_reduction=("horizon_reduction", "min"),
        mean_action_minute=("mean_action_minute", "mean"),
    )


def apply_selected(
    outer_scores: pd.DataFrame, selected_configs: pd.DataFrame, outer_event: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event = outer_scores.loc[outer_scores.event_id.eq(outer_event)].copy()
    horizon = float(event.loc[event.checkpoint_sec.eq(600), "future_growth"].sum())
    metrics, selections = [], []
    for row in selected_configs.itertuples(index=False):
        config = HazardConfig(
            int(row.max_budget), float(row.min_predicted_impact),
            float(row.min_predicted_wait_loss),
        )
        chosen = hazard_select(event, config)
        blocked = float(chosen.dynamic_preventable_impact.sum())
        metrics.append({
            "outer_event": outer_event, "policy_name": row.policy_name,
            "config_id": config.config_id, "max_budget": config.max_budget,
            "min_predicted_impact": config.min_predicted_impact,
            "min_predicted_wait_loss": config.min_predicted_wait_loss,
            "interventions": len(chosen),
            "positive_interventions": int(chosen.dynamic_preventable_impact.gt(0).sum()),
            "blocked_impact": blocked, "horizon_future_at_10m": horizon,
            "horizon_reduction": blocked / horizon if horizon else 0.0,
            "mean_action_minute": (
                float(chosen.action_checkpoint_sec.mean() / 60.0)
                if len(chosen) else np.nan
            ),
            "used_fallback": bool(row.used_fallback),
        })
        if len(chosen):
            selections.append(chosen.assign(
                outer_event=outer_event, policy_name=row.policy_name
            )[[
                "outer_event", "policy_name", "config_id", "thread_id", "event_id",
                "action_checkpoint_sec", "prediction", "hazard_prediction",
                "predicted_impact", "predicted_wait_loss",
                "dynamic_preventable_impact", "future_growth",
            ]])
    return pd.DataFrame(metrics), pd.concat(selections, ignore_index=True)
