"""Fold-gated quiet-thread acceleration RF utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score


CHECKPOINTS = (600, 1200, 1800, 2400, 3000)
QUIET_QUANTILE = 0.60
MODEL_PARAMS = {
    "n_estimators": 300, "max_depth": 8, "min_samples_leaf": 3,
    "max_features": 0.8, "n_jobs": 1, "random_state": 42,
}


def fit_predict_quiet(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str], params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(train.event_id) & set(test.event_id):
        raise ValueError("event overlap in quiet acceleration fold")
    quiet_train_chunks, quiet_test_chunks, gate_rows = [], [], []
    for checkpoint in CHECKPOINTS:
        train_cp = train.loc[train.checkpoint_sec.eq(checkpoint)]
        test_cp = test.loc[test.checkpoint_sec.eq(checkpoint)]
        threshold = float(
            train_cp["cumulative__cum_time_since_last_activity"].quantile(QUIET_QUANTILE)
        )
        quiet_train = train_cp.loc[
            train_cp["cumulative__cum_time_since_last_activity"].ge(threshold)
        ].copy()
        quiet_test = test_cp.loc[
            test_cp["cumulative__cum_time_since_last_activity"].ge(threshold)
        ].copy()
        if quiet_train.empty or quiet_test.empty:
            raise ValueError(f"empty quiet fold at checkpoint {checkpoint}")
        quiet_train_chunks.append(quiet_train)
        quiet_test_chunks.append(quiet_test.assign(quiet_threshold_sec=threshold))
        gate_rows.append({
            "checkpoint_sec": checkpoint, "quiet_threshold_sec": threshold,
            "quiet_train_rows": len(quiet_train), "quiet_test_rows": len(quiet_test),
            "quiet_train_events": quiet_train.event_id.nunique(),
        })
    pooled_train = pd.concat(quiet_train_chunks, ignore_index=True)
    pooled_test = pd.concat(quiet_test_chunks, ignore_index=True)
    model = RandomForestRegressor(**params).fit(
        pooled_train[columns].to_numpy(dtype=np.float32),
        pooled_train.next10_positive_acceleration_y.to_numpy(dtype=np.float32),
    )
    prediction = model.predict(pooled_test[columns].to_numpy(dtype=np.float32))
    result = pooled_test[[
        "thread_id", "event_id", "checkpoint_sec",
        "previous10_reply_count", "next10_reply_count",
        "next10_signed_acceleration", "next10_positive_acceleration",
        "quiet_threshold_sec",
    ]].copy()
    result["prediction"] = prediction
    result["predicted_positive_acceleration"] = np.maximum(np.expm1(prediction), 0.0)
    if result.duplicated(["thread_id", "event_id", "checkpoint_sec"]).any():
        raise ValueError("duplicate quiet prediction")
    return result, pd.DataFrame(gate_rows)


def metric_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (event, checkpoint), group in scored.groupby(["event_id", "checkpoint_sec"]):
        actual = group.next10_positive_acceleration.to_numpy(dtype=float)
        prediction = group.predicted_positive_acceleration.to_numpy(dtype=float)
        positive = actual > 0
        k = min(20, len(group))
        chosen = np.argsort(-prediction, kind="stable")[:k]
        oracle = np.argsort(-actual, kind="stable")[:k]
        oracle_total = float(actual[oracle].sum())
        corr = pd.Series(actual).corr(pd.Series(prediction), method="spearman")
        rows.append({
            "event_id": event, "checkpoint_sec": int(checkpoint), "quiet_rows": len(group),
            "positive_rows": int(positive.sum()),
            "positive_prevalence": float(positive.mean()),
            "mae": float(np.mean(np.abs(actual - prediction))),
            "spearman": float(corr) if np.isfinite(corr) else 0.0,
            "roc_auc": float(roc_auc_score(positive, prediction))
            if 0 < positive.sum() < len(group) else np.nan,
            "pr_auc": float(average_precision_score(positive, prediction))
            if 0 < positive.sum() < len(group) else np.nan,
            "top20_acceleration_capture": (
                float(actual[chosen].sum() / oracle_total) if oracle_total else 0.0
            ),
            "top20_positive_hits": int(positive[chosen].sum()),
        })
    return pd.DataFrame(rows)
