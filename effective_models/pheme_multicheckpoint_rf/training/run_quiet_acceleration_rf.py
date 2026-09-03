"""Nested event-separated quiet-thread acceleration learnability experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd

import quiet_acceleration_common as common


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
ELIGIBLE_EVENTS = (
    "charliehebdo", "ferguson", "germanwings-crash", "ottawashooting",
    "prince-toronto", "putinmissing", "sydneysiege",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest_materialization() -> Path:
    candidates = []
    for path in OUT_ROOT.glob("*_quiet_acceleration_materialization"):
        record = path / "run_record.json"
        data = path / "pheme_quiet_acceleration_features.csv"
        if record.is_file() and data.is_file():
            if json.loads(record.read_text(encoding="utf-8")).get("status") == "complete":
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("no complete quiet acceleration materialization")
    return sorted(candidates)[-1]


def run(mode: str) -> None:
    if mode not in {"smoke", "full"}:
        raise ValueError(mode)
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_quiet_acceleration_rf_{mode}"
    )
    output.mkdir(parents=True, exist_ok=False)
    source = latest_materialization()
    params = dict(common.MODEL_PARAMS)
    if mode == "smoke":
        params.update(n_estimators=20, max_depth=5)
    record = {
        "status": "running", "started_at": now(), "mode": mode,
        "model": "pooled Window/Delta RF on fold-gated quiet rows",
        "model_params": params, "input": str(source.resolve()),
        "split": "outer event held out; fresh inner-event OOF also saved",
        "checkpoints_seconds": list(common.CHECKPOINTS),
        "quiet_gate": (
            "per-checkpoint 60th percentile of cumulative seconds since last activity, "
            "computed from model-training events only"
        ),
        "target": "log1p(max(next10 replies - previous10 replies, 0))",
        "research_safety": (
            "Acceleration is outcome-only. Every gate threshold and fitted model excludes the "
            "event it predicts; inner folds additionally exclude the current outer event."
        ),
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print(f"Starting {mode} quiet acceleration RF experiment.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/7] Loading acceleration data and feature schema...", flush=True)
        frame = pd.read_csv(
            source / "pheme_quiet_acceleration_features.csv",
            dtype={"thread_id": str, "event_id": str},
        )
        schema = json.loads((source / "feature_schema.json").read_text(encoding="utf-8"))
        columns = list(schema["window_delta_feature_columns"])
        events = sorted(frame.event_id.unique())
        outer_events = list(ELIGIBLE_EVENTS[:1] if mode == "smoke" else ELIGIBLE_EVENTS)
        inner_limit = 3 if mode == "smoke" else None
        print("[2/7] Auditing cutoff feature allowlist and target boundaries...", flush=True)
        forbidden = ("next10", "future", "impact", "acceleration", "target", "label")
        if any(token in c.lower() for c in columns for token in forbidden):
            raise ValueError("future-like acceleration feature")
        if (frame.next10_positive_acceleration < 0).any():
            raise ValueError("negative positive-acceleration target")

        inner_predictions, inner_gates = [], []
        outer_predictions, outer_gates = [], []
        print("[3/7] Fitting pooled quiet-only inner and outer RF folds...", flush=True)
        for outer_index, outer_event in enumerate(outer_events, 1):
            inner_events = [event for event in events if event != outer_event]
            if inner_limit:
                inner_events = inner_events[:inner_limit]
            print(
                f"  outer {outer_index}/{len(outer_events)}: {outer_event}; "
                f"inner folds={len(inner_events)}", flush=True,
            )
            for inner_index, inner_event in enumerate(inner_events, 1):
                print(f"    inner {inner_index}/{len(inner_events)}: {inner_event}", flush=True)
                train = frame.loc[
                    frame.event_id.ne(outer_event) & frame.event_id.ne(inner_event)
                ]
                validation = frame.loc[frame.event_id.eq(inner_event)]
                scored, gates = common.fit_predict_quiet(train, validation, columns, params)
                scored["outer_event"] = outer_event
                gates["outer_event"] = outer_event
                gates["predicted_event"] = inner_event
                gates["fold_type"] = "inner"
                inner_predictions.append(scored)
                inner_gates.append(gates)
            train_outer = frame.loc[frame.event_id.ne(outer_event)]
            test_outer = frame.loc[frame.event_id.eq(outer_event)]
            scored, gates = common.fit_predict_quiet(train_outer, test_outer, columns, params)
            gates["outer_event"] = outer_event
            gates["predicted_event"] = outer_event
            gates["fold_type"] = "outer"
            outer_predictions.append(scored)
            outer_gates.append(gates)

        print("[4/7] Calculating event/checkpoint acceleration metrics...", flush=True)
        inner_prediction = pd.concat(inner_predictions, ignore_index=True)
        outer_prediction = pd.concat(outer_predictions, ignore_index=True)
        outer_metrics = common.metric_rows(outer_prediction)
        macro = outer_metrics.groupby("checkpoint_sec", as_index=False).agg(
            outer_events=("event_id", "nunique"),
            total_quiet_rows=("quiet_rows", "sum"),
            total_positive_rows=("positive_rows", "sum"),
            mean_positive_prevalence=("positive_prevalence", "mean"),
            mean_mae=("mae", "mean"), mean_spearman=("spearman", "mean"),
            mean_roc_auc=("roc_auc", "mean"), mean_pr_auc=("pr_auc", "mean"),
            mean_top20_acceleration_capture=("top20_acceleration_capture", "mean"),
            total_top20_positive_hits=("top20_positive_hits", "sum"),
        )
        overall = {
            "mean_spearman": float(outer_metrics.spearman.mean()),
            "mean_roc_auc": float(outer_metrics.roc_auc.mean()),
            "mean_pr_auc": float(outer_metrics.pr_auc.mean()),
            "mean_top20_acceleration_capture": float(
                outer_metrics.top20_acceleration_capture.mean()
            ),
        }
        print("[5/7] Checking split isolation, gate provenance, and identities...", flush=True)
        if (inner_prediction.outer_event == inner_prediction.event_id).any():
            raise ValueError("outer event in inner acceleration predictions")
        if outer_prediction.duplicated(["thread_id", "event_id", "checkpoint_sec"]).any():
            raise ValueError("duplicate outer acceleration prediction")
        print("[6/7] Writing OOF predictions, gate thresholds, and metrics...", flush=True)
        inner_prediction.to_csv(output / "inner_quiet_acceleration_oof.csv", index=False)
        pd.concat(inner_gates, ignore_index=True).to_csv(output / "inner_gate_thresholds.csv", index=False)
        outer_prediction.to_csv(output / "outer_quiet_acceleration_oof.csv", index=False)
        pd.concat(outer_gates, ignore_index=True).to_csv(output / "outer_gate_thresholds.csv", index=False)
        outer_metrics.to_csv(output / "outer_acceleration_metrics.csv", index=False)
        macro.to_csv(output / "outer_acceleration_macro.csv", index=False)
        print("[7/7] Completing result and run record...", flush=True)
        result = {
            "mode": mode, "overall": overall,
            "checkpoint_macro": macro.to_dict(orient="records"),
            "interpretation_boundary": (
                "Quiet acceleration learnability only; no intervention-policy improvement is claimed. "
                "This post-result hypothesis is exploratory on the consumed PHEME events."
            ),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        record.update({
            "status": "complete", "completed_at": now(), "outer_events": outer_events,
            "feature_count": len(columns),
            "model_fits": len(outer_events) * ((inner_limit or 8) + 1),
            "outer_quiet_prediction_rows": len(outer_prediction),
            "metrics": overall,
            "output_files": [
                "inner_quiet_acceleration_oof.csv", "inner_gate_thresholds.csv",
                "outer_quiet_acceleration_oof.csv", "outer_gate_thresholds.csv",
                "outer_acceleration_metrics.csv", "outer_acceleration_macro.csv",
                "result.json", "run_record.json",
            ],
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: {mode} quiet acceleration RF saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "output_files": sorted(p.name for p in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: {mode} quiet acceleration RF preserved at: {output.resolve()}", flush=True)
        raise
