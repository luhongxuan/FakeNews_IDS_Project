"""Nested-LOEO quiet rescue expert layered after corrected hybrid Top-50."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import analyze_quiet_false_negative_separability as sep


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
HYBRID_RUN = OUT_ROOT / "20260903_214222_297421_corrected_quiet_tier_hybrid_full"
OUTER_SCORES = HYBRID_RUN / "outer_scores.csv"
NESTED_SCORES = HYBRID_RUN / "nested_inner_oof_scores.csv"
ELIGIBLE_EVENTS = (
    "charliehebdo", "ferguson", "germanwings-crash", "ottawashooting",
    "prince-toronto", "putinmissing", "sydneysiege",
)
BASE_BUDGET = 50
RESCUE_CAPS = (5, 10, 15, 20)
SEED = 42


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_features() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    features = pd.read_csv(sep.RELATIONAL_TABLE, dtype={"thread_id": str, "event_id": str})
    topology_time = pd.read_csv(sep.TOPOLOGY_TIME_BLOCK, dtype={"thread_id": str})
    features = features.merge(topology_time, on="thread_id", how="left", validate="one_to_one")
    features = features.copy()
    sep.add_hypothesis_interactions(features)
    model_columns = [name for name in features if sep.feature_family(name) is not None]
    if not np.isfinite(features[model_columns].to_numpy(float)).all():
        raise ValueError("non-finite protected rescue feature")
    families = {
        family: [name for name in model_columns if sep.feature_family(name) == family]
        for family in sorted({sep.feature_family(name) for name in model_columns})
    }
    configs = {
        "relational_only": families["relational"],
        "hypothesis_interactions_only": families["hypothesis_interaction"],
        "activity_temporal_topology": families["activity"] + families["temporal"] + families["topology"],
    }
    return features[["thread_id", "event_id", *model_columns]], configs


def event_pool(event: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    ordered_oracle = event.sort_values(
        ["preventable_impact", "thread_id"], ascending=[False, True], kind="stable"
    )
    ordered_model = event.sort_values(
        ["hybrid_calibrated_score", "thread_id"], ascending=[False, True], kind="stable"
    )
    oracle_ids = set(ordered_oracle.head(BASE_BUDGET).thread_id)
    baseline = ordered_model.head(BASE_BUDGET).copy()
    baseline_ids = set(baseline.thread_id)
    pool = event.loc[event.expert.eq("quiet") & ~event.thread_id.isin(baseline_ids)].copy()
    pool["rescue_target"] = pool.thread_id.isin(oracle_ids).astype(int)
    return pool, baseline, oracle_ids


def add_features(scores: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    joined = scores.merge(features, on=["thread_id", "event_id"], how="left", validate="many_to_one")
    model_columns = [name for name in features if sep.feature_family(name) is not None]
    if len(joined) != len(scores) or joined[model_columns].isna().any().any():
        raise ValueError("rescue score/feature identity join failed")
    return joined


def score_selection(pool: pd.DataFrame, score: np.ndarray, cap: int, total_impact: float) -> dict[str, float]:
    ranked = pool.assign(rescue_score=score).sort_values(
        ["rescue_score", "thread_id"], ascending=[False, True], kind="stable"
    )
    chosen = ranked.head(min(cap, len(ranked)))
    return {
        "selected": len(chosen),
        "incremental_impact": float(chosen.preventable_impact.sum()),
        "incremental_reduction": float(chosen.preventable_impact.sum()) / total_impact if total_impact else 0.0,
        "quiet_oracle_hits": int(chosen.rescue_target.sum()),
    }


def select_inner_models(
    inner: pd.DataFrame,
    configs: dict[str, list[str]],
    outer_event: str,
) -> tuple[dict[int, tuple[str, str]], pd.DataFrame]:
    events = sorted(inner.event_id.unique())
    pools = {event_id: event_pool(group)[0] for event_id, group in inner.groupby("event_id", sort=True)}
    rows = []
    total = len(configs) * 2 * len(events)
    completed = 0
    for config_name, columns in configs.items():
        for model_name in ("balanced_logistic", "balanced_rf"):
            for fold_index, validation_event in enumerate(events):
                train = pd.concat([pools[event] for event in events if event != validation_event], ignore_index=True)
                test = pools[validation_event]
                score = sep.fit_score(model_name, train, test, columns, SEED + fold_index)
                total_impact = float(inner.loc[inner.event_id.eq(validation_event), "preventable_impact"].sum())
                for cap in RESCUE_CAPS:
                    rows.append({
                        "outer_event": outer_event, "validation_event": validation_event,
                        "configuration": config_name, "model": model_name,
                        "rescue_cap": cap, "feature_count": len(columns),
                        **score_selection(test, score, cap, total_impact),
                    })
                completed += 1
                if completed == 1 or completed % 12 == 0 or completed == total:
                    print(f"    inner rescue fits {completed}/{total}", flush=True)
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["outer_event", "configuration", "model", "rescue_cap", "feature_count"], as_index=False
    ).agg(
        inner_events=("validation_event", "nunique"),
        mean_incremental_reduction=("incremental_reduction", "mean"),
        mean_incremental_impact=("incremental_impact", "mean"),
        total_quiet_oracle_hits=("quiet_oracle_hits", "sum"),
        total_selected=("selected", "sum"),
    )
    selected = {}
    for cap, group in summary.groupby("rescue_cap", sort=True):
        best = group.sort_values(
            ["mean_incremental_reduction", "total_quiet_oracle_hits", "configuration", "model"],
            ascending=[False, False, True, True], kind="stable",
        ).iloc[0]
        selected[int(cap)] = (str(best.configuration), str(best.model))
    return selected, summary


def run(mode: str) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    outer_events = ELIGIBLE_EVENTS[:1] if mode == "smoke" else ELIGIBLE_EVENTS
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_quiet_rescue_expert_{mode}"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(), "mode": mode,
        "experiment": "nested-LOEO quiet rescue expert after corrected hybrid Top-50",
        "inputs": {"outer_scores": str(OUTER_SCORES.resolve()), "nested_scores": str(NESTED_SCORES.resolve()), "relational_features": str(sep.RELATIONAL_TABLE.resolve()), "topology_time_features": str(sep.TOPOLOGY_TIME_BLOCK.resolve())},
        "split": "eligible outer LOEO; remaining six eligible events use inner LOEO to select expert separately at each fixed rescue cap",
        "cutoff_seconds": 1800, "base_budget": BASE_BUDGET,
        "rescue_caps": list(RESCUE_CAPS), "seed": SEED,
        "candidate_models": ["balanced_logistic", "balanced_rf"],
        "candidate_feature_configs": ["relational_only", "hypothesis_interactions_only", "activity_temporal_topology"],
        "selection_metric": "inner-event mean incremental reduction at each fixed rescue cap; Oracle hits only break ties",
        "research_safety": "Outer event is excluded from feature-model fitting and model selection. Only corrected outcomes define training labels/evaluation. Every model input is from validated strict-30 protected features. Rescue caps are predeclared and all are reported.",
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print(f"Starting {mode} nested quiet rescue expert.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/5] Loading corrected outer/nested OOF and protected features...", flush=True)
        outer = pd.read_csv(OUTER_SCORES, dtype={"thread_id": str, "event_id": str})
        outer["expert"] = np.where(outer.quiet.astype(bool), "quiet", "active")
        nested = pd.read_csv(NESTED_SCORES, dtype={"thread_id": str, "event_id": str, "outer_event": str})
        features, configs = load_features()
        outer = add_features(outer, features)
        nested = add_features(nested, features)
        if tuple(sorted(outer.event_id.unique())) != tuple(sorted(ELIGIBLE_EVENTS)):
            raise ValueError("outer events do not match protected eligible event protocol")

        print("[2/5] Reconstructing nested rescue pools and selecting candidates...", flush=True)
        metric_rows, selection_rows, chosen_rows = [], [], []
        for outer_index, outer_event in enumerate(outer_events, 1):
            print(f"  outer event {outer_index}/{len(outer_events)}: {outer_event}", flush=True)
            inner_events = [event for event in ELIGIBLE_EVENTS if event != outer_event]
            inner = nested.loc[nested.outer_event.eq(outer_event) & nested.event_id.isin(inner_events)].copy()
            if sorted(inner.event_id.unique()) != sorted(inner_events):
                raise ValueError(f"{outer_event}: incomplete eligible inner events")
            selected_models, inner_summary = select_inner_models(inner, configs, outer_event)
            selection_rows.append(inner_summary)

            train_pools = [event_pool(group)[0] for _, group in inner.groupby("event_id", sort=True)]
            train = pd.concat(train_pools, ignore_index=True)
            test_event = outer.loc[outer.event_id.eq(outer_event)].copy()
            test_pool, baseline, oracle50_ids = event_pool(test_event)
            total_impact = float(test_event.preventable_impact.sum())
            baseline_impact = float(baseline.preventable_impact.sum())
            score_cache: dict[tuple[str, str], np.ndarray] = {}
            for cap in RESCUE_CAPS:
                config_name, model_name = selected_models[cap]
                key = (config_name, model_name)
                if key not in score_cache:
                    score_cache[key] = sep.fit_score(model_name, train, test_pool, configs[config_name], SEED + 1000 + outer_index)
                ranked = test_pool.assign(rescue_score=score_cache[key]).sort_values(["rescue_score", "thread_id"], ascending=[False, True], kind="stable")
                rescued = ranked.head(min(cap, len(ranked))).copy()
                existing_quiet = test_pool.sort_values(
                    ["hybrid_calibrated_score", "thread_id"],
                    ascending=[False, True], kind="stable",
                ).head(min(cap, len(test_pool)))
                oracle_quiet = test_pool.sort_values(
                    ["preventable_impact", "thread_id"],
                    ascending=[False, True], kind="stable",
                ).head(min(cap, len(test_pool)))
                combined_ids = set(baseline.thread_id) | set(rescued.thread_id)
                combined = test_event.loc[test_event.thread_id.isin(combined_ids)]
                oracle_total = test_event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(BASE_BUDGET + cap)
                blocked = float(combined.preventable_impact.sum())
                metric_rows.append({
                    "outer_event": outer_event, "rescue_cap": cap,
                    "selected_configuration": config_name, "selected_model": model_name,
                    "base_selected": len(baseline), "rescue_selected": len(rescued),
                    "total_selected": len(combined), "baseline_impact": baseline_impact,
                    "incremental_rescue_impact": float(rescued.preventable_impact.sum()),
                    "existing_quiet_incremental_impact": float(existing_quiet.preventable_impact.sum()),
                    "rescue_minus_existing_quiet_impact": float(rescued.preventable_impact.sum() - existing_quiet.preventable_impact.sum()),
                    "random_quiet_expected_impact": float(cap * test_pool.preventable_impact.mean()),
                    "rescue_minus_random_expected_impact": float(rescued.preventable_impact.sum() - cap * test_pool.preventable_impact.mean()),
                    "quiet_pool_oracle_impact_at_cap": float(oracle_quiet.preventable_impact.sum()),
                    "quiet_pool_oracle_efficiency": float(rescued.preventable_impact.sum()) / float(oracle_quiet.preventable_impact.sum()) if float(oracle_quiet.preventable_impact.sum()) else 0.0,
                    "total_blocked_impact": blocked,
                    "baseline_reduction": baseline_impact / total_impact if total_impact else 0.0,
                    "total_reduction": blocked / total_impact if total_impact else 0.0,
                    "incremental_reduction": float(rescued.preventable_impact.sum()) / total_impact if total_impact else 0.0,
                    "oracle_efficiency_at_total_budget": blocked / float(oracle_total.preventable_impact.sum()),
                    "quiet_oracle_top50_rescued": int(rescued.thread_id.isin(oracle50_ids).sum()),
                    "existing_quiet_oracle_top50_hits": int(existing_quiet.thread_id.isin(oracle50_ids).sum()),
                    "random_quiet_expected_oracle_top50_hits": float(cap * test_pool.rescue_target.mean()),
                    "quiet_oracle_top50_remaining": int(test_pool.rescue_target.sum() - rescued.rescue_target.sum()),
                })
                chosen_rows.append(rescued.assign(outer_event=outer_event, rescue_cap=cap, selected_configuration=config_name, selected_model=model_name)[["outer_event", "rescue_cap", "selected_configuration", "selected_model", "thread_id", "event_id", "rescue_score", "preventable_impact", "rescue_target"]])

        print("[3/5] Aggregating fixed-cap intervention results...", flush=True)
        metrics = pd.DataFrame(metric_rows)
        macro = metrics.groupby("rescue_cap", as_index=False).agg(
            events=("outer_event", "nunique"), mean_total_selected=("total_selected", "mean"),
            mean_baseline_reduction=("baseline_reduction", "mean"),
            mean_incremental_reduction=("incremental_reduction", "mean"),
            mean_total_reduction=("total_reduction", "mean"),
            mean_oracle_efficiency=("oracle_efficiency_at_total_budget", "mean"),
            mean_rescue_minus_existing_quiet_impact=("rescue_minus_existing_quiet_impact", "mean"),
            mean_rescue_minus_random_expected_impact=("rescue_minus_random_expected_impact", "mean"),
            mean_quiet_pool_oracle_efficiency=("quiet_pool_oracle_efficiency", "mean"),
            total_quiet_oracle_top50_rescued=("quiet_oracle_top50_rescued", "sum"),
            total_existing_quiet_oracle_top50_hits=("existing_quiet_oracle_top50_hits", "sum"),
            total_random_quiet_expected_oracle_top50_hits=("random_quiet_expected_oracle_top50_hits", "sum"),
            total_quiet_oracle_top50_remaining=("quiet_oracle_top50_remaining", "sum"),
        )
        print("[4/5] Validating selection and output integrity...", flush=True)
        if metrics.duplicated(["outer_event", "rescue_cap"]).any() or len(metrics) != len(outer_events) * len(RESCUE_CAPS):
            raise ValueError("incomplete rescue evaluation")
        if (metrics.total_selected != BASE_BUDGET + metrics.rescue_cap).any():
            raise ValueError("rescue policy violated fixed intervention count")

        print("[5/5] Writing scores, metrics, and complete run record...", flush=True)
        pd.concat(selection_rows, ignore_index=True).to_csv(output / "inner_model_selection.csv", index=False)
        metrics.to_csv(output / "per_event_rescue_metrics.csv", index=False)
        macro.to_csv(output / "rescue_cap_summary.csv", index=False)
        pd.concat(chosen_rows, ignore_index=True).to_csv(output / "selected_rescue_threads.csv", index=False)
        record.update({"status": "complete", "completed_at": now(), "outer_events": list(outer_events), "metrics": macro.to_dict(orient="records"), "output_files": ["inner_model_selection.csv", "per_event_rescue_metrics.csv", "rescue_cap_summary.csv", "selected_rescue_threads.csv", "run_record.json"]})
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: {mode} quiet rescue expert saved to: {output.resolve()}", flush=True)
        return output
    except BaseException as error:
        record.update({"status": "failed", "failed_at": now(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "output_files": sorted(path.name for path in output.iterdir())})
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: {mode} quiet rescue expert preserved at: {output.resolve()}", flush=True)
        raise
