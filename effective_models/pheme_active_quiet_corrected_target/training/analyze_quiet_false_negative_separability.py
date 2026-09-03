"""Can strict-30 features separate missed quiet Oracle Top-50 threads?"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT_ROOT = HERE.parent / "experiments"
HYBRID_RUN = OUT_ROOT / "20260903_214222_297421_corrected_quiet_tier_hybrid_full"
RELATIONAL_TABLE = (
    ROOT / "effective_models" / "pheme_quiet_tier_group_search" / "experiments"
    / "20260902_145009_008419_strict30_relational_feature_materialization"
    / "strict30_protected_relational_snapshot_features.csv"
)
TOPOLOGY_TIME_BLOCK = (
    ROOT / "effective_models" / "pheme_quiet_tier_group_search" / "experiments"
    / "20260902_145443_669471_strict30_topology_time_feature_materialization"
    / "topology_time_feature_block.csv"
)
TOP_K = 50
SEED = 42


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def feature_family(name: str) -> str | None:
    prefixes = (
        ("account_age_", "account_age"), ("activity_", "activity"),
        ("semantic_", "semantic_summary"), ("sentiment_", "sentiment"),
        ("temporal_", "temporal"), ("topology_", "topology"),
        ("rel_", "relational"), ("tt_", "topology_time"),
        ("hx_", "hypothesis_interaction"),
    )
    if name.startswith(("source_text_", "reply_text_", "snapshot_text_")):
        return "text_surface"
    for prefix, family in prefixes:
        if name.startswith(prefix):
            return family
    return None


def add_hypothesis_interactions(frame: pd.DataFrame) -> list[str]:
    replies = np.maximum(frame["activity_observed_replies"].to_numpy(float), 1.0)
    semantic_distance = 1.0 - frame["rel_source_reply_centroid_cosine"].to_numpy(float)
    span_scale = np.log1p(np.maximum(frame["rel_reply_active_span_sec"].to_numpy(float), 0.0))
    values = {
        "hx_reply_question_rate": frame["reply_text_question_count"] / replies,
        "hx_reply_exclamation_rate": frame["reply_text_exclamation_count"] / replies,
        "hx_reply_url_rate": frame["reply_text_url_count"] / replies,
        "hx_sentiment_disagreement_x_late": frame["rel_source_reply_sentiment_delta_abs_mean"] * frame["temporal_last10_fraction"],
        "hx_cosine_dispersion_x_late": frame["rel_source_reply_cosine_std"] * frame["temporal_last10_fraction"],
        "hx_semantic_distance_x_span": semantic_distance * span_scale,
        "hx_depth_x_semantic_distance": frame["topology_max_depth"] * semantic_distance,
        "hx_reply_diversity_x_span": frame["reply_text_unique_token_ratio"] * span_scale,
        "hx_polarized_questions": frame["sentiment_compound_std"] * (frame["reply_text_question_count"] / replies),
        "hx_late_branching": frame["temporal_last10_fraction"] * frame["topology_root_children"] / replies,
        "hx_source_reply_question_tension": frame["source_text_question_count"] * (frame["reply_text_question_count"] / replies),
        "hx_semantic_variability_x_late": frame["semantic_graph_std"] * frame["temporal_last10_fraction"],
    }
    for name, value in values.items():
        frame[name] = np.asarray(value, dtype=float)
    return list(values)


def event_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(["event_id", "rescue_target"])["thread_id"].transform("size")
    return 1.0 / counts.to_numpy(float)


def fit_score(model_name: str, train: pd.DataFrame, test: pd.DataFrame, columns: list[str], seed: int) -> np.ndarray:
    if model_name == "balanced_logistic":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed))
        weight_key = "logisticregression__sample_weight"
    elif model_name == "balanced_rf":
        model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=4, max_features="sqrt", class_weight="balanced_subsample", random_state=seed, n_jobs=1))
        weight_key = "randomforestclassifier__sample_weight"
    else:
        raise ValueError(f"unknown model: {model_name}")
    model.fit(train[columns], train.rescue_target, **{weight_key: event_balanced_weights(train)})
    return model.predict_proba(test[columns])[:, 1]


def univariate_effects(pool: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        directions, event_aucs = [], []
        for _, event in pool.groupby("event_id", sort=True):
            case = event.loc[event.rescue_target.eq(1), column].to_numpy(float)
            control = event.loc[event.rescue_target.eq(0), column].to_numpy(float)
            if not len(case) or not len(control):
                continue
            pooled = np.r_[case, control]
            scale = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
            if scale > 1e-12:
                directions.append((float(np.mean(case)) - float(np.mean(control))) / scale)
            truth = np.r_[np.ones(len(case)), np.zeros(len(control))]
            auc = float(roc_auc_score(truth, np.r_[case, control]))
            event_aucs.append(max(auc, 1.0 - auc))
        if event_aucs:
            mean_effect = float(np.mean(directions)) if directions else 0.0
            rows.append({
                "feature": column, "family": feature_family(column),
                "mean_event_standardized_effect": mean_effect,
                "mean_abs_event_standardized_effect": float(np.mean(np.abs(directions))) if directions else 0.0,
                "mean_orientation_free_event_auc": float(np.mean(event_aucs)),
                "events_same_direction_as_mean": int(sum(abs(x) > 1e-12 and np.sign(x) == np.sign(mean_effect) for x in directions)),
                "events_with_nonconstant_scale": len(directions),
                "events_evaluated": len(event_aucs),
            })
    return pd.DataFrame(rows).sort_values(["mean_orientation_free_event_auc", "mean_abs_event_standardized_effect"], ascending=False, kind="stable")


def main() -> None:
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_quiet_false_negative_separability")
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started_at": now(),
        "purpose": "strict-30 case-control and LOEO separability of missed quiet Oracle Top-50 threads",
        "inputs": {"corrected_hybrid_run": str(HYBRID_RUN.resolve()), "relational_features": str(RELATIONAL_TABLE.resolve()), "topology_time_features": str(TOPOLOGY_TIME_BLOCK.resolve())},
        "split": "seven-event LOEO; every predeclared configuration is reported, with no held-out selection",
        "cutoff_seconds": 1800, "seed": SEED,
        "label": "corrected event-global Oracle Top-50 membership among quiet threads not selected by the current hybrid",
        "research_safety": "All model columns come from validated strict-30 feature artifacts. Corrected impact and current hybrid score define outcomes/cohorts only and are excluded from model inputs. Protected inputs are read-only.",
    }
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting quiet false-negative separability diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading corrected OOF outcomes and protected strict-30 features...", flush=True)
        scores = pd.read_csv(HYBRID_RUN / "outer_scores.csv", dtype={"thread_id": str, "event_id": str})
        features = pd.read_csv(RELATIONAL_TABLE, dtype={"thread_id": str, "event_id": str})
        topology_time = pd.read_csv(TOPOLOGY_TIME_BLOCK, dtype={"thread_id": str})
        features = features.merge(topology_time, on="thread_id", how="left", validate="one_to_one")
        keep = [name for name in features if feature_family(name) is not None]
        frame = scores[["thread_id", "event_id", "quiet", "preventable_impact", "hybrid_calibrated_score"]].merge(features[["thread_id", "event_id", *keep]], on=["thread_id", "event_id"], validate="one_to_one")
        if len(frame) != len(scores) or frame.duplicated(["event_id", "thread_id"]).any():
            raise ValueError("feature join changed corrected OOF identities")
        if not np.isfinite(frame[["preventable_impact", "hybrid_calibrated_score", *keep]].to_numpy(float)).all():
            raise ValueError("non-finite diagnostic input")

        print("[2/6] Reconstructing deterministic Oracle and hybrid Top-50 cohorts...", flush=True)
        frame = frame.copy()
        frame["oracle_top50"] = False
        frame["hybrid_top50"] = False
        for _, event in frame.groupby("event_id", sort=True):
            frame.loc[event.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").head(TOP_K).index, "oracle_top50"] = True
            frame.loc[event.sort_values(["hybrid_calibrated_score", "thread_id"], ascending=[False, True], kind="stable").head(TOP_K).index, "hybrid_top50"] = True
        frame["quiet"] = frame.quiet.astype(bool)
        quiet = frame.loc[frame.quiet].copy()
        quiet["cohort"] = np.select(
            [quiet.oracle_top50 & ~quiet.hybrid_top50, quiet.oracle_top50 & quiet.hybrid_top50, ~quiet.oracle_top50 & quiet.hybrid_top50],
            ["missed_oracle_case", "selected_oracle_hit", "selected_false_positive"],
            default="unselected_nonoracle_control",
        )
        pool = quiet.loc[~quiet.hybrid_top50].copy()
        pool["rescue_target"] = pool.oracle_top50.astype(int)
        if int(pool.rescue_target.sum()) != 68:
            raise ValueError(f"expected 68 missed quiet cases, found {int(pool.rescue_target.sum())}")

        print("[3/6] Constructing predeclared real-world interaction hypotheses...", flush=True)
        hypothesis_columns = add_hypothesis_interactions(pool)
        keep = [name for name in pool if feature_family(name) is not None]
        families = {family: [name for name in keep if feature_family(name) == family] for family in sorted({feature_family(name) for name in keep})}
        configurations = {
            "activity_temporal_topology": families["activity"] + families["temporal"] + families["topology"],
            "content_context": families["account_age"] + families["semantic_summary"] + families["sentiment"] + families["text_surface"],
            "relational_only": families["relational"],
            "topology_time_only": families["topology_time"],
            "hypothesis_interactions_only": hypothesis_columns,
            "activity_plus_relational": families["activity"] + families["temporal"] + families["topology"] + families["relational"],
            "content_plus_relational": families["account_age"] + families["semantic_summary"] + families["sentiment"] + families["text_surface"] + families["relational"],
            "all_protected_plus_interactions": keep,
        }
        if any(not columns for columns in configurations.values()):
            raise ValueError("empty predeclared feature configuration")

        print("[4/6] Measuring event-conditioned single-feature effects...", flush=True)
        effects = univariate_effects(pool, keep)
        print("[5/6] Running bounded LOEO combination diagnostic (112 fits)...", flush=True)
        prediction_rows, metric_rows = [], []
        events = sorted(pool.event_id.unique())
        total_fits = len(events) * len(configurations) * 2
        completed = 0
        for config_name, columns in configurations.items():
            for model_name in ("balanced_logistic", "balanced_rf"):
                for fold_index, outer_event in enumerate(events):
                    train = pool.loc[pool.event_id.ne(outer_event)]
                    test = pool.loc[pool.event_id.eq(outer_event)].copy()
                    score = fit_score(model_name, train, test, columns, SEED + fold_index)
                    truth = test.rescue_target.to_numpy(int)
                    prevalence = float(truth.mean())
                    ap = float(average_precision_score(truth, score))
                    auc = float(roc_auc_score(truth, score))
                    k = int(truth.sum())
                    order = np.lexsort((test.thread_id.to_numpy(str), -score))
                    hits = int(truth[order[:k]].sum())
                    metric_rows.append({"configuration": config_name, "model": model_name, "outer_event": outer_event, "features": len(columns), "test_threads": len(test), "missed_cases": k, "prevalence": prevalence, "average_precision": ap, "ap_lift_over_prevalence": ap - prevalence, "roc_auc": auc, "recall_at_case_k": hits / k, "hits_at_case_k": hits})
                    prediction_rows.append(pd.DataFrame({"configuration": config_name, "model": model_name, "outer_event": outer_event, "thread_id": test.thread_id, "rescue_target": truth, "score": score}))
                    completed += 1
                    if completed == 1 or completed % 14 == 0 or completed == total_fits:
                        print(f"  LOEO fits {completed}/{total_fits}", flush=True)
        metrics = pd.DataFrame(metric_rows)
        macro = metrics.groupby(["configuration", "model"], as_index=False).agg(feature_count=("features", "first"), events=("outer_event", "nunique"), mean_average_precision=("average_precision", "mean"), mean_prevalence=("prevalence", "mean"), mean_ap_lift=("ap_lift_over_prevalence", "mean"), mean_roc_auc=("roc_auc", "mean"), total_hits_at_case_k=("hits_at_case_k", "sum"), total_missed_cases=("missed_cases", "sum"), mean_recall_at_case_k=("recall_at_case_k", "mean"))
        macro["micro_recall_at_case_k"] = macro.total_hits_at_case_k / macro.total_missed_cases
        macro.sort_values(["mean_ap_lift", "micro_recall_at_case_k"], ascending=False, inplace=True, kind="stable")

        print("[6/6] Writing cohort, effects, LOEO evidence, and run record...", flush=True)
        quiet.groupby(["event_id", "cohort"], as_index=False).size().to_csv(output / "quiet_cohort_counts.csv", index=False)
        quiet[["thread_id", "event_id", "preventable_impact", "hybrid_calibrated_score", "cohort"]].to_csv(output / "quiet_case_control_membership.csv", index=False)
        effects.to_csv(output / "univariate_event_conditioned_effects.csv", index=False)
        metrics.to_csv(output / "loeo_per_event_metrics.csv", index=False)
        macro.to_csv(output / "loeo_combination_summary.csv", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_csv(output / "loeo_predictions.csv", index=False)
        pd.DataFrame([{"feature": name, "family": feature_family(name)} for name in keep]).to_csv(output / "model_feature_schema.csv", index=False)
        record.update({
            "status": "complete", "completed_at": now(), "quiet_threads": len(quiet),
            "unselected_quiet_pool": len(pool), "missed_oracle_cases": int(pool.rescue_target.sum()),
            "selected_false_positives": int(quiet.cohort.eq("selected_false_positive").sum()),
            "feature_count": len(keep), "configurations": {key: len(value) for key, value in configurations.items()},
            "models": ["balanced_logistic", "balanced_rf"], "fits": total_fits,
            "best_exploratory_configuration": macro.iloc[0].to_dict(),
            "interpretation_boundary": "The best row is descriptive, not a selected final model. A future candidate must select its configuration using inner folds only.",
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: quiet false-negative separability saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update({"status": "failed", "failed_at": now(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(), "output_files": sorted(path.name for path in output.iterdir())})
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: quiet false-negative separability preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
