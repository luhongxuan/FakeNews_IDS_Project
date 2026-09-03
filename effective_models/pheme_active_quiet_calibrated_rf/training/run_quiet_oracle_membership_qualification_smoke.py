"""Strict-30 quiet Oracle-membership qualification smoke.

The classifier learns a binary intervention qualification, not preventable
impact.  Inner event OOF predictions select probability thresholds subject to
an average candidate-cap constraint.  The held-out event is evaluation-only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TIER_ROOT = ROOT / "effective_models" / "pheme_quiet_tier_group_search"
sys.path.insert(0, str(TIER_ROOT))
from strict30_protected_feature_registry import STRICT30_ROOT, feature_group


MATERIALIZATION = (
    TIER_ROOT
    / "experiments"
    / "20260902_030441_849111_strict30_protected_feature_materialization"
)
FEATURE_TABLE = MATERIALIZATION / "strict30_protected_snapshot_features.csv"
MATERIALIZATION_RECORD = MATERIALIZATION / "run_record.json"
DATASET = (
    STRICT30_ROOT
    / "pheme_graphs_roberta_30min_replyv4_preventableimpact_semantic.pt"
)
BASELINE_RUN = (
    HERE.parent
    / "experiments"
    / "20260902_132700_413954_quiet_tier_hybrid_policy_full"
)
BASELINE_SCORES = BASELINE_RUN / "outer_scores.csv"
OUT_ROOT = HERE.parent / "experiments"

OUTER_EVENT = "charliehebdo"
INNER_EVENTS = ("ferguson", "germanwings-crash", "putinmissing")
QUIET_QUANTILE = 0.60
ORACLE_K = 50
CANDIDATE_CAPS = (5, 10, 15, 20)
REPLACEMENT_LIMITS = (1, 3, 5)
THRESHOLDS = tuple(np.linspace(0.05, 0.95, 37))
SEED = 42
TREES = 100


def save_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def add_oracle_label(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["oracle_global_top50"] = False
    for _, event in result.groupby("event_id", sort=True):
        indices = event.sort_values(
            ["preventable_impact", "thread_id"],
            ascending=[False, True],
            kind="stable",
        ).head(ORACLE_K).index
        result.loc[indices, "oracle_global_top50"] = True
    return result


def snapshot_inputs(graphs: list) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    recency = {}
    embeddings = {}
    for index, graph in enumerate(graphs, 1):
        temporal = graph.temporal_features.detach().cpu().numpy().reshape(-1)
        source = torch.nonzero(graph.is_source_mask, as_tuple=False).flatten()
        if (
            len(temporal) != 11
            or not np.isfinite(temporal).all()
            or source.numel() != 1
            or graph.x.shape[1] != 781
        ):
            raise ValueError(f"{graph.thread_id}: invalid protected snapshot")
        root = int(source.item())
        vectors = graph.x[:, 13:].detach().cpu().numpy().astype(np.float32)
        replies = np.delete(vectors, root, axis=0)
        reply_mean = replies.mean(0) if len(replies) else np.zeros(768, np.float32)
        recency[str(graph.thread_id)] = float(temporal[5])
        embeddings[str(graph.thread_id)] = np.r_[vectors[root], reply_mean]
        if index % 500 == 0 or index == len(graphs):
            print(f"  protected snapshot audit {index}/{len(graphs)}", flush=True)
    return recency, embeddings


def quiet_partition(
    frame: pd.DataFrame,
    recency: dict[str, float],
    train_events: list[str],
    test_event: str,
    gate_events: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    gate_events = train_events if gate_events is None else gate_events
    gate_train = frame.loc[frame.event_id.isin(gate_events)].copy()
    train_all = frame.loc[frame.event_id.isin(train_events)].copy()
    test_all = frame.loc[frame.event_id.eq(test_event)].copy()
    threshold = float(np.quantile(gate_train.thread_id.map(recency), QUIET_QUANTILE))
    train = train_all.loc[train_all.thread_id.map(recency) >= threshold].copy()
    test = test_all.loc[test_all.thread_id.map(recency) >= threshold].copy()
    if train.oracle_global_top50.nunique() != 2 or test.empty:
        raise ValueError(f"{test_event}: degenerate quiet split")
    return train, test, threshold


def prepare_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scalar_columns: list[str],
    embeddings: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_scalar = train[scalar_columns].to_numpy(np.float32)
    test_scalar = test[scalar_columns].to_numpy(np.float32)
    scalar_scaler = StandardScaler().fit(train_scalar)
    train_embed = np.vstack([embeddings[value] for value in train.thread_id])
    test_embed = np.vstack([embeddings[value] for value in test.thread_id])
    embed_scaler = StandardScaler().fit(train_embed)
    pca = PCA(n_components=16, random_state=seed).fit(
        embed_scaler.transform(train_embed)
    )
    train_x = np.c_[
        scalar_scaler.transform(train_scalar),
        pca.transform(embed_scaler.transform(train_embed)),
    ]
    test_x = np.c_[
        scalar_scaler.transform(test_scalar),
        pca.transform(embed_scaler.transform(test_embed)),
    ]
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError("Non-finite qualification matrix")
    return train_x, test_x


def fit_scores(
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    scalar_columns: list[str],
    embeddings: dict[str, np.ndarray],
    seed: int,
) -> np.ndarray:
    train_x, test_x = prepare_matrices(
        train, test, scalar_columns, embeddings, seed
    )
    target = train.oracle_global_top50.to_numpy(bool)
    if model_name == "balanced_logistic":
        model = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        )
    elif model_name == "balanced_extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=TREES,
            min_samples_leaf=3,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.fit(train_x, target)
    classes = list(model.classes_)
    return model.predict_proba(test_x)[:, classes.index(True)]


def threshold_metrics(frame: pd.DataFrame, threshold: float) -> dict:
    selected = frame.loc[frame.probability >= threshold]
    positives = int(frame.oracle_global_top50.sum())
    hits = int(selected.oracle_global_top50.sum())
    return {
        "candidates": len(selected),
        "hits": hits,
        "recall": hits / positives if positives else 0.0,
        "precision": hits / len(selected) if len(selected) else 0.0,
    }


def choose_thresholds(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, model_rows in oof.groupby("model", sort=True):
        # A coarse fixed grid can falsely report that a small candidate cap is
        # infeasible when calibrated probabilities concentrate above 0.95.
        # Actual OOF probability boundaries are outer-train information and
        # permit every empirically distinguishable non-empty qualification set.
        probability_boundaries = np.unique(
            model_rows.probability.to_numpy(dtype=float)
        )
        threshold_candidates = np.unique(
            np.r_[np.asarray(THRESHOLDS, dtype=float), probability_boundaries]
        )
        for cap in CANDIDATE_CAPS:
            candidates = []
            for threshold in threshold_candidates:
                folds = []
                for _, event in model_rows.groupby("validation_event", sort=True):
                    folds.append(threshold_metrics(event, float(threshold)))
                values = {
                    "model": model_name,
                    "candidate_cap": cap,
                    "threshold": float(threshold),
                    "mean_candidates": float(np.mean([x["candidates"] for x in folds])),
                    "mean_recall": float(np.mean([x["recall"] for x in folds])),
                    "mean_precision": float(np.mean([x["precision"] for x in folds])),
                }
                if values["mean_candidates"] <= cap and values["mean_candidates"] > 0:
                    candidates.append(values)
            # Tree probabilities may contain a large tied maximum, so a model
            # can genuinely have no non-empty set below a very small cap.  Skip
            # that model/cap; another declared model may still be feasible.
            if not candidates:
                continue
            chosen = sorted(
                candidates,
                key=lambda x: (
                    -x["mean_recall"],
                    -x["mean_precision"],
                    -x["mean_candidates"],
                    x["threshold"],
                ),
            )[0]
            rows.append(chosen)
    selected = pd.DataFrame(rows)
    missing_caps = sorted(set(CANDIDATE_CAPS) - set(selected.candidate_cap))
    if missing_caps:
        raise ValueError(
            f"No declared classifier has a non-empty feasible threshold for caps {missing_caps}"
        )
    # Select the model independently inside each declared candidate capacity.
    selected = (
        selected.sort_values(
            ["candidate_cap", "mean_recall", "mean_precision", "model"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .groupby("candidate_cap", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return selected


def qualification_comparison(
    quiet: pd.DataFrame, baseline: pd.DataFrame, config: pd.Series
) -> tuple[dict, pd.DataFrame]:
    threshold = float(config.threshold)
    qualified = quiet.loc[quiet.probability >= threshold].copy()
    n = len(qualified)
    baseline_quiet = baseline.loc[baseline.quiet].sort_values(
        ["hybrid_calibrated_score", "thread_id"],
        ascending=[False, True],
        kind="stable",
    ).head(n)
    classifier_hits = int(qualified.oracle_global_top50.sum())
    rf_hits = int(baseline_quiet.thread_id.isin(
        set(quiet.loc[quiet.oracle_global_top50, "thread_id"])
    ).sum())
    metrics = {
        "candidate_cap": int(config.candidate_cap),
        "model": str(config.model),
        "threshold": threshold,
        "held_out_candidates": n,
        "held_out_classifier_hits": classifier_hits,
        "held_out_classifier_precision": classifier_hits / n if n else 0.0,
        "held_out_classifier_recall": classifier_hits / max(int(quiet.oracle_global_top50.sum()), 1),
        "same_size_quiet_rf_hits": rf_hits,
    }
    qualified["candidate_cap"] = int(config.candidate_cap)
    qualified["qualification_model"] = str(config.model)
    qualified["qualification_threshold"] = threshold
    return metrics, qualified


def replacement_metrics(
    baseline: pd.DataFrame,
    qualified_ids: set[str],
    cap: int,
    limit: int,
) -> dict:
    ranked = baseline.sort_values(
        ["hybrid_calibrated_score", "thread_id"],
        ascending=[False, True],
        kind="stable",
    )
    original = ranked.head(ORACLE_K).copy()
    original_ids = set(original.thread_id)
    challengers = ranked.loc[
        ranked.thread_id.isin(qualified_ids) & ~ranked.thread_id.isin(original_ids)
    ].head(limit)
    removable = original.loc[~original.thread_id.isin(qualified_ids)].sort_values(
        ["hybrid_calibrated_score", "thread_id"],
        ascending=[True, True],
        kind="stable",
    ).head(len(challengers))
    chosen = pd.concat(
        [original.loc[~original.thread_id.isin(set(removable.thread_id))], challengers],
        ignore_index=True,
    )
    oracle = ranked.sort_values(
        ["preventable_impact", "thread_id"],
        ascending=[False, True],
        kind="stable",
    ).head(ORACLE_K)
    oracle_ids = set(oracle.thread_id)
    return {
        "candidate_cap": cap,
        "replacement_limit": limit,
        "actual_replacements": len(challengers),
        "blocked_impact": float(chosen.preventable_impact.sum()),
        "model_reduction": float(chosen.preventable_impact.sum())
        / float(baseline.preventable_impact.sum()),
        "oracle_top50_hits": len(set(chosen.thread_id) & oracle_ids),
        "baseline_blocked_impact": float(original.preventable_impact.sum()),
        "baseline_oracle_top50_hits": len(original_ids & oracle_ids),
    }


def main() -> None:
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_quiet_oracle_membership_qualification_smoke"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke",
        "outer_event": OUTER_EVENT,
        "inner_events": list(INNER_EVENTS),
        "cutoff_seconds": 1800,
        "label": "event-global Oracle Top-50 membership; outcome-only",
        "models": ["balanced_logistic", "balanced_extra_trees"],
        "candidate_caps": list(CANDIDATE_CAPS),
        "replacement_limits": list(REPLACEMENT_LIMITS),
        "seed": SEED,
        "research_safety": (
            "All features are protected strict-30 inputs. PCA/scalers/models/gates "
            "and thresholds use outer-train events only. Held-out labels are metrics only."
        ),
    }
    save_record(output, record)
    try:
        print("Starting quiet Oracle-membership qualification smoke.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading and validating protected strict-30 artifacts...", flush=True)
        materialization = json.loads(MATERIALIZATION_RECORD.read_text(encoding="utf-8"))
        if materialization.get("status") != "complete":
            raise ValueError("Protected feature materialization is incomplete")
        frame = pd.read_csv(FEATURE_TABLE, dtype={"thread_id": str, "event_id": str})
        if len(frame) != 2402 or frame.thread_id.duplicated().any():
            raise ValueError("Invalid protected feature table")
        scalar_columns = [column for column in frame if feature_group(column) is not None]
        if not scalar_columns or not np.isfinite(frame[scalar_columns].to_numpy(float)).all():
            raise ValueError("Invalid protected scalar features")
        frame = add_oracle_label(frame)
        sizes = frame.groupby("event_id").size()
        eligible = sorted(sizes[sizes >= 100].index.tolist())
        artifact_events = sorted(frame.event_id.unique().tolist())
        graphs = torch.load(DATASET, weights_only=False)
        recency, embeddings = snapshot_inputs(graphs)

        print("[2/6] Producing outer-train inner-event OOF probabilities...", flush=True)
        oof_rows = []
        model_names = ("balanced_logistic", "balanced_extra_trees")
        for fold_index, validation_event in enumerate(INNER_EVENTS, 1):
            print(
                f"  inner fold {fold_index}/{len(INNER_EVENTS)}: {validation_event}",
                flush=True,
            )
            train_events = [
                event
                for event in eligible
                if event not in {OUTER_EVENT, validation_event}
            ]
            train, validation, _ = quiet_partition(
                frame,
                recency,
                train_events,
                validation_event,
                [event for event in artifact_events if event not in {OUTER_EVENT, validation_event}],
            )
            for model_name in model_names:
                probability = fit_scores(
                    model_name,
                    train,
                    validation,
                    scalar_columns,
                    embeddings,
                    SEED + fold_index * 100,
                )
                oof_rows.extend(
                    {
                        "validation_event": validation_event,
                        "thread_id": str(row.thread_id),
                        "model": model_name,
                        "probability": float(value),
                        "oracle_global_top50": bool(row.oracle_global_top50),
                    }
                    for row, value in zip(validation.itertuples(index=False), probability)
                )
        oof = pd.DataFrame(oof_rows)

        print("[3/6] Selecting binary thresholds under candidate-cap constraints...", flush=True)
        selected_configs = choose_thresholds(oof)
        for row in selected_configs.itertuples(index=False):
            print(
                f"  cap={row.candidate_cap}: {row.model}, threshold={row.threshold:.3f}, "
                f"inner recall={row.mean_recall:.3f}",
                flush=True,
            )

        print("[4/6] Fitting outer-train classifiers and qualifying held-out quiet threads...", flush=True)
        outer_train, outer_quiet, gate_threshold = quiet_partition(
            frame,
            recency,
            [event for event in eligible if event != OUTER_EVENT],
            OUTER_EVENT,
            [event for event in artifact_events if event != OUTER_EVENT],
        )
        probabilities = {}
        for model_name in set(selected_configs.model):
            probabilities[model_name] = fit_scores(
                model_name,
                outer_train,
                outer_quiet,
                scalar_columns,
                embeddings,
                SEED + 10000,
            )
        baseline = pd.read_csv(
            BASELINE_SCORES, dtype={"thread_id": str, "event_id": str}
        )
        baseline = baseline.loc[baseline.event_id.eq(OUTER_EVENT)].copy()
        if len(baseline) != int(sizes.loc[OUTER_EVENT]):
            raise ValueError("Held-out baseline identity mismatch")

        print("[5/6] Comparing same-size quiet RF and bounded replacement policies...", flush=True)
        comparison_rows = []
        qualified_rows = []
        replacement_rows = []
        for config in selected_configs.itertuples(index=False):
            evaluated = outer_quiet.copy()
            evaluated["probability"] = probabilities[str(config.model)]
            metrics, qualified = qualification_comparison(
                evaluated, baseline, pd.Series(config._asdict())
            )
            comparison_rows.append(metrics)
            qualified_rows.append(qualified)
            qualified_ids = set(qualified.thread_id)
            for limit in REPLACEMENT_LIMITS:
                replacement_rows.append(
                    replacement_metrics(
                        baseline, qualified_ids, int(config.candidate_cap), limit
                    )
                )
        comparisons = pd.DataFrame(comparison_rows)
        replacements = pd.DataFrame(replacement_rows)
        qualifications = pd.concat(qualified_rows, ignore_index=True)

        print("[6/6] Writing smoke evidence...", flush=True)
        oof.to_csv(output / "inner_oof_probabilities.csv", index=False)
        selected_configs.to_csv(output / "inner_selected_thresholds.csv", index=False)
        comparisons.to_csv(output / "held_out_qualification_comparison.csv", index=False)
        replacements.to_csv(output / "held_out_replacement_safety.csv", index=False)
        qualifications.to_csv(output / "held_out_qualified_threads.csv", index=False)
        record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "eligible_events": eligible,
                "scalar_feature_count": len(scalar_columns),
                "outer_quiet_gate_threshold": gate_threshold,
                "outer_train_quiet_threads": len(outer_train),
                "outer_test_quiet_threads": len(outer_quiet),
                "selected_thresholds": selected_configs.to_dict(orient="records"),
                "held_out_qualification": comparisons.to_dict(orient="records"),
                "held_out_replacement": replacements.to_dict(orient="records"),
                "output_files": [
                    "inner_oof_probabilities.csv",
                    "inner_selected_thresholds.csv",
                    "held_out_qualification_comparison.csv",
                    "held_out_replacement_safety.csv",
                    "held_out_qualified_threads.csv",
                ],
            }
        )
        save_record(output, record)
        print(
            f"SUCCESS: quiet Oracle-membership qualification smoke saved to: {output.resolve()}",
            flush=True,
        )
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        save_record(output, record)
        print(
            f"FAILURE: qualification smoke preserved at: {output.resolve()}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
