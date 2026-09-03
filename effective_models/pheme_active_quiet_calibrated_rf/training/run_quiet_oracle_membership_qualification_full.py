"""Seven-event nested LOEO quiet Oracle-membership qualification run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch

import run_quiet_oracle_membership_qualification_smoke as common


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE.parent / "experiments"
SEED = 42
FULL_TREES = 300


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def main() -> None:
    common.TREES = FULL_TREES
    output = OUT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_quiet_oracle_membership_qualification_full"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "full",
        "split": "seven eligible-event outer LOEO; six-event inner OOF threshold/model selection",
        "cutoff_seconds": 1800,
        "label": "event-global Oracle Top-50 membership; outcome-only",
        "models": ["balanced_logistic", "balanced_extra_trees"],
        "extra_trees": FULL_TREES,
        "candidate_caps": list(common.CANDIDATE_CAPS),
        "replacement_limits": list(common.REPLACEMENT_LIMITS),
        "seed": SEED,
        "inputs": {
            "features": str(common.FEATURE_TABLE.resolve()),
            "graphs": str(common.DATASET.resolve()),
            "baseline_scores": str(common.BASELINE_SCORES.resolve()),
        },
        "research_safety": (
            "Every outer event is absent from gate/model/PCA/threshold selection. "
            "Small artifact events may contribute only to the unsupervised recency "
            "gate; they do not contribute classifier labels or model training."
        ),
    }
    write_record(output, record)
    try:
        print("Starting full quiet Oracle-membership qualification LOEO.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading and validating protected strict-30 artifacts...", flush=True)
        materialization = json.loads(
            common.MATERIALIZATION_RECORD.read_text(encoding="utf-8")
        )
        if materialization.get("status") != "complete":
            raise ValueError("Protected materialization is incomplete")
        frame = pd.read_csv(
            common.FEATURE_TABLE, dtype={"thread_id": str, "event_id": str}
        )
        if len(frame) != 2402 or frame.thread_id.duplicated().any():
            raise ValueError("Invalid protected feature table")
        scalar_columns = [column for column in frame if common.feature_group(column) is not None]
        if not scalar_columns or not np.isfinite(frame[scalar_columns].to_numpy(float)).all():
            raise ValueError("Invalid protected scalar feature matrix")
        frame = common.add_oracle_label(frame)
        sizes = frame.groupby("event_id").size()
        eligible = sorted(sizes[sizes >= 100].index.tolist())
        artifact_events = sorted(frame.event_id.unique().tolist())
        if len(eligible) != 7:
            raise ValueError(f"Expected seven eligible events, found {eligible}")
        graphs = torch.load(common.DATASET, weights_only=False)
        recency, embeddings = common.snapshot_inputs(graphs)
        baseline_all = pd.read_csv(
            common.BASELINE_SCORES, dtype={"thread_id": str, "event_id": str}
        )
        if baseline_all.thread_id.duplicated().any():
            raise ValueError("Duplicate held-out baseline identities")

        all_configs = []
        all_comparisons = []
        all_replacements = []
        all_qualified = []
        print("[2/6] Running seven outer folds and nested inner OOF selection...", flush=True)
        for outer_index, outer_event in enumerate(eligible, 1):
            print(f"  outer fold {outer_index}/{len(eligible)}: {outer_event}", flush=True)
            inner_events = [event for event in eligible if event != outer_event]
            oof_rows = []
            for inner_index, validation_event in enumerate(inner_events, 1):
                print(
                    f"    inner fold {inner_index}/{len(inner_events)}: {validation_event}",
                    flush=True,
                )
                classifier_train_events = [
                    event
                    for event in eligible
                    if event not in {outer_event, validation_event}
                ]
                gate_events = [
                    event
                    for event in artifact_events
                    if event not in {outer_event, validation_event}
                ]
                train, validation, _ = common.quiet_partition(
                    frame,
                    recency,
                    classifier_train_events,
                    validation_event,
                    gate_events,
                )
                for model_name in ("balanced_logistic", "balanced_extra_trees"):
                    probabilities = common.fit_scores(
                        model_name,
                        train,
                        validation,
                        scalar_columns,
                        embeddings,
                        SEED + outer_index * 1000 + inner_index * 100,
                    )
                    oof_rows.extend(
                        {
                            "validation_event": validation_event,
                            "thread_id": str(row.thread_id),
                            "model": model_name,
                            "probability": float(probability),
                            "oracle_global_top50": bool(row.oracle_global_top50),
                        }
                        for row, probability in zip(
                            validation.itertuples(index=False), probabilities
                        )
                    )
            selected_configs = common.choose_thresholds(pd.DataFrame(oof_rows))
            selected_configs["outer_event"] = outer_event
            all_configs.append(selected_configs)

            classifier_train, outer_quiet, gate_threshold = common.quiet_partition(
                frame,
                recency,
                [event for event in eligible if event != outer_event],
                outer_event,
                [event for event in artifact_events if event != outer_event],
            )
            outer_probabilities = {}
            for model_name in set(selected_configs.model):
                outer_probabilities[model_name] = common.fit_scores(
                    model_name,
                    classifier_train,
                    outer_quiet,
                    scalar_columns,
                    embeddings,
                    SEED + outer_index * 10000,
                )
            baseline = baseline_all.loc[baseline_all.event_id.eq(outer_event)].copy()
            if len(baseline) != int(sizes.loc[outer_event]):
                raise ValueError(f"{outer_event}: baseline identity mismatch")
            for config in selected_configs.itertuples(index=False):
                evaluated = outer_quiet.copy()
                evaluated["probability"] = outer_probabilities[str(config.model)]
                metrics, qualified = common.qualification_comparison(
                    evaluated, baseline, pd.Series(config._asdict())
                )
                metrics.update(
                    {
                        "outer_event": outer_event,
                        "quiet_gate_threshold": gate_threshold,
                        "outer_quiet_threads": len(outer_quiet),
                    }
                )
                all_comparisons.append(metrics)
                qualified["outer_event"] = outer_event
                all_qualified.append(qualified)
                qualified_ids = set(qualified.thread_id)
                for limit in common.REPLACEMENT_LIMITS:
                    replacement = common.replacement_metrics(
                        baseline,
                        qualified_ids,
                        int(config.candidate_cap),
                        limit,
                    )
                    replacement["outer_event"] = outer_event
                    all_replacements.append(replacement)
            pd.DataFrame(all_comparisons).to_csv(
                output / "outer_qualification_partial.csv", index=False
            )
            pd.DataFrame(all_replacements).to_csv(
                output / "outer_replacement_partial.csv", index=False
            )

        print("[3/6] Aggregating qualification-set performance...", flush=True)
        configs = pd.concat(all_configs, ignore_index=True)
        comparisons = pd.DataFrame(all_comparisons)
        replacements = pd.DataFrame(all_replacements)
        qualified = pd.concat(all_qualified, ignore_index=True)
        qualification_summary = comparisons.groupby("candidate_cap", as_index=False).agg(
            eligible_events=("outer_event", "nunique"),
            mean_candidates=("held_out_candidates", "mean"),
            total_classifier_hits=("held_out_classifier_hits", "sum"),
            total_same_size_quiet_rf_hits=("same_size_quiet_rf_hits", "sum"),
            mean_classifier_precision=("held_out_classifier_precision", "mean"),
            mean_classifier_recall=("held_out_classifier_recall", "mean"),
        )
        print("[4/6] Aggregating bounded replacement safety...", flush=True)
        replacement_summary = replacements.groupby(
            ["candidate_cap", "replacement_limit"], as_index=False
        ).agg(
            eligible_events=("outer_event", "nunique"),
            total_actual_replacements=("actual_replacements", "sum"),
            mean_model_reduction=("model_reduction", "mean"),
            total_oracle_top50_hits=("oracle_top50_hits", "sum"),
            mean_baseline_reduction=(
                "baseline_blocked_impact",
                lambda values: float(
                    np.mean(
                        [
                            float(value)
                            / float(
                                baseline_all.loc[
                                    baseline_all.event_id.eq(event), "preventable_impact"
                                ].sum()
                            )
                            for value, event in zip(
                                values,
                                replacements.loc[values.index, "outer_event"],
                            )
                        ]
                    )
                ),
            ),
            total_baseline_oracle_top50_hits=("baseline_oracle_top50_hits", "sum"),
        )

        print("[5/6] Writing fold and summary artifacts...", flush=True)
        configs.to_csv(output / "outer_selected_thresholds.csv", index=False)
        comparisons.to_csv(output / "outer_qualification_comparison.csv", index=False)
        replacements.to_csv(output / "outer_replacement_safety.csv", index=False)
        qualified.to_csv(output / "outer_qualified_threads.csv", index=False)
        qualification_summary.to_csv(output / "qualification_summary.csv", index=False)
        replacement_summary.to_csv(output / "replacement_summary.csv", index=False)

        print("[6/6] Finalizing complete run record...", flush=True)
        record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "eligible_events": eligible,
                "scalar_feature_count": len(scalar_columns),
                "qualification_summary": qualification_summary.to_dict(orient="records"),
                "replacement_summary": replacement_summary.to_dict(orient="records"),
                "output_files": [
                    "outer_selected_thresholds.csv",
                    "outer_qualification_comparison.csv",
                    "outer_replacement_safety.csv",
                    "outer_qualified_threads.csv",
                    "qualification_summary.csv",
                    "replacement_summary.csv",
                ],
            }
        )
        write_record(output, record)
        print(
            f"SUCCESS: full quiet Oracle-membership qualification saved to: {output.resolve()}",
            flush=True,
        )
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "output_files": sorted(
                    path.name for path in output.iterdir() if path.is_file()
                ),
            }
        )
        write_record(output, record)
        print(
            f"FAILURE: full qualification run preserved at: {output.resolve()}",
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
