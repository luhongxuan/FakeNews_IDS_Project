"""One-time held-out test evaluation for the frozen v2 weak-order candidate.

The configuration was selected only from the v2 validation run.  This script
does not compare or tune configurations: it refits the selected candidate on
train+validation and evaluates the protected test split once.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from head_focused_v2_strict_common import BASE, DATA, PROJECT_ROOT, RECORDS, load_train_validation_only
import run_validation_head_focused_scalar as v1

SELECTION_RUN = BASE / "reference_result" / "20260901_135117_434080_validation_head_focused_scalar_v2_strict_weak_order"
OUT = BASE / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_test_head_focused_scalar_v2_strict_weak_order")
FROZEN_ORDER_LAMBDA = 0.05
FROZEN_SOFT_TOPK_LAMBDA = 0.0
IMPACT_TARGETS = (0.10, 0.20, 0.40, 0.60)


def load_test_only(features: list[str]) -> pd.DataFrame:
    """Parse only test rows after selection/training configuration is frozen."""
    split_index = pd.read_csv(RECORDS, usecols=["split"], dtype={"split": "string"})
    non_test_lines = set((np.flatnonzero(~split_index["split"].eq("test")) + 1).tolist())
    test = pd.read_csv(
        RECORDS,
        skiprows=lambda line_number: line_number in non_test_lines,
        dtype={"corpus": "string", "thread_id": "string", "split": "string"},
    )
    if len(test) == 0 or set(test["split"].dropna().unique()) != {"test"}:
        raise ValueError("Test-only loader did not return exactly the test split.")
    if not np.isfinite(test[features].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite test feature value.")
    if not np.allclose(
        test["preventable_y"].to_numpy(dtype=float),
        np.log1p(test["preventable_impact"].to_numpy(dtype=float)),
    ):
        raise ValueError("Test preventable_y integrity failure.")
    return test


def min_threads_rows(pool: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    total = float(pool.preventable_impact.sum())
    ranked = pool.sort_values(["priority_score", "thread_id"], ascending=[False, True], kind="stable")
    oracle = pool.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable")
    model_cumulative = ranked.preventable_impact.cumsum().to_numpy(dtype=float)
    oracle_cumulative = oracle.preventable_impact.cumsum().to_numpy(dtype=float)
    for target in IMPACT_TARGETS:
        required = target * total
        model_k = int(np.searchsorted(model_cumulative, required, side="left") + 1)
        oracle_k = int(np.searchsorted(oracle_cumulative, required, side="left") + 1)
        rows.append({
            "impact_reduction_target": target,
            "model_threads_required": min(model_k, len(pool)),
            "oracle_threads_required": min(oracle_k, len(pool)),
            "model_reached_target": bool(len(model_cumulative) and model_cumulative[-1] >= required),
            "oracle_reached_target": bool(len(oracle_cumulative) and oracle_cumulative[-1] >= required),
        })
    return rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    selection = json.loads((SELECTION_RUN / "run_record.json").read_text(encoding="utf-8"))
    if selection.get("status") != "complete" or selection.get("chosen_config") != "0.05_0.0":
        raise ValueError("Frozen v2 selection record is missing or does not select 0.05_0.0.")
    print("Starting one-time held-out test evaluation for frozen weak-order scalar candidate...", flush=True)
    print(f"Output directory: {OUT.resolve()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "dataset": str(DATA),
        "selection_source": str(SELECTION_RUN),
        "frozen_selected_config": {"order_lambda": FROZEN_ORDER_LAMBDA, "soft_topk_lambda": FROZEN_SOFT_TOPK_LAMBDA},
        "selection_rule": "Frozen before this test run: v2 validation mean Oracle efficiency across both corpora and budgets (1,3,5,10,20,40).",
        "fit_split": "train + validation",
        "evaluation_split": "test, evaluated once after configuration freeze",
        "features": "31 source-anchored <=30-minute activity, fine-temporal, and graph-topology features",
        "seeds": list(v1.SEEDS),
        "epochs": v1.EPOCHS,
        "research_safety": "Test outcomes are loaded only for this final evaluation, never for fitting, scaling, pair construction, loss computation, or configuration selection.",
    }
    (OUT / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("[1/5] Loading protected train/validation records for fitting; test rows remain unparsed...", flush=True)
        train, validation, features, protected_test_count = load_train_validation_only()
        fit = pd.concat([train, validation], ignore_index=True)
        print(f"  fit rows={len(fit)}, protected test rows={protected_test_count}; features={len(features)}.", flush=True)
        print("[2/5] Loading test rows only after configuration has been verified frozen...", flush=True)
        test = load_test_only(features)
        print(f"  final evaluation rows={len(test)} across {test.corpus.nunique()} corpora.", flush=True)
        print("[3/5] Refitting one corpus-specific selected model per seed on train+validation...", flush=True)
        predictions: list[pd.DataFrame] = []
        losses: list[dict] = []
        for corpus in sorted(fit.corpus.unique()):
            fit_corpus = fit[fit.corpus == corpus]
            test_corpus = test[test.corpus == corpus]
            if len(test_corpus) == 0:
                raise ValueError(f"No test records for {corpus}.")
            print(f"  corpus={corpus}: fit={len(fit_corpus)}, test={len(test_corpus)}", flush=True)
            for seed in v1.SEEDS:
                prediction, training_losses = v1.train_one(
                    fit_corpus, test_corpus, features, seed, FROZEN_ORDER_LAMBDA, FROZEN_SOFT_TOPK_LAMBDA
                )
                losses.append({"corpus": corpus, "seed": seed, **training_losses})
                predictions.append(pd.DataFrame({"corpus": corpus, "thread_id": test_corpus.thread_id, "seed": seed, "priority_score": prediction}))
        print("[4/5] Computing complete held-out ranking, budget, and impact-target curves...", flush=True)
        all_predictions = pd.concat(predictions, ignore_index=True)
        mean_predictions = all_predictions.groupby(["corpus", "thread_id"], as_index=False).priority_score.mean()
        ranking = test.merge(mean_predictions, on=["corpus", "thread_id"], validate="one_to_one")
        ranking["priority_rank"] = ranking.groupby("corpus").priority_score.rank(method="first", ascending=False).astype(int)
        budget_rows: list[dict] = []
        target_rows: list[dict] = []
        for corpus, pool in ranking.groupby("corpus", sort=True):
            for row in v1.curve(pool, pool.priority_score.to_numpy()):
                budget_rows.append({"corpus": corpus, **row})
            for row in min_threads_rows(pool):
                target_rows.append({"corpus": corpus, **row})
        print("[5/5] Writing immutable test results and run record...", flush=True)
        ranking.sort_values(["corpus", "priority_rank"]).to_csv(OUT / "test_priority_ranking_any_k.csv", index=False)
        pd.DataFrame(budget_rows).to_csv(OUT / "test_budget_curve.csv", index=False)
        pd.DataFrame(target_rows).to_csv(OUT / "test_min_threads_by_reduction_target.csv", index=False)
        pd.DataFrame(losses).to_csv(OUT / "training_loss_summary.csv", index=False)
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "test_rows": len(test), "output_files": ["test_priority_ranking_any_k.csv", "test_budget_curve.csv", "test_min_threads_by_reduction_target.csv", "training_loss_summary.csv"]})
        (OUT / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: frozen weak-order scalar test evaluation saved to: {OUT.resolve()}", flush=True)
    except Exception as error:
        record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        (OUT / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: test evaluation output preserved at: {OUT.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
