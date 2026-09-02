"""Strict development ablation: utility-only versus weak Oracle-head order.

This intentionally does not retry soft-Top-K.  Version 1 already showed it
hurts validation, so this run changes one hypothesis at a time: whether a
small order weight can help without overwhelming the utility objective.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

from head_focused_v2_strict_common import BASE, DATA, load_train_validation_only
import run_validation_head_focused_scalar as v1

OUT = BASE / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_validation_head_focused_scalar_v2_strict_weak_order")
CONFIGS = ((0.0, 0.0), (0.02, 0.0), (0.05, 0.0), (0.10, 0.0))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    print("Starting strict train/validation weak-order scalar ablation...", flush=True)
    print(f"Output directory: {OUT.resolve()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "dataset": str(DATA),
        "split": "Train fits; validation selects; test rows are skipped before feature/target parsing.",
        "hypothesis": "A small head-order weight may complement tail-weighted utility without the v1 loss domination caused by lambda 0.5 or 1.0.",
        "controlled_change_from_v1": "Soft-Top-K is disabled. Pair construction, 31 strict-30-minute features, model, seeds, epochs, selection metric, and split remain unchanged. Only order_lambda grid is reduced.",
        "configs": [{"order_lambda": order, "soft_topk_lambda": soft} for order, soft in CONFIGS],
        "selection": "Mean validation Oracle efficiency across Twitter15/Twitter16 and budgets (1,3,5,10,20,40).",
        "seeds": list(v1.SEEDS),
        "epochs": v1.EPOCHS,
        "research_safety": "All inputs are source-anchored <=30-minute features. preventable impact is an outcome-only train target and validation metric. Test targets/features are not loaded.",
    }
    (OUT / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("[1/4] Loading train/validation records; skipping every test row before parsing targets...", flush=True)
        train, validation, features, test_count = load_train_validation_only()
        print(f"  train={len(train)}, validation={len(validation)}, protected test rows skipped={test_count}; features={len(features)}.", flush=True)
        print("[2/4] Training corpus-specific scalar models over weak-order configurations...", flush=True)
        scores: list[pd.DataFrame] = []
        runs: list[dict] = []
        for config_number, (order_lambda, soft_lambda) in enumerate(CONFIGS, 1):
            config = f"{order_lambda}_{soft_lambda}"
            print(f"  config {config_number}/{len(CONFIGS)}: order_lambda={order_lambda}, soft_topk_lambda={soft_lambda}", flush=True)
            for corpus in sorted(train.corpus.unique()):
                train_corpus = train[train.corpus == corpus]
                validation_corpus = validation[validation.corpus == corpus]
                for seed in v1.SEEDS:
                    prediction, losses = v1.train_one(train_corpus, validation_corpus, features, seed, order_lambda, soft_lambda)
                    runs.append({"config": config, "order_lambda": order_lambda, "soft_topk_lambda": soft_lambda, "corpus": corpus, "seed": seed, **losses})
                    scores.append(pd.DataFrame({"thread_id": validation_corpus.thread_id, "corpus": corpus, "config": config, "seed": seed, "priority_score": prediction}))
        print("[3/4] Selecting solely on validation multi-budget Oracle efficiency...", flush=True)
        all_scores = pd.concat(scores, ignore_index=True)
        metrics: list[dict] = []
        for config, grouped_scores in all_scores.groupby("config"):
            for corpus, corpus_scores in grouped_scores.groupby("corpus"):
                mean_score = corpus_scores.groupby("thread_id", as_index=False).priority_score.mean()
                pool = validation[validation.corpus == corpus].merge(mean_score, on="thread_id", validate="one_to_one")
                for row in v1.curve(pool, pool.priority_score.to_numpy()):
                    metrics.append({"config": config, "corpus": corpus, **row})
        metric_frame = pd.DataFrame(metrics)
        selection = metric_frame.groupby("config").oracle_efficiency.mean().sort_values(ascending=False)
        chosen = str(selection.index[0])
        print(f"  selected={chosen}; validation mean multi-budget Oracle efficiency={selection.iloc[0]:.4f}", flush=True)
        print("[4/4] Writing validation rankings, curves, losses, and run record...", flush=True)
        chosen_scores = all_scores[all_scores.config == chosen].groupby(["corpus", "thread_id"], as_index=False).priority_score.mean()
        ranking = validation.merge(chosen_scores, on=["corpus", "thread_id"], validate="one_to_one")
        ranking["priority_rank"] = ranking.groupby("corpus").priority_score.rank(method="first", ascending=False).astype(int)
        ranking.sort_values(["corpus", "priority_rank"]).to_csv(OUT / "validation_priority_ranking_any_k.csv", index=False)
        metric_frame.to_csv(OUT / "validation_budget_curve_all_configs.csv", index=False)
        pd.DataFrame(runs).to_csv(OUT / "training_loss_summary.csv", index=False)
        selection.rename("mean_validation_oracle_efficiency_across_corpora_and_budgets").to_csv(OUT / "validation_config_selection.csv")
        record.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "chosen_config": chosen, "validation_selection": selection.to_dict(), "output_files": ["validation_priority_ranking_any_k.csv", "validation_budget_curve_all_configs.csv", "training_loss_summary.csv", "validation_config_selection.csv"]})
        (OUT / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"SUCCESS: strict weak-order scalar validation result saved to: {OUT.resolve()}", flush=True)
    except Exception as error:
        record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        (OUT / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FAILURE: strict weak-order scalar output preserved at: {OUT.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
