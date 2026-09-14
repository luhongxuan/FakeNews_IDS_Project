from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
import pandas as pd

from effective_models.twitter_weak_order_scalar_mlp import config
from effective_models.twitter_weak_order_scalar_mlp.features.strict30_features import load_non_test, load_test
from effective_models.twitter_weak_order_scalar_mlp.model.scalar_mlp import train_predict
from effective_models.twitter_weak_order_scalar_mlp.evaluation.metrics import budget_curve


def _now(): return datetime.now(timezone.utc).isoformat()
def _save(path: Path, value: dict): path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run(mode: str) -> Path:
    if mode not in {"smoke", "full"}: raise ValueError(mode)
    output = config.EXPERIMENTS / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_weak_order_scalar_{mode}"
    output.mkdir(parents=True, exist_ok=False)
    epochs = 2 if mode == "smoke" else config.EPOCHS
    seeds = config.SEEDS[:1] if mode == "smoke" else config.SEEDS
    record = {"status": "running", "started_at": _now(), "mode": mode,
              "model": "31-input weak-order scalar MLP", "dataset": str(config.ARTIFACT.resolve()),
              "split": "smoke=train->validation without test; full=train+validation->frozen test",
              "cutoff_seconds": config.CUTOFF_SECONDS, "seeds": seeds, "epochs": epochs,
              "frozen_config": {"order_lambda": config.ORDER_LAMBDA, "soft_topk_lambda": 0.0},
              "selection_source": str(config.REFERENCE_SELECTION.resolve())}
    _save(output / "run_record.json", record)
    try:
        print(f"Starting weak-order scalar {mode}.\nOutput directory: {output.resolve()}", flush=True)
        print("[1/5] Loading train/validation while protecting test rows...", flush=True)
        train, validation, features, protected_count = load_non_test()
        if len(features) != 31: raise ValueError(f"Expected 31 features, got {len(features)}")
        selection = json.loads((config.REFERENCE_SELECTION / "run_record.json").read_text(encoding="utf-8"))
        if selection.get("chosen_config") != "0.05_0.0": raise ValueError("Frozen selection record mismatch")
        if mode == "smoke":
            corpora = [sorted(train.corpus.unique())[0]]; fit, evaluation = train, validation
            print(f"[2/5] Smoke uses one corpus and {epochs} epochs; {protected_count} test rows remain unparsed...", flush=True)
        else:
            corpora = sorted(train.corpus.unique()); fit = pd.concat([train, validation], ignore_index=True)
            print("[2/5] Frozen configuration verified; loading protected test once for reproduction...", flush=True)
            evaluation = load_test(features)
        print("[3/5] Training corpus-specific seed ensemble...", flush=True)
        predictions, losses = [], []
        for corpus in corpora:
            fit_part, eval_part = fit.loc[fit.corpus.eq(corpus)], evaluation.loc[evaluation.corpus.eq(corpus)]
            print(f"  corpus={corpus}: fit={len(fit_part)}, evaluate={len(eval_part)}", flush=True)
            for seed in seeds:
                prediction, loss = train_predict(fit_part, eval_part, features, seed, epochs, config.ORDER_LAMBDA)
                predictions.append(pd.DataFrame({"corpus": corpus, "thread_id": eval_part.thread_id, "seed": seed, "priority_score": prediction}))
                losses.append({"corpus": corpus, "seed": seed, **loss})
        print("[4/5] Aggregating rankings and budget curves...", flush=True)
        raw = pd.concat(predictions, ignore_index=True)
        mean = raw.groupby(["corpus", "thread_id"], as_index=False).priority_score.mean()
        ranking = evaluation.loc[evaluation.corpus.isin(corpora)].merge(mean, on=["corpus", "thread_id"], validate="one_to_one")
        ranking["priority_rank"] = ranking.groupby("corpus").priority_score.rank(method="first", ascending=False).astype(int)
        rows = [{"corpus": corpus, **row} for corpus, pool in ranking.groupby("corpus") for row in budget_curve(pool)]
        print("[5/5] Writing outputs and complete run record...", flush=True)
        ranking.sort_values(["corpus", "priority_rank"]).to_csv(output / "priority_ranking.csv", index=False)
        pd.DataFrame(rows).to_csv(output / "budget_curve.csv", index=False)
        pd.DataFrame(losses).to_csv(output / "training_loss_summary.csv", index=False)
        result = {"corpora": corpora, "features": len(features), "rows": rows,
                  "smoke_is_not_efficacy_evidence": mode == "smoke",
                  "test_already_consumed_do_not_tune_from_reproduction": mode == "full"}
        _save(output / "result.json", result)
        record.update(status="complete", completed_at=_now(), metrics_results=result,
                      output_file_inventory=["priority_ranking.csv", "budget_curve.csv", "training_loss_summary.csv", "result.json", "run_record.json"],
                      failure_details=None,
                      research_safety={"artifact_and_split_unchanged": True, "cutoff_30_minutes": True,
                                       "test_not_loaded_in_smoke": mode == "smoke",
                                       "frozen_config_not_selected_on_test": True})
        _save(output / "run_record.json", record)
        print(f"SUCCESS: weak-order scalar {mode} saved to: {output.resolve()}", flush=True); return output
    except BaseException as error:
        record.update(status="failed", completed_at=_now(), failure_details={"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        _save(output / "run_record.json", record); print(f"FAILURE: weak-order scalar {mode} preserved at: {output.resolve()}", flush=True); raise

