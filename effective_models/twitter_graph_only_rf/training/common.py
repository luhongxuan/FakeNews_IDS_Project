"""Single canonical smoke/full pipeline for Twitter15/16 graph-only RF."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from effective_models.twitter_graph_only_rf.config import (
    CUTOFF_SECONDS,
    DATA_ROOT,
    EARLY_GRAPH_FEATURES,
    EXPERIMENT_ROOT,
    FULL_PARAMETERS,
    SEED,
    SMOKE_PARAMETERS,
    SPLIT_ROOT,
    TOP_K,
)
from effective_models.twitter_graph_only_rf.features.graph_only_features import load_protected_data


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def fit_rows(frame: pd.DataFrame, params: dict[str, object]) -> RandomForestRegressor:
    if frame.empty:
        raise ValueError("Cannot fit graph-only RF on an empty frame")
    return RandomForestRegressor(**params).fit(
        frame[list(EARLY_GRAPH_FEATURES)], frame.preventable_y
    )


def evaluate(
    model: RandomForestRegressor, frame: pd.DataFrame
) -> tuple[dict[str, float | int], pd.DataFrame]:
    if frame.empty:
        raise ValueError("Cannot evaluate graph-only RF on an empty frame")
    pool = frame.copy()
    pool["score"] = model.predict(pool[list(EARLY_GRAPH_FEATURES)])
    selected = pool.sort_values(
        ["score", "thread_id"], ascending=[False, True], kind="stable"
    ).head(TOP_K).copy()
    oracle = pool.sort_values(
        ["preventable_impact", "thread_id"], ascending=[False, True], kind="stable"
    ).head(TOP_K)
    total = float(pool.preventable_impact.sum())
    captured = float(selected.preventable_impact.sum())
    oracle_captured = float(oracle.preventable_impact.sum())
    result = {
        "candidates": int(len(pool)),
        "selected_preventable_impact": captured,
        "total_preventable_impact": total,
        "crr_at_10": captured / total if total else 0.0,
        "oracle_selected_preventable_impact": oracle_captured,
        "oracle_crr_at_10": oracle_captured / total if total else 0.0,
        "impact_efficiency_vs_oracle": captured / oracle_captured if oracle_captured else 0.0,
        "exact_oracle_overlap": int(selected.thread_id.isin(set(oracle.thread_id)).sum()),
    }
    selected["selected_by_model"] = True
    selected["selected_by_oracle"] = selected.thread_id.isin(set(oracle.thread_id))
    return result, selected


def _run_corpus(
    frame: pd.DataFrame,
    corpus: str,
    parameters: dict[str, dict[str, object]],
    include_test: bool,
) -> tuple[dict, pd.DataFrame | None]:
    train = frame[(frame.corpus == corpus) & (frame.split == "train")]
    validation = frame[(frame.corpus == corpus) & (frame.split == "validation")]
    validation_scores: dict[str, float] = {}
    for name, params in parameters.items():
        result, _ = evaluate(fit_rows(train, params), validation)
        validation_scores[name] = result["crr_at_10"]
        print(f"    validation {name}: CRR@10={result['crr_at_10']:.4f}", flush=True)
    chosen = max(parameters, key=lambda name: (validation_scores[name], name))
    result: dict = {
        "chosen_on_validation": chosen,
        "validation_crr_at_10": validation_scores,
    }
    if not include_test:
        return result, None
    train_validation = frame[
        (frame.corpus == corpus) & frame.split.isin(["train", "validation"])
    ]
    test = frame[(frame.corpus == corpus) & (frame.split == "test")]
    test_result, selected = evaluate(
        fit_rows(train_validation, parameters[chosen]), test
    )
    result["test"] = test_result
    return result, selected.assign(corpus=corpus)


def run_experiment(mode: str) -> Path:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")
    is_smoke = mode == "smoke"
    parameters = SMOKE_PARAMETERS if is_smoke else FULL_PARAMETERS
    suffix = "twitter15_16_graph_only_rf_smoke" if is_smoke else "twitter15_16_graph_only_rf_full"
    output = EXPERIMENT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{suffix}")
    output.mkdir(parents=True, exist_ok=False)
    record_path = output / "run_record.json"
    record = {
        "status": "running",
        "started_at": now(),
        "mode": mode,
        "model": "Twitter15/16 graph-only RandomForestRegressor",
        "configuration": {
            "cutoff_seconds": CUTOFF_SECONDS,
            "top_k": TOP_K,
            "seed": SEED,
            "features": list(EARLY_GRAPH_FEATURES),
            "parameters": parameters,
            "smoke_difference": (
                "one corpus, validation-only, five trees per candidate"
                if is_smoke
                else None
            ),
        },
        "input_artifacts": {
            "records": str((DATA_ROOT / "thread_level_records.csv").resolve()),
            "assignments": str((SPLIT_ROOT / "assignments.csv").resolve()),
            "split_manifest": str((SPLIT_ROOT / "manifest.json").resolve()),
        },
        "split": (
            "fixed source-level train/validation; test not evaluated"
            if is_smoke
            else "fixed source-level train/validation/test; validation selects and frozen test is evaluated once"
        ),
        "cutoff_seconds": CUTOFF_SECONDS,
        "seed": SEED,
        "metrics_results": {},
        "output_file_inventory": [],
        "failure_details": None,
    }
    write_json(record_path, record)
    print(f"Starting Twitter graph-only RF {mode} run.", flush=True)
    print(f"Output directory: {output.resolve()}", flush=True)
    try:
        print("[1/4] Loading immutable records and predefined source split...", flush=True)
        frame, manifest = load_protected_data()
        corpora = sorted(frame.corpus.astype(str).unique())
        if is_smoke:
            corpora = corpora[:1]
        print(f"[2/4] Running shared selection pipeline for {len(corpora)} corpus/corpora...", flush=True)
        selected_rows: list[pd.DataFrame] = []
        for index, corpus in enumerate(corpora, 1):
            print(f"  corpus {index}/{len(corpora)}: {corpus}", flush=True)
            result, selected = _run_corpus(frame, corpus, parameters, include_test=not is_smoke)
            record["metrics_results"][corpus] = result
            if selected is not None:
                selected_rows.append(selected)
            write_json(record_path, record)
        print("[3/4] Validating outputs and research-safety boundary...", flush=True)
        if is_smoke and any("test" in row for row in record["metrics_results"].values()):
            raise AssertionError("Smoke run must not evaluate the protected test split")
        if not is_smoke and len(selected_rows) != len(corpora):
            raise AssertionError("Full run did not produce one test selection per corpus")
        print("[4/4] Writing run record and selected-thread output...", flush=True)
        if selected_rows:
            pd.concat(selected_rows, ignore_index=True).to_csv(
                output / "test_model_selected_threads.csv", index=False
            )
        record.update(
            {
                "status": "complete",
                "completed_at": now(),
                "split_manifest": manifest,
                "research_safety": {
                    "predefined_split_unchanged": True,
                    "future_impact_used_as_feature": False,
                    "test_used_for_model_selection": False,
                    "protected_test_evaluated": not is_smoke,
                },
                "output_file_inventory": sorted(
                    path.name for path in output.iterdir() if path.is_file()
                ),
            }
        )
        write_json(record_path, record)
        print(f"SUCCESS: Twitter graph-only RF {mode} saved to: {output.resolve()}", flush=True)
        return output
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "completed_at": now(),
                "failure_details": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "output_file_inventory": sorted(
                    path.name for path in output.iterdir() if path.is_file()
                ),
            }
        )
        write_json(record_path, record)
        print(f"FAILURE: Twitter graph-only RF {mode} preserved at: {output.resolve()}", flush=True)
        raise
