from __future__ import annotations
import numpy as np
import pandas as pd
from benchmarks.pheme_fixed30_unified import config

def load_validate() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    allow = pd.read_csv(config.ALLOWLIST, dtype={"thread_id": str, "event_id": str})
    scores = pd.read_csv(config.OOF, dtype={"thread_id": str, "event_id": str})
    if len(allow) != 2373 or allow.thread_id.duplicated().any(): raise ValueError("Unexpected allowlist identity")
    models = sorted(scores.model.unique())
    event_map = allow.set_index("thread_id").event_id.astype(str)
    impact_map = allow.set_index("thread_id").preventable_impact.astype(float)
    for model, rows in scores.groupby("model"):
        if len(rows) != len(allow) or rows.thread_id.duplicated().any() or set(rows.thread_id) != set(allow.thread_id): raise ValueError(f"OOF coverage mismatch: {model}")
        if not np.isfinite(rows.prediction.to_numpy(float)).all(): raise ValueError(f"Non-finite score: {model}")
        indexed = rows.set_index("thread_id").loc[event_map.index]
        if not indexed.event_id.astype(str).eq(event_map).all(): raise ValueError(f"Event mismatch: {model}")
        if not np.array_equal(indexed.preventable_impact.to_numpy(float), impact_map.to_numpy(float)): raise ValueError(f"Target mismatch: {model}")
    eligible = sorted(allow.groupby("event_id").size().loc[lambda x: x >= config.ELIGIBLE_MIN_THREADS].index)
    if len(eligible) != 7: raise ValueError("Expected seven eligible events")
    return allow, scores, eligible

def evaluate(scores: pd.DataFrame, eligible: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows=[]
    for model, model_rows in scores.loc[scores.event_id.isin(eligible)].groupby("model"):
        for event, pool in model_rows.groupby("event_id"):
            ranked=pool.sort_values(["prediction","thread_id"],ascending=[False,True],kind="stable"); total=float(pool.preventable_impact.sum())
            for budget in config.BUDGETS:
                count=min(budget,len(pool)); blocked=float(ranked.head(count).preventable_impact.sum())
                rows.append({"model":model,"event_id":event,"budget":budget,"actual_budget":count,"corrected_impact_capture":blocked/total if total else 0.0})
    per_event=pd.DataFrame(rows)
    macro=per_event.groupby(["model","budget"],as_index=False).agg(eligible_events=("event_id","nunique"),macro_corrected_impact_capture=("corrected_impact_capture","mean"),mean_actual_budget=("actual_budget","mean"))
    random=[]
    for event,pool in scores.loc[scores.event_id.isin(eligible)].drop_duplicates("thread_id").groupby("event_id"):
        for budget in config.BUDGETS: random.append({"model":"random_expected","event_id":event,"budget":budget,"actual_budget":min(budget,len(pool)),"corrected_impact_capture":min(budget,len(pool))/len(pool)})
    random=pd.DataFrame(random); per_event=pd.concat([per_event,random],ignore_index=True)
    random_macro=random.groupby(["model","budget"],as_index=False).agg(eligible_events=("event_id","nunique"),macro_corrected_impact_capture=("corrected_impact_capture","mean"),mean_actual_budget=("actual_budget","mean"))
    return per_event, pd.concat([macro,random_macro],ignore_index=True).sort_values(["model","budget"])
