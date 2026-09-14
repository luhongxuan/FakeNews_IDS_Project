from __future__ import annotations
import numpy as np
import pandas as pd
from benchmarks.pheme_static_dynamic_bridge import config

def load_validate():
    fixed=pd.read_csv(config.FIXED_OOF,dtype={"thread_id":str,"event_id":str}); dynamic=pd.read_csv(config.DYNAMIC_OOF,dtype={"thread_id":str,"event_id":str})
    if fixed.duplicated(["model","thread_id"]).any() or dynamic.duplicated(["model","thread_id","checkpoint_sec"]).any(): raise ValueError("Duplicate OOF identity")
    if not np.isfinite(fixed.prediction.to_numpy(float)).all() or not np.isfinite(dynamic.prediction.to_numpy(float)).all(): raise ValueError("Non-finite prediction")
    counts=fixed.drop_duplicates("thread_id").groupby("event_id").size(); eligible=sorted(counts[counts>=config.ELIGIBLE_MIN_THREADS].index)
    if len(eligible)!=7 or set(dynamic.checkpoint_sec.astype(int))!=set(config.CHECKPOINTS): raise ValueError("Unexpected bridge contract")
    ids=set(fixed.loc[fixed.event_id.isin(eligible),"thread_id"].astype(str))
    for model,rows in fixed.groupby("model"):
        eligible_rows=rows.loc[rows.event_id.isin(eligible)]
        if set(eligible_rows.thread_id.astype(str))!=ids: raise ValueError(f"Fixed identity mismatch: {model}")
    for model,rows in dynamic.groupby("model"):
        if len(rows)!=len(ids)*len(config.CHECKPOINTS) or set(rows.thread_id.astype(str))!=ids: raise ValueError(f"Dynamic coverage mismatch: {model}")
    fixed_target=fixed.loc[fixed.event_id.isin(eligible)].drop_duplicates("thread_id").set_index("thread_id").preventable_impact.astype(float)
    dynamic_30=dynamic.loc[dynamic.checkpoint_sec.eq(1800)].drop_duplicates("thread_id").set_index("thread_id").dynamic_preventable_impact.astype(float)
    if not np.array_equal(fixed_target.loc[sorted(ids)].to_numpy(),dynamic_30.loc[sorted(ids)].to_numpy()): raise ValueError("Fixed/dynamic target mismatch at 30 minutes")
    return fixed,dynamic,eligible

def evaluate(fixed,dynamic,eligible):
    horizons={event:float(pool.loc[pool.checkpoint_sec.eq(600)].drop_duplicates("thread_id").future_growth.sum()) for event,pool in dynamic.groupby("event_id")}
    rows=[]; selections=[]
    random_source=fixed.loc[fixed.event_id.isin(eligible)].drop_duplicates("thread_id")
    for event,pool in random_source.groupby("event_id"):
        expected_blocked=float(pool.preventable_impact.sum())*min(config.TOTAL_BUDGET,len(pool))/len(pool)
        rows.append({"model":"random_expected_static","decision_design":"fixed_30min","event_id":event,"interventions":min(config.TOTAL_BUDGET,len(pool)),"blocked_corrected_future_nodes":expected_blocked,"horizon_future_nodes_at_10min":horizons[event],"unified_horizon_capture_at_budget50":expected_blocked/horizons[event],"mean_action_minute":30.})
    for model,model_rows in fixed.loc[fixed.event_id.isin(eligible)].groupby("model"):
        for event,pool in model_rows.groupby("event_id"):
            chosen=pool.sort_values(["prediction","thread_id"],ascending=[False,True],kind="stable").head(50).copy(); blocked=float(chosen.preventable_impact.sum()); chosen["action_minute"]=30.; chosen["model"]=model; selections.append(chosen)
            rows.append({"model":model,"decision_design":"fixed_30min","event_id":event,"interventions":len(chosen),"blocked_corrected_future_nodes":blocked,"horizon_future_nodes_at_10min":horizons[event],"unified_horizon_capture_at_budget50":blocked/horizons[event],"mean_action_minute":30.})
    for model,model_rows in dynamic.loc[dynamic.event_id.isin(eligible)].groupby("model"):
        for event,pool in model_rows.groupby("event_id"):
            ids=set(); parts=[]
            for checkpoint,quota in zip(config.CHECKPOINTS,config.QUOTAS):
                available=pool.loc[pool.checkpoint_sec.eq(checkpoint)&~pool.thread_id.isin(ids)].sort_values(["prediction","thread_id"],ascending=[False,True],kind="stable"); chosen=available.head(quota).copy()
                if len(chosen)!=quota: raise ValueError("Insufficient dynamic candidates")
                chosen["action_minute"]=checkpoint/60.; parts.append(chosen); ids.update(chosen.thread_id)
            chosen=pd.concat(parts,ignore_index=True); blocked=float(chosen.dynamic_preventable_impact.sum()); selections.append(chosen)
            rows.append({"model":model,"decision_design":"dynamic_10_to_60min","event_id":event,"interventions":len(chosen),"blocked_corrected_future_nodes":blocked,"horizon_future_nodes_at_10min":horizons[event],"unified_horizon_capture_at_budget50":blocked/horizons[event],"mean_action_minute":float(chosen.action_minute.mean())})
    per=pd.DataFrame(rows); macro=per.groupby(["model","decision_design"],as_index=False).agg(eligible_events=("event_id","nunique"),mean_interventions=("interventions","mean"),total_blocked_corrected_future_nodes=("blocked_corrected_future_nodes","sum"),total_horizon_future_nodes_at_10min=("horizon_future_nodes_at_10min","sum"),macro_unified_horizon_capture_at_budget50=("unified_horizon_capture_at_budget50","mean"),mean_action_minute=("mean_action_minute","mean")); macro["pooled_unified_horizon_capture_at_budget50"]=macro.total_blocked_corrected_future_nodes/macro.total_horizon_future_nodes_at_10min
    return per,macro,pd.concat(selections,ignore_index=True)
