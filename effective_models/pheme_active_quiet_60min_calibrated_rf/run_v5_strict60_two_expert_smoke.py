"""Calibrated active/quiet RF smoke on protected-v5 strict-60 snapshots."""
from __future__ import annotations
from datetime import datetime, timezone
import json, traceback, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
DATASET=HERE/"experiments"/"20260902_231543_841935_v5_strict60_materialization_full"/"pheme_graphs_roberta_v5_strict60_preventableimpact.pt"
REPLIES=ROOT/"data"/"protected_research_assets"/"pheme_v5_strict30"/"pheme_reply_level_v4.csv"
OUT_ROOT=HERE/"experiments"; SEED=42; TREES=50; OUTER="charliehebdo"
OOF_EVENTS=("ferguson","germanwings-crash","putinmissing")
TEMPORAL_NAMES=("log1p_count_0_10m","log1p_count_10_20m","log1p_count_20_30m","log1p_count_30_40m","log1p_count_40_50m","log1p_count_50_60m","log1p_count_recent_5m","log1p_count_recent_10m","log1p_seconds_since_last_activity","log1p_median_interarrival_sec","log1p_std_interarrival_sec","observed_leaf_fraction","max_observed_children")
sys.path.insert(0,str(ROOT/"effective_models"/"pheme_active_quiet_calibrated_rf"/"training"))
from pheme_quiet_expert_calibration_common import fit_calibrators

def write(out,value): (out/"run_record.json").write_text(json.dumps(value,indent=2),encoding="utf-8")
def recency(graphs): return np.asarray([float(g.temporal_features[8]) for g in graphs],np.float32)
def interval_map():
    f=pd.read_csv(REPLIES,dtype={"thread_id":str,"tweet_id":str,"parent_id":str},usecols=["thread_id","tweet_id","parent_id","is_source","offset_sec"],low_memory=False)
    src=f.loc[f.is_source.eq(1),["thread_id","tweet_id"]].rename(columns={"tweet_id":"source"}); r=f.loc[f.is_source.ne(1)&f.offset_sec.gt(1800)&f.offset_sec.le(3600)].merge(src,on="thread_id",how="left")
    r["rr"]=r.parent_id.ne(r.source); branch=r.groupby(["thread_id","parent_id"]).size().groupby(level=0).max()
    a=r.groupby("thread_id").agg(n=("tweet_id","size"),rr=("rr","sum")); a["branch"]=branch
    return {str(i):(float(np.log1p(x.n)),float(x.rr/max(x.n,1)),float(np.log1p(x.branch))) for i,x in a.iterrows()}
def scalar(g,interval):
    t=g.temporal_features.numpy().astype(np.float32); depth=g.x[:,0].numpy(); extra=interval.get(str(g.thread_id),(0.,0.,0.)); early=float(t[:3].sum()); late=float(t[3:6].sum())
    return np.r_[t,np.log1p(g.num_nodes),depth.mean(),depth.max(),extra,late-early].astype(np.float32)
def text_raw(graphs):
    rows=[]
    for g in graphs:
        emb=g.x[:,13:].numpy().astype(np.float32); idx=np.flatnonzero(g.x[:,3].numpy()>0.5)
        if len(idx)!=1 or emb.shape[1]!=768: raise ValueError(f"Invalid source/text layout {g.thread_id}")
        reply=np.delete(emb,idx[0],axis=0); rows.append(np.r_[emb[idx[0]],reply.mean(0) if len(reply) else np.zeros(768,np.float32)])
    return np.vstack(rows)
def fit_predict(train,test,interval,seed):
    xs=np.vstack([scalar(g,interval) for g in train]); xt=np.vstack([scalar(g,interval) for g in test]); tr=text_raw(train); te=text_raw(test)
    s=StandardScaler().fit(tr); p=PCA(n_components=min(32,len(train)-1),random_state=seed).fit(s.transform(tr)); xs=np.c_[xs,p.transform(s.transform(tr))]; xt=np.c_[xt,p.transform(s.transform(te))]
    ages=np.asarray([g.x[:,12].mean().item() for g in train])[:,None]; aget=np.asarray([g.x[:,12].mean().item() for g in test])[:,None]; xs=np.c_[xs,ages];xt=np.c_[xt,aget]
    scale=StandardScaler().fit(xs); m=RandomForestRegressor(n_estimators=TREES,max_depth=8,min_samples_leaf=3,max_features=.8,n_jobs=1,random_state=seed).fit(scale.transform(xs),[float(g.preventable_y) for g in train])
    return m.predict(scale.transform(xt)).astype(np.float32)
def gated(train,test,interval,seed):
    threshold=float(np.quantile(recency(train),.6)); tq=recency(train)>=threshold; sq=recency(test)>=threshold; pred=np.empty(len(test),np.float32)
    for q in (False,True):
        tr=[g for g,v in zip(train,tq) if v==q]; te=[g for g,v in zip(test,sq) if v==q]
        if te: pred[sq==q]=fit_predict(tr,te,interval,seed+int(q))
    return pred,sq,threshold
def calibrator_frame(rows):
    f=pd.DataFrame(rows); n=f.validation_event.nunique(); f["event_weight"]=f.groupby("validation_event").thread_id.transform(lambda x:len(f)/(n*len(x))); return f
def apply(raw,quiet,cal):
    out=np.empty(len(raw),np.float32)
    for name,mask in (("quiet",quiet),("active",~quiet)):
        if mask.any(): out[mask]=cal[name]["coefficient"]*raw[mask]+cal[name]["intercept"]
    return out
def main():
    out=OUT_ROOT/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+"_v5_strict60_two_expert_smoke");out.mkdir(parents=True,exist_ok=False)
    rec={"status":"running","started_at":datetime.now(timezone.utc).isoformat(),"dataset":str(DATASET),"cutoff_seconds":3600,"outer_event":OUTER,"inner_oof_events":list(OOF_EVENTS),"trees":TREES,"features":"safe strict-60 temporal/depth/text/account-age plus explicit 30-60 reply-to-reply, branching, and acceleration","research_safety":"All model inputs are <=60 minutes; calibration is fitted on outer-train event OOF predictions; outer outcomes are evaluation-only."};write(out,rec)
    try:
        print("Starting v5 strict-60 calibrated two-expert smoke.",flush=True);print(f"Output directory: {out.resolve()}",flush=True);print("[1/5] Loading and validating strict-60 inputs...",flush=True)
        graphs=torch.load(DATASET,weights_only=False); interval=interval_map(); train=[g for g in graphs if str(g.event_id)!=OUTER]; test=[g for g in graphs if str(g.event_id)==OUTER]
        print("[2/5] Producing three outer-train OOF calibration folds...",flush=True);rows=[]
        for i,event in enumerate(OOF_EVENTS,1):
            print(f"  OOF {i}/3: {event}",flush=True); val=[g for g in train if str(g.event_id)==event]; inner=[g for g in train if str(g.event_id)!=event]; raw,q,_=gated(inner,val,interval,SEED+i)
            rows.extend({"validation_event":event,"thread_id":str(g.thread_id),"expert":"quiet" if z else "active","raw_score":float(s),"target":float(g.preventable_y)} for g,s,z in zip(val,raw,q))
        cal,details=fit_calibrators(calibrator_frame(rows));print("[3/5] Fitting outer-train active/quiet experts...",flush=True);raw,q,threshold=gated(train,test,interval,SEED+1000);score=apply(raw,q,cal)
        print("[4/5] Computing held-out Top-50 intervention result...",flush=True);f=pd.DataFrame({"thread_id":[str(g.thread_id) for g in test],"event_id":[str(g.event_id) for g in test],"preventable_impact_60m":[float(g.preventable_impact) for g in test],"score":score,"quiet":q});rank=f.sort_values(["score","thread_id"],ascending=[False,True]);oracle=f.sort_values(["preventable_impact_60m","thread_id"],ascending=[False,True]);blocked=float(rank.head(50).preventable_impact_60m.sum());optimal=float(oracle.head(50).preventable_impact_60m.sum());metrics={"reduction_at_50":blocked/float(f.preventable_impact_60m.sum()),"oracle_efficiency_at_50":blocked/optimal,"oracle_top50_hits":len(set(rank.head(50).thread_id)&set(oracle.head(50).thread_id))}
        print("[5/5] Writing smoke evidence...",flush=True);f.to_csv(out/"outer_scores.csv",index=False);rec.update({"status":"complete","completed_at":datetime.now(timezone.utc).isoformat(),"graphs":len(graphs),"gate_threshold":threshold,"calibration":details,"metrics":metrics,"output_files":["outer_scores.csv"]});write(out,rec);print(f"SUCCESS: v5 strict-60 calibrated two-expert smoke saved to: {out.resolve()}",flush=True)
    except BaseException as e: rec.update({"status":"failed","failed_at":datetime.now(timezone.utc).isoformat(),"error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()});write(out,rec);print(f"FAILURE: v5 strict-60 two-expert smoke preserved at: {out.resolve()}",flush=True);raise
if __name__=="__main__":main()
