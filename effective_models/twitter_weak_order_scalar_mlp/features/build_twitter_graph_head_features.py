"""Create a versioned strict-30-minute graph feature artifact for scalar ranking.

No split, label, or existing artifact is modified.  Raw tree reconstruction is
delegated to the audited Graph13 source-anchored Snowflake snapshot routine.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
MODEL_ROOT = BASE.parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from raw_graph_feature_builder import raw_graph_features  # noqa: E402

INPUT = ROOT / "data" / "derived" / "20260901_121751_764451_twitter15_16_to_pheme_raw_30min_transfer_v2_source_deduplicated"
OUT = MODEL_ROOT / "artifacts" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_twitter15_16_strict30_graph_head_features_v1")
BASE_FEATURES = ["log1p_observed_nodes", "log1p_count_0_10m", "log1p_count_10_20m", "log1p_count_20_30m", "log1p_count_recent_5m", "log1p_count_recent_10m", "log1p_seconds_since_last_activity", "log1p_median_interarrival_sec", "log1p_std_interarrival_sec", "late_activity_frac"]
NEW_FEATURES = ["raw_time_actions_bin_1_5m", "raw_time_actions_bin_2_5m", "raw_time_actions_bin_3_5m", "raw_time_actions_bin_4_5m", "raw_time_actions_bin_5_5m", "raw_time_actions_bin_6_5m", "raw_time_last_gap_seconds", "raw_time_interarrival_p90", "raw_time_burstiness", "raw_time_late_to_early_ratio", "raw_graph_max_depth", "raw_graph_mean_depth", "raw_graph_depth_std", "raw_graph_depth_p90", "raw_graph_depth_entropy", "raw_graph_leaf_fraction", "raw_graph_late_leaf_count", "raw_graph_root_children", "raw_graph_branch_hhi", "raw_graph_branch_gini", "raw_graph_width_to_depth"]
FEATURES = BASE_FEATURES + NEW_FEATURES

def main() -> None:
    if OUT.exists(): raise FileExistsError(f"Refusing to overwrite: {OUT}")
    print("Starting strict-30 Twitter15/16 graph-head feature build (no training)...", flush=True)
    print(f"Output directory: {OUT.resolve()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=False)
    rec = {"status":"running", "input":str(INPUT), "cutoff_seconds":1800, "features":FEATURES, "label":{"training_regression":"preventable_y = log1p(preventable_impact)", "future_use":"outcome only; never a model input"}, "research_safety":"Only source-anchored nodes and edges with both endpoints <=30 minutes are reconstructed. Existing split/labels are copied, not changed."}
    (OUT / "run_record.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
    try:
        print("[1/3] Loading immutable source-deduplicated Twitter candidate artifact...", flush=True)
        schema=json.loads((INPUT / "schema.json").read_text()); rows=pd.read_csv(INPUT / "twitter15_16_train_candidates.csv",dtype={"sample_id":"string","thread_id":"string","corpus":"string","original_label":"string"}); split=pd.read_csv(INPUT / "twitter15_16_source_split.csv",dtype={"sample_id":"string","split":"string"}); frame=rows.merge(split[["sample_id","split"]],on="sample_id",validate="one_to_one")
        if set(schema["feature_columns"]) != set(BASE_FEATURES) or set(frame.split) != {"train","validation","test"}: raise ValueError("Unexpected protected source artifact")
        print(f"  loaded {len(frame)} candidates: " + ", ".join(f"{k}={v}" for k,v in frame.groupby('split').size().sort_index().items()), flush=True)
        print("[2/3] Reconstructing pre-specified observed-forest depth and fine-temporal features...", flush=True)
        raw=[]
        for index,item in enumerate(frame[["corpus","thread_id"]].itertuples(index=False),start=1):
            if index==1 or index%100==0 or index==len(frame): print(f"  snapshots {index}/{len(frame)} ({item.corpus})",flush=True)
            raw.append(raw_graph_features(str(item.corpus),str(item.thread_id)))
        extra=pd.DataFrame(raw)[["corpus","thread_id"]+NEW_FEATURES]
        derived=frame.merge(extra,on=["corpus","thread_id"],how="left",validate="one_to_one")
        if derived[FEATURES].isna().any().any() or not np.isfinite(derived[FEATURES].to_numpy(float)).all() or not np.allclose(derived.preventable_y.to_numpy(float),np.log1p(derived.preventable_impact.to_numpy(float))): raise ValueError("Feature or target integrity failure")
        print(f"  validated {len(FEATURES)} safe early features and protected target integrity.",flush=True)
        print("[3/3] Writing versioned artifact and feature provenance...",flush=True)
        derived.to_csv(OUT / "thread_level_records.csv",index=False)
        provenance=[]
        for feature in FEATURES:
            provenance.append({"feature":feature,"group":"base_activity" if feature in BASE_FEATURES else "new_graph_or_timing","source":"existing strict-30 artifact" if feature in BASE_FEATURES else "raw tree reconstructed through Graph13 source-anchored Snowflake routine","availability":"<=30-minute nodes; observed edges only"})
        pd.DataFrame(provenance).to_csv(OUT / "feature_provenance.csv",index=False)
        (OUT / "schema.json").write_text(json.dumps({"metadata_columns":["corpus","sample_id","thread_id","original_label","split"],"target_columns":["preventable_impact","preventable_y"],"feature_columns":FEATURES,"cutoff_seconds":1800,"candidate_policy":"Twitter labels false/true/unverified; non-rumor excluded by immutable source artifact"},indent=2),encoding="utf-8")
        rec.update({"status":"complete","completed_at":datetime.now(timezone.utc).isoformat(),"records":len(derived),"feature_count":len(FEATURES),"output_files":["thread_level_records.csv","feature_provenance.csv","schema.json"]}); (OUT / "run_record.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
        print(f"SUCCESS: strict-30 graph-head feature artifact saved to: {OUT.resolve()}",flush=True)
    except Exception as error:
        rec.update({"status":"failed","error":f"{type(error).__name__}: {error}"});(OUT / "run_record.json").write_text(json.dumps(rec,indent=2),encoding="utf-8");print(f"FAILURE: feature artifact preserved at: {OUT.resolve()}",flush=True);raise
if __name__=="__main__": main()
