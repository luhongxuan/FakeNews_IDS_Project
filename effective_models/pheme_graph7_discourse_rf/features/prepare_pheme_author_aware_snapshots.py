"""Create strict 30-minute PHEME author-aware thread records (no training)."""
from __future__ import annotations
import json, math, sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data_pipeline.audit_rumdetect2017_timing import parse_created_at

CUT = timedelta(minutes=30)
PROFILE = ("followers_count", "friends_count", "statuses_count", "verified", "created_at")
ROOT = PROJECT_ROOT / "data" / "raw" / "pheme"
OUTPUT = BASE_DIR / "artifacts" / "partial_observation" / "20260830_143000_pheme_author_roles_30min_v1"

def load(path): return json.loads(path.read_text(encoding="utf-8", errors="replace"))
def structure_maps(tree):
    parents, depth = {}, {}
    def walk(values, parent=None, d=0):
        for key, children in values.items():
            if key in parents: raise ValueError("repeated structure node")
            parents[str(key)], depth[str(key)] = parent, d
            if isinstance(children, dict): walk(children, str(key), d + 1)
    walk(tree); return parents, depth
def files(folder): return [p for p in folder.glob("*.json") if not p.name.startswith("._")]
def quant(values, prefix):
    a=np.asarray(values,dtype=float)
    return {prefix+"_mean":float(a.mean()),prefix+"_max":float(a.max()),prefix+"_p90":float(np.quantile(a,.9))}
def quant_or_zero(values, prefix):
    return quant(values, prefix) if values else {prefix+"_mean":0.0,prefix+"_max":0.0,prefix+"_p90":0.0}

def scan(root, status_path):
    threads=[]; actions=[]; excluded=[]
    scanned=0
    for event_dir in sorted(root.glob("*-all-rnr-threads")):
      event=event_dir.name.removesuffix("-all-rnr-threads")
      for category in ("rumours","non-rumours"):
       for td in sorted((event_dir/category).glob("*")):
        if not td.is_dir(): continue
        scanned += 1
        if scanned % 100 == 0:
          status_path.write_text(json.dumps({"status":"running","stage":"raw_action_scan","scanned_threads":scanned,"valid_threads":len(threads),"excluded_threads":len(excluded),"retained_actions":len(actions),"updated_at":datetime.now(timezone.utc).isoformat()}))
          print(f"Scanned {scanned} PHEME threads", flush=True)
        try:
          parents, depth=structure_maps(load(td/"structure.json")); tid=td.name
          tweets={}
          for p in files(td/"source-tweets")+files(td/"reactions"):
            t=load(p); tweets[str(t["id"])] = t
          roots=[k for k,v in parents.items() if v is None]
          if len(roots)!=1 or roots[0] not in tweets: raise ValueError("missing_raw_source")
          rootid=roots[0]; source=parse_created_at(tweets[rootid].get("created_at"))
          if source is None: raise ValueError("missing_source_timestamp")
          rows=[]
          # Extra reaction JSON not referenced by structure.json is not part
          # of the propagation tree and must not exclude this thread.
          for aid in parents:
            if aid not in tweets: raise ValueError("missing_raw_tree_action")
            t=tweets[aid]
            u=t.get("user") or {}; uid=str(u.get("id_str") or u.get("id") or "")
            when=parse_created_at(t.get("created_at"))
            if not uid or when is None or when < source: raise ValueError("missing_or_pre_source_action")
            # Retaining the full Twitter user object for every action can use
            # gigabytes of memory.  Keep only values this builder aggregates.
            rows.append({"id":aid,"user":uid,"time":when,"followers":float(u.get("followers_count") or 0),"parent":parents[aid],"depth":depth[aid]})
          by_id={row["id"]:row for row in rows}
          if any(row["parent"] is not None and by_id[row["parent"]]["time"] > row["time"] for row in rows):
            raise ValueError("parent_timestamp_after_child")
          actions.extend((row["time"],row["user"],tid,row["id"]) for row in rows)
          threads.append((tid,event,category,source,rows))
        except (OSError,KeyError,ValueError,json.JSONDecodeError) as e: excluded.append({"thread_id":td.name,"event_id":event,"reason":str(e)})
    return threads,actions,excluded

def main():
 out=OUTPUT
 if out.exists(): raise FileExistsError(out)
 out.mkdir(parents=True)
 status=out/"build_status.json"
 status.write_text(json.dumps({"status":"running","stage":"raw_action_scan","started_at":datetime.now(timezone.utc).isoformat()}))
 print("Starting partial-observation author-aware 30-minute dataset build...", flush=True)
 print(f"Output: {out}", flush=True)
 threads,actions,excluded=scan(ROOT, status); actions.sort(key=lambda x:x[0]); hist=Counter(); hmap={}; i=0
 status.write_text(json.dumps({"status":"running","stage":"strict_prior_history","valid_threads":len(threads),"valid_actions":len(actions)}))
 while i<len(actions):
  j=i+1
  while j<len(actions) and actions[j][0]==actions[i][0]: j+=1
  for _,u,t,n in actions[i:j]: hmap[(t,n)]=hist[u]
  for _,u,_,_ in actions[i:j]: hist[u]+=1
  i=j
 records=[]
 for tid,event,category,source,rows in threads:
  obs=[r for r in rows if r["time"]<=source+CUT]; future=[r for r in rows if r["time"]>source+CUT]
  obs_ids={r["id"] for r in obs}; children=defaultdict(list)
  for r in rows:
   if r["parent"] is not None: children[r["parent"]].append(r["id"])
  offsets=sorted((r["time"]-source).total_seconds() for r in obs); users=Counter(r["user"] for r in obs); histories=[hmap[(tid,r["id"])] for r in obs]
  followers=[math.log1p(max(0,r["followers"])) for r in obs]
  source_row=next(r for r in obs if r["parent"] is None)
  reply_rows=[r for r in obs if r["id"] != source_row["id"]]
  reply_histories=[hmap[(tid,r["id"])] for r in reply_rows]
  reply_followers=[math.log1p(max(0,r["followers"])) for r in reply_rows]
  visible_children=[sum(c in obs_ids for c in children[r["id"]]) for r in obs]
  # With validated temporal parent→child order and intervention on every
  # observed node (including the source), each future node lies on an
  # unbroken future-only suffix beginning at an intervened observed ancestor.
  preventable=len(future)
  rec={"thread_id":tid,"event_id":event,"category":category,"observed_actions":len(obs),"observed_unique_users":len(users),"observed_user_hhi":float(sum((v/len(obs))**2 for v in users.values())),"log1p_count_0_10m":math.log1p(sum(x<=600 for x in offsets)),"log1p_count_10_20m":math.log1p(sum(600<x<=1200 for x in offsets)),"log1p_count_20_30m":math.log1p(sum(1200<x<=1800 for x in offsets)),"seconds_since_last_action":1800-max(offsets),"observed_leaf_fraction":float(np.mean(np.asarray(visible_children)==0)),"mean_observed_children":float(np.mean(visible_children)),"preventable_impact":preventable,"preventable_y":math.log1p(preventable)}
  rec["seconds_since_last_action"]=float(CUT.total_seconds()-max(offsets))
  for start_minute in range(30,int(CUT.total_seconds()/60),10):
   end_minute=min(start_minute+10,int(CUT.total_seconds()/60))
   rec[f"log1p_count_{start_minute}_{end_minute}m"]=math.log1p(sum(start_minute*60 < x <= end_minute*60 for x in offsets))
  rec.update(quant(histories,"prior_user_actions")); rec.update(quant(followers,"log1p_followers"))
  rec.update({"source_prior_user_actions":float(hmap[(tid,source_row["id"])]),"source_log1p_followers":math.log1p(max(0,source_row["followers"]))})
  rec.update(quant_or_zero(reply_histories,"reply_prior_user_actions")); rec.update(quant_or_zero(reply_followers,"reply_log1p_followers"))
  rec["source_minus_reply_prior_history_mean"]=rec["source_prior_user_actions"]-rec["reply_prior_user_actions_mean"]
  rec["source_minus_reply_log1p_followers_mean"]=rec["source_log1p_followers"]-rec["reply_log1p_followers_mean"]
  records.append(rec)
 frame=pd.DataFrame(records)
 if frame.empty or not np.isfinite(frame.select_dtypes("number").to_numpy()).all(): raise ValueError("empty_or_nonfinite_records")
 metadata_columns=["thread_id","event_id","category"]
 target_columns=["preventable_impact","preventable_y"]
 feature_columns=[column for column in frame.columns if column not in {*metadata_columns,*target_columns}]
 frame.to_csv(out/"thread_level_records.csv",index=False); pd.DataFrame(excluded).to_csv(out/"excluded_threads.csv",index=False)
 (out/"schema.json").write_text(json.dumps({"metadata_columns":metadata_columns,"target_columns":target_columns,"feature_columns":feature_columns,"csv_dtypes":{"thread_id":"string","event_id":"string","category":"string"}},indent=2))
 (out/"manifest.json").write_text(json.dumps({"protocol":"partial_observation: tree-referenced raw JSON only; unreferenced reaction JSON ignored","cutoff_minutes":CUT.total_seconds()/60,"history_rule":"strictly earlier action timestamps; equal timestamps excluded","features":"Only <= cutoff action/profile aggregates; identifiers/category are metadata, not model inputs","target":"preventable_impact: future nodes reached through a future-only suffix after intervening on all observed nodes; temporal parent-child order is validated","records":len(frame),"excluded":len(excluded)},indent=2))
 status.write_text(json.dumps({"status":"complete","records":len(frame),"excluded":len(excluded),"completed_at":datetime.now(timezone.utc).isoformat()}))
 print(f"Created {len(frame)} records at {out}")
if __name__=="__main__": main()
