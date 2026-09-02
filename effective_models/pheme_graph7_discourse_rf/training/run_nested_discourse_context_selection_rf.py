"""Nested LOEO selection across role, event-context, and discourse bundles."""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
BASE=Path(__file__).resolve().parent.parent; DATA=BASE/'artifacts'/'20260830_170000_early_discourse_response_v1'; OUT=BASE/'experiments'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_nested_discourse_context_selection'); CUTOFF_MINUTES=30; P={'n_estimators':300,'max_depth':8,'min_samples_leaf':3,'max_features':0.8,'n_jobs':1,'random_state':42}
def evaluate(train,test,features):
 m=RandomForestRegressor(**P).fit(train[features],train.preventable_y); pool=test[test.category=='rumours'].copy(); pool['score']=m.predict(pool[features]); total=float(pool.preventable_impact.sum()); return {'n_candidate_rumours':int(len(pool)),'crr_at_10':float(pool.nlargest(10,'score').preventable_impact.sum())/total if total else 0.0}
def main():
 print('Loading discourse/context dataset for nested LOEO...',flush=True); s=json.loads((DATA/'schema.json').read_text()); d=pd.read_csv(DATA/'thread_level_records.csv',dtype=s['csv_dtypes']); discourse=s['discourse_feature_columns']; context=[x for x in s['base_feature_columns'] if x.startswith('event_context_')]; role=[x for x in s['base_feature_columns'] if x not in context]; variants={'role_base':role,'plus_event_context':role+context,'plus_event_context_discourse':role+context+discourse}
 if any(set(v)&set(s['metadata_columns']+s['target_columns']) for v in variants.values()): raise ValueError('Unsafe feature selection')
 OUT.mkdir(parents=True,exist_ok=False); events=sorted(d.event_id.unique()); outer={}
 for outer_event in events:
  print(f'Outer event: {outer_event}',flush=True); inner={k:[] for k in variants}
  for val in (e for e in events if e!=outer_event):
   tr=d[~d.event_id.isin([outer_event,val])]; te=d[d.event_id==val]
   for name,features in variants.items():
    r=evaluate(tr,te,features)
    if r['n_candidate_rumours']>=100: inner[name].append(r['crr_at_10'])
  means={k:float(np.mean(v)) for k,v in inner.items()}; chosen=max(variants,key=lambda k:(means[k],k=='role_base')); r=evaluate(d[d.event_id!=outer_event],d[d.event_id==outer_event],variants[chosen]); outer[outer_event]={'chosen_variant':chosen,'inner_mean_crr':means,'outer':r}; print(f'  chose={chosen}; outer CRR@10={r["crr_at_10"]:.4f}',flush=True)
 vals=[x['outer']['crr_at_10'] for x in outer.values() if x['outer']['n_candidate_rumours']>=100]; report={'config':{'split':'nested LOEO','cutoff_minutes':CUTOFF_MINUTES,'variants':list(variants),'candidate_pool':'oracle rumour category','params':P,'safety':'All representation selection uses inner events only; discourse/context features are cutoff-safe and label-free.'},'outer_events':outer,'nested_mean_crr_at_10_excluding_small_events':float(np.mean(vals))}; (OUT/'result.json').write_text(json.dumps(report,indent=2)); print(f'Nested discourse/context result saved to: {OUT}',flush=True)
if __name__=='__main__': main()
