"""Formal fixed-config LOEO RF for the partial-observation protocol."""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASE_DIR=Path(__file__).resolve().parent.parent
DATASET=BASE_DIR/'artifacts'/'20260830_133000_pheme_author_aware_30min_schema_v2'
OUTPUT=BASE_DIR/'experiments'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_author_aware_rf_partial_formal')
SEED=42
PARAMS={'n_estimators':300,'max_depth':8,'min_samples_leaf':3,'max_features':0.8,'n_jobs':1,'random_state':SEED}
def crr(frame):
 return float(frame.preventable_impact.sum())
def main():
 print('Loading schema-locked partial-observation dataset...',flush=True)
 schema=json.loads((DATASET/'schema.json').read_text(encoding='utf-8'))
 d=pd.read_csv(DATASET/'thread_level_records.csv',dtype=schema['csv_dtypes']); f=schema['feature_columns']
 if set(f)&set(schema['metadata_columns']+schema['target_columns']): raise ValueError('Unsafe schema')
 if not np.isfinite(d[f].to_numpy(float)).all(): raise ValueError('Non-finite feature')
 OUTPUT.mkdir(parents=True,exist_ok=False); rows={}; rng=np.random.default_rng(SEED)
 for event in sorted(d.event_id.unique()):
  train,test=d[d.event_id!=event],d[d.event_id==event]
  if set(train.event_id)&set(test.event_id): raise ValueError('Event leakage')
  print(f'Training LOEO fold: {event}...',flush=True)
  model=RandomForestRegressor(**PARAMS).fit(train[f],train.preventable_y)
  pool=test[test.category=='rumours'].copy(); pool['prediction']=model.predict(pool[f])
  total=crr(pool); model_value=crr(pool.nlargest(10,'prediction')); size_value=crr(pool.nlargest(10,'observed_actions'))
  random_values=[crr(pool.iloc[rng.choice(len(pool),size=min(10,len(pool)),replace=False)]) for _ in range(200)] if len(pool) else [0.0]
  rows[event]={'n_candidate_rumours':int(len(pool)),'crr_model':model_value/total if total else 0.0,'crr_size':size_value/total if total else 0.0,'crr_random':float(np.mean(random_values))/total if total else 0.0,'blocked_model':model_value,'candidate_future_total':total}
  print(f'[{event}] CRR@10 model={rows[event]["crr_model"]:.4f} size={rows[event]["crr_size"]:.4f} random={rows[event]["crr_random"]:.4f}',flush=True)
 inc=[x for x in rows.values() if x['n_candidate_rumours']>=100]
 report={'config':{'protocol':'partial_observation','split':'LOEO by event','candidate_pool':'oracle rumour category; never a feature','cutoff_minutes':30,'model':'RandomForestRegressor','params':PARAMS,'features':f,'selection':'top 10','safety':'schema-enforced features; event-disjoint folds; fixed configuration copied from earlier RF baseline, with no outer-test tuning'},'events':rows,'mean_excluding_small_events':{k:float(np.mean([x[k] for x in inc])) for k in ('crr_model','crr_size','crr_random')}}
 (OUTPUT/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(f'Formal result saved to: {OUTPUT}',flush=True)
if __name__=='__main__': main()
