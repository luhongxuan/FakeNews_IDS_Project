"""Create strictly-past event-context features for 30-minute PHEME snapshots."""
from __future__ import annotations
import json
from datetime import timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import sys
BASE_DIR=Path(__file__).resolve().parent; ROOT=BASE_DIR.parent; sys.path.insert(0,str(ROOT))
from data_pipeline.audit_rumdetect2017_timing import parse_created_at
INPUT=BASE_DIR/'artifacts'/'partial_observation'/'20260830_143000_pheme_author_roles_30min_v1'
OUTPUT=BASE_DIR/'artifacts'/'event_context'/'20260830_160000_strict_past_event_context_v1'
RAW=ROOT/'data'/'raw'/'pheme'; BASE_FEATURES=['observed_actions','observed_unique_users','observed_user_hhi','log1p_count_20_30m','seconds_since_last_action','prior_user_actions_mean','log1p_followers_mean']
CUT_MINUTES=30
def source_times():
 out={}; print('Reading raw PHEME source timestamps...',flush=True)
 for event_dir in sorted(RAW.glob('*-all-rnr-threads')):
  event=event_dir.name.removesuffix('-all-rnr-threads'); print(f'  event={event}',flush=True)
  for category in ('rumours','non-rumours'):
   for td in (event_dir/category).glob('*'):
    files=[p for p in (td/'source-tweets').glob('*.json') if not p.name.startswith('._')]
    if len(files)!=1: continue
    try:
     value=json.loads(files[0].read_text(encoding='utf-8',errors='replace')); when=parse_created_at(value.get('created_at'))
     if when is not None: out[str(value['id'])]=when
    except (OSError,KeyError,json.JSONDecodeError): pass
 return out
def main():
 if OUTPUT.exists(): raise FileExistsError(f'Refusing to overwrite: {OUTPUT}')
 print('Loading schema-locked role-aware records...',flush=True)
 schema=json.loads((INPUT/'schema.json').read_text()); d=pd.read_csv(INPUT/'thread_level_records.csv',dtype=schema['csv_dtypes']); times=source_times()
 if d.thread_id.isin(times).sum()!=len(d): raise ValueError('Missing raw source timestamp for an included record')
 d['decision_time']=[times[x]+timedelta(minutes=CUT_MINUTES) for x in d.thread_id]
 additions=[]
 for event,frame in d.groupby('event_id',sort=True):
  print(f'Building strict-past context: {event} ({len(frame)} threads)',flush=True); frame=frame.sort_values('decision_time',kind='stable'); history=[]
  for _,batch in frame.groupby('decision_time',sort=True):
   batch_rows=list(batch.itertuples(index=False))
   for row in batch_rows:
    values={'event_context_prior_threads':float(len(history))}
    for col in BASE_FEATURES:
     past=np.asarray([getattr(x,col) for x in history],dtype=float)
     current=float(getattr(row,col))
     values[f'event_context_{col}_percentile']=float(np.mean(past<=current)) if past.size else 0.5
     values[f'event_context_{col}_zscore']=(current-float(past.mean()))/(float(past.std())+1e-6) if past.size else 0.0
    values['thread_id']=row.thread_id; additions.append(values)
   history.extend(batch_rows)
 context=pd.DataFrame(additions); d=d.merge(context,on='thread_id',validate='one_to_one')
 context_cols=[c for c in context.columns if c!='thread_id']; features=schema['feature_columns']+context_cols
 if not np.isfinite(d[features].to_numpy(float)).all(): raise ValueError('Non-finite event-context feature')
 OUTPUT.mkdir(parents=True); d.drop(columns=['decision_time']).to_csv(OUTPUT/'thread_level_records.csv',index=False); pd.read_csv(INPUT/'excluded_threads.csv').to_csv(OUTPUT/'excluded_threads.csv',index=False)
 (OUTPUT/'schema.json').write_text(json.dumps({'metadata_columns':schema['metadata_columns'],'target_columns':schema['target_columns'],'feature_columns':features,'base_feature_columns':schema['feature_columns'],'event_context_feature_columns':context_cols,'csv_dtypes':schema['csv_dtypes']},indent=2),encoding='utf-8')
 (OUTPUT/'manifest.json').write_text(json.dumps({'protocol':'strict_past_event_context','input':str(INPUT),'decision_time':f'raw source created_at + {CUT_MINUTES} minutes','context_rule':'same-event snapshots with strictly earlier decision_time only; no labels or same/later snapshots','context_base_features':BASE_FEATURES,'research_safety':'Event context is an operational feature, not a test normalization statistic.'},indent=2),encoding='utf-8')
 print(f'Created strict-past event-context dataset: {OUTPUT}',flush=True)
if __name__=='__main__': main()
