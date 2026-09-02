"""Build time-safe early-discourse response features for nested evaluation."""
from __future__ import annotations
import json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent; ROOT=BASE.parent; sys.path.insert(0,str(ROOT))
from data_pipeline.audit_rumdetect2017_timing import parse_created_at
INPUT=BASE/'artifacts'/'event_context'/'20260830_160000_strict_past_event_context_v1'; TEXT=BASE/'artifacts'/'text_semantic'/'20260830_153000_partial_30min_frozen_roberta.npz'; RAW=ROOT/'data'/'raw'/'pheme'; OUTPUT=BASE/'artifacts'/'discourse_response'/'20260830_170000_early_discourse_response_v1'; CUTOFF_MINUTES=30
ENQUIRY=re.compile(r'\b(?:is|are|was|were)\s+(?:this|that|it)\s+(?:true|real|confirmed)\b|\b(?:can|could)\s+(?:anyone|someone)\s+(?:confirm|verify)\b|\b(?:source|proof|evidence)\s*\?',re.I); CORRECTION=re.compile(r'\b(?:fake|hoax|false|debunk(?:ed)?|not\s+true)\b',re.I); EVIDENCE=re.compile(r'\b(?:source|proof|evidence|confirm|verify)\b',re.I)
def node_ids(tree,out=None):
 out=set() if out is None else out
 for key,value in tree.items(): out.add(str(key)); node_ids(value,out) if isinstance(value,dict) else None
 return out
def score(text):
 text=text or ''; return {'enquiry':int(bool(ENQUIRY.search(text))),'correction':int(bool(CORRECTION.search(text))),'evidence':int(bool(EVIDENCE.search(text))),'question':int('?' in text)}
def main():
 if OUTPUT.exists(): raise FileExistsError(f'Refusing to overwrite: {OUTPUT}')
 print('Loading strict-past event-context records...',flush=True); s=json.loads((INPUT/'schema.json').read_text()); d=pd.read_csv(INPUT/'thread_level_records.csv',dtype=s['csv_dtypes']); wanted=set(d.thread_id); rows=[]
 for ed in sorted(RAW.glob('*-all-rnr-threads')):
  event=ed.name.removesuffix('-all-rnr-threads'); print(f'Extracting early discourse features: {event}',flush=True)
  for cat in ('rumours','non-rumours'):
   for td in (ed/cat).glob('*'):
    tid=td.name
    if tid not in wanted: continue
    try:
     ids=node_ids(json.loads((td/'structure.json').read_text(encoding='utf-8'))); tweets={}
     for folder in ('source-tweets','reactions'):
      for p in (td/folder).glob('*.json'):
       if not p.name.startswith('._'):
        t=json.loads(p.read_text(encoding='utf-8',errors='replace')); tweets[str(t['id'])]=t
     source=tweets[tid]; start=parse_created_at(source.get('created_at')); early=[t for key,t in tweets.items() if key in ids and (when:=parse_created_at(t.get('created_at'))) is not None and start<=when<=start+pd.Timedelta(minutes=CUTOFF_MINUTES)]
     values=[score(t.get('text','')) for t in early]; n=max(len(values),1); rows.append({'thread_id':tid,'discourse_early_text_actions':len(values),**{f'discourse_{k}_count':sum(v[k] for v in values) for k in ('enquiry','correction','evidence','question')},**{f'discourse_{k}_fraction':sum(v[k] for v in values)/n for k in ('enquiry','correction','evidence','question')}})
    except (OSError,KeyError,json.JSONDecodeError,TypeError): raise
 discourse=pd.DataFrame(rows)
 if len(discourse)!=len(d): raise ValueError(f'Missing discourse rows: {len(d)-len(discourse)}')
 z=np.load(TEXT); order=d.thread_id.astype(str).to_numpy()
 if not np.array_equal(z['thread_id'].astype(str),order): raise ValueError('Text artifact order mismatch')
 src=z['source_embedding']; rep=z['early_reply_embedding_mean']; denom=np.linalg.norm(src,axis=1)*np.linalg.norm(rep,axis=1); semantic=pd.DataFrame({'thread_id':order,'semantic_source_reply_cosine':np.divide((src*rep).sum(axis=1),denom,out=np.zeros(len(d)),where=denom>0),'semantic_reply_centroid_norm':np.linalg.norm(rep,axis=1),'semantic_has_early_reply_text':(z['early_reply_count']>0).astype(float)})
 d=d.merge(discourse,on='thread_id',validate='one_to_one').merge(semantic,on='thread_id',validate='one_to_one'); added=[c for c in d.columns if c.startswith('discourse_') or c.startswith('semantic_')]; features=s['feature_columns']+added
 if not np.isfinite(d[features].to_numpy(float)).all(): raise ValueError('Non-finite discourse feature')
 OUTPUT.mkdir(parents=True); d.to_csv(OUTPUT/'thread_level_records.csv',index=False); pd.read_csv(INPUT/'excluded_threads.csv').to_csv(OUTPUT/'excluded_threads.csv',index=False); (OUTPUT/'schema.json').write_text(json.dumps({'metadata_columns':s['metadata_columns'],'target_columns':s['target_columns'],'feature_columns':features,'base_feature_columns':s['feature_columns'],'discourse_feature_columns':added,'csv_dtypes':s['csv_dtypes']},indent=2)); (OUTPUT/'manifest.json').write_text(json.dumps({'protocol':'early_discourse_response','cutoff_minutes':CUTOFF_MINUTES,'patterns':'pre-specified enquiry/correction/evidence/question regexes; source and tree-referenced replies only','semantic':'source-reply cosine and reply-centroid norm from frozen time-safe embeddings','research_safety':'Only raw texts at or before cutoff; no labels in feature construction.'},indent=2)); print(f'Created early-discourse dataset: {OUTPUT}',flush=True)
if __name__=='__main__': main()
