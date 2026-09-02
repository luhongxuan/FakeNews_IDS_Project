"""Train/validate a corpus-specific, head-focused scalar priority model.

This development run deliberately does not score the previously inspected test
split.  It exports a complete validation ranking: choosing Top-K is a user
policy applied after scoring, not part of the model architecture.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

BASE=Path(__file__).resolve().parent.parent
PROJECT_ROOT=Path(__file__).resolve().parents[3]
DATA=BASE/'artifacts'/'20260901_133740_887252_twitter15_16_strict30_graph_head_features_v1'
OUT=BASE/'experiments'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_validation_head_focused_scalar_v1')
BUDGETS=(1,3,5,10,20,40); SEEDS=(13,42,71); EPOCHS=160
CONFIGS=((0.0,0.0),(0.5,0.0),(0.5,0.25),(1.0,0.25)) # order_lambda, soft_topk_lambda

class ScalarNet(nn.Module):
 def __init__(self,d:int): super().__init__();self.layers=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Dropout(.05),nn.Linear(32,1))
 def forward(self,x): return self.layers(x).squeeze(-1)

def head_pairs(y:np.ndarray,seed:int):
 order=np.argsort(-y,kind='stable'); head=order[:min(10,len(order))]; mid=order[min(10,len(order)):min(40,len(order))]; rest=order[min(40,len(order)):]
 rng=np.random.default_rng(seed); high=[];low=[]
 for h in head:
  candidates=np.concatenate([mid,rng.choice(rest,size=min(16,len(rest)),replace=False) if len(rest) else np.array([],dtype=int)])
  candidates=candidates[y[h]>y[candidates]]
  high.extend([h]*len(candidates));low.extend(candidates.tolist())
 return np.asarray(high,dtype=int),np.asarray(low,dtype=int)

def soft_membership(scores:torch.Tensor,k:int,tau:float=.25):
 lo=(scores.min()-20).detach();hi=(scores.max()+20).detach()
 for _ in range(24):
  mid=(lo+hi)/2;count=torch.sigmoid((scores-mid)/tau).sum().detach()
  if count>k: lo=mid
  else: hi=mid
 return torch.sigmoid((scores-((lo+hi)/2).detach())/tau)

def train_one(train:pd.DataFrame,val:pd.DataFrame,features:list[str],seed:int,order_lam:float,soft_lam:float):
 torch.manual_seed(seed); scaler=StandardScaler().fit(train[features]); x=torch.tensor(scaler.transform(train[features]),dtype=torch.float32);z=torch.tensor(scaler.transform(val[features]),dtype=torch.float32)
 y_raw=train.preventable_impact.to_numpy(float);y=torch.tensor(np.log1p(y_raw),dtype=torch.float32);q=max(float(np.quantile(y_raw,.9)),1.);weights=torch.tensor(1+2*np.clip(y_raw/q,0,1),dtype=torch.float32)
 h,l=head_pairs(y_raw,seed);h=torch.tensor(h);l=torch.tensor(l);pair_w=torch.tensor(np.clip(np.abs(y_raw[h]-y_raw[l])/q,.25,3),dtype=torch.float32) if len(h) else torch.ones(1)
 model=ScalarNet(len(features));opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
 for epoch in range(1,EPOCHS+1):
  opt.zero_grad();scores=model(x);utility=(F.smooth_l1_loss(scores,y,reduction='none')*weights).mean();order=(F.softplus(-(scores[h]-scores[l]))*pair_w).mean() if len(h) else scores.sum()*0;soft=[]
  total=float(y_raw.sum())
  for k in BUDGETS:
   if k<len(scores): soft.append(-(soft_membership(scores,k)*torch.tensor(y_raw/total,dtype=torch.float32)).sum())
  soft_loss=torch.stack(soft).mean() if soft else scores.sum()*0;loss=utility+order_lam*order+soft_lam*soft_loss
  if not torch.isfinite(loss):raise ValueError('Non-finite loss')
  loss.backward();opt.step()
  if epoch==1 or epoch%80==0 or epoch==EPOCHS:print(f'    seed={seed} epoch {epoch}/{EPOCHS}: utility={utility.item():.4f} order={order.item():.4f} soft={soft_loss.item():.4f}',flush=True)
 with torch.no_grad():return model(z).numpy(),{'utility':float(utility.item()),'order':float(order.item()),'soft_topk':float(soft_loss.item()),'head_pairs':int(len(h))}

def curve(frame:pd.DataFrame,score:np.ndarray):
 rows=[]; ranked=frame.assign(priority_score=score).sort_values(['priority_score','thread_id'],ascending=[False,True],kind='stable');oracle=frame.sort_values(['preventable_impact','thread_id'],ascending=[False,True],kind='stable');total=float(frame.preventable_impact.sum())
 for k in BUDGETS:
  n=min(k,len(frame));got=float(ranked.preventable_impact.iloc[:n].sum());best=float(oracle.preventable_impact.iloc[:n].sum());rows.append({'budget':k,'actual_budget':n,'blocked_future_nodes':got,'oracle_blocked_future_nodes':best,'reduction':got/total if total else 0.,'oracle_efficiency':got/best if best else 0.})
 return rows

def main():
 if OUT.exists():raise FileExistsError(f'Refusing to overwrite: {OUT}')
 print('Starting train/validation-only head-focused scalar priority training...',flush=True);print(f'Output directory: {OUT.resolve()}',flush=True);OUT.mkdir(parents=True,exist_ok=False)
 schema=json.loads((DATA/'schema.json').read_text());features=schema['feature_columns'];rec={'status':'running','dataset':str(DATA),'split':'train fits; validation selects; test explicitly not loaded or scored','label':{'utility':'preventable_y = log1p(preventable_impact)','order':'within-train-corpus Oracle-head impact ordering; outcome-only','soft_topk':'within-train-corpus differentiable capture of preventable_impact; outcome-only'},'features':features,'user_budget':'At inference sort priority_score descending; choose any Top-K. Reported validation budgets are '+str(BUDGETS),'configs':[{'order_lambda':a,'soft_topk_lambda':b} for a,b in CONFIGS],'seeds':SEEDS,'research_safety':'All inputs are source-anchored <=30-minute features. Test split is deliberately excluded.'};(OUT/'run_record.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
 try:
  print('[1/4] Loading strict-30 artifact and enforcing test exclusion...',flush=True);frame=pd.read_csv(DATA/'thread_level_records.csv',dtype={'corpus':'string','thread_id':'string','split':'string'});train=frame[frame.split=='train'].copy();val=frame[frame.split=='validation'].copy()
  if len(train)+len(val)+int((frame.split=='test').sum())!=len(frame) or not np.isfinite(frame[features].to_numpy(float)).all() or not np.allclose(frame.preventable_y,np.log1p(frame.preventable_impact)):raise ValueError('Artifact integrity failure')
  print(f'  train={len(train)}, validation={len(val)}, test excluded={int((frame.split=="test").sum())}; features={len(features)}.',flush=True)
  print('[2/4] Training separate Twitter15/Twitter16 scalar models across validation-selected loss settings...',flush=True);scores=[];runs=[]
  for oi,(ol,sl) in enumerate(CONFIGS,1):
   print(f'  config {oi}/{len(CONFIGS)}: order_lambda={ol}, soft_topk_lambda={sl}',flush=True)
   for corpus in sorted(train.corpus.unique()):
    tr=train[train.corpus==corpus];va=val[val.corpus==corpus]
    for seed in SEEDS:
     prediction,loss=train_one(tr,va,features,seed,ol,sl);runs.append({'config':f'{ol}_{sl}','order_lambda':ol,'soft_topk_lambda':sl,'corpus':corpus,'seed':seed,**loss});scores.append(pd.DataFrame({'thread_id':va.thread_id,'corpus':corpus,'config':f'{ol}_{sl}','seed':seed,'priority_score':prediction}))
  print('[3/4] Selecting configuration using validation multi-budget Oracle efficiency only...',flush=True);all_scores=pd.concat(scores,ignore_index=True);metrics=[]
  for config,g in all_scores.groupby('config'):
   for corpus,part in g.groupby('corpus'):
    mean_score=part.groupby('thread_id',as_index=False).priority_score.mean();pool=val[val.corpus==corpus].merge(mean_score,on='thread_id',validate='one_to_one')
    for row in curve(pool,pool.priority_score.to_numpy()):metrics.append({'config':config,'corpus':corpus,**row})
  metrics=pd.DataFrame(metrics);select=metrics.groupby('config').oracle_efficiency.mean().sort_values(ascending=False);chosen=str(select.index[0]);print(f'  selected={chosen}; validation mean multi-budget Oracle efficiency={select.iloc[0]:.4f}',flush=True)
  print('[4/4] Writing validation rankings, per-budget curves, and complete run record...',flush=True);chosen_scores=all_scores[all_scores.config==chosen].groupby(['corpus','thread_id'],as_index=False).priority_score.mean();ranking=val.merge(chosen_scores,on=['corpus','thread_id'],validate='one_to_one');ranking['priority_rank']=ranking.groupby('corpus').priority_score.rank(method='first',ascending=False).astype(int);ranking=ranking.sort_values(['corpus','priority_rank'])
  ranking.to_csv(OUT/'validation_priority_ranking_any_k.csv',index=False);metrics.to_csv(OUT/'validation_budget_curve_all_configs.csv',index=False);pd.DataFrame(runs).to_csv(OUT/'training_loss_summary.csv',index=False);select.rename('mean_validation_oracle_efficiency_across_corpora_and_budgets').to_csv(OUT/'validation_config_selection.csv')
  rec.update({'status':'complete','completed_at':datetime.now(timezone.utc).isoformat(),'chosen_config':chosen,'validation_selection':select.to_dict(),'output_files':['validation_priority_ranking_any_k.csv','validation_budget_curve_all_configs.csv','training_loss_summary.csv','validation_config_selection.csv']});(OUT/'run_record.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');print(f'SUCCESS: validation-only scalar priority result saved to: {OUT.resolve()}',flush=True)
 except Exception as error:rec.update({'status':'failed','error':f'{type(error).__name__}: {error}'});(OUT/'run_record.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');print(f'FAILURE: validation-only scalar output preserved at: {OUT.resolve()}',flush=True);raise
if __name__=='__main__':main()
