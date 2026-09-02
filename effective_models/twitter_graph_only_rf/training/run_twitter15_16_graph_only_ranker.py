"""Final protected-split graph-only intervention ranking for Twitter15/16."""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
from twitter_graph_only_ranker_common import EARLY_GRAPH_FEATURES, PARAMETERS, evaluate, fit, load_protected_data

BASE=Path(__file__).resolve().parent.parent; OUTPUT=BASE/'experiments'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_twitter15_16_graph_only_ranker')
def main():
 print('Starting final graph-only Twitter15/16 ranking experiment...',flush=True); print(f'Output directory: {OUTPUT}',flush=True); OUTPUT.mkdir(parents=True,exist_ok=False)
 try:
  print('Loading immutable combined records and fixed source-level split...',flush=True); frame,manifest=load_protected_data(); report={'status':'running','protocol':{'task':'30-minute source-anchored graph-only preventable-impact proxy ranking','split':'fixed source-level train/validation/test; auxiliary only, not event-separated','candidate_pool':'all retained threads within each corpus test cohort','top_k':10,'features':EARLY_GRAPH_FEATURES,'excluded_inputs':'text, user data, original_label, mutable counts, all future/tree-total fields'},'split_manifest':manifest,'corpora':{}}; (OUTPUT/'run_record.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
  selected_rows=[]
  for corpus in sorted(frame.corpus.unique()):
   print(f'[{corpus}] selecting RF configuration on validation only...',flush=True); validation=frame[(frame.corpus==corpus)&(frame.split=='validation')]; validation_scores={}
   for name,params in PARAMETERS.items():
    model=fit(frame,corpus,params); result,_=evaluate(model,validation); validation_scores[name]=result['crr_at_10']; print(f'  validation {name}: CRR@10={result["crr_at_10"]:.4f}',flush=True)
   chosen=max(PARAMETERS,key=lambda name:(validation_scores[name],name)); print(f'  chosen={chosen}; refitting on train+validation, then evaluating test once...',flush=True)
   trainval=frame[(frame.corpus==corpus)&(frame.split.isin(['train','validation']))]; from sklearn.ensemble import RandomForestRegressor
   model=RandomForestRegressor(**PARAMETERS[chosen]).fit(trainval[EARLY_GRAPH_FEATURES],trainval.preventable_y); test=frame[(frame.corpus==corpus)&(frame.split=='test')]; result,selected=evaluate(model,test); selected_rows.append(selected.assign(corpus=corpus)); report['corpora'][corpus]={'chosen_on_validation':chosen,'validation_crr_at_10':validation_scores,'test':result}; print(f'  TEST candidates={result["candidates"]}; model CRR@10={result["crr_at_10"]:.4f}; oracle={result["oracle_crr_at_10"]:.4f}; overlap={result["exact_oracle_overlap"]}/10',flush=True)
  report['status']='completed'; report['research_safety']='Only the fixed train split fits validation candidates. Validation selects configuration. Test is evaluated once after selection; future impact is target/evaluation only. Original label is excluded from model features.'; (OUTPUT/'run_record.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); __import__('pandas').concat(selected_rows,ignore_index=True).to_csv(OUTPUT/'test_model_selected_threads.csv',index=False); print(f'Final graph-only Twitter15/16 result saved to: {OUTPUT}',flush=True)
 except Exception as error:
  (OUTPUT/'run_record.json').write_text(json.dumps({'status':'failed','error_type':type(error).__name__,'error_message':str(error)},indent=2),encoding='utf-8'); print(f'Final graph-only Twitter15/16 experiment failed. Partial record saved to: {OUTPUT}',flush=True); raise
if __name__=='__main__': main()
