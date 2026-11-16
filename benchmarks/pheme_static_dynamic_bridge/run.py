from __future__ import annotations
from datetime import datetime,timezone
import json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from benchmarks.pheme_static_dynamic_bridge import config
from benchmarks.pheme_static_dynamic_bridge.evaluation.common import load_validate,evaluate
def main(mode="full"):
 output=config.EXPERIMENTS/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+f"_bridge_a_{mode}");output.mkdir(parents=True,exist_ok=False);record={"status":"running","started_at":datetime.now(timezone.utc).isoformat(),"mode":mode,"input_artifacts":{"fixed_oof":str(config.FIXED_OOF.resolve()),"dynamic_oof":str(config.DYNAMIC_OOF.resolve())},"split":"frozen event-separated OOF; seven-event macro","cutoff_seconds":[600,1200,1800,2400,3000,3600],"seed":42,"configuration":{"budget":50,"quotas":config.QUOTAS}}
 (output/"run_record.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
 try:
  print(f"Starting Bridge A {mode}.\nOutput directory: {output.resolve()}",flush=True);print("[1/4] Loading frozen static and dynamic OOF...",flush=True);fixed,dynamic,eligible=load_validate();print("[2/4] Validating common event and timing contract...",flush=True)
  if mode=="smoke":eligible=eligible[:1]
  print(f"[3/4] Replaying Budget-50 policies for {len(eligible)} event(s)...",flush=True);per,macro,selected=evaluate(fixed,dynamic,eligible);per.to_csv(output/"per_event_bridge_a_metrics.csv",index=False);macro.to_csv(output/"unified_bridge_a_model_comparison.csv",index=False);selected.to_csv(output/"selected_threads.csv",index=False)
  print("[4/4] Writing complete run record...",flush=True);result={"eligible_events":eligible,"metrics":macro.to_dict(orient="records"),"smoke_is_not_research_result":mode=="smoke"};(output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8");record.update(status="complete",completed_at=datetime.now(timezone.utc).isoformat(),metrics_results=result,output_file_inventory=["per_event_bridge_a_metrics.csv","unified_bridge_a_model_comparison.csv","selected_threads.csv","result.json","run_record.json"],failure_details=None,research_safety={"models_not_retrained":True,"frozen_oof_unchanged":True,"budget_equal":True,"denominator_common":True});(output/"run_record.json").write_text(json.dumps(record,indent=2),encoding="utf-8");print(f"SUCCESS: Bridge A saved to: {output.resolve()}",flush=True)
 except BaseException as e:
  record.update(status="failed",completed_at=datetime.now(timezone.utc).isoformat(),failure_details={"error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()});(output/"run_record.json").write_text(json.dumps(record,indent=2),encoding="utf-8");print(f"FAILURE: Bridge A preserved at: {output.resolve()}",flush=True);raise
if __name__=="__main__":main("smoke" if "--smoke" in sys.argv else "full")
