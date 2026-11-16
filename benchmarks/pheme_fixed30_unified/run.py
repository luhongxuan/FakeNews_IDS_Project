from __future__ import annotations
from datetime import datetime, timezone
import json, sys, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from benchmarks.pheme_fixed30_unified import config
from benchmarks.pheme_fixed30_unified.evaluation.common import load_validate, evaluate

def main(mode="full"):
    output=config.EXPERIMENTS/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+f"_fixed30_unified_{mode}"); output.mkdir(parents=True,exist_ok=False)
    record={"status":"running","started_at":datetime.now(timezone.utc).isoformat(),"mode":mode,"input_artifacts":{"allowlist":str(config.ALLOWLIST.resolve()),"frozen_oof":str(config.OOF.resolve())},"split":"frozen nine-event OOF; seven >=100-thread macro events","cutoff_seconds":1800,"seed":42,"configuration":{"budgets":config.BUDGETS}}
    (output/"run_record.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    try:
        print(f"Starting fixed-30 benchmark {mode}.\nOutput directory: {output.resolve()}",flush=True); print("[1/4] Loading immutable allowlist and frozen OOF...",flush=True)
        allow,scores,eligible=load_validate(); print("[2/4] Validating identity and target coverage...",flush=True)
        if mode=="smoke": eligible=eligible[:1]
        print(f"[3/4] Recomputing capture curves for {len(eligible)} event(s)...",flush=True); per_event,macro=evaluate(scores,eligible)
        per_event.to_csv(output/"per_event_capture_curve.csv",index=False); macro.to_csv(output/"unified_model_comparison.csv",index=False)
        print("[4/4] Writing complete run record...",flush=True); result={"rows":len(allow),"eligible_events":eligible,"metrics":macro.to_dict(orient="records"),"smoke_is_not_research_result":mode=="smoke"}
        (output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); record.update(status="complete",completed_at=datetime.now(timezone.utc).isoformat(),metrics_results=result,output_file_inventory=["per_event_capture_curve.csv","unified_model_comparison.csv","result.json","run_record.json"],failure_details=None,research_safety={"models_not_retrained":True,"frozen_oof_unchanged":True,"common_identity":True})
        (output/"run_record.json").write_text(json.dumps(record,indent=2),encoding="utf-8"); print(f"SUCCESS: fixed-30 benchmark saved to: {output.resolve()}",flush=True)
    except BaseException as e:
        record.update(status="failed",completed_at=datetime.now(timezone.utc).isoformat(),failure_details={"error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()}); (output/"run_record.json").write_text(json.dumps(record,indent=2),encoding="utf-8"); print(f"FAILURE: fixed-30 benchmark preserved at: {output.resolve()}",flush=True); raise

if __name__=="__main__": main("smoke" if "--smoke" in sys.argv else "full")
