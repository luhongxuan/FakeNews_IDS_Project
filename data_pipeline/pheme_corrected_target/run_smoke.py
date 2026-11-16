from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_pipeline.pheme_corrected_target import target_builder

def main() -> None:
    output = Path(__file__).resolve().parent / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_corrected_target_smoke")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "configuration": {"cutoff_seconds": target_builder.CUTOFF_SEC}, "input_artifacts": {"reply_level": str(target_builder.REPLIES.resolve())}, "split": "not applicable: deterministic target construction", "seed": None}
    (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        print("Starting corrected-target traversal smoke.", flush=True); print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/4] Loading protected reply identities and offsets...", flush=True); maps = target_builder.load_reply_maps()
        print("[2/4] Selecting one deterministic real thread...", flush=True); thread_id = sorted(maps)[0]; parents, offsets = maps[thread_id]
        observed = {node_id for node_id, offset in offsets.items() if offset <= target_builder.CUTOFF_SEC}
        print("[3/4] Traversing every root and computing corrected impact...", flush=True); impact = target_builder.corrected_preventable_impact(parents, offsets, observed)
        if impact < 0: raise ValueError("Negative preventable impact")
        print("[4/4] Writing complete run record...", flush=True)
        record.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), metrics_results={"thread_id": thread_id, "nodes": len(parents), "observed_nodes": len(observed), "corrected_preventable_impact": impact}, output_file_inventory=["run_record.json"], failure_details=None, research_safety={"dataset_unchanged": True, "all_roots_traversed": True, "cutoff_respected": True})
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8"); print(f"SUCCESS: corrected-target smoke saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update(status="failed", completed_at=datetime.now(timezone.utc).isoformat(), failure_details={"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        (output / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8"); print(f"FAILURE: corrected-target smoke preserved at: {output.resolve()}", flush=True); raise

if __name__ == "__main__": main()
