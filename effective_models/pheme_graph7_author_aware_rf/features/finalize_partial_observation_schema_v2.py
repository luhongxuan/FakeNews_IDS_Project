"""Create a schema-locked copy of the completed partial-observation dataset."""
from __future__ import annotations
import hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / "artifacts" / "partial_observation" / "20260830_123800_pheme_author_aware_30min"
OUTPUT = BASE_DIR / "artifacts" / "partial_observation" / "20260830_133000_pheme_author_aware_30min_schema_v2"

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    print("Loading completed partial-observation records...", flush=True)
    records_path = SOURCE / "thread_level_records.csv"
    records = pd.read_csv(records_path, dtype={"thread_id": "string", "event_id": "string", "category": "string"})
    metadata = ["thread_id", "event_id", "category"]
    targets = ["preventable_impact", "preventable_y"]
    features = [column for column in records.columns if column not in {*metadata, *targets}]
    if records[metadata].isna().any().any() or records["thread_id"].duplicated().any():
        raise ValueError("Invalid metadata columns in completed source dataset")
    OUTPUT.mkdir(parents=True)
    print("Writing schema-locked copy (no raw PHEME rescan)...", flush=True)
    records.to_csv(OUTPUT / "thread_level_records.csv", index=False)
    shutil.copy2(SOURCE / "excluded_threads.csv", OUTPUT / "excluded_threads.csv")
    shutil.copy2(SOURCE / "build_status.json", OUTPUT / "build_status.json")
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"schema_locked_from": str(SOURCE), "source_records_sha256": digest(records_path), "created_at": datetime.now(timezone.utc).isoformat()})
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT / "schema.json").write_text(json.dumps({"metadata_columns": metadata, "target_columns": targets, "feature_columns": features, "csv_dtypes": {key: "string" for key in metadata}, "research_safety": "Only feature_columns may be passed to a model."}, indent=2), encoding="utf-8")
    print(f"Created schema-locked dataset: {OUTPUT}", flush=True)

if __name__ == "__main__":
    main()
