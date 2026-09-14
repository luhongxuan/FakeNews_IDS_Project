"""Combine already validated Twitter15 and Twitter16 derived records safely."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def load(path: Path, corpus: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"thread_id": str, "original_label": str})
    if frame.thread_id.duplicated().any() or frame.original_label.isna().any():
        raise ValueError(f"Invalid derived records: {path}")
    frame.insert(0, "corpus", corpus)
    frame.insert(1, "sample_id", corpus + ":" + frame.thread_id)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine time-safe Twitter15 and Twitter16 derived records")
    parser.add_argument("--twitter15", type=Path, required=True)
    parser.add_argument("--twitter16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing combined records: {output}")
    paths = {"twitter15": args.twitter15.resolve(), "twitter16": args.twitter16.resolve()}
    frames = [load(path, corpus) for corpus, path in paths.items()]
    expected_columns = list(frames[0].columns)
    if any(list(frame.columns) != expected_columns for frame in frames[1:]):
        raise ValueError("Twitter15 and Twitter16 record schemas differ")
    combined = pd.concat(frames, ignore_index=True)
    if combined.sample_id.duplicated().any() or not combined.thread_id.notna().all():
        raise ValueError("Combined sample IDs are not unique")
    output.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output / "thread_level_records.csv", index=False)
    manifest = {
        "corpora": {name: str(path) for name, path in paths.items()},
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        "record_file": "thread_level_records.csv",
        "sample_id": "corpus:thread_id",
        "counts_by_corpus": combined.corpus.value_counts().sort_index().to_dict(),
        "research_safety": "New combined derived records only. Inputs are immutable derived files; raw data, labels, and source splits are unchanged.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created combined Twitter15+16 records for {len(combined)} threads: {output}")


if __name__ == "__main__":
    main()
