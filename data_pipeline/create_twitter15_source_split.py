"""Create a fixed source-level diagnostic split for derived Twitter15 records.

RumDetect2017 Twitter15 provides no reliable event identifier.  This is
therefore *not* an event-generalization protocol and must never replace PHEME
LOEO evaluation.  It is a deterministic, label-stratified source split for
checking an auxiliary Twitter15 task and for keeping its validation/test rows
out of any auxiliary model fitting.

The split assignment deliberately reads only ``thread_id`` and the original
static Twitter15 label.  It never reads a future-growth or preventable-impact
target, hydration status, tweet text, or any PHEME data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SEED = 42
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REQUIRED_COLUMNS = {"original_label"}


def stable_rank(thread_id: str) -> str:
    return hashlib.sha256(f"twitter15-source-split-v1:{SEED}:{thread_id}".encode("utf-8")).hexdigest()


def counts_for_group(size: int) -> dict[str, int]:
    train = int(size * RATIOS["train"])
    validation = int(size * RATIOS["validation"])
    return {"train": train, "validation": validation, "test": size - train - validation}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed Twitter15 source-level diagnostic split")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records_path, output = args.records.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing split manifest: {output}")
    records = pd.read_csv(records_path, dtype={"thread_id": str, "original_label": str})
    if not REQUIRED_COLUMNS <= set(records.columns):
        raise ValueError(f"Records missing required columns: {REQUIRED_COLUMNS - set(records.columns)}")
    id_column = "sample_id" if "sample_id" in records.columns else "thread_id"
    if records[id_column].duplicated().any() or records[id_column].isna().any():
        raise ValueError(f"{id_column} values must be unique and present")
    if records.original_label.isna().any():
        raise ValueError("original_label values must be present")

    assignments: list[dict[str, str]] = []
    summary: dict[str, dict[str, int]] = {}
    for label, group in records.loc[:, [id_column, "original_label"]].groupby("original_label", sort=True):
        ordered = group.assign(_rank=group[id_column].map(stable_rank)).sort_values(["_rank", id_column])
        counts = counts_for_group(len(ordered))
        boundaries = (counts["train"], counts["train"] + counts["validation"])
        split_values = [
            "train" if index < boundaries[0] else "validation" if index < boundaries[1] else "test"
            for index in range(len(ordered))
        ]
        assignments.extend(
            {id_column: getattr(row, id_column), "original_label": label, "split": split}
            for row, split in zip(ordered.itertuples(index=False), split_values)
        )
        summary[str(label)] = counts

    assignment_frame = pd.DataFrame(assignments).sort_values(id_column).reset_index(drop=True)
    if len(assignment_frame) != len(records) or assignment_frame[id_column].duplicated().any():
        raise ValueError("Split assignment does not provide exactly one row per source thread")
    if set(assignment_frame.split) != set(RATIOS):
        raise ValueError("At least one split is unexpectedly empty")

    output.mkdir(parents=True, exist_ok=False)
    assignment_frame.to_csv(output / "assignments.csv", index=False)
    manifest = {
        "protocol": "Twitter15 source-level label-stratified deterministic diagnostic split",
        "version": 1,
        "seed": SEED,
        "ratios": RATIOS,
        "records": str(records_path),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "assignment_file": "assignments.csv",
        "assignment_id_column": id_column,
        "counts_by_original_label": summary,
        "overall_counts": assignment_frame.split.value_counts().sort_index().to_dict(),
        "split_inputs": f"{id_column} and original_label only",
        "excluded_inputs": "preventable_impact, preventable_y, future-node information, text, hydration fields, and PHEME data",
        "limitation": (
            "Twitter15 lacks a reliable event identifier. This source-level split is for auxiliary-task diagnostics only "
            "and must not be reported as event-separated or used to replace PHEME nested LOEO model selection."
        ),
        "research_safety": (
            "New manifest only; raw and derived source data are unchanged. The builder refuses overwrite and does not fit "
            "a model, create features, or access future-derived target columns."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created deterministic Twitter15 split for {len(assignment_frame)} threads: {output}")


if __name__ == "__main__":
    main()
