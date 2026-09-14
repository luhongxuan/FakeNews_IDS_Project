"""External semantic proxy audit on matched Twitter15 future-growth pairs.

Only the source and hydrated reply texts from the already selected matched
pairs are embedded with the locally cached frozen Twitter RoBERTa model.  No
text is written to output and no model parameters are trained or updated.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from data_pipeline.audit_rumdetect2017_timing import parse_tree, snowflake_time  # noqa: E402


MODEL_NAME = "cardiffnlp/twitter-roberta-base"
CUTOFF_MINUTES = 30.0
MIN_REPLY_TEXTS = 2
SEED = 42
BOOTSTRAP_DRAWS = 10_000


def source_texts(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        tweet_id, separator, text = raw.partition("\t")
        if separator and tweet_id.isdigit() and text.strip():
            result[tweet_id] = text
    return result


def hydrated_texts(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(row, dict) and row.get("status") == "ok"
                and isinstance(row.get("tweet_id"), str)
                and isinstance(row.get("text"), str) and row["text"].strip()
            ):
                result.setdefault(row["tweet_id"], row["text"])
    return result


def observed_reply_ids(dataset: Path, source_id: str) -> list[str]:
    source_time = snowflake_time(source_id)
    if source_time is None:
        raise ValueError(f"Invalid source ID: {source_id}")
    delays, _, malformed = parse_tree(dataset / "tree" / f"{source_id}.txt")
    if malformed:
        raise ValueError(f"Malformed tree: {source_id}")
    result = []
    for tweet_id in delays:
        timestamp = snowflake_time(tweet_id)
        if timestamp is None:
            raise ValueError(f"Invalid node ID: {tweet_id}")
        offset = (timestamp - source_time).total_seconds() / 60.0
        if tweet_id != source_id and 0 <= offset <= CUTOFF_MINUTES:
            result.append(tweet_id)
    return result


def embed(text_by_id: dict[str, str]) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True).to(device)
    model.eval()
    ids = list(text_by_id)
    output = {}
    with torch.no_grad():
        for start in range(0, len(ids), 32):
            batch_ids = ids[start:start + 32]
            encoded = tokenizer([text_by_id[item] for item in batch_ids], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            values = ((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)).cpu().numpy()
            output.update({item: value.astype(np.float32) for item, value in zip(batch_ids, values)})
    return output


def semantic_row(source_id: str, reply_ids: list[str], embeddings: dict[str, np.ndarray]) -> dict:
    source = embeddings.get(source_id)
    replies = [embeddings[item] for item in reply_ids if item in embeddings]
    if source is None or len(replies) < MIN_REPLY_TEXTS:
        return {"thread_id": source_id, "usable_reply_texts": len(replies), "semantic_eligible": False}
    source_norm = np.linalg.norm(source)
    reply_array = np.vstack(replies)
    reply_norm = np.linalg.norm(reply_array, axis=1)
    if source_norm == 0 or np.any(reply_norm == 0):
        raise ValueError(f"Zero embedding norm in {source_id}")
    cosine = reply_array.dot(source) / (reply_norm * source_norm)
    return {
        "thread_id": source_id,
        "usable_reply_texts": len(replies),
        "semantic_eligible": True,
        "source_reply_cosine_mean": float(cosine.mean()),
        "source_reply_cosine_std": float(cosine.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched Twitter15 semantic external proxy audit")
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/rumdetect2017/twitter15"))
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    pairs = pd.read_csv(args.pairs, dtype={"oracle_thread_id": str, "control_thread_id": str, "label": str})
    thread_ids = set(pairs.oracle_thread_id) | set(pairs.control_thread_id)
    source = source_texts(args.dataset / "source_tweets.txt")
    replies = hydrated_texts(args.dataset / "hydrated_tweets.jsonl")
    texts = {thread_id: source[thread_id] for thread_id in thread_ids if thread_id in source}
    reply_ids_by_thread = {}
    for thread_id in thread_ids:
        reply_ids = observed_reply_ids(args.dataset, thread_id)
        reply_ids_by_thread[thread_id] = reply_ids
        texts.update({tweet_id: replies[tweet_id] for tweet_id in reply_ids if tweet_id in replies})
    embeddings = embed(texts)
    rows = [semantic_row(thread_id, reply_ids_by_thread[thread_id], embeddings) for thread_id in sorted(thread_ids)]
    semantics = pd.DataFrame(rows)
    oracle = semantics.rename(columns={column: f"oracle_{column}" for column in semantics.columns if column != "thread_id"}).rename(columns={"thread_id": "oracle_thread_id"})
    control = semantics.rename(columns={column: f"control_{column}" for column in semantics.columns if column != "thread_id"}).rename(columns={"thread_id": "control_thread_id"})
    paired = pairs.merge(oracle, on="oracle_thread_id", validate="one_to_one").merge(control, on="control_thread_id", validate="one_to_one")
    paired = paired[paired.oracle_semantic_eligible & paired.control_semantic_eligible].copy()
    for feature in ("source_reply_cosine_mean", "source_reply_cosine_std"):
        paired[f"delta_{feature}"] = paired[f"oracle_{feature}"] - paired[f"control_{feature}"]
    rng = np.random.default_rng(SEED)
    summary = []
    for feature in ("source_reply_cosine_mean", "source_reply_cosine_std"):
        per_label = paired.groupby("label", sort=True)[f"delta_{feature}"].mean().to_numpy(dtype=float)
        draws = rng.choice(per_label, size=(BOOTSTRAP_DRAWS, len(per_label)), replace=True).mean(axis=1)
        summary.append({
            "feature": feature,
            "mean_label_oracle_minus_control": float(per_label.mean()),
            "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
            "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
            "labels": len(per_label),
            "eligible_pairs": len(paired),
        })
    args.output.mkdir(parents=True, exist_ok=False)
    semantics.to_csv(args.output / "semantic_coverage_by_thread.csv", index=False)
    paired.to_csv(args.output / "matched_semantic_pairs.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output / "matched_semantic_summary.csv", index=False)
    metadata = {
        "dataset": str(args.dataset.resolve()), "pairs": str(args.pairs.resolve()), "cutoff_minutes": CUTOFF_MINUTES,
        "model": MODEL_NAME, "local_files_only": True, "min_usable_reply_texts": MIN_REPLY_TEXTS,
        "scope": "External future-growth proxy analysis, not causal intervention evaluation.",
        "research_safety": "Frozen local encoder inference only. No model fitting, training, network request, mutable engagement field, or tweet text export.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Semantic audit saved to: {args.output}; eligible matched pairs={len(paired)}")


if __name__ == "__main__":
    main()
