"""Build an AI-readable strict-30 blind feature package and separate answer key."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parent
ROOT = HERE.parents[2]
OUT_ROOT = MODEL_ROOT / "experiments"
CUTOFF_SECONDS = 1800.0

BASE_FEATURES = (
    ROOT / "effective_models" / "pheme_quiet_tier_group_search" / "experiments"
    / "20260902_145009_008419_strict30_relational_feature_materialization"
    / "strict30_protected_relational_snapshot_features.csv"
)
TOPOLOGY_TIME = (
    ROOT / "effective_models" / "pheme_quiet_tier_group_search" / "experiments"
    / "20260902_145443_669471_strict30_topology_time_feature_materialization"
    / "topology_time_feature_block.csv"
)
USER_OVERLAP = (
    ROOT / "effective_models" / "pheme_active_quiet_calibrated_rf" / "experiments"
    / "20260903_023432_316758_strict30_global_raw_user_overlap_diagnostic"
    / "strict30_user_overlap_features.csv"
)
REPLIES = (
    ROOT / "data" / "protected_research_assets" / "pheme_v5_strict30"
    / "pheme_reply_level_v4.csv"
)
CORRECTED_TARGET_AUDIT = (
    ROOT / "effective_models" / "pheme_active_quiet_calibrated_rf" / "experiments"
    / "20260903_141125_153501_strict30_preventable_target_consistency_audit"
    / "per_thread_target_consistency.csv"
)

FORBIDDEN_BLIND_TOKENS = (
    "preventable", "oracle", "future", "post30", "target", "label", "outcome",
    "prediction", "score", "rank", "thread_id", "event_id", "decision_time",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _assert_identity(frame: pd.DataFrame, name: str) -> None:
    required = {"thread_id", "event_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name} missing identity columns")
    if frame[list(required)].isna().any().any() or frame.thread_id.duplicated().any():
        raise ValueError(f"{name} has invalid or duplicate thread identity")


def _text_block(raw: pd.DataFrame) -> pd.DataFrame:
    observed = raw.loc[raw.offset_sec.le(CUTOFF_SECONDS)].copy()
    observed.sort_values(["thread_id", "offset_sec", "tweet_id"], kind="stable", inplace=True)
    rows = []
    for thread_id, group in observed.groupby("thread_id", sort=False):
        source = group.loc[group.is_source.eq(1)]
        if len(source) != 1:
            raise ValueError(f"{thread_id}: expected exactly one source before cutoff")
        replies = group.loc[group.is_source.ne(1), "text"].fillna("").astype(str).tolist()
        rows.append(
            {
                "thread_id": str(thread_id),
                "source_text": str(source.iloc[0].text or ""),
                "observed_reply_texts_json": json.dumps(replies, ensure_ascii=False),
                "observed_text_reply_count": int(sum(bool(value.strip()) for value in replies)),
                "observed_reply_count_from_raw": len(replies),
            }
        )
    return pd.DataFrame(rows)


def _answer_key(raw: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    replies = raw.loc[raw.is_source.ne(1)].copy()
    counts = replies.groupby("thread_id", as_index=False).agg(total_reply_count=("tweet_id", "size"))
    future = (
        replies.loc[replies.offset_sec.gt(CUTOFF_SECONDS)]
        .groupby("thread_id", as_index=False)
        .agg(post30_reply_count=("tweet_id", "size"))
    )
    answer = audit[["thread_id", "event_id", "all_roots_impact"]].copy()
    answer.rename(columns={"all_roots_impact": "corrected_preventable_impact"}, inplace=True)
    answer = answer.merge(counts, on="thread_id", how="left", validate="one_to_one")
    answer = answer.merge(future, on="thread_id", how="left", validate="one_to_one")
    answer[["total_reply_count", "post30_reply_count"]] = answer[
        ["total_reply_count", "post30_reply_count"]
    ].fillna(0).astype(int)
    answer["oracle_intervention_rank"] = 0
    answer["oracle_post30_volume_rank"] = 0
    for _, indices in answer.groupby("event_id", sort=True).groups.items():
        subset = answer.loc[indices]
        impact_order = subset.sort_values(
            ["corrected_preventable_impact", "thread_id"],
            ascending=[False, True], kind="stable",
        ).index
        volume_order = subset.sort_values(
            ["post30_reply_count", "thread_id"],
            ascending=[False, True], kind="stable",
        ).index
        answer.loc[impact_order, "oracle_intervention_rank"] = np.arange(1, len(impact_order) + 1)
        answer.loc[volume_order, "oracle_post30_volume_rank"] = np.arange(1, len(volume_order) + 1)
    answer["oracle_intervention_top50"] = answer.oracle_intervention_rank.le(50)
    answer["oracle_post30_volume_top50"] = answer.oracle_post30_volume_rank.le(50)
    event_sizes = answer.groupby("event_id").thread_id.transform("size")
    answer["official_eligible_event"] = event_sizes.ge(100)
    return answer


def _feature_description(column: str) -> tuple[str, str]:
    if column in {"case_id", "event_group"}:
        return "metadata", "Anonymous identifier; not an outcome."
    if column == "source_text":
        return "observed_text", "Source-post text available by the decision time."
    if column == "observed_reply_texts_json":
        return "observed_text", "JSON list of reply texts observed at or before 30 minutes."
    prefix = column.split("_", 1)[0]
    groups = {
        "activity": "activity", "temporal": "temporal", "topology": "topology",
        "source": "text_surface", "reply": "text_surface", "snapshot": "text_surface",
        "sentiment": "sentiment", "account": "account_age", "semantic": "semantic",
        "rel": "source_reply_relation", "tt": "topology_time_interaction",
        "user": "prior_user_overlap", "coordination": "prior_user_overlap",
        "same": "prior_user_overlap", "recent": "prior_user_overlap",
        "retweeter": "prior_user_overlap", "observed": "observed_summary",
    }
    return groups.get(prefix, "cutoff_safe_numeric"), column.replace("_", " ")


def build(mode: str = "full", max_threads: int | None = None) -> Path:
    suffix = "strict30_ai_blind_dataset_smoke" if mode == "smoke" else "strict30_ai_blind_dataset_full"
    output = OUT_ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + suffix)
    output.mkdir(parents=True, exist_ok=False)
    inputs = [BASE_FEATURES, TOPOLOGY_TIME, USER_OVERLAP, REPLIES, CORRECTED_TARGET_AUDIT]
    record = {
        "status": "running", "started_at": now(), "mode": mode,
        "purpose": "AI-readable strict-30 blind feature package with a physically separate answer key",
        "inputs": [str(path.resolve()) for path in inputs],
        "split": "no model split; rows retain anonymous event groups for event-separated evaluation",
        "cutoff_seconds": CUTOFF_SECONDS, "seed": None,
        "research_safety": (
            "Blind input contains only fields available by 1800 seconds. Raw thread/event IDs and all "
            "future outcomes, Oracle fields, ranks, scores, and targets are excluded. Future information "
            "is written only to the separate DO_NOT_GIVE_TO_AI answer key."
        ),
    }
    save_record(output, record)
    try:
        print(f"Starting {mode} strict-30 AI blind dataset build.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Validating existing cutoff-safe feature artifacts...", flush=True)
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
        base = pd.read_csv(BASE_FEATURES, dtype={"thread_id": str, "event_id": str})
        topology = pd.read_csv(TOPOLOGY_TIME, dtype={"thread_id": str})
        users = pd.read_csv(USER_OVERLAP, dtype={"thread_id": str, "event_id": str})
        for frame, name in ((base, "base"), (users, "user overlap")):
            _assert_identity(frame, name)
        if topology.thread_id.isna().any() or topology.thread_id.duplicated().any():
            raise ValueError("topology-time feature table has invalid identity")

        print("[2/6] Reading reply provenance and enforcing offset <= 1800 for AI text...", flush=True)
        raw = pd.read_csv(
            REPLIES,
            usecols=["thread_id", "tweet_id", "event_id", "is_source", "offset_sec", "text"],
            dtype={"thread_id": str, "tweet_id": str, "event_id": str}, low_memory=False,
        )
        raw["offset_sec"] = pd.to_numeric(raw.offset_sec, errors="coerce")
        if raw.offset_sec.isna().any() or (raw.offset_sec < 0).any():
            raise ValueError("reply provenance contains invalid offsets")
        text = _text_block(raw)

        print("[3/6] Joining cutoff-safe numeric and text feature groups...", flush=True)
        base = base.drop(columns=["preventable_impact", "preventable_y"], errors="raise")
        topology = topology.drop(columns=["event_id"], errors="ignore")
        users = users.drop(columns=["event_id", "decision_time"], errors="raise")
        joined = base.merge(topology, on="thread_id", validate="one_to_one")
        joined = joined.merge(users, on="thread_id", validate="one_to_one")
        joined = joined.merge(text, on="thread_id", validate="one_to_one")
        expected_ids = set(base.thread_id)
        if len(joined) != len(base) or set(joined.thread_id) != expected_ids:
            raise ValueError("feature join changed candidate membership")
        joined.sort_values(["event_id", "thread_id"], kind="stable", inplace=True)
        if max_threads is not None:
            joined = joined.groupby("event_id", sort=True, group_keys=False).head(max_threads)

        print("[4/6] Constructing anonymous cases and physically separate future answer key...", flush=True)
        event_map = {
            event: f"event_{index:02d}"
            for index, event in enumerate(sorted(joined.event_id.unique()), 1)
        }
        anonymous = pd.DataFrame(
            {
                "case_id": [f"PHEME30_{index:06d}" for index in range(1, len(joined) + 1)],
                "event_group": joined.event_id.map(event_map).to_numpy(),
            },
            index=joined.index,
        )
        joined = pd.concat([anonymous, joined], axis=1).copy()
        identity = joined[["case_id", "event_group", "thread_id", "event_id"]].copy()
        blind = joined.drop(columns=["thread_id", "event_id"])
        bad_columns = [
            column for column in blind.columns
            if any(token in column.lower() for token in FORBIDDEN_BLIND_TOKENS)
        ]
        if bad_columns:
            raise ValueError(f"future/identity-like columns reached blind input: {bad_columns}")
        numeric = blind.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("blind input contains non-finite numeric values")

        audit = pd.read_csv(CORRECTED_TARGET_AUDIT, dtype={"thread_id": str, "event_id": str})
        answer = _answer_key(raw, audit).merge(
            identity, on=["thread_id", "event_id"], how="inner", validate="one_to_one"
        )
        if len(answer) != len(blind) or set(answer.case_id) != set(blind.case_id):
            raise ValueError("answer-key identity does not exactly match blind cases")
        answer = answer[[
            "case_id", "event_group", "thread_id", "event_id",
            "corrected_preventable_impact", "post30_reply_count", "total_reply_count",
            "oracle_intervention_rank", "oracle_intervention_top50",
            "oracle_post30_volume_rank", "oracle_post30_volume_top50",
            "official_eligible_event",
        ]].sort_values("case_id", kind="stable")

        print("[5/6] Writing blind input, answer key, dictionary, and AI task prompt...", flush=True)
        blind.to_csv(output / "ai_blind_strict30_features.csv", index=False)
        answer.to_csv(output / "answer_key_DO_NOT_GIVE_TO_AI.csv", index=False)
        dictionary_rows = []
        for column in blind.columns:
            group, description = _feature_description(column)
            dictionary_rows.append({
                "column": column, "role": group, "available_by_seconds": CUTOFF_SECONDS,
                "description": description,
            })
        pd.DataFrame(dictionary_rows).to_csv(output / "feature_dictionary.csv", index=False)
        prompt = f"""# Blind strict-30 future-explosion task

You receive `{(output / 'ai_blind_strict30_features.csv').name}`. Every predictive
field is available at or before 30 minutes. No future outcome is present.

For every `event_group`, identify up to 50 cases most likely to produce high
post-30-minute propagation and, more importantly, high preventable future
impact if intervened at minute 30. Return CSV columns:

`case_id,event_group,predicted_priority,confidence_0_to_1,short_reason`

Rank only within the same event group. Freeze predictions for every group, but
compute the official macro result only on groups containing at least 100 cases,
matching the preserved PHEME protocol. Do not browse the web or infer original
tweet IDs. Do not request or inspect `answer_key_DO_NOT_GIVE_TO_AI.csv` until
all predictions have been frozen.
"""
        (output / "AI_TASK_PROMPT.md").write_text(prompt, encoding="utf-8")
        card = {
            "candidate_pool": "2402 protected PHEME v5 rumour-only threads" if mode == "full" else "bounded smoke subset",
            "observation_cutoff_seconds": CUTOFF_SECONDS,
            "blind_rows": len(blind), "blind_columns": len(blind.columns),
            "numeric_feature_columns": len(numeric.columns),
            "text_columns": ["source_text", "observed_reply_texts_json"],
            "primary_answer": "corrected_preventable_impact / oracle_intervention_top50",
            "secondary_answer": "post30_reply_count / oracle_post30_volume_top50",
            "official_evaluation": "event groups with at least 100 candidate threads",
            "official_eligible_event_groups": int(
                answer.loc[answer.official_eligible_event, "event_group"].nunique()
            ),
            "limitation": "Candidate pool is rumour-only; this package cannot measure non-rumour false interventions.",
        }
        (output / "dataset_card.json").write_text(
            json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("[6/6] Verifying saved files and completing run record...", flush=True)
        reread_blind = pd.read_csv(output / "ai_blind_strict30_features.csv")
        reread_answer = pd.read_csv(output / "answer_key_DO_NOT_GIVE_TO_AI.csv")
        if len(reread_blind) != len(blind) or len(reread_answer) != len(answer):
            raise ValueError("saved dataset row count mismatch")
        record.update({
            "status": "complete", "completed_at": now(), "metrics": card,
            "event_group_sizes": blind.groupby("event_group").size().to_dict(),
            "output_files": [
                "ai_blind_strict30_features.csv", "answer_key_DO_NOT_GIVE_TO_AI.csv",
                "feature_dictionary.csv", "AI_TASK_PROMPT.md", "dataset_card.json",
                "run_record.json",
            ],
        })
        save_record(output, record)
        print(f"SUCCESS: strict-30 AI blind dataset saved to: {output.resolve()}", flush=True)
        return output
    except BaseException as error:
        record.update({
            "status": "failed", "failed_at": now(),
            "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
            "output_files": sorted(path.name for path in output.iterdir()),
        })
        save_record(output, record)
        print(f"FAILURE: strict-30 AI blind dataset preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    build()
