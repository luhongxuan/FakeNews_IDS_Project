"""Strict 30-minute Twitter15/16 graph-only Oracle/control feature-gap audit.

The protected source-level test split is read only.  Matching uses exactly the
15 previously approved graph-only inputs; additional graph and timing features
are reconstructed only after matching and are descriptive, never model inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data_pipeline.audit_rumdetect2017_timing import parse_tree, snowflake_time  # noqa: E402
from data_pipeline.prepare_twitter15_preventable_impact_dataset import forward_forest  # noqa: E402
from twitter_graph_only_ranker_common import EARLY_GRAPH_FEATURES, load_protected_data  # noqa: E402

RAW_ROOT = ROOT / "data" / "raw" / "rumdetect2017"
OUT = BASE.parent / "experiments" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_twitter15_16_strict30_comprehensive_graph_oracle_gap")
CUTOFF_SECONDS = 1800.0
TOP_K = 10
SEED = 42


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    p = values[values > 0] / total
    return float(-(p * np.log(p)).sum())


def gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) == 0 or float(x.sum()) <= 0:
        return 0.0
    return float(2 * np.dot(np.arange(1, len(x) + 1), x) / (len(x) * x.sum()) - (len(x) + 1) / len(x))


def write_record(status: str, **extra: object) -> None:
    record = {
        "status": status,
        "script": str(Path(__file__).resolve()),
        "protocol": "Twitter15/16 fixed-test-cohort strict-30-minute source-anchored graph-only Oracle/control feature-gap diagnostic",
        "split": "existing fixed source-level test split; diagnostic only; no fitting or model selection",
        "cutoff_seconds": CUTOFF_SECONDS,
        "top_k": TOP_K,
        "matching_rule": "Within each corpus test cohort, match each future-impact Oracle Top-10 thread to its nearest non-Oracle thread using only the 15 pre-existing approved early graph features, standardized within corpus.",
        "research_safety": "Future preventable_impact identifies post-hoc Oracle rows only. New inputs are reconstructed from raw tree IDs/edges whose Twitter Snowflake timestamps are <= source + 30 minutes; observed edges require both endpoints <= cutoff. No labels, text, hydrated data, mutable metadata, full-tree totals, or future edges are inputs.",
        "interpretation": "Exploratory held-out-cohort diagnostic, not a candidate model result. Any apparent feature is hypothesis-generating only and cannot be selected on this test cohort.",
        **extra,
    }
    (OUT / "run_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def raw_graph_features(corpus: str, thread_id: str) -> dict[str, float | str]:
    tree_path = RAW_ROOT / corpus / "tree" / f"{thread_id}.txt"
    node_delays, raw_edges, malformed = parse_tree(tree_path)
    if malformed:
        raise ValueError(f"Malformed raw tree {tree_path}")
    source_time = snowflake_time(thread_id)
    node_ids = set(node_delays) | {thread_id}
    times = {node_id: snowflake_time(node_id) for node_id in node_ids}
    if source_time is None or any(value is None for value in times.values()):
        raise ValueError(f"Invalid Snowflake timestamp in {corpus}/{thread_id}")
    valid_times = {node_id: value for node_id, value in times.items() if value is not None}
    if any(value < source_time for value in valid_times.values()):
        raise ValueError(f"Pre-source node should have been excluded: {corpus}/{thread_id}")
    parent_map, children, _ = forward_forest(thread_id, node_ids, raw_edges, valid_times)
    offsets = {node_id: (valid_times[node_id] - source_time).total_seconds() for node_id in parent_map}
    observed = {node_id for node_id, offset in offsets.items() if offset <= CUTOFF_SECONDS}
    if thread_id not in observed or any(offsets[node_id] > CUTOFF_SECONDS for node_id in observed):
        raise ValueError(f"Unsafe observed snapshot {corpus}/{thread_id}")
    edges = [(parent, child) for child, parent in parent_map.items() if parent in observed and child in observed]
    if len(edges) != len(observed) - 1:
        raise ValueError(f"Observed forest unexpectedly disconnected {corpus}/{thread_id}")
    depths = {thread_id: 0}
    stack = [thread_id]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, []):
            if child in observed:
                depths[child] = depths[parent] + 1
                stack.append(child)
    if set(depths) != observed:
        raise ValueError(f"Could not derive all observed depths {corpus}/{thread_id}")
    reply_offsets = np.sort(np.asarray([offsets[node_id] for node_id in observed if node_id != thread_id], dtype=float))
    observed_depths = np.asarray(list(depths.values()), dtype=float)
    child_counts = np.asarray([sum(child in observed for child in children.get(node, [])) for node in observed], dtype=float)
    leaf = child_counts == 0
    width = Counter(int(value) for value in observed_depths)
    width_values = np.asarray(list(width.values()), dtype=float)
    bins = np.histogram(reply_offsets, bins=np.arange(0, CUTOFF_SECONDS + 1, 300))[0].astype(float)
    gaps = np.diff(reply_offsets)
    depth_time_corr = 0.0
    observable_offsets = np.asarray([offsets[node_id] for node_id in depths], dtype=float)
    if len(observable_offsets) >= 3 and observable_offsets.std() > 0 and observed_depths.std() > 0:
        depth_time_corr = float(np.corrcoef(observable_offsets, observed_depths)[0, 1])
    result: dict[str, float | str] = {"corpus": corpus, "thread_id": thread_id}
    result.update({
        "raw_graph_nodes": float(len(observed)), "raw_graph_edges": float(len(edges)),
        "raw_graph_max_depth": float(observed_depths.max()), "raw_graph_mean_depth": float(observed_depths.mean()),
        "raw_graph_depth_std": float(observed_depths.std()), "raw_graph_depth_p90": float(np.quantile(observed_depths, .9)),
        "raw_graph_depth_entropy": entropy(np.bincount(observed_depths.astype(int))), "raw_graph_depth_time_correlation": depth_time_corr,
        "raw_graph_root_children": float(sum(parent == thread_id for parent, _ in edges)),
        "raw_graph_leaf_fraction": float(leaf.mean()), "raw_graph_leaf_count": float(leaf.sum()),
        "raw_graph_late_leaf_count": float(np.sum([depths[node] > 0 and offsets[node] > 1200 and sum(child in observed for child in children.get(node, [])) == 0 for node in observed])),
        "raw_graph_branch_std": float(child_counts.std()), "raw_graph_branch_p90": float(np.quantile(child_counts, .9)),
        "raw_graph_branch_gini": gini(child_counts), "raw_graph_branch_hhi": ratio(float(np.square(child_counts).sum()), float(child_counts.sum() ** 2)),
        "raw_graph_width_max": float(width_values.max()), "raw_graph_width_entropy": entropy(width_values),
        "raw_graph_width_to_depth": ratio(float(width_values.max()), float(observed_depths.max() + 1)),
        "raw_time_reply_count": float(len(reply_offsets)), "raw_time_span_seconds": float(reply_offsets.max()) if len(reply_offsets) else 0.0,
        "raw_time_first_reply_seconds": float(reply_offsets.min()) if len(reply_offsets) else CUTOFF_SECONDS,
        "raw_time_last_gap_seconds": float(CUTOFF_SECONDS - reply_offsets.max()) if len(reply_offsets) else CUTOFF_SECONDS,
        "raw_time_interarrival_mean": float(gaps.mean()) if len(gaps) else 0.0, "raw_time_interarrival_std": float(gaps.std()) if len(gaps) else 0.0,
        "raw_time_interarrival_cv": ratio(float(gaps.std()), float(gaps.mean())) if len(gaps) else 0.0,
        "raw_time_interarrival_p90": float(np.quantile(gaps, .9)) if len(gaps) else 0.0,
        "raw_time_interarrival_max": float(gaps.max()) if len(gaps) else 0.0,
        "raw_time_burstiness": ratio(float(gaps.std() - gaps.mean()), float(gaps.std() + gaps.mean())) if len(gaps) else 0.0,
        "raw_time_bin_entropy": entropy(bins), "raw_time_late20_30_fraction": float(np.mean(reply_offsets > 1200)) if len(reply_offsets) else 0.0,
        "raw_time_last5_fraction": float(np.mean(reply_offsets > 1500)) if len(reply_offsets) else 0.0,
        "raw_time_late_minus_early": float(bins[4:].sum() - bins[:2].sum()), "raw_time_late_to_early_ratio": ratio(float(bins[4:].sum()), float(bins[:2].sum())),
    })
    for index, value in enumerate(bins, start=1):
        result[f"raw_time_actions_bin_{index}_5m"] = float(value)
    return result


def match(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for corpus, pool in frame.groupby("corpus", sort=True):
        print(f"Matching Oracle Top-{TOP_K} to nearest non-Oracle test controls: {corpus}", flush=True)
        pool = pool.reset_index(drop=True)
        matrix = StandardScaler().fit_transform(pool[EARLY_GRAPH_FEATURES].to_numpy(float))
        order = pool.sort_values(["preventable_impact", "thread_id"], ascending=[False, True], kind="stable").index.to_numpy()
        oracle = order[:TOP_K]
        controls = np.setdiff1d(np.arange(len(pool)), oracle, assume_unique=True)
        for oracle_index in oracle:
            distances = np.sum((matrix[controls] - matrix[oracle_index]) ** 2, axis=1)
            control_index = int(controls[int(np.argmin(distances))])
            rows.append({"corpus": corpus, "oracle_thread_id": str(pool.loc[oracle_index, "thread_id"]), "oracle_rank": int(np.where(order == oracle_index)[0][0] + 1), "oracle_preventable_impact": float(pool.loc[oracle_index, "preventable_impact"]), "control_thread_id": str(pool.loc[control_index, "thread_id"]), "control_preventable_impact": float(pool.loc[control_index, "preventable_impact"]), "impact_ratio_oracle_over_control": ratio(float(pool.loc[oracle_index, "preventable_impact"]), max(float(pool.loc[control_index, "preventable_impact"]), 1.0)), "approved15_standardized_distance": float(np.sqrt(distances.min()))})
    return pd.DataFrame(rows)


def descriptive_summary(paired: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        corpus_effects = []
        for corpus, group in paired.groupby("corpus", sort=True):
            oracle = group.loc[group.group == "oracle", feature].to_numpy(float)
            control = group.loc[group.group == "control", feature].to_numpy(float)
            combined = np.concatenate([oracle, control])
            corpus_effects.append(ratio(float((oracle - control).mean()), float(combined.std(ddof=0))))
        oracle_all = paired.loc[paired.group == "oracle", feature].to_numpy(float)
        control_all = paired.loc[paired.group == "control", feature].to_numpy(float)
        difference = oracle_all - control_all
        rows.append({"feature": feature, "feature_group": "approved_15" if feature in EARLY_GRAPH_FEATURES else "new_raw_graph_or_time", "oracle_mean": float(oracle_all.mean()), "control_mean": float(control_all.mean()), "oracle_median": float(np.median(oracle_all)), "control_median": float(np.median(control_all)), "paired_standardized_effect_dz": ratio(float(difference.mean()), float(difference.std(ddof=1))), "twitter15_effect": corpus_effects[0], "twitter16_effect": corpus_effects[1], "same_direction_corpora": bool(np.sign(corpus_effects[0]) == np.sign(corpus_effects[1]) and np.sign(corpus_effects[0]) != 0)})
    return pd.DataFrame(rows).sort_values("paired_standardized_effect_dz", key=lambda value: np.abs(value), ascending=False)


def make_figures(matches: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame) -> list[str]:
    colors = {"approved_15": "#4c78a8", "new_raw_graph_or_time": "#e45756"}
    ordered = summary.sort_values("paired_standardized_effect_dz", key=lambda value: np.abs(value), ascending=True)
    figure, axes = plt.subplots(1, 2, figsize=(18, max(9, len(ordered) * .22)), constrained_layout=True)
    y = np.arange(len(ordered)); axes[0].axvline(0, color="black", linewidth=.8)
    axes[0].scatter(ordered.paired_standardized_effect_dz, y, c=[colors[group] for group in ordered.feature_group], s=28)
    axes[0].set_yticks(y, ordered.feature, fontsize=7); axes[0].set_xlabel("Oracle − nearest control: paired standardized effect")
    axes[0].set_title("All legal 30-minute Twitter15/16 graph features")
    for corpus, marker in (("twitter15", "o"), ("twitter16", "s")):
        axes[1].scatter(ordered[f"{corpus}_effect"], y, marker=marker, s=28, label=corpus)
    axes[1].axvline(0, color="black", linewidth=.8); axes[1].set_yticks(y, ordered.feature, fontsize=7); axes[1].set_xlabel("Corpus-specific standardized effect")
    axes[1].set_title("Does the direction reproduce across corpora?"); axes[1].legend(frameon=False)
    filename = "all_twitter_graph_features_paired_effects.png"; figure.savefig(OUT / filename, dpi=220, bbox_inches="tight"); plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for corpus, values in matches.groupby("corpus", sort=True): axis.scatter(values.approved15_standardized_distance, values.impact_ratio_oracle_over_control, label=corpus, s=55, alpha=.85)
    axis.set_yscale("log"); axis.set_xlabel("Distance under approved 15 early graph features"); axis.set_ylabel("Oracle / matched-control future impact"); axis.set_title("Twitter15/16: early graph similarity versus future-impact gap"); axis.legend(frameon=False)
    outcome = "future_impact_gap_vs_approved15_distance.png"; figure.savefig(OUT / outcome, dpi=220, bbox_inches="tight"); plt.close(figure)
    pages = []
    feature_order = summary.feature.tolist()
    for start in range(0, len(feature_order), 12):
        page = feature_order[start:start + 12]; figure, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
        for axis, feature in zip(axes.flat, page):
            labels, control, oracle = [], [], []
            for corpus, values in paired.groupby("corpus", sort=True):
                labels.append(corpus); control.append(float(values.loc[values.group == "control", feature].mean())); oracle.append(float(values.loc[values.group == "oracle", feature].mean()))
            y2 = np.arange(len(labels))
            for row, (left, right) in enumerate(zip(control, oracle)): axis.plot([left, right], [row, row], color="#999999", linewidth=.9)
            axis.scatter(control, y2, color="#4c78a8", label="control", s=28); axis.scatter(oracle, y2, color="#e45756", marker="D", label="Oracle", s=30)
            stat = summary.loc[summary.feature == feature].iloc[0]; axis.set_yticks(y2, labels); axis.set_title(f"{feature}\ndz={stat.paired_standardized_effect_dz:.2f}", fontsize=8); axis.grid(axis="x", alpha=.2)
        for axis in axes.flat[len(page):]: axis.axis("off")
        axes.flat[0].legend(frameon=False, fontsize=7); filename = f"feature_by_feature_graph_{start // 12 + 1:02d}.png"; figure.savefig(OUT / filename, dpi=200, bbox_inches="tight"); plt.close(figure); pages.append(filename)
    return ["all_twitter_graph_features_paired_effects.png", outcome] + pages


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    print("Starting Twitter15/16 strict-30 graph-only Oracle/control feature-gap audit (no training)...", flush=True)
    print(f"Output directory: {OUT.resolve()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        print("[1/5] Loading protected source-level split and validating the 15 approved early features...", flush=True)
        records, split_manifest = load_protected_data()
        test = records.loc[records.split == "test"].copy()
        expected = {"twitter15": 237, "twitter16": 109}
        if test.groupby("corpus").size().to_dict() != expected or not np.isfinite(test[EARLY_GRAPH_FEATURES].to_numpy(float)).all():
            raise ValueError("Unexpected protected test cohort")
        if not np.allclose(test.preventable_y.to_numpy(float), np.log1p(test.preventable_impact.to_numpy(float))):
            raise ValueError("Protected target relationship mismatch")
        print("  validated protected test cohorts: twitter15=237, twitter16=109.", flush=True)
        write_record("running", protected_split_manifest=split_manifest, test_candidates=expected, approved_feature_count=len(EARLY_GRAPH_FEATURES))
        print("[2/5] Matching Oracle Top-10 to nearest controls with the approved 15 inputs only...", flush=True)
        matches = match(test)
        selected = pd.concat([matches[["corpus", "oracle_thread_id"]].rename(columns={"oracle_thread_id": "thread_id"}).assign(group="oracle"), matches[["corpus", "control_thread_id"]].rename(columns={"control_thread_id": "thread_id"}).assign(group="control")], ignore_index=True)
        print(f"  fixed {len(matches)} pairs across both held-out cohorts.", flush=True)
        print("[3/5] Reconstructing additional source-anchored graph/timing features from observed tree nodes only...", flush=True)
        raw_rows = []; unique = selected[["corpus", "thread_id"]].drop_duplicates().sort_values(["corpus", "thread_id"]).reset_index(drop=True)
        for index, item in enumerate(unique.itertuples(index=False), start=1):
            if index == 1 or index % 10 == 0 or index == len(unique): print(f"  raw graph extraction: {index}/{len(unique)} ({item.corpus})", flush=True)
            raw_rows.append(raw_graph_features(str(item.corpus), str(item.thread_id)))
        raw = pd.DataFrame(raw_rows); raw_features = [column for column in raw.columns if column not in {"corpus", "thread_id"}]
        if not np.isfinite(raw[raw_features].to_numpy(float)).all(): raise ValueError("Non-finite reconstructed graph feature")
        print(f"  reconstructed {len(raw_features)} additional graph/timing features for {len(raw)} unique threads.", flush=True)
        print("[4/5] Computing every feature's paired contrast and generating charts...", flush=True)
        paired = selected.merge(test[["corpus", "thread_id", "preventable_impact"] + EARLY_GRAPH_FEATURES], on=["corpus", "thread_id"], how="left", validate="many_to_one").merge(raw, on=["corpus", "thread_id"], how="left", validate="many_to_one")
        features = EARLY_GRAPH_FEATURES + raw_features
        if len(paired) != 40 or paired[features].isna().any().any(): raise ValueError("Incomplete pair-feature merge")
        summary = descriptive_summary(paired, features); chart_files = make_figures(matches, summary, paired)
        matches.to_csv(OUT / "oracle_nearest_nonoracle_matches_approved15.csv", index=False); paired.to_csv(OUT / "paired_oracle_control_all_early_graph_features.csv", index=False); summary.to_csv(OUT / "all_early_graph_feature_paired_gap_summary.csv", index=False)
        print(f"  compared {len(features)} legal graph-only early features and wrote {len(chart_files)} charts.", flush=True)
        print("[5/5] Writing run record...", flush=True)
        write_record("complete", pair_count=int(len(matches)), unique_selected_threads=int(len(raw)), additional_raw_feature_count=len(raw_features), total_feature_count=len(features), output_files=["oracle_nearest_nonoracle_matches_approved15.csv", "paired_oracle_control_all_early_graph_features.csv", "all_early_graph_feature_paired_gap_summary.csv"] + chart_files)
        print(f"SUCCESS: Twitter15/16 comprehensive graph Oracle/control audit saved to: {OUT.resolve()}", flush=True)
    except Exception as error:
        write_record("failed", error_type=type(error).__name__, error_message=str(error)); print(f"FAILURE: Twitter15/16 graph audit saved to: {OUT.resolve()}", flush=True); raise


if __name__ == "__main__": main()
