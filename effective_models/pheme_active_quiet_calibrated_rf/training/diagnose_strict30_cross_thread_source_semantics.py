"""Nested-LOEO diagnostic for cross-thread source semantic context.

The deployable protocol compares each strict-30 source embedding only with
sources from reference events.  A separate within-event analysis is saved as a
non-deployable ceiling because the protected artifact has no absolute source
timestamps with which to prove that every peer source already existed.

This script does not train a model or modify any protected artifact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from pheme_account_age_v5_common import CUTOFF_SECONDS, DATASET, load_graphs, load_node_offsets
from pheme_quiet_expert_common import QUIET_QUANTILE, recency_values


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent / "experiments"
HYBRID_RUN = (
    EXPERIMENT_ROOT
    / "20260902_132700_413954_quiet_tier_hybrid_policy_full"
)
HYBRID_SCORES = HYBRID_RUN / "outer_scores.csv"

SEED = 42
ORACLE_KS = (5, 10)
ADMISSION_CAPS = (1, 2, 3, 5, 10)
SCORE_NAMES = (
    "reference_max_similarity",
    "reference_top3_mean_similarity",
    "reference_top10_mean_similarity",
    "reference_dense_neighbor_count",
    "reference_max_novelty",
    "reference_top3_novelty",
    "reference_top10_novelty",
    "reference_sparse_neighbor_score",
    "reference_pair_specificity",
    "reference_cluster_concentration",
)
PRIMARY_CAP = 2
PRIMARY_ORACLE_K = 5
MIN_PRIMARY_HITS = 4
MIN_NONWORSE_EVENTS = 6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_record(output: Path, record: dict) -> None:
    (output / "run_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


def normalized_source_embeddings(graphs: list, offsets: dict[tuple[str, str], float]) -> np.ndarray:
    rows = []
    for index, graph in enumerate(graphs, 1):
        source = torch.nonzero(graph.is_source_mask, as_tuple=False).flatten()
        if source.numel() != 1 or graph.x.shape[1] != 781:
            raise ValueError(f"{graph.thread_id}: invalid protected source embedding")
        if graph.edge_index.numel() and (
            int(graph.edge_index.min()) < 0
            or int(graph.edge_index.max()) >= int(graph.num_nodes)
        ):
            raise ValueError(f"{graph.thread_id}: edge references an unobserved node")
        for tweet_id in graph.node_ids:
            key = (str(graph.thread_id), str(tweet_id))
            if key not in offsets or offsets[key] > CUTOFF_SECONDS + 1e-9:
                raise ValueError(f"{graph.thread_id}: post-cutoff or missing node {tweet_id}")
        vector = graph.x[int(source.item()), 13:].detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm <= 0:
            raise ValueError(f"{graph.thread_id}: non-finite or zero source embedding")
        rows.append(vector / norm)
        if index % 500 == 0 or index == len(graphs):
            print(f"  protected source audit {index}/{len(graphs)}", flush=True)
    return np.vstack(rows).astype(np.float32)


def add_oracle_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for k in ORACLE_KS:
        column = f"oracle_top{k}"
        result[column] = False
        for _, event in result.groupby("event_id", sort=True):
            chosen = event.sort_values(
                ["preventable_impact", "thread_id"],
                ascending=[False, True],
                kind="stable",
            ).head(k).index
            result.loc[chosen, column] = True
    return result


def reference_features(
    similarity: np.ndarray,
    query_indices: np.ndarray,
    reference_indices: np.ndarray,
) -> pd.DataFrame:
    if len(reference_indices) < 10:
        raise ValueError("semantic reference bank has fewer than ten sources")
    values = similarity[np.ix_(query_indices, reference_indices)]
    ordered = np.sort(values, axis=1)[:, ::-1]

    # The density threshold is fitted only from the reference bank.  It is the
    # median nearest-neighbour similarity of reference sources, excluding self.
    reference_square = similarity[np.ix_(reference_indices, reference_indices)].copy()
    np.fill_diagonal(reference_square, -np.inf)
    density_threshold = float(np.median(reference_square.max(axis=1)))
    maximum = ordered[:, 0]
    top3 = ordered[:, :3].mean(axis=1)
    top10 = ordered[:, :10].mean(axis=1)
    dense_count = (values >= density_threshold).sum(axis=1)
    return pd.DataFrame(
        {
            "reference_max_similarity": maximum,
            "reference_top3_mean_similarity": top3,
            "reference_top10_mean_similarity": top10,
            "reference_dense_neighbor_count": dense_count,
            "reference_max_novelty": -maximum,
            "reference_top3_novelty": -top3,
            "reference_top10_novelty": -top10,
            "reference_sparse_neighbor_score": -dense_count,
            # A near-duplicate claim should have one unusually close peer even
            # when its broader topic neighbourhood is not generally dense.
            "reference_pair_specificity": maximum - top10,
            "reference_cluster_concentration": top3 - top10,
            "reference_density_threshold": density_threshold,
        }
    )


def within_event_ceiling_features(
    similarity: np.ndarray, event_indices: np.ndarray
) -> pd.DataFrame:
    square = similarity[np.ix_(event_indices, event_indices)].copy()
    np.fill_diagonal(square, -np.inf)
    ordered = np.sort(square, axis=1)[:, ::-1]
    maximum = ordered[:, 0]
    top3 = ordered[:, :3].mean(axis=1)
    top10 = ordered[:, :10].mean(axis=1)
    return pd.DataFrame(
        {
            "within_event_max_similarity": maximum,
            "within_event_top3_mean_similarity": top3,
            "within_event_top10_mean_similarity": top10,
            "within_event_max_novelty": -maximum,
            "within_event_top3_novelty": -top3,
            "within_event_top10_novelty": -top10,
            "within_event_pair_specificity": maximum - top10,
            "within_event_cluster_concentration": top3 - top10,
        }
    )


def quiet_mask_for_reference(graphs: list, query_indices: np.ndarray, reference_indices: np.ndarray) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(recency_values([graphs[i] for i in reference_indices]), QUIET_QUANTILE))
    quiet = recency_values([graphs[i] for i in query_indices]) >= threshold
    return quiet, threshold


def selected_hits(frame: pd.DataFrame, score: str, cap: int, oracle_k: int) -> int:
    selected = frame.sort_values(
        [score, "thread_id"], ascending=[False, True], kind="stable"
    ).head(min(cap, len(frame)))
    return int(selected[f"oracle_top{oracle_k}"].sum())


def choose_score(inner_rows: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    summaries = []
    for score in SCORE_NAMES:
        row = {"score_name": score}
        for k in ORACLE_KS:
            for cap in ADMISSION_CAPS:
                hits = sum(
                    selected_hits(event, score, cap, k)
                    for _, event in inner_rows.groupby("validation_event", sort=True)
                )
                row[f"top{k}_hits_at_cap{cap}"] = int(hits)
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.sort_values(
        ["top5_hits_at_cap2", "top5_hits_at_cap1", "top10_hits_at_cap2", "score_name"],
        ascending=[False, False, False, True],
        kind="stable",
        inplace=True,
    )
    return str(summary.iloc[0].score_name), summary


def event_metrics(frame: pd.DataFrame, score: str, protocol: str) -> list[dict]:
    rows = []
    for k in ORACLE_KS:
        positives = int(frame[f"oracle_top{k}"].sum())
        labels = frame[f"oracle_top{k}"].to_numpy(bool)
        auc = float(roc_auc_score(labels, frame[score])) if 0 < positives < len(frame) else np.nan
        ap = float(average_precision_score(labels, frame[score])) if positives else np.nan
        for cap in ADMISSION_CAPS:
            hits = selected_hits(frame, score, cap, k)
            rows.append(
                {
                    "protocol": protocol,
                    "event_id": str(frame.event_id.iloc[0]),
                    "score_name": score,
                    "oracle_k": k,
                    "admission_cap": cap,
                    "quiet_candidates": len(frame),
                    "quiet_oracle_positives": positives,
                    "hits": hits,
                    "precision": hits / min(cap, len(frame)) if len(frame) else 0.0,
                    "recall": hits / positives if positives else np.nan,
                    "roc_auc": auc,
                    "average_precision": ap,
                }
            )
    return rows


def main() -> None:
    output = EXPERIMENT_ROOT / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + "_strict30_cross_thread_source_semantic_diagnostic"
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "started_at": utc_now(),
        "purpose": "nested-LOEO signal-ceiling diagnostic for cross-thread source semantic context",
        "dataset": str(DATASET.resolve()),
        "baseline_scores": str(HYBRID_SCORES.resolve()),
        "split": "seven eligible outer LOEO events; score definition selected by inner LOEO",
        "cutoff_seconds": 1800,
        "seed": SEED,
        "oracle_ks": list(ORACLE_KS),
        "admission_caps": list(ADMISSION_CAPS),
        "success_gate": {
            "primary_oracle_k": PRIMARY_ORACLE_K,
            "primary_cap": PRIMARY_CAP,
            "minimum_hits": MIN_PRIMARY_HITS,
            "minimum_events_nonworse_than_saved_hybrid_quiet_score": MIN_NONWORSE_EVENTS,
        },
        "research_safety": (
            "Deployable features use only strict-30 source embeddings and reference events that exclude the query event. "
            "Every snapshot node is re-audited at offset<=1800. Inner LOEO chooses the score definition without outer labels. "
            "Within-event similarity is saved only as a non-deployable ceiling because absolute source timestamps are unavailable."
        ),
    }
    write_record(output, record)
    try:
        print("Starting strict-30 cross-thread source semantic diagnostic.", flush=True)
        print(f"Output directory: {output.resolve()}", flush=True)
        print("[1/6] Loading protected graphs and saved held-out hybrid scores...", flush=True)
        graphs, _, eligible = load_graphs()
        offsets = load_node_offsets()
        saved = pd.read_csv(HYBRID_SCORES, dtype={"thread_id": str, "event_id": str})
        required = {"thread_id", "event_id", "preventable_impact", "hybrid_calibrated_score", "quiet"}
        if not required.issubset(saved.columns) or saved.thread_id.duplicated().any():
            raise ValueError("invalid saved held-out hybrid score artifact")
        if sorted(saved.event_id.unique()) != sorted(eligible):
            raise ValueError("saved hybrid events do not match eligible events")

        graph_ids = [str(graph.thread_id) for graph in graphs]
        if len(graph_ids) != len(set(graph_ids)):
            raise ValueError("protected graph thread identifiers are not unique")
        index_by_id = {thread_id: index for index, thread_id in enumerate(graph_ids)}
        if not set(saved.thread_id).issubset(index_by_id):
            raise ValueError("saved scores contain thread IDs absent from protected graphs")
        graph_frame = pd.DataFrame(
            {
                "thread_id": graph_ids,
                "event_id": [str(graph.event_id) for graph in graphs],
                "preventable_impact": [float(graph.preventable_impact.item()) for graph in graphs],
                "graph_index": np.arange(len(graphs)),
            }
        )
        graph_frame = add_oracle_labels(graph_frame)

        print("[2/6] Auditing cutoff safety and caching cosine similarities...", flush=True)
        embedding = normalized_source_embeddings(graphs, offsets)
        similarity = embedding @ embedding.T
        if similarity.shape != (len(graphs), len(graphs)) or not np.isfinite(similarity).all():
            raise ValueError("invalid source cosine similarity matrix")

        print("[3/6] Running nested inner-LOEO score selection...", flush=True)
        outer_predictions = []
        inner_summaries = []
        fold_details = []
        for outer_number, outer_event in enumerate(sorted(eligible), 1):
            print(f"  outer fold {outer_number}/{len(eligible)}: {outer_event}", flush=True)
            outer_reference = graph_frame.loc[~graph_frame.event_id.eq(outer_event)]
            inner_rows = []
            for inner_event in sorted(event for event in eligible if event != outer_event):
                inner_query = graph_frame.loc[graph_frame.event_id.eq(inner_event)].copy()
                inner_reference = outer_reference.loc[~outer_reference.event_id.eq(inner_event)]
                query_idx = inner_query.graph_index.to_numpy(int)
                ref_idx = inner_reference.graph_index.to_numpy(int)
                features = reference_features(similarity, query_idx, ref_idx)
                quiet, _ = quiet_mask_for_reference(graphs, query_idx, ref_idx)
                inner_query = pd.concat([inner_query.reset_index(drop=True), features], axis=1)
                inner_query = inner_query.loc[quiet].copy()
                inner_query["validation_event"] = inner_event
                inner_rows.append(inner_query)
            inner_frame = pd.concat(inner_rows, ignore_index=True)
            selected_score, selection = choose_score(inner_frame)
            selection.insert(0, "outer_event", outer_event)
            selection["selected"] = selection.score_name.eq(selected_score)
            inner_summaries.append(selection)

            outer_query = graph_frame.loc[graph_frame.event_id.eq(outer_event)].copy()
            query_idx = outer_query.graph_index.to_numpy(int)
            ref_idx = outer_reference.graph_index.to_numpy(int)
            outer_features = reference_features(similarity, query_idx, ref_idx)
            quiet, threshold = quiet_mask_for_reference(graphs, query_idx, ref_idx)
            outer_query = pd.concat([outer_query.reset_index(drop=True), outer_features], axis=1)
            outer_query["quiet_recomputed"] = quiet
            outer_query["selected_score_name"] = selected_score
            saved_event = saved.loc[saved.event_id.eq(outer_event)].copy()
            outer_query = outer_query.merge(
                saved_event[["thread_id", "quiet", "hybrid_calibrated_score"]],
                on="thread_id", how="left", validate="one_to_one",
            )
            if outer_query[["quiet", "hybrid_calibrated_score"]].isna().any().any():
                raise ValueError(f"{outer_event}: incomplete saved-score alignment")
            if not np.array_equal(outer_query.quiet.astype(bool), outer_query.quiet_recomputed):
                raise ValueError(f"{outer_event}: recomputed quiet gate differs from official hybrid")
            fold_details.append(
                {
                    "outer_event": outer_event,
                    "selected_score": selected_score,
                    "quiet_threshold": threshold,
                    "quiet_candidates": int(quiet.sum()),
                    "reference_sources": len(ref_idx),
                }
            )
            outer_predictions.append(outer_query.loc[outer_query.quiet_recomputed].copy())

        predictions = pd.concat(outer_predictions, ignore_index=True)
        print("[4/6] Evaluating deployable admission signals and saved-score comparator...", flush=True)
        metric_rows = []
        for event_id, event in predictions.groupby("event_id", sort=True):
            selected_score = str(event.selected_score_name.iloc[0])
            metric_rows.extend(event_metrics(event, selected_score, "nested_reference_semantics"))
            metric_rows.extend(event_metrics(event, "hybrid_calibrated_score", "saved_hybrid_quiet_score"))

        print("[5/6] Computing explicitly non-deployable within-event ceiling...", flush=True)
        ceiling_predictions = []
        for event_id, event in graph_frame.loc[graph_frame.event_id.isin(eligible)].groupby("event_id", sort=True):
            event = event.copy().reset_index(drop=True)
            indices = event.graph_index.to_numpy(int)
            features = within_event_ceiling_features(similarity, indices)
            event = pd.concat([event, features], axis=1)
            official_quiet = saved.loc[saved.event_id.eq(event_id), ["thread_id", "quiet"]]
            event = event.merge(official_quiet, on="thread_id", how="left", validate="one_to_one")
            ceiling_predictions.append(event.loc[event.quiet.astype(bool)].copy())
        ceiling = pd.concat(ceiling_predictions, ignore_index=True)
        ceiling_scores = (
            "within_event_top3_mean_similarity",
            "within_event_top3_novelty",
            "within_event_pair_specificity",
        )
        for _, event in ceiling.groupby("event_id", sort=True):
            for ceiling_score in ceiling_scores:
                metric_rows.extend(
                    event_metrics(
                        event,
                        ceiling_score,
                        f"non_deployable_{ceiling_score}",
                    )
                )

        metrics = pd.DataFrame(metric_rows)
        pooled = metrics.groupby(
            ["protocol", "oracle_k", "admission_cap"], as_index=False
        ).agg(
            eligible_events=("event_id", "nunique"),
            total_quiet_candidates=("quiet_candidates", "sum"),
            total_quiet_oracle_positives=("quiet_oracle_positives", "sum"),
            total_hits=("hits", "sum"),
            mean_event_precision=("precision", "mean"),
            mean_event_recall=("recall", "mean"),
            mean_event_auc=("roc_auc", "mean"),
            mean_event_ap=("average_precision", "mean"),
        )
        primary = metrics.loc[
            metrics.protocol.eq("nested_reference_semantics")
            & metrics.oracle_k.eq(PRIMARY_ORACLE_K)
            & metrics.admission_cap.eq(PRIMARY_CAP)
        ].set_index("event_id")
        comparator = metrics.loc[
            metrics.protocol.eq("saved_hybrid_quiet_score")
            & metrics.oracle_k.eq(PRIMARY_ORACLE_K)
            & metrics.admission_cap.eq(PRIMARY_CAP)
        ].set_index("event_id")
        nonworse = int((primary.hits >= comparator.hits).sum())
        primary_hits = int(primary.hits.sum())
        gate_passed = primary_hits >= MIN_PRIMARY_HITS and nonworse >= MIN_NONWORSE_EVENTS
        decision = {
            "passed": bool(gate_passed),
            "primary_hits": primary_hits,
            "primary_possible_admissions": len(eligible) * PRIMARY_CAP,
            "events_nonworse_than_saved_hybrid_quiet_score": nonworse,
            "required_hits": MIN_PRIMARY_HITS,
            "required_nonworse_events": MIN_NONWORSE_EVENTS,
            "interpretation": (
                "Proceed to a separately validated admission-policy experiment."
                if gate_passed
                else "Do not integrate this semantic channel into the hybrid policy."
            ),
        }

        print("[6/6] Writing diagnostic evidence and final decision...", flush=True)
        predictions.drop(columns="graph_index").to_csv(output / "outer_quiet_semantic_predictions.csv", index=False)
        pd.concat(inner_summaries, ignore_index=True).to_csv(output / "inner_score_selection.csv", index=False)
        metrics.to_csv(output / "per_event_admission_metrics.csv", index=False)
        pooled.to_csv(output / "pooled_admission_metrics.csv", index=False)
        ceiling.drop(columns="graph_index").to_csv(output / "non_deployable_within_event_ceiling_predictions.csv", index=False)
        (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
        record.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "eligible_events": sorted(eligible),
                "fold_details": fold_details,
                "metrics": pooled.to_dict(orient="records"),
                "decision": decision,
                "output_files": [
                    "outer_quiet_semantic_predictions.csv",
                    "inner_score_selection.csv",
                    "per_event_admission_metrics.csv",
                    "pooled_admission_metrics.csv",
                    "non_deployable_within_event_ceiling_predictions.csv",
                    "decision.json",
                    "run_record.json",
                ],
            }
        )
        write_record(output, record)
        print(f"  decision: {'PASS' if gate_passed else 'FAIL'} ({primary_hits} Top-5 hits; {nonworse}/7 events non-worse)", flush=True)
        print(f"SUCCESS: strict-30 cross-thread source semantic diagnostic saved to: {output.resolve()}", flush=True)
    except BaseException as error:
        record.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "output_files": sorted(path.name for path in output.iterdir()),
            }
        )
        write_record(output, record)
        print(f"FAILURE: strict-30 cross-thread source semantic diagnostic preserved at: {output.resolve()}", flush=True)
        raise


if __name__ == "__main__":
    main()
