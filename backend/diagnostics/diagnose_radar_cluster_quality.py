"""Live Radar Event Discovery V2, phase 1 -- READ-ONLY diagnostic.

Reads every existing RadarThread row and reports on clustering quality.
Never writes, updates, or deletes a RadarThread, VerificationReport, or
InterventionDecision row -- only SELECT queries against the real DB, plus
local (no-network, no-training) re-embedding of the SAME already-stored
member post texts via the same frozen sentence-transformers model
event_clustering.py already uses for inference (never trained or
fine-tuned here).

Produces, in a new timestamped output directory under backend/diagnostics/:
  - run_record.json             -- AGENTS.md-style run record (status,
                                    timestamps, input source, output files)
  - cluster_quality_summary.json -- the aggregate numbers this phase is for
  - cluster_inventory.csv        -- one row per existing RadarThread
  - multi_member_pair_audit.csv  -- one row per (cluster, member) pair
                                    audited for entity/location conflicts
  - singleton_samples.csv        -- a sample of singleton clusters with
                                    their display-eligibility classification
  - conclusion.md                -- human-readable write-up

Run: python diagnose_radar_cluster_quality.py
(from backend/diagnostics/, using the project venv -- see the repo's own
backend/dev_server.py for the same DATABASE_URL default this script uses)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://admin:secret@localhost:15432/misinfo_db")

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np

from app.database import SessionLocal
from app.models.schema import RadarThread
from app.services.event_clustering import SIMILARITY_THRESHOLD as V1_SIMILARITY_THRESHOLD
from app.services.event_clustering import embed
from app.services.event_clustering_v2 import classify_cluster_merge, cosine_similarity, is_cluster_claim_ambiguous
from app.services.radar_event_signature import EventSignature, extract_event_signature
from app.services.radar_thread_quality import (
    DISPLAY_ELIGIBLE,
    SEARCH_NOISE_CANDIDATE,
    SINGLETON_MONITORING,
    classify_radar_thread_quality,
)

THRESHOLD_SWEEP = (0.55, 0.60, 0.65, 0.70, 0.75)
SINGLETON_SAMPLE_SIZE = 25


def _total_engagement(members: list[dict]) -> int:
    return sum(m.get("reply_count", 0) * 3 + m.get("repost_count", 0) for m in members)


def _load_threads_readonly(db) -> list[RadarThread]:
    # .all() with no .update()/.delete() anywhere in this file -- read-only by
    # construction. Ordered for determinism across reruns of this same script.
    return db.query(RadarThread).order_by(RadarThread.first_seen_at.asc()).all()


# --- cluster inventory / per-cluster stats -----------------------------------

def _build_cluster_inventory(threads: list[RadarThread]) -> list[dict]:
    rows = []
    for t in threads:
        members = t.members or []
        reply_counts = [m.get("reply_count", 0) for m in members]
        repost_counts = [m.get("repost_count", 0) for m in members]
        rows.append({
            "cluster_id": str(t.id),
            "member_count": len(members),
            "matched_keywords": ";".join(t.matched_keywords or []),
            "sum_reply_count": sum(reply_counts),
            "max_reply_count": max(reply_counts, default=0),
            "sum_repost_count": sum(repost_counts),
            "total_engagement": _total_engagement(members),
            "first_seen_at": t.first_seen_at.isoformat() if t.first_seen_at else None,
            "last_seen_at": t.last_seen_at.isoformat() if t.last_seen_at else None,
            "representative_text_preview": (t.representative_text or "")[:100].replace("\n", " "),
        })
    return rows


# --- multi-member entity/location conflict audit -----------------------------

def _build_multi_member_pair_audit(threads: list[RadarThread]) -> tuple[list[dict], int]:
    """One row per (cluster, member) with that member's own extracted
    signature and whether it conflicts (disjoint explicit locations, or
    disjoint incident types) with the CLUSTER'S representative post's
    signature -- i.e. would event_clustering_v2.classify_cluster_merge
    have refused to merge this member into this cluster's representative,
    had v2 been in effect when this cluster was assembled. v1 has no such
    check at merge time (see event_clustering.py), so an existing cluster
    can absolutely contain one regardless.

    Returns (rows, n_ambiguous_clusters) -- the second number is what
    cluster_quality_summary.json reports as "clusters with a possible
    entity/location conflict".
    """
    rows = []
    n_ambiguous = 0
    for t in threads:
        members = t.members or []
        if len(members) < 2:
            continue
        representative_signature = extract_event_signature(t.representative_text or "")
        member_signatures = [extract_event_signature(m.get("text", "")) for m in members]
        cluster_ambiguous = is_cluster_claim_ambiguous(member_signatures)
        if cluster_ambiguous:
            n_ambiguous += 1
        for member, signature in zip(members, member_signatures):
            location_conflict = bool(
                signature.locations and representative_signature.locations
                and signature.locations.isdisjoint(representative_signature.locations)
            )
            incident_conflict = bool(
                signature.incident_types and representative_signature.incident_types
                and signature.incident_types.isdisjoint(representative_signature.incident_types)
            )
            rows.append({
                "cluster_id": str(t.id),
                "cluster_ambiguous": cluster_ambiguous,
                "member_uri": member.get("uri"),
                "member_text_preview": (member.get("text") or "")[:80].replace("\n", " "),
                "member_locations": ";".join(sorted(signature.locations)),
                "member_incident_types": ";".join(sorted(signature.incident_types)),
                "representative_locations": ";".join(sorted(representative_signature.locations)),
                "representative_incident_types": ";".join(sorted(representative_signature.incident_types)),
                "location_conflict_vs_representative": location_conflict,
                "incident_conflict_vs_representative": incident_conflict,
            })
    return rows, n_ambiguous


# --- singleton display-eligibility sampling ----------------------------------

def _build_singleton_samples(threads: list[RadarThread]) -> list[dict]:
    rows = []
    for t in threads:
        members = t.members or []
        if len(members) != 1:
            continue
        classification = classify_radar_thread_quality({
            "member_count": len(members),
            "members": members,
            "matched_keywords": t.matched_keywords or [],
            "representative_text": t.representative_text,
        })
        rows.append({
            "cluster_id": str(t.id),
            "matched_keywords": ";".join(t.matched_keywords or []),
            "reply_count": members[0].get("reply_count", 0) if members else 0,
            "repost_count": members[0].get("repost_count", 0) if members else 0,
            "display_status": classification["display_status"],
            "reasons": " | ".join(classification["reasons"]),
            "representative_text_preview": (t.representative_text or "")[:100].replace("\n", " "),
        })
    rows.sort(key=lambda r: -(r["reply_count"] + r["repost_count"]))
    return rows[:SINGLETON_SAMPLE_SIZE]


# --- threshold sensitivity sweep ----------------------------------------------

def _flatten_posts(threads: list[RadarThread]) -> list[dict]:
    posts = []
    seen_uris = set()
    for t in threads:
        for m in t.members or []:
            uri = m.get("uri")
            if not uri or uri in seen_uris or not m.get("text"):
                continue
            seen_uris.add(uri)
            posts.append(m)
    posts.sort(key=lambda p: p.get("created_at") or "")
    return posts


def _simulate_clustering(posts: list[dict], threshold: float, use_v2: bool) -> list[dict]:
    """Single-pass greedy clustering over the FLATTENED real member posts,
    re-embedding each with the same frozen model -- inference only, no
    training, no gradient step, nothing persisted. Mirrors event_
    clustering.assign_post_to_cluster's own greedy best-match logic so the
    comparison across thresholds/v1-vs-v2 is apples to apples, but is a
    SEPARATE simulation over already-collected real text, never touching
    the actual RadarThread rows those posts came from."""
    clusters: list[dict] = []
    for post in posts:
        vector = embed(post["text"])
        signature = extract_event_signature(post["text"])
        best_cluster = None
        best_similarity = 0.0
        for cluster in clusters:
            similarity = cosine_similarity(vector, cluster["centroid"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        merge = False
        if best_cluster is not None:
            if use_v2:
                decision = classify_cluster_merge(
                    vector, signature, best_cluster["centroid"],
                    best_cluster["representative_vector"], best_cluster["signature"],
                    similarity_threshold=threshold,
                )
                merge = decision.merge
            else:
                merge = best_similarity >= threshold

        if merge:
            n = len(best_cluster["members"])
            old_centroid = np.array(best_cluster["centroid"])
            best_cluster["centroid"] = ((old_centroid * n + np.array(vector)) / (n + 1)).tolist()
            best_cluster["members"].append(post)
        else:
            clusters.append({
                "members": [post], "centroid": vector, "representative_vector": vector,
                "representative_text": post["text"], "signature": signature,
            })
    return clusters


def _summarize_simulation(clusters: list[dict]) -> dict:
    n_total = len(clusters)
    n_singleton = sum(1 for c in clusters if len(c["members"]) == 1)
    n_multi = n_total - n_singleton
    n_flagged_false_merge = 0
    n_legitimate_duplicate_retention = 0
    for c in clusters:
        if len(c["members"]) < 2:
            continue
        signatures = [extract_event_signature(m["text"]) for m in c["members"]]
        if is_cluster_claim_ambiguous(signatures):
            n_flagged_false_merge += 1
            continue
        # weak proxy for "these really do look like the same story": ANY pair
        # of members in this (non-ambiguous) cluster shares a named entity or
        # distinctive phrase -- see radar_event_signature.py.
        all_entities = [s.named_entities | s.distinctive_phrases for s in signatures]
        if any(a & b for i, a in enumerate(all_entities) for b in all_entities[i + 1:]):
            n_legitimate_duplicate_retention += 1
    return {
        "n_total_clusters": n_total,
        "n_singleton": n_singleton,
        "singleton_rate": round(n_singleton / n_total, 4) if n_total else None,
        "n_multi_member_clusters": n_multi,
        "n_rule_flagged_false_merge": n_flagged_false_merge,
        "n_legitimate_duplicate_retention_proxy": n_legitimate_duplicate_retention,
    }


def _threshold_sensitivity_table(posts: list[dict]) -> dict:
    table = {}
    for threshold in THRESHOLD_SWEEP:
        table[f"v1_centroid_only_{threshold}"] = _summarize_simulation(_simulate_clustering(posts, threshold, use_v2=False))
        table[f"v2_entity_aware_{threshold}"] = _summarize_simulation(_simulate_clustering(posts, threshold, use_v2=True))
    return table


# --- keyword-level noise stats -------------------------------------------------

def _keyword_stats(threads: list[RadarThread]) -> dict:
    per_keyword: dict[str, dict] = {}
    for t in threads:
        classification = classify_radar_thread_quality({
            "member_count": len(t.members or []),
            "members": t.members or [],
            "matched_keywords": t.matched_keywords or [],
            "representative_text": t.representative_text,
        })
        for kw in (t.matched_keywords or []):
            bucket = per_keyword.setdefault(kw, {"n_clusters": 0, "n_singleton": 0, "n_search_noise_candidate": 0, "n_display_eligible": 0})
            bucket["n_clusters"] += 1
            if len(t.members or []) == 1:
                bucket["n_singleton"] += 1
            if classification["display_status"] == SEARCH_NOISE_CANDIDATE:
                bucket["n_search_noise_candidate"] += 1
            if classification["display_status"] == DISPLAY_ELIGIBLE:
                bucket["n_display_eligible"] += 1
    for kw, bucket in per_keyword.items():
        bucket["singleton_rate"] = round(bucket["n_singleton"] / bucket["n_clusters"], 4) if bucket["n_clusters"] else None
        bucket["effective_event_rate"] = round(bucket["n_display_eligible"] / bucket["n_clusters"], 4) if bucket["n_clusters"] else None
    return per_keyword


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = BACKEND / "diagnostics" / f"{timestamp}_radar_cluster_quality"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[diagnose_radar_cluster_quality] started_at={started_at}", flush=True)
    print(f"      output directory: {out_dir}", flush=True)
    print("      READ-ONLY: no RadarThread row will be created, modified, or deleted.", flush=True)

    run_record_base = {
        "script": "backend/diagnostics/diagnose_radar_cluster_quality.py",
        "started_at": started_at,
        "read_only": True,
        "input_source": "radar_threads table (real Postgres DB, live radar data)",
        "output_files": [
            "run_record.json", "cluster_quality_summary.json", "cluster_inventory.csv",
            "multi_member_pair_audit.csv", "singleton_samples.csv", "conclusion.md",
        ],
    }
    (out_dir / "run_record.json").write_text(
        json.dumps({**run_record_base, "status": "running", "completed_at": None}, indent=2, ensure_ascii=False), encoding="utf-8")

    db = SessionLocal()
    try:
        print("[1/6] Loading radar_threads (read-only)...", flush=True)
        threads = _load_threads_readonly(db)
        n_threads = len(threads)
        n_singleton = sum(1 for t in threads if len(t.members or []) == 1)
        print(f"      {n_threads} clusters, {n_singleton} singleton ({round(100 * n_singleton / n_threads, 1) if n_threads else 0}%)", flush=True)

        print("[2/6] Building cluster inventory...", flush=True)
        inventory_rows = _build_cluster_inventory(threads)

        print("[3/6] Auditing multi-member clusters for entity/location conflicts...", flush=True)
        pair_audit_rows, n_ambiguous_clusters = _build_multi_member_pair_audit(threads)
        print(f"      {n_ambiguous_clusters} multi-member clusters flagged as internally ambiguous", flush=True)

        print("[4/6] Classifying display eligibility + sampling singletons...", flush=True)
        singleton_rows = _build_singleton_samples(threads)
        n_display_eligible = sum(
            1 for t in threads
            if classify_radar_thread_quality({
                "member_count": len(t.members or []), "members": t.members or [],
                "matched_keywords": t.matched_keywords or [], "representative_text": t.representative_text,
            })["display_status"] == DISPLAY_ELIGIBLE
        )
        n_search_noise = sum(
            1 for t in threads
            if classify_radar_thread_quality({
                "member_count": len(t.members or []), "members": t.members or [],
                "matched_keywords": t.matched_keywords or [], "representative_text": t.representative_text,
            })["display_status"] == SEARCH_NOISE_CANDIDATE
        )
        keyword_stats = _keyword_stats(threads)

        print("[5/6] Threshold sensitivity sweep (0.55/0.60/0.65/0.70/0.75, v1 vs v2, re-embedding real stored text)...", flush=True)
        flattened_posts = _flatten_posts(threads)
        print(f"      {len(flattened_posts)} distinct member posts to re-cluster per threshold", flush=True)
        sensitivity_table = _threshold_sensitivity_table(flattened_posts)

        print("[6/6] Writing outputs...", flush=True)
        summary = {
            "n_total_clusters": n_threads,
            "n_singleton": n_singleton,
            "singleton_rate": round(n_singleton / n_threads, 4) if n_threads else None,
            "member_count_distribution": {
                str(k): sum(1 for t in threads if len(t.members or []) == k)
                for k in sorted({len(t.members or []) for t in threads})
            },
            "n_display_eligible_singletons": sum(
                1 for t in threads if len(t.members or []) == 1
                and classify_radar_thread_quality({
                    "member_count": 1, "members": t.members or [],
                    "matched_keywords": t.matched_keywords or [], "representative_text": t.representative_text,
                })["display_status"] == DISPLAY_ELIGIBLE
            ),
            "n_display_eligible_total": n_display_eligible,
            "n_search_noise_candidate": n_search_noise,
            "n_multi_member_clusters_with_possible_entity_conflict": n_ambiguous_clusters,
            "n_multi_member_clusters_total": sum(1 for t in threads if len(t.members or []) > 1),
            "per_keyword_stats": keyword_stats,
            "threshold_sensitivity_table": sensitivity_table,
            "current_production_v1_threshold": V1_SIMILARITY_THRESHOLD,
            "note": "display-eligibility thresholds (radar_thread_quality.py) and the v2 "
                    "similarity threshold used here are ENGINEERING defaults, not values "
                    "validated/optimized against a labeled ground truth -- there is no "
                    "'this is really the same event' label available for live Bluesky data.",
        }
        (out_dir / "cluster_quality_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(out_dir / "cluster_inventory.csv", inventory_rows)
        _write_csv(out_dir / "multi_member_pair_audit.csv", pair_audit_rows)
        _write_csv(out_dir / "singleton_samples.csv", singleton_rows)

        search_noise_share_of_singletons = round(100 * n_search_noise / n_singleton, 1) if n_singleton else 0.0
        v1_at_threshold = sensitivity_table.get(f"v1_centroid_only_{V1_SIMILARITY_THRESHOLD}", {})
        v2_at_threshold = sensitivity_table.get(f"v2_entity_aware_{V1_SIMILARITY_THRESHOLD}", {})
        conclusion_lines = [
            "# Live Radar Event Discovery V2 -- Phase 1 Diagnostic Conclusion",
            "",
            f"Generated {started_at}. Read-only: {n_threads} existing RadarThread rows inspected, none modified.",
            "",
            "## Headline numbers",
            f"- Total clusters: {n_threads}",
            f"- Singleton clusters: {n_singleton} ({summary['singleton_rate']:.1%})" if summary["singleton_rate"] is not None else "- Singleton clusters: 0",
            f"- display_eligible (any member_count): {n_display_eligible}",
            f"- search_noise_candidate: {n_search_noise} ({search_noise_share_of_singletons}% of the {n_singleton} singletons -- "
            "this is the EXACT share, not 'most'; the remaining singletons' cause is unknown, see below)",
            f"- Multi-member clusters with a possible entity/location conflict: {n_ambiguous_clusters} / {summary['n_multi_member_clusters_total']}",
            "",
            "## Interpretation (corrected per review -- do not restate the earlier, wrong 'most singleton "
            "clusters are search noise' claim; it contradicted this file's own numbers)",
            f"Only {search_noise_share_of_singletons}% of singletons are flagged search_noise_candidate by the "
            "current heuristic. The remaining singletons' cause is UNKNOWN from this diagnostic alone -- each of "
            "the following is possible and none is ruled out: a real event with genuinely no other found source "
            "post, search noise the heuristic's own limits miss (see radar_thread_quality._looks_like_search_noise's "
            "docstring), a real duplicate this run's clustering failed to merge, or simply a story Bluesky only "
            "has one post about right now.",
            "",
            "v1's pure embedding-centroid clustering can merge multiple DIFFERENT real incidents into one cluster "
            "when their breaking-news phrasing is similar (see multi_member_pair_audit.csv rows with "
            "location_conflict_vs_representative=True) -- confirmed at this run's numbers: v1 at the current "
            f"production threshold ({V1_SIMILARITY_THRESHOLD}) shows "
            f"n_rule_flagged_false_merge={v1_at_threshold.get('n_rule_flagged_false_merge')}. The v2 (entity-aware) "
            f"column at the SAME threshold shows n_rule_flagged_false_merge={v2_at_threshold.get('n_rule_flagged_false_merge')}, "
            f"but at the cost of singleton_rate rising to {v2_at_threshold.get('singleton_rate')} (v1: "
            f"{v1_at_threshold.get('singleton_rate')}) and multi-member clusters falling to "
            f"{v2_at_threshold.get('n_multi_member_clusters')} (v1: {v1_at_threshold.get('n_multi_member_clusters')}) -- "
            "i.e. v2's stricter merge gate is NOT free: it trades away multi-member clusters (the exact thing this "
            "whole effort wants MORE of) to remove false merges. This table does not by itself justify wiring v2 "
            "into live ingestion; see this diagnostic's own README/report for the recommended next step (a small, "
            "manually-triggered query-expansion pilot with human-judged same-event/different-event labels) before "
            "that decision.",
            "",
            "n_rule_flagged_false_merge and n_legitimate_duplicate_retention_proxy are BOTH computed by the same "
            "event-signature heuristic being evaluated -- they are self-referential proxies, not accuracy against "
            "a human-labeled ground truth. No threshold value here is claimed to be validated or optimal.",
        ]
        (out_dir / "conclusion.md").write_text("\n".join(conclusion_lines), encoding="utf-8")

        completed_at = datetime.now(timezone.utc).isoformat()
        (out_dir / "run_record.json").write_text(
            json.dumps({**run_record_base, "status": "complete", "completed_at": completed_at, "metrics_summary": {
                "n_total_clusters": n_threads, "n_singleton": n_singleton, "singleton_rate": summary["singleton_rate"],
                "n_ambiguous_multi_member_clusters": n_ambiguous_clusters,
            }}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSUCCESS -- {out_dir}", flush=True)
    except Exception:
        (out_dir / "run_record.json").write_text(
            json.dumps({**run_record_base, "status": "failed", "completed_at": datetime.now(timezone.utc).isoformat(),
                        "error_traceback": traceback.format_exc()}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFAILURE -- {out_dir}", flush=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
