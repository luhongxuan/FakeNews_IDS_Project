"""Incremental same-event clustering for the live radar, via sentence
embeddings (no training -- `sentence-transformers` is already a backend
dependency, this just uses a small pretrained model to embed post text and
group posts by cosine similarity to a rolling per-cluster centroid). This is
the same technique used by news aggregators for story clustering: cheap,
keyless, no labelled data required.

Plain keyword-overlap (e.g. grouping everything that matched "shooting")
was tried first and rejected -- it lumps together unrelated posts that
happen to share a word. Embedding similarity groups posts by what they are
actually about.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from sqlalchemy.orm import Session

from app.models.schema import RadarThread

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.55
CLUSTER_LOOKBACK = timedelta(days=3)


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed(text: str) -> list[float]:
    vector = _get_model().encode(text, normalize_embeddings=True)
    return vector.tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def _post_score(post: dict) -> int:
    return post.get("reply_count", 0) * 3 + post.get("repost_count", 0)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def assign_post_to_cluster(db: Session, post: dict, matched_keyword: str) -> RadarThread:
    """Embed `post` and either merge it into the most similar recent
    RadarThread or create a new one. `post` must have at least
    uri/author_handle/text/created_at/reply_count/repost_count/like_count.
    """
    vector = embed(post["text"])
    cutoff = datetime.now(timezone.utc) - CLUSTER_LOOKBACK
    candidates = (
        db.query(RadarThread)
        .filter(RadarThread.last_seen_at >= cutoff)
        .all()
    )

    best_match: RadarThread | None = None
    best_similarity = 0.0
    for cluster in candidates:
        existing_uris = {member["uri"] for member in cluster.members}
        if post["uri"] in existing_uris:
            return cluster  # already ingested, no-op
        similarity = _cosine(vector, cluster.centroid_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = cluster

    now = datetime.now(timezone.utc)
    member_record = {**post, "matched_keyword": matched_keyword}

    if best_match is not None and best_similarity >= SIMILARITY_THRESHOLD:
        member_count = len(best_match.members)
        old_centroid = np.array(best_match.centroid_embedding)
        new_centroid = (old_centroid * member_count + np.array(vector)) / (member_count + 1)
        best_match.centroid_embedding = new_centroid.tolist()
        best_match.members = [*best_match.members, member_record]
        best_match.last_seen_at = now
        if matched_keyword not in best_match.matched_keywords:
            best_match.matched_keywords = [*best_match.matched_keywords, matched_keyword]
        if _post_score(post) > _post_score(
            next((m for m in best_match.members if m["uri"] == best_match.representative_uri), {})
        ):
            best_match.representative_text = post["text"]
            best_match.representative_uri = post["uri"]
            best_match.representative_created_at = _parse_iso(post.get("created_at"))
        db.add(best_match)
        db.commit()
        db.refresh(best_match)
        return best_match

    cluster = RadarThread(
        representative_text=post["text"],
        representative_uri=post["uri"],
        representative_created_at=_parse_iso(post.get("created_at")),
        matched_keywords=[matched_keyword],
        members=[member_record],
        centroid_embedding=vector,
        first_seen_at=now,
        last_seen_at=now,
    )
    db.add(cluster)
    db.commit()
    db.refresh(cluster)
    return cluster
