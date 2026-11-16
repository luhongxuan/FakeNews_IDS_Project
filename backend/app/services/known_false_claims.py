"""RAG knowledge base of already-verified `likely_false` claims for the live
radar pipeline.

One-time setup required before this module's writes/reads will work at all
(a stock `postgres:15` image does not have this extension available -- see
docker-compose.yml's `db` service, switched to pgvector/pgvector:pg15):

    CREATE EXTENSION IF NOT EXISTS vector;

Run once against the running `db` container, then let this project's normal
table-creation path (Base.metadata.create_all, wherever app/database.py's
startup already does that) pick up the new `known_false_claims` table --
there is no Alembic migration system here, consistent with every other
table in app/models/schema.py.

Two uses of the SAME retrieval (find_similar), both real RAG, not just a
similarity score:
1. Priority-queue boosting -- a brand-new unverified candidate whose text is
   close to a PAST confirmed-false claim is very likely the same rumour
   reworded (this project found exactly that live: the "3 gunmen" Charlie
   Hebdo claim resurfaced across many differently-worded posts, all
   independently confirmed false). A caller deciding what to verify next
   (e.g. LiveScenarioSimulator's auto-verify picker) can use a close match
   to jump such a candidate to the front of the queue, well before it would
   otherwise be reached by predicted_impact/percentile_score alone.
2. Retrieved context handed BACK to the verification agent -- the matched
   past claim's summary/evidence can be injected into
   verification_agent.run_verification's prompt (via its own retrieval
   param) so the agent starts from "a semantically similar claim was
   already checked and found false, here is what was found" instead of
   rediscovering the same evidence from a blank slate every time.

Only ever written by record() from a REAL verification agent result --
never seeded, hand-edited, or backfilled from the offline PHEME/
Twitter15-16 benchmarks. Mixing retrospective benchmark labels into this
store would silently make the live radar's own priority scores look like
they are grounded in real-world outcomes when they would actually be
grounded in a decade-old, differently-distributed dataset (see this
project's own PHEME/Twitter15-16 realism analysis, which is the whole
reason this module exists: the live radar should build its OWN, current,
non-decayed evidence base over time instead of leaning further on those).
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.schema import KnownFalseClaim
from app.services.event_clustering import embed

# Tighter than event_clustering.SIMILARITY_THRESHOLD (0.55, used for "is this
# the same NEWS STORY, worth grouping under one radar thread"). "Is this the
# same FALSE CLAIM reworded" is a stronger, more specific claim to make, and
# a false positive here directly inflates a candidate's verification
# priority (and, if used as RAG context, could bias the agent toward a
# stale prior) -- so it deliberately requires a much closer semantic match
# than mere same-topic clustering does.
SIMILARITY_THRESHOLD = 0.75
DEFAULT_TOP_K = 5


def record(
    db: Session, thread_id: str, claim_text: str,
    credibility: str, confidence: float | None, summary: str | None,
) -> None:
    """Store one real verification result in the RAG knowledge base.

    Only persists credibility == "likely_false" -- an "unverified" or
    "likely_true" result isn't a known FALSE claim to match future
    candidates against, and recording it here would let a merely-unresolved
    claim silently start boosting other candidates' priority as if it had
    been confirmed false. Best-effort: swallows the error (with a rollback,
    so the caller's own session isn't left in an aborted-transaction state)
    if the table/extension doesn't exist yet, e.g. before the one-time
    `CREATE EXTENSION vector` migration has been run -- this is an
    enrichment side-effect of verification, not something that should ever
    break the verification response itself.
    """
    if credibility != "likely_false":
        return
    try:
        vector = embed(claim_text)
        db.add(KnownFalseClaim(
            source_thread_id=thread_id, claim_text=claim_text, embedding=vector,
            credibility=credibility, confidence=confidence, summary=summary,
        ))
        db.commit()
    except Exception:
        db.rollback()


def find_similar(
    db: Session, claim_text: str,
    top_k: int = DEFAULT_TOP_K, min_similarity: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Nearest known-false claims to `claim_text`, closest first.

    Returns [] (never raises) when nothing clears min_similarity, or when
    the table/extension doesn't exist yet -- a fresh install with no
    verified-false history at all, or before the one-time migration, must
    degrade to "no prior match", not break every candidate's priority
    scoring or verification call.
    """
    try:
        vector = embed(claim_text)
        rows = (
            db.query(KnownFalseClaim, KnownFalseClaim.embedding.cosine_distance(vector).label("distance"))
            .order_by("distance")
            .limit(top_k)
            .all()
        )
    except Exception:
        db.rollback()
        return []

    results = []
    for claim, distance in rows:
        similarity = 1.0 - float(distance)
        if similarity < min_similarity:
            continue
        results.append({
            "source_thread_id": claim.source_thread_id,
            "claim_text": claim.claim_text,
            "credibility": claim.credibility,
            "confidence": claim.confidence,
            "summary": claim.summary,
            "similarity": round(similarity, 4),
            "verified_at": claim.verified_at.isoformat() if claim.verified_at else None,
        })
    return results
