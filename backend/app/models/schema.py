from sqlalchemy import Column, String, DateTime, Boolean, Text, Float, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime, timezone
import uuid
from app.database import Base

class ClaimCache(Base):
    __tablename__ = "claim_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, index=True, nullable=False)
    claim_text = Column(Text, nullable=False)
    claimant = Column(String, nullable=True)
    claim_date = Column(DateTime, nullable=True)
    rating = Column(String, nullable=True)
    reviewer = Column(String, nullable=True)
    review_url = Column(String, nullable=True)
    review_date = Column(DateTime, nullable=True)
    review_title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

class Newscache(Base):
    __tablename__ = "news_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, index=True, nullable=False)
    title = Column(String, nullable=False)
    url = Column(String, unique=True, nullable=False)
    source = Column(String, nullable=True)
    date = Column(DateTime, nullable=True)
    country = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))


class WebSearchCache(Base):
    __tablename__ = "web_search_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, index=True, nullable=False)
    title = Column(String, nullable=False)
    url = Column(String, nullable=False)
    snippet = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PropagationEvent(Base):
    """One post/share/comment event from social-frontend's WebSocket bridge.

    Field names/semantics are fixed by social-frontend/src/lib/backend.js
    (EVENT_TYPE, makePostEvent/makeShareEvent/makeCommentEvent) -- this is
    the receiving side of that contract, not an independent schema.
    """
    __tablename__ = "propagation_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_event_id = Column(String, index=True, nullable=True)
    type = Column(String, index=True, nullable=True)
    thread_id = Column(String, index=True, nullable=True)
    source_post_id = Column(String, nullable=True)
    from_user = Column(String, nullable=True)
    to_user = Column(String, nullable=True)
    author_user = Column(String, nullable=True)
    content = Column(Text, nullable=True)
    is_rumour = Column(Boolean, nullable=True)
    event_cluster_id = Column(String, nullable=True)
    event_ts = Column(DateTime, nullable=True)
    raw_jsonb = Column(JSONB, nullable=False)
    received_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RadarThread(Base):
    """One clustered 'story' from the live Bluesky radar -- a group of
    public posts judged similar enough (by sentence-embedding cosine
    similarity, see app/services/event_clustering.py) to represent the same
    real-world event, not a single post.

    Shared by both the general-user "trending now" view and the research
    "pending review" queue (app/routers/radar.py); `reviewed` only matters
    for the research workflow. This is a live-monitoring surface, never fed
    into early-detection model training/evaluation.
    """
    __tablename__ = "radar_threads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    representative_text = Column(Text, nullable=False)
    representative_uri = Column(String, nullable=False)
    representative_created_at = Column(DateTime, nullable=True)
    matched_keywords = Column(JSONB, nullable=False, default=list)
    members = Column(JSONB, nullable=False, default=list)
    centroid_embedding = Column(JSONB, nullable=False)
    reviewed = Column(Boolean, nullable=False, default=False)
    first_seen_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class VerificationReport(Base):
    """Cached output of the post-hoc verification agent for one thread.

    This is decision-support evidence gathered AFTER a thread has already
    been ranked by the early-detection model. It must never be read back
    into that model's features -- see app/services/verification_agent.py.
    """
    __tablename__ = "verification_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String, index=True, nullable=False, unique=True)
    event_id = Column(String, index=True, nullable=False)
    model_name = Column(String, nullable=False)
    report_jsonb = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class RadarQueryExpansionAudit(Base):
    """Live Radar Event Discovery V2, phase 3 -- one row per event-specific
    query-expansion attempt for one seed source post (app/services/
    radar_query_expansion.py). A NEW, independent table -- adding it does
    not alter RadarThread's existing schema or rows, and Base.metadata.
    create_all() (see app/main.py) only needs to CREATE this table, never
    ALTER an existing one, so this is safe to add without a migration.

    Purely an audit trail: every issued query, every candidate URI found,
    which of those were new after dedup against already-known URIs, and
    (once a merge decision is made for a candidate -- see
    event_clustering_v2.classify_cluster_merge) the reasons behind it, all
    recorded here so an expansion round can be reconstructed after the
    fact. Never itself creates or edits a RadarThread row.
    """
    __tablename__ = "radar_query_expansion_audits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    seed_uri = Column(String, index=True, nullable=False)
    cluster_thread_id = Column(String, index=True, nullable=True)
    queries_issued = Column(JSONB, nullable=False, default=list)
    candidate_uris_found = Column(JSONB, nullable=False, default=list)
    new_uris_after_dedup = Column(JSONB, nullable=False, default=list)
    merge_decisions = Column(JSONB, nullable=False, default=list)
    skipped_cooldown = Column(Boolean, nullable=False, default=False)
    error = Column(Text, nullable=True)
    expanded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class InterventionDecision(Base):
    """One audit-trail row from intervention_agent.py's triage loop for one
    radar thread at one checkpoint.

    Recommendation only -- nothing here ever deletes or hides content. Also
    the input get_budget_state (see intervention_agent.py) reads from, so
    the agent can see how many "escalate_now" slots this checkpoint tier
    has already used before deciding whether to use another one.
    """
    __tablename__ = "intervention_decisions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(String, index=True, nullable=False)
    post_uri = Column(String, nullable=False)
    checkpoint_minutes = Column(Integer, nullable=False)
    action = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=False)
    report_jsonb = Column(JSONB, nullable=False)
    decided_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))