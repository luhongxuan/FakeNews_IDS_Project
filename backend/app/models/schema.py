from sqlalchemy import Column, String, DateTime, Boolean, Text, Float
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