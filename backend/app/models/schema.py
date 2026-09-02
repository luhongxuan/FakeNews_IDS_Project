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