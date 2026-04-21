from sqlalchemy import Column, String, DateTime, Boolean, Text, Float
from sqlalchemy.dialects.postgresql import UUID
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