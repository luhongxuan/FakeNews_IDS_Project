# backend/app/models/graph_schema.py

from sqlalchemy import (
    Column, String, Integer, BigInteger, Boolean,
    Numeric, Text, DateTime, Date, ForeignKey, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.database import Base


class Event(Base):
    __tablename__ = "events"

    event_id         = Column(BigInteger, primary_key=True, autoincrement=True)
    pheme_event_id   = Column(String(100), unique=True, nullable=False, index=True)
    title            = Column(String(255))
    query_text       = Column(Text)
    normalized_claim = Column(Text)
    status      = Column(String(30), default="pending")
    severity         = Column(String(20), default="medium")
    first_seen_at    = Column(Date)
    node_count       = Column(Integer, default=0)
    rumour_count     = Column(Integer, default=0)
    description      = Column(Text)
    summary_jsonb    = Column(JSONB)
    built_at         = Column(DateTime)
    created_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    nodes = relationship("GraphNode", back_populates="event",
                         cascade="all, delete-orphan", lazy="dynamic")
    edges = relationship("GraphEdge", back_populates="event",
                         cascade="all, delete-orphan", lazy="dynamic")


class GraphNode(Base):
    __tablename__ = "graph_nodes"
    __table_args__ = (
        UniqueConstraint("event_id", "screen_name", name="uq_node_event_screen"),
    )

    node_id           = Column(BigInteger, primary_key=True, autoincrement=True)
    event_id          = Column(BigInteger, ForeignKey("events.event_id", ondelete="CASCADE"),
                               nullable=False, index=True)
    screen_name       = Column(String(255), nullable=False, index=True)
    node_type         = Column(String(50), default="user")
    is_rumour         = Column(Boolean, default=False)
    degree_score      = Column(Numeric)
    betweenness_score = Column(Numeric)
    pagerank          = Column(Numeric)
    risk_score        = Column(Numeric)
    created_at        = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    event = relationship("Event", back_populates="nodes")


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    edge_id         = Column(BigInteger, primary_key=True, autoincrement=True)
    event_id        = Column(BigInteger, ForeignKey("events.event_id", ondelete="CASCADE"),
                             nullable=False, index=True)
    src_screen_name = Column(String(255), nullable=False)
    dst_screen_name = Column(String(255), nullable=False)
    src_node_id     = Column(BigInteger, ForeignKey("graph_nodes.node_id"), nullable=True)
    dst_node_id     = Column(BigInteger, ForeignKey("graph_nodes.node_id"), nullable=True)
    edge_type       = Column(String(50), default="reply")
    is_rumour       = Column(Boolean, default=False)
    thread_id       = Column(String(255))
    weight          = Column(Numeric, default=1.0)
    created_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    event = relationship("Event", back_populates="edges")