from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ClaimResult(BaseModel):
    claim_text: str
    claimant: Optional[str]
    claim_date: Optional[datetime]
    rating: Optional[str]
    reviewer: Optional[str]
    review_url: Optional[str]
    review_date: Optional[datetime]
    review_title: Optional[str]

    class Config:
        from_attributes = True

class NewsResult(BaseModel):
    title: str
    url: str
    source: Optional[str]
    date: Optional[datetime]
    country: Optional[str]

    class Config:
        from_attributes = True

class SearchResponse(BaseModel):
    query: str
    claims: list[ClaimResult]
    news: list[NewsResult]

class TimelineItem(BaseModel):
    date: datetime
    type: str
    title: Optional[str]
    url: Optional[str]
    source: Optional[str]
    rating: Optional[str]

class TimelineResponse(BaseModel):
    query: str
    total: int
    items: list[TimelineItem]