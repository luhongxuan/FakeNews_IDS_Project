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
        orm_mode = True

class NewsResult(BaseModel):
    title: str
    url: str
    source: Optional[str]
    date: Optional[datetime]
    country: Optional[str]

    class Config:
        orm_mode = True

class SearchResponse(BaseModel):
    query: str
    claims: list[ClaimResult]
    news: list[NewsResult]