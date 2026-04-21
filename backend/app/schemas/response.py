from pydantic import BaseModel
from typing import Optional

class ClaimResult(BaseModel):
    claim_text: str
    claimant: Optional[str]
    claim_date: Optional[str]
    rating: Optional[str]
    reviewer: Optional[str]
    review_url: Optional[str]
    review_date: Optional[str]

    class Config:
        orm_mode = True

class NewsResult(BaseModel):
    title: str
    url: str
    source: Optional[str]
    date: Optional[str]
    country: Optional[str]

    class Config:
        orm_mode = True

class SearchResponse(BaseModel):
    query: str
    claims: list[ClaimResult]
    news: list[NewsResult]