from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from app.database import get_db
from app.services.gdelt import fetch_news
from app.services.fact_check import fetch_claims
from app.schemas.response import TimelineResponse, TimelineItem
from app.models.schema import ClaimCache, Newscache

router = APIRouter()

@router.get("/timeline", response_model=TimelineResponse)
async def timeline(query: str, db: Session = Depends(get_db)):
    
    timeline_items = []

    cached_Claims = db.query(ClaimCache).filter(ClaimCache.query == query).all()

    if cached_Claims:
        for claim in cached_Claims:
            if claim.review_date:
                timeline_items.append(TimelineItem(
                    date=claim.review_date,
                    type="claim_review_date",
                    title=claim.claim_text,
                    url=claim.review_url,
                    source=claim.reviewer,
                    rating=claim.rating
                ))
            if claim.claim_date:
                timeline_items.append(TimelineItem(
                    date=claim.claim_date,
                    type="claim_claim_date",
                    title=claim.claim_text,
                    url=None,
                    source=claim.claimant,
                    rating=None
                ))
    else:
        claims = await fetch_claims(query)

        for claim in claims:
            if claim["review_date"]:
                timeline_items.append(TimelineItem(
                    date=claim["review_date"],
                    type="claim_review_date",
                    title=claim["claim_text"],
                    url=claim["review_url"],
                    source=claim["reviewer"],
                    rating=claim["rating"]
                ))
            if claim["claim_date"]:
                timeline_items.append(TimelineItem(
                    date=claim["claim_date"],
                    type="claim_claim_date",
                    title=claim["claim_text"],
                    url=None,
                    source=claim["claimant"],
                    rating=None
                ))

            db.add(ClaimCache(query=query, **claim))
        db.commit()
    
    cached_News = db.query(Newscache).filter(Newscache.query == query).all()

    if cached_News:
        for new in cached_News:
            if new.date:
                timeline_items.append(TimelineItem(
                    date=new.date,
                    type="news_date",
                    title=new.title,
                    url=new.url,
                    source=new.source,
                    rating=None
                ))
    else:
        news = await fetch_news(query, max_results=30)

        for new in news:
            if new["date"]:
                timeline_items.append(TimelineItem(
                    date=new["date"],
                    type="news_date",
                    title=new["title"],
                    url=new["url"],
                    source=new["source"],
                    rating=None
                ))

            db.add(Newscache(query=query, **new))
        db.commit()
    
    timeline_items.sort(key=lambda x: x.date, reverse=True)

    return TimelineResponse(
        query=query,
        total=len(timeline_items),
        items=timeline_items
    )