from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.fact_check import fetch_claims
from app.services.gdelt import fetch_news
from app.schemas.response import SearchResponse
from app.models.schema import ClaimCache, Newscache

router = APIRouter()

@router.get("/search", response_model=SearchResponse)
async def search(query: str, db: Session = Depends(get_db)):
    
    cached_Claims = db.query(ClaimCache).filter(ClaimCache.query == query).all()
    
    if cached_Claims:
        # claims = []
        # for claim in cached_Claims:
        #     claims.append({
        #         "claim_text": claim.claim_text,
        #         "claimant": claim.claimant,
        #         "claim_date": claim.claim_date,
        #         "reviewer": claim.reviewer,
        #         "rating": claim.rating,
        #         "review_url": claim.review_url,
        #         "review_date": claim.review_date,
        #         "review_title": claim.review_title
        #     })
        #claims = [claim.__dict__ for claim in cached_Claims]
        claims = cached_Claims
    else:
        claims = await fetch_claims(query)

        for claim in claims:
            # claim_cache = ClaimCache(
            #     query=query,
            #     claim_text = claim["claim_text"],
            #     claimant = claim["claimant"],
            #     claim_date = claim["claim_date"],
            #     reviewer = claim["reviewer"],
            #     rating = claim["rating"],
            #     review_url = claim["review_url"],
            #     review_date = claim["review_date"],
            #     review_title = claim["review_title"],
            # )
            db.add(ClaimCache(query=query, **claim))
        db.commit()

    cached_News = db.query(Newscache).filter(Newscache.query == query).all()

    if cached_News:
        # for new in cached_News:
        #     news.append({
        #         "title": new.title,
        #         "url": new.url,
        #         "source": new.source,
        #         "date": new.date,
        #         "country": new.country
        #     })
        #news = [new.__dict__ for new in cached_News]
        news = cached_News
    else:
        news = await fetch_news(query)

        for new in news:
            # news_cache = Newscache(
            #     query=query,
            #     title = new["title"],
            #     url = new["url"],
            #     source = new["source"],
            #     date = new["date"],
            #     country = new["country"]
            # )
            db.add(Newscache(query=query, **new))
        db.commit()

    return SearchResponse(query=query, claims=claims, news=news)