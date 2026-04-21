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
        print(cached_Claims)
    else:
        claims = await fetch_claims(query)

        for claim in claims:
            claim_cache = ClaimCache(
                query=query,
                claim_text = claim["claim_text"],
                claimant = claim["claimant"],
                claim_date = claim["claim_date"],
                reviewer = claim["reviewer"],
                rating = claim["rating"],
                review_url = claim["review_url"],
                review_date = claim["review_date"],
                review_title = claim["review_title"],
            )
            db.add(claim_cache)
        db.commit()