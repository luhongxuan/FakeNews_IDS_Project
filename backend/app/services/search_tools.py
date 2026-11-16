"""Cache-aware wrappers around the existing GDELT news and Google Fact Check
services, reused as tools for the verification agent.

Mirrors the same query-string cache pattern already used in
app/routers/timeline.py (ClaimCache / Newscache), so repeated agent tool
calls for the same query don't re-hit the external APIs.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.schema import ClaimCache, Newscache, WebSearchCache
from app.services.fact_check import fetch_claims
from app.services.gdelt import fetch_news
from app.services.web_search import fetch_web_results


async def search_news(db: Session, query: str, max_results: int = 10) -> list[dict]:
    cached = db.query(Newscache).filter(Newscache.query == query).all()
    if cached:
        return [
            {"title": row.title, "url": row.url, "source": row.source,
             "date": row.date.isoformat() if row.date else None, "country": row.country}
            for row in cached[:max_results]
        ]
    articles = await fetch_news(query, max_results=max_results)
    for article in articles:
        db.add(Newscache(query=query, **article))
    db.commit()
    return [
        {**article, "date": article["date"].isoformat() if article.get("date") else None}
        for article in articles
    ]


async def search_web(db: Session, query: str, max_results: int = 8) -> list[dict]:
    cached = db.query(WebSearchCache).filter(WebSearchCache.query == query).all()
    if cached:
        return [{"title": row.title, "url": row.url, "snippet": row.snippet} for row in cached[:max_results]]
    results = await fetch_web_results(query, max_results=max_results)
    for item in results:
        db.add(WebSearchCache(query=query, **item))
    db.commit()
    return results


async def search_fact_checks(db: Session, query: str, max_results: int = 10) -> list[dict]:
    cached = db.query(ClaimCache).filter(ClaimCache.query == query).all()
    if cached:
        rows = cached[:max_results]
    else:
        claims = await fetch_claims(query, max_results=max_results)
        for claim in claims:
            db.add(ClaimCache(query=query, **claim))
        db.commit()
        rows = db.query(ClaimCache).filter(ClaimCache.query == query).all()[:max_results]
    return [
        {
            "claim_text": row.claim_text, "claimant": row.claimant,
            "rating": row.rating, "reviewer": row.reviewer, "review_url": row.review_url,
            "review_date": row.review_date.isoformat() if row.review_date else None,
        }
        for row in rows
    ]
