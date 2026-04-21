import os
import httpx
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

async def fetch_claims(query: str, max_results: int = 10) -> dict:
    params = {
        "key": API_KEY,
        "query": query,
        "languageCode": "en",
        "pageSize": max_results,
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        raw = response.json()
    
    results = []
    for claim in raw.get("claims", []):
        for review in claim.get("claimReview", []):

            results.append({
                "claim_text": claim.get("text"),
                "claimant": claim.get("claimant"),
                #2023-09-20T00:00:00Z
                "claim_date": datetime.strptime(claim.get("claimDate"), "%Y-%m-%dT%H:%M:%SZ") if claim.get("claimDate") else None,
                "reviewer": review.get("publisher", {}).get("name"),
                "rating": review.get("textualRating"),
                "review_url": review.get("url"),
                "review_date": datetime.strptime(review.get("reviewDate"), "%Y-%m-%dT%H:%M:%SZ") if review.get("reviewDate") else None,
                "review_title": review.get("title"),
            })
    
    return results