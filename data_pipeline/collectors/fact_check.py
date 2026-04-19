import os
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY  = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def search_claims(query: str, language: str = "en", max_results: int = 10) -> dict:
    """
    打 Google Fact Check API，回傳原始結果
    """
    params = {
        "key"         : API_KEY,
        "query"       : query,
        "languageCode": language,
        "pageSize"    : max_results,
    }

    response = httpx.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()


def parse_claims(raw: dict) -> list[dict]:
    """
    把原始 API 回傳整理成乾淨的 list
    """
    results = []

    for claim in raw.get("claims", []):
        for review in claim.get("claimReview", []):
            results.append({
                "claim_text" : claim.get("text"),
                "claimant"   : claim.get("claimant"),
                "claim_date" : claim.get("claimDate"),
                "reviewer"   : review.get("publisher", {}).get("name"),
                "rating"     : review.get("textualRating"),
                "review_url" : review.get("url"),
                "review_date": review.get("reviewDate"),
                "review_title": review.get("title"),
            })

    return results