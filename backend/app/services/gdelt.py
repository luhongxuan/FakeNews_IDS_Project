import httpx
from datetime import datetime

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

async def fetch_news(query: str, max_results: int = 20) -> list[dict]:
    params = {
        "query"     : query,
        "mode"      : "artlist",
        "maxrecords": max_results,
        "format"    : "json",
        "sort"      : "DateDesc",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

    articles = []
    seen_urls = set()

    for article in data.get("articles", []):
        url = article.get("url")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)

        articles.append({
            "title"  : article.get("title"),
            "url"    : url,
            "source" : article.get("domain"),
            #20260421T061500Z
            "date"   : datetime.strptime(article.get("seendate"), "%Y%m%dT%H%M%SZ") if article.get("seendate") else None,
            "country": article.get("countrycode"),
        })

    articles.sort(key=lambda x: x["date"] or "", reverse=True)

    return articles[:max_results]