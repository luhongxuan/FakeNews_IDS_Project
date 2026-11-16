"""Free, keyless general web search via DuckDuckGo's HTML endpoint.

DuckDuckGo's HTML result page (html.duckduckgo.com/html/) is a JS-free,
plain-HTML search results page. There is no official API for it and no
published terms explicitly licensing this use -- treat it as best-effort
and rate-limited by the calling agent (a handful of queries per
verification, not bulk scraping), not as a guaranteed-stable dependency. If
this stops working, the natural upgrade path is a real search API (e.g.
Tavily's free tier), swapped in behind the same search_web() signature.
"""
from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (compatible; MisinfoResearchAgent/1.0; research use)"


def _resolve_redirect_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    if "uddg" in query:
        return unquote(query["uddg"][0])
    return href


async def fetch_web_results(query: str, max_results: int = 8) -> list[dict]:
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=15) as client:
        response = await client.get(DDG_HTML_URL, params={"q": query})
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for result in soup.select(".result"):
        link = result.select_one(".result__a")
        snippet = result.select_one(".result__snippet")
        if not link or not link.get("href"):
            continue
        results.append({
            "title": link.get_text(strip=True),
            "url": _resolve_redirect_url(link["href"]),
            "snippet": snippet.get_text(strip=True) if snippet else "",
        })
        if len(results) >= max_results:
            break
    return results
