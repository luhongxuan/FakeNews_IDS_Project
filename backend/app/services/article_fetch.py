"""Fetch and extract the main text of one article URL.

Used by the verification agent's `fetch_article` tool: search results give
only a title + one-line snippet, which is often not enough to tell whether
an article actually supports or refutes a claim. This lets the agent read
further before deciding -- but only for URLs that already came from a prior
search result (enforced by the caller), never an arbitrary URL the model
invents.
"""
from __future__ import annotations

from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; MisinfoResearchAgent/1.0; research use)"

_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _is_safe_public_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    if not host or host in _BLOCKED_HOSTS:
        return False
    if host.startswith("192.168.") or host.startswith("10.") or host.startswith("169.254."):
        return False
    if host.startswith("172."):
        try:
            second_octet = int(host.split(".")[1])
            if 16 <= second_octet <= 31:
                return False
        except (IndexError, ValueError):
            pass
    return True


async def fetch_article_text(url: str, max_chars: int = 1500) -> str | None:
    if not _is_safe_public_url(url):
        return None
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=15, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    body = "\n".join(lines)
    return body[:max_chars]
