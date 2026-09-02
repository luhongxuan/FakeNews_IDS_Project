"""Post-hoc verification agent.

Given the text of a thread the early-detection model has already flagged as
high priority, this agent decides what to search for, calls GDELT news,
Google Fact Check, and general web search (DuckDuckGo, keyless) tools via
app/services/search_tools.py, and produces a structured credibility report.

Division of responsibility, kept strict on purpose:
- The early-detection RF models (effective_models/) decide, from only the
  first 30 minutes of a thread, whether it is worth prioritizing. That
  decision and its evaluation must never depend on this agent.
- This agent runs strictly AFTER that ranking, as decision support for a
  human operator. Its output (credibility, evidence) is never written back
  into any training/evaluation artifact under effective_models/, and never
  becomes a feature for the cutoff-based models. If that boundary is ever
  crossed, treat it as a leakage bug per AGENTS.md.

The agent talks to a local Ollama server (no cloud API key required). Model
and host are configurable via OLLAMA_MODEL / OLLAMA_API_BASE.
"""
from __future__ import annotations

import json
import os
import re

import httpx
from sqlalchemy.orm import Session

from app.services.search_tools import search_fact_checks, search_news, search_web

# Deliberately not named OLLAMA_HOST: Ollama itself reserves that variable
# for its own bind address (e.g. "0.0.0.0:11434", no scheme), and this
# machine already has it set system-wide -- reusing the name here silently
# picked up that value and broke httpx (missing http:// scheme).
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
MAX_TOOL_ITERATIONS = 5
REQUEST_TIMEOUT = 120.0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": "Search recent news articles (via GDELT) for a keyword query. Use this to find independent news coverage that would support or contradict the claim.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Short keyword search query, not the full claim text."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "General web search (via DuckDuckGo) for a keyword query. Use this for broader context, primary sources, or when search_news and search_fact_checks found nothing -- e.g. to check what reputable outlets or official sources say about the claim.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Short keyword search query, not the full claim text."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_fact_checks",
            "description": "Search published fact-checks (via Google Fact Check Tools) for a keyword query. Use this to find whether professional fact-checkers have already rated this or a similar claim.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Short keyword search query, not the full claim text."}},
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a claim-verification research assistant. You are given the text of one social media post that an upstream model has flagged as high-priority to check. Your job is NOT to decide whether to intervene -- that decision has already been made. Your job is to gather independent evidence about whether the claim in the post is true, false, or unverified.

You must not answer from your own background knowledge or training data. You have almost certainly not seen live search results for this specific claim, and even for events you recognize, your job here is to demonstrate what the SEARCH TOOLS actually return, not what you already believe.

Process:
1. Identify the concrete, checkable claim(s) in the post text.
2. You MUST call at least one of search_news, search_fact_checks, or search_web, with short keyword queries (not the raw post text), before giving any final verdict. search_news and search_fact_checks are narrower and more authoritative when they return something; search_web (general DuckDuckGo search) is broader and a good fallback when the other two come back empty. You may call tools multiple times with different queries if the first results are not useful.
3. Once you have enough evidence (or tool results are consistently empty/irrelevant after a couple of tries), stop calling tools and output your final verdict, based only on what the tools actually returned. If every tool call came back empty, your credibility MUST be "unverified" and your summary must say plainly that no independent evidence was found -- do not fill the gap with what you happen to already know about the event.

Your final answer MUST be ONLY a single JSON object (no prose before or after, no markdown fences) with exactly this shape:
{
  "credibility": "likely_true" | "disputed" | "likely_false" | "unverified",
  "confidence": <number between 0 and 1>,
  "summary": "<2-4 sentence plain-language summary of what the evidence shows>",
  "evidence": [
    {"title": "<title>", "url": "<url>", "source": "<source/reviewer>", "date": "<date or null>", "stance": "supports" | "refutes" | "context" | "unrelated"}
  ],
  "queries_used": ["<query1>", "<query2>"]
}
Only cite title/url pairs that actually appeared in a tool result. If no useful evidence was found at all, use "unverified", low confidence, an empty evidence list, and say so in the summary. Do not invent URLs."""


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "in", "on", "at", "of",
    "for", "with", "that", "which", "who", "whose", "this", "these", "those", "and", "or", "but",
    "to", "from", "by", "as", "it", "its", "their", "his", "her", "he", "she", "they", "them",
    "we", "you", "i", "not", "no", "do", "does", "did", "has", "have", "had", "will", "would",
    "can", "could", "should", "may", "might", "says", "said", "one", "into", "over", "after",
    "carrying", "there", "here", "just", "also", "than", "then", "so", "if", "up", "out",
}


def _heuristic_query(claim_text: str, year: str | None = None, max_words: int = 8) -> str:
    """First-pass search query with no LLM involved.

    A search engine matches keywords, not sentences: dropping URLs, hashtag
    markers, and function words gets much closer to how a person would
    actually search than truncating the raw sentence does (e.g. "Israeli
    news is carrying a live interview with a woman whose niece is one of
    the hostages in #Paris" -> "Israeli news live interview woman niece
    hostages Paris", not "Israeli news is carrying a live interview with").
    Capitalized (likely proper-noun) words are kept first since they anchor
    the query to the specific event/people involved.

    ``year`` (the post's own posting year, e.g. "2015") is appended when
    known. Generic keyword combinations like "Israeli hostage niece" are
    dominated by whatever is most prominent on the web right now; without a
    year anchor, search results for an old, minor claim get drowned out by
    unrelated recent events that happen to share the same keywords.
    """
    text = re.sub(r"https?://\S+", "", claim_text)
    text = re.sub(r"[#@]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    words = [word for word in text.split() if len(word) > 1]
    content_words = [word for word in words if word.lower() not in _STOPWORDS]
    if not content_words:
        content_words = words
    proper_nouns = [word for word in content_words if word[0].isupper()]
    rest = [word for word in content_words if not word[0].isupper()]
    ordered, seen = [], set()
    for word in proper_nouns + rest:
        key = word.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(word)
    query = " ".join(ordered[:max_words]).strip()
    if year:
        query = f"{query} {year}".strip()
    return query


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


async def _call_ollama(client: httpx.AsyncClient, messages: list[dict], use_tools: bool) -> dict:
    payload = {"model": OLLAMA_MODEL, "stream": False, "messages": messages}
    if use_tools:
        payload["tools"] = TOOLS
    response = await client.post(f"{OLLAMA_API_BASE}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


async def _run_tool(db: Session, name: str, arguments: dict) -> list[dict]:
    query = str(arguments.get("query", "")).strip()
    if not query:
        return []
    try:
        if name == "search_news":
            return await search_news(db, query)
        if name == "search_web":
            return await search_web(db, query)
        if name == "search_fact_checks":
            return await search_fact_checks(db, query)
    except httpx.HTTPError:
        # A single unreachable/erroring external API should not abort the
        # whole verification -- the agent still has whatever the other tool
        # returned, and can say so in its summary instead of failing outright.
        return []
    return []


async def run_verification(db: Session, thread_id: str, event_id: str, claim_text: str, post_date: str | None = None) -> dict:
    """Run the tool-calling agent loop; returns the structured report dict.

    ``post_date`` (YYYY-MM-DD, from the source tweet's own timestamp) anchors
    searches to when the post was actually made -- see _heuristic_query's
    docstring for why this matters for old/minor claims.

    Never raises for a well-formed but unreachable Ollama host / model --
    callers should still handle httpx errors (e.g. connection refused if
    Ollama is not running) and surface them as a clear failure, not a fake
    verdict.
    """
    date_note = f" (posted on {post_date})" if post_date else ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Post text to verify{date_note}:\n\n{claim_text}\n\n"
                + (
                    f"When you write your own search queries, include the year {post_date[:4]} "
                    "unless it's clearly unnecessary -- generic keywords alone tend to surface "
                    "unrelated, more recent events that happen to share the same words."
                    if post_date else ""
                )
            ),
        },
    ]
    seen_urls: set[str] = set()
    queries_used: list[str] = []
    tool_call_log: list[dict] = []

    # Small local models don't reliably honor "you must call a tool" -- and
    # Ollama's chat API does not support forcing tool use (tool_choice is
    # accepted but silently ignored by this model). Rather than depend on
    # instruction-following for the one thing that matters most (grounding),
    # always run one real search deterministically before the model reasons
    # at all. The model can still call more tools afterward if it wants
    # different evidence; this just guarantees the loop never starts empty.
    initial_query = _heuristic_query(claim_text, year=post_date[:4] if post_date else None)
    if initial_query:
        initial_results = await _run_tool(db, "search_web", {"query": initial_query})
        queries_used.append(initial_query)
        for item in initial_results:
            if item.get("url"):
                seen_urls.add(item["url"])
        tool_call_log.append({"name": "search_web", "arguments": {"query": initial_query}, "result_count": len(initial_results), "automatic": True})
        messages.append({
            "role": "user",
            "content": (
                f"Automatic initial web search for \"{initial_query}\" returned:\n"
                f"{json.dumps(initial_results, ensure_ascii=False)}\n\n"
                "Review these. Call search_news, search_fact_checks, or search_web again with a "
                "different query if you need better evidence, otherwise give your final verdict "
                "based on what is actually here."
            ),
        })

    async with httpx.AsyncClient() as client:
        final_content = None
        for _ in range(MAX_TOOL_ITERATIONS):
            result = await _call_ollama(client, messages, use_tools=True)
            message = result.get("message", {})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final_content = message.get("content", "")
                break
            messages.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls})
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                query = str(arguments.get("query", "")).strip()
                if query:
                    queries_used.append(query)
                tool_result = await _run_tool(db, name, arguments)
                for item in tool_result:
                    if item.get("url"):
                        seen_urls.add(item["url"])
                tool_call_log.append({"name": name, "arguments": arguments, "result_count": len(tool_result)})
                messages.append({"role": "tool", "content": json.dumps(tool_result, ensure_ascii=False)})
        else:
            # Ran out of iterations still calling tools; force a final answer without tools.
            messages.append({"role": "user", "content": "Stop searching now. Output only the final JSON verdict as specified, using the evidence already gathered."})
            result = await _call_ollama(client, messages, use_tools=False)
            final_content = result.get("message", {}).get("content", "")

    parsed = _extract_json(final_content or "") or {
        "credibility": "unverified", "confidence": 0.0,
        "summary": "Agent did not return a parseable verdict.", "evidence": [], "queries_used": [],
    }
    # Drop any evidence citing a URL that never actually appeared in a tool result (anti-hallucination guard).
    parsed["evidence"] = [item for item in parsed.get("evidence", []) if item.get("url") in seen_urls]
    # Hard backstop, independent of prompt compliance: a model can ignore the
    # "must call a tool" instruction and answer from parametric memory. If no
    # tool ever ran, or every tool call came back empty, the verdict cannot
    # be grounded in anything this agent actually retrieved.
    if not tool_call_log or not seen_urls:
        parsed["credibility"] = "unverified"
        parsed["confidence"] = min(float(parsed.get("confidence", 0.0)) if isinstance(parsed.get("confidence"), (int, float)) else 0.0, 0.2)
        parsed["summary"] = (
            "No independent evidence was retrieved by the search tools (either no tool was called, "
            "or all searches returned nothing usable). " + str(parsed.get("summary", ""))
        ).strip()
    parsed["queries_used"] = queries_used
    parsed["model"] = OLLAMA_MODEL
    parsed["tool_calls"] = tool_call_log
    return parsed
