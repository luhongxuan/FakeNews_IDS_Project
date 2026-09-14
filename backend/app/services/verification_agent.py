"""Post-hoc verification agent.

Given the text of a thread the early-detection model has already flagged as
high priority, this agent decides what to search for, calls GDELT news,
Google Fact Check, and general web search (DuckDuckGo, keyless) tools via
app/services/search_tools.py, can read a full article's text via
app/services/article_fetch.py when a snippet isn't enough, and produces a
structured credibility report with server-computed source-credibility tags.

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

import asyncio
import json
import os
import re
from urllib.parse import urlparse

import httpx
from sqlalchemy.orm import Session

from app.services.article_fetch import fetch_article_text
from app.services.search_tools import search_fact_checks, search_news, search_web
from app.services import verification_progress

# Deliberately not named OLLAMA_HOST: Ollama itself reserves that variable
# for its own bind address (e.g. "0.0.0.0:11434", no scheme), and this
# machine already has it set system-wide -- reusing the name here silently
# picked up that value and broke httpx (missing http:// scheme).
# .rstrip("/"): a trailing slash in the configured base (very easy to paste
# in by accident, e.g. copying a RunPod proxy URL straight from its console)
# turned every request below into ".../api/chat" with a doubled slash, which
# some reverse proxies (RunPod's included) 307-redirect to the normalized
# single-slash path -- and since httpx.AsyncClient() below defaults to
# follow_redirects=False, that redirect surfaced as a raise_for_status()
# error ("Redirect response '307 Temporary Redirect'...") instead of ever
# reaching the model, i.e. a config-formatting typo that looked exactly like
# the remote server being broken. Normalizing here makes the join always
# correct regardless of how the operator formatted the env var.
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-vl:8b-instruct")

# Bump whenever SYSTEM_PROMPT's grounding/reasoning rules change in a way
# that could flip a past verdict (e.g. the "different event != refutation"
# rule added for the Isabela, Puerto Rico earthquake false-refutation case:
# a real, accurate USGS-confirmed post was wrongly marked "likely_false"
# because the model treated a similar-but-different nearby quake record as
# a contradiction). Callers should fold this into their cache key so an
# old VerificationReport produced under a since-fixed prompt is never
# served as if it still reflects current reasoning -- see factcheck.py's
# and radar.py's _cache_key/_verify_cache_key.
# v3: minimum-search-rounds-before-abstaining rule (step 2) -- observed a
# claim that mixed an unusual name/number combination (a Trump/JFK-era-figure
# mashup) get "unverified" on some runs and "likely_false" on others purely
# because the FIRST search query's exact phrasing sometimes missed the real
# coverage; the model was allowed to give up after one try instead of
# rephrasing. Also see OLLAMA_TEMPERATURE just below -- same underlying
# symptom (run-to-run inconsistency on an unchanged claim), two different
# causes (giving up too early vs. high sampling variance), both worth fixing.
# v4: rule 7, the mirror image of rule 6's "different event != refutation".
# Same Trump/"wrestled a second gunman"/Jacqueline Kennedy claim, AFTER v3:
# the agent now reliably found and fetch_article'd several detailed reports
# on the real 2026 White House Correspondents' Dinner shooting, correctly
# recognized none of them mention the claimed detail -- and then still
# landed on "unverified" instead of "likely_false", because nothing told it
# that silence in thorough coverage of a dramatic claim's own event IS
# evidence, not a null result. Rule 6 stops the model from over-reading a
# near-miss as a contradiction; rule 7 stops it from under-reading a real,
# well-covered non-mention as "no evidence either way" -- two failure modes
# in opposite directions, not one fix generalizing to the other.
VERIFICATION_PROMPT_VERSION = "v4"
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
    {
        "type": "function",
        "function": {
            "name": "fetch_article",
            "description": "Fetch and read the full text of one article, when its title and one-line snippet from a previous search result are not enough to tell whether it actually supports or refutes the claim. The url MUST be one that already appeared in a previous search_web/search_news/search_fact_checks result -- you cannot fetch an arbitrary URL you did not get from a search.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "A URL that appeared in an earlier tool result."}},
                "required": ["url"],
            },
        },
    },
]

# Domains judged reputable enough to weight evidence toward, computed
# server-side (not left to the model to eyeball a URL) so it's consistent
# and auditable. Deliberately conservative and small -- absence from this
# list is not itself evidence of unreliability, just "no signal either way".
_REPUTABLE_DOMAINS = {
    "reuters.com": "wire_service", "apnews.com": "wire_service", "afp.com": "wire_service",
    "bbc.com": "major_outlet", "bbc.co.uk": "major_outlet", "npr.org": "major_outlet",
    "nytimes.com": "major_outlet", "theguardian.com": "major_outlet", "washingtonpost.com": "major_outlet",
    "wsj.com": "major_outlet", "cnn.com": "major_outlet", "usatoday.com": "major_outlet",
    "snopes.com": "fact_checker", "factcheck.org": "fact_checker", "politifact.com": "fact_checker",
    "fullfact.org": "fact_checker", "afpfactcheck.com": "fact_checker",
    "usgs.gov": "official", "cdc.gov": "official", "who.int": "official", "noaa.gov": "official",
    "wikipedia.org": "reference",
}


def _source_tier(url: str | None) -> str:
    if not url:
        return "unknown"
    host = (urlparse(url).hostname or "").lower()
    for domain, tier in _REPUTABLE_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return tier
    return "unknown"

SYSTEM_PROMPT = """You are a claim-verification research assistant. You are given the text of one social media post that an upstream model has flagged as high-priority to check. Your job is NOT to decide whether to intervene -- that decision has already been made. Your job is to gather independent evidence about whether the claim in the post is true, false, or unverified.

You must not answer from your own background knowledge or training data. You have almost certainly not seen live search results for this specific claim, and even for events you recognize, your job here is to demonstrate what the SEARCH TOOLS actually return, not what you already believe.

Process:
1. Identify the concrete, checkable claim(s) in the post text.
2. You MUST call at least one of search_news, search_fact_checks, or search_web, with short keyword queries (not the raw post text), before giving any final verdict. search_news and search_fact_checks are narrower and more authoritative when they return something; search_web (general DuckDuckGo search) is broader and a good fallback when the other two come back empty. If your FIRST round of searches does not turn up a result that clearly discusses the same specific event as the claim, you MUST try at least one more search with a differently-worded query (synonyms, fewer/different keywords, drop or add the year) before concluding "unverified" -- a claim that mixes an unusual combination of names/numbers/places is exactly the case where the first query's exact phrasing is most likely to miss the real coverage, and giving up after one try is indistinguishable from never having checked at all.
3. Each search result includes a "source_tier" field (e.g. "wire_service", "major_outlet", "fact_checker", "official", "reference", or "unknown"), computed independently of you -- not something you assign. Weight tiered sources more heavily than "unknown" ones when they disagree, and say so in your summary when it matters.
4. If a result's title/snippet alone isn't enough to tell whether it supports or refutes the claim, call fetch_article with that result's exact url to read the full text before deciding -- don't guess from the snippet alone when the full article is one call away.
5. Once you have enough evidence (or tool results are consistently empty/irrelevant after a couple of tries), stop calling tools and output your final verdict, based only on what the tools actually returned. If every tool call came back empty, your credibility MUST be "unverified" and your summary must say plainly that no independent evidence was found -- do not fill the gap with what you happen to already know about the event.
6. For a PRECISE, checkable claim (an exact number, date, magnitude, distance, or other specific measurement -- e.g. "M3.2 earthquake 27 km N of X on Sept 7"), a source only counts as "refutes" if it is clearly reporting on THAT SAME specific event and states a conflicting fact about it (e.g. multiple outlets giving a different death toll for the same confirmed incident). Finding a DIFFERENT event of the same general kind -- a similar magnitude quake near the same place but a different date, distance, or depth; a different fire; a different flood -- is NOT evidence the claimed event is false. Your search tools have no access to authoritative structured databases (seismic networks, official statistics registries, etc.), so failing to find an exact match is usually a coverage gap, not a contradiction. If the closest thing you found is a related-but-distinct record, say so plainly ("I found a similar but distinct event; I could not confirm or refute this specific one") and use "unverified" -- never promote "the nearest match I found doesn't line up" into "likely_false" or a "refutes" stance. This applies symmetrically: don't let it become "likely_true" either just because something superficially similar exists.
7. The opposite gap: if the claim attaches a specific, notable, dramatic action or detail (something that, if it really happened, would obviously be reported -- e.g. "the President personally wrestled a gunman to the ground", "a celebrity was present and injured") to a real event you HAVE found, and you have fetch_article'd multiple detailed, full-text reports specifically about that same event (same date/location/incident), and NONE of that detailed coverage mentions the claimed action or person at all -- that silence IS evidence against the claim, not merely "insufficient evidence". Thorough contemporaneous reporting on a real, well-covered incident does not casually omit a detail that dramatic if it happened; the most likely explanation is that the specific detail was invented and grafted onto a real event. Call this "likely_false" (not "unverified"), cite the detailed real-event articles you fetched with stance "refutes", and say plainly in the summary that you checked multiple detailed accounts of the actual event and none mention the specific claimed detail. Only apply this when you actually fetched full article text (not just snippets) from more than one source about the real event -- a single snippet or a single article is not "thorough coverage", and minor/background details can legitimately go unreported even when true, so don't apply this to claims about small, easily-overlooked specifics.

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
Only cite title/url pairs that actually appeared in a tool result (including from fetch_article, which echoes back the url you gave it). If no useful evidence was found at all, use "unverified", low confidence, an empty evidence list, and say so in the summary. Do not invent URLs."""


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


OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
# Ollama/qwen2.5 defaults to a fairly high sampling temperature (~0.7-0.8),
# tuned for open-ended chat, not for a task that should give the SAME verdict
# on the SAME evidence every time. This is a real, observed source of the
# agent flip-flopping between runs on an identical claim+evidence (e.g. the
# "3 gunmen"/Trump-JFK cases this session): not a bug in any one run, just
# high-variance sampling on a task that wants low variance. Low but nonzero
# (not 0) so the model can still break ties/vary phrasing in its summary
# without changing its actual verdict run-to-run.
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.15"))


async def _call_ollama(client: httpx.AsyncClient, messages: list[dict], use_tools: bool) -> dict:
    payload = {
        "model": OLLAMA_MODEL, "stream": False, "messages": messages,
        # Ollama defaults num_ctx to 4096 regardless of the model's real max
        # context, to save memory. This agent's conversation grows fast
        # (search results, then full article text from fetch_article), and
        # once it silently overflows 4096 tokens the model loses track of
        # what it already searched for -- observed directly as the model
        # re-issuing near-identical queries a few tool calls in. Explicit
        # num_ctx trades a bit more KV-cache memory for the model actually
        # remembering its own tool history.
        "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": OLLAMA_TEMPERATURE},
    }
    if use_tools:
        payload["tools"] = TOOLS
    response = await client.post(f"{OLLAMA_API_BASE}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


async def _run_tool(db: Session, name: str, arguments: dict, seen_urls: set[str]) -> list[dict]:
    if name == "fetch_article":
        url = str(arguments.get("url", "")).strip()
        if not url or url not in seen_urls:
            return [{"error": "That url did not appear in a previous search result. Only fetch urls returned by search_web, search_news, or search_fact_checks."}]
        try:
            text = await fetch_article_text(url)
        except httpx.HTTPError:
            return [{"error": "Could not fetch that article (unreachable or the site refused the request)."}]
        if text is None:
            return [{"error": "That url could not be fetched (not a public http/https address)."}]
        return [{"url": url, "text": text, "source_tier": _source_tier(url)}]

    query = str(arguments.get("query", "")).strip()
    if not query:
        return []
    try:
        if name == "search_news":
            results = await search_news(db, query)
        elif name == "search_web":
            results = await search_web(db, query)
        elif name == "search_fact_checks":
            raw = await search_fact_checks(db, query)
            # Normalize to the same {title, url, ...} shape the other two
            # tools use -- ClaimCache rows use "review_url"/"claim_text"
            # instead. Without this, a citation from a fact-check result
            # never lands in `seen_urls` (which only ever looks at "url"),
            # so the anti-hallucination filter below silently strips out
            # every fact-check citation even when it was real.
            results = [{**item, "url": item.get("review_url"), "title": item.get("claim_text")} for item in raw]
        else:
            return []
    except httpx.HTTPError:
        # A single unreachable/erroring external API should not abort the
        # whole verification -- the agent still has whatever the other tool
        # returned, and can say so in its summary instead of failing outright.
        return []
    for item in results:
        item["source_tier"] = _source_tier(item.get("url"))
    return results


async def run_verification(
    db: Session, thread_id: str, event_id: str, claim_text: str, post_date: str | None = None,
    progress_id: str | None = None,
) -> dict:
    """Run the tool-calling agent loop; returns the structured report dict.

    ``post_date`` (YYYY-MM-DD, from the source tweet's own timestamp) anchors
    searches to when the post was actually made -- see _heuristic_query's
    docstring for why this matters for old/minor claims.

    ``progress_id``: when given, every tool call this run makes is also
    pushed to verification_progress AS IT HAPPENS (not just in the final
    returned report), so a caller polling verification_progress.get(...)
    with the same id can show live progress during the ~10-90s this
    usually takes. Purely additive -- None (the default) skips this
    entirely, same behavior as before this existed.

    Never raises for a well-formed but unreachable Ollama host / model --
    callers should still handle httpx errors (e.g. connection refused if
    Ollama is not running) and surface them as a clear failure, not a fake
    verdict. Still marks the progress feed done (with the error message) in
    that case via the finally block below, so a polling frontend does not
    spin forever.
    """
    verification_progress.start(progress_id)
    try:
        result = await _run_verification_inner(db, thread_id, event_id, claim_text, post_date, progress_id)
    except Exception as error:
        verification_progress.finish(progress_id, error=str(error))
        raise
    verification_progress.finish(progress_id)
    return result


async def _run_verification_inner(
    db: Session, thread_id: str, event_id: str, claim_text: str, post_date: str | None, progress_id: str | None,
) -> dict:
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
    # always run real searches deterministically before the model reasons at
    # all. search_web and search_fact_checks both run in parallel here
    # (confirmed reachable); search_news (GDELT) is deliberately left out of
    # this automatic pass -- unreachable from this network, so calling it
    # automatically would just burn a round trip for nothing every time. It
    # stays available for the model to call itself, in case a future
    # deployment's network can actually reach it. The model can still call
    # more tools afterward if it wants different evidence; this just
    # guarantees the loop never starts empty.
    initial_query = _heuristic_query(claim_text, year=post_date[:4] if post_date else None)
    if initial_query:
        initial_web, initial_facts = await asyncio.gather(
            _run_tool(db, "search_web", {"query": initial_query}, seen_urls),
            _run_tool(db, "search_fact_checks", {"query": initial_query}, seen_urls),
        )
        queries_used.append(initial_query)
        for item in initial_web + initial_facts:
            if item.get("url"):
                seen_urls.add(item["url"])
        _log_call = {"name": "search_web", "arguments": {"query": initial_query}, "result_count": len(initial_web), "automatic": True}
        tool_call_log.append(_log_call)
        verification_progress.append_tool_call(progress_id, _log_call)
        _log_call = {"name": "search_fact_checks", "arguments": {"query": initial_query}, "result_count": len(initial_facts), "automatic": True}
        tool_call_log.append(_log_call)
        verification_progress.append_tool_call(progress_id, _log_call)
        messages.append({
            "role": "user",
            "content": (
                f"Automatic initial search for \"{initial_query}\":\n"
                f"search_web results:\n{json.dumps(initial_web, ensure_ascii=False)}\n\n"
                f"search_fact_checks results:\n{json.dumps(initial_facts, ensure_ascii=False)}\n\n"
                "Review these. Call fetch_article on a specific url if the snippet isn't enough, or "
                "call search_news / search_fact_checks / search_web again with a different query if "
                "you need better evidence. Otherwise give your final verdict based on what is "
                "actually here."
            ),
        })

    # Small local models frequently re-issue a near-identical query (or the
    # same fetch_article url) several times in a row instead of recognizing
    # they already have that result -- observed in testing to blow up a
    # single verification from ~60s to 4+ minutes with no new evidence
    # gained. Cache by (tool name, normalized arguments) so a repeat costs
    # no network round trip, and tell the model plainly that it repeated
    # itself so it has a reason to actually change approach or stop.
    call_cache: dict[str, list[dict]] = {}

    def _cache_key(name: str, arguments: dict) -> str:
        normalized = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in arguments.items()}
        return name + "|" + json.dumps(normalized, sort_keys=True)

    # follow_redirects=True: defense-in-depth alongside the OLLAMA_API_BASE
    # normalization above -- a remote proxy (RunPod's included) issuing a
    # benign redirect (protocol upgrade, trailing-slash normalization, etc.)
    # should transparently succeed, not surface as a fake "server error".
    async with httpx.AsyncClient(follow_redirects=True) as client:
        final_content = None
        for _ in range(MAX_TOOL_ITERATIONS):
            result = await _call_ollama(client, messages, use_tools=True)
            message = result.get("message", {})
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final_content = message.get("content", "")
                break
            messages.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls})

            # Parse every call in this batch first, then run whichever ones
            # aren't already cached concurrently -- a model turn often asks
            # for 2-3 tools (e.g. fetch_article on several search hits) that
            # have no dependency on each other, and running them one at a
            # time was pure wasted wall-clock time.
            parsed = []
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                parsed.append((name, arguments, _cache_key(name, arguments)))

            keys_before_batch = set(call_cache)
            to_run: dict[str, tuple] = {}
            for name, arguments, key in parsed:
                if key not in call_cache and key not in to_run:
                    to_run[key] = (name, arguments)
            if to_run:
                fresh_results = await asyncio.gather(*(_run_tool(db, n, a, seen_urls) for n, a in to_run.values()))
                for key, tool_result in zip(to_run.keys(), fresh_results):
                    call_cache[key] = tool_result

            emitted_this_batch: set[str] = set()
            for name, arguments, key in parsed:
                query = str(arguments.get("query", "")).strip()
                if query:
                    queries_used.append(query)
                repeated = key in keys_before_batch or key in emitted_this_batch
                emitted_this_batch.add(key)
                tool_result = call_cache[key]
                for item in tool_result:
                    if item.get("url"):
                        seen_urls.add(item["url"])
                _log_call = {"name": name, "arguments": arguments, "result_count": len(tool_result), "repeated": repeated}
                tool_call_log.append(_log_call)
                verification_progress.append_tool_call(progress_id, _log_call)
                content = json.dumps(tool_result, ensure_ascii=False)
                if repeated:
                    content = (
                        "(You already made this exact call before -- this is the same cached result, "
                        "not a fresh search. Repeating it again will not produce new evidence. Try a "
                        "meaningfully different query, or stop calling tools and give your final "
                        "verdict now.) " + content
                    )
                messages.append({"role": "tool", "content": content})
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
    # Server-computed, not left to the model: a small, auditable domain
    # allowlist (see _source_tier) so the frontend can show readers which
    # citations are wire services/fact-checkers vs. unknown sources.
    for item in parsed["evidence"]:
        item["source_tier"] = _source_tier(item.get("url"))
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
