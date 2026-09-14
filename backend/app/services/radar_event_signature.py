"""Cutoff-independent event-signature extraction from a live radar seed
post's SOURCE TEXT ONLY (Live Radar Event Discovery V2, phase 3).

Every signal here is visible in the seed post's own text at the moment it
was ingested -- no reply content, no post-hoc fetched data, no future
information. This is deliberately NOT a NER model (no new dependency was
approved -- see AGENTS.md Section 14 -- and this is explicitly not a model
training task): incident type, locations, and named entities are extracted
with small, transparent, regex/heuristic rules, not a trained recognizer.
That is a real precision/recall tradeoff, not a hidden one -- every
extracted token is directly traceable to the substring that matched, and
every function below documents exactly what it will and will not catch.

Two things this module produces, both pure and offline:
  1. `extract_event_signature(text)` -- an `EventSignature` capturing
     incident type, locations, named entities (persons/orgs, NOT
     separately distinguished -- see EventSignature's own docstring),
     casualty-count expressions, hashtags, external domains, and
     distinctive multi-word phrases.
  2. `build_expansion_queries(signature, ...)` -- turns that signature into
     a handful of specific Bluesky search queries, meant to re-find OTHER
     source posts about the SAME real-world event that the original
     keyword search missed (see radar_query_expansion.py for the part that
     actually issues these queries and enforces per-seed budgets/cooldown).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

# --- incident type -----------------------------------------------------------

# Matches DEFAULT_KEYWORDS' own event domain (bluesky_source.py) -- deliberately
# a closed, small set rather than an open-vocabulary classifier.
INCIDENT_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "shooting": ("shooting", "shooter", "gunman", "gunfire", "shots fired", "gunmen"),
    "explosion": ("explosion", "blast", "bomb", "bombing"),
    "hostage": ("hostage", "siege", "barricaded"),
    "plane_crash": ("plane crash", "aircraft crash", "airliner crash", "crash landing", "cargo plane crash"),
    "earthquake": ("earthquake", "quake", "tremor", "aftershock"),
    "wildfire": ("wildfire", "brush fire", "forest fire", "bushfire", "fire evacuation"),
    "flood": ("flood", "flash flood", "flooding", "flood advisory", "flood warning"),
    "protest": ("protest", "riot", "unrest", "demonstration", "rioters"),
}


def _detect_incident_types(lower_text: str) -> set[str]:
    return {
        incident_type
        for incident_type, phrases in INCIDENT_TYPE_KEYWORDS.items()
        if any(phrase in lower_text for phrase in phrases)
    }


# --- casualty / count expressions --------------------------------------------

_CASUALTY_RE = re.compile(
    r"\b(\d+)\s+(dead|killed|injured|wounded|hurt|hospitalized|missing|evacuated)\b",
    re.IGNORECASE,
)


def _extract_casualty_expressions(text: str) -> set[str]:
    return {f"{count} {word.lower()}" for count, word in _CASUALTY_RE.findall(text)}


# --- hashtags / URLs ----------------------------------------------------------

_HASHTAG_RE = re.compile(r"#(\w+)")
_URL_RE = re.compile(r"https?://([^\s/]+)")


def _extract_hashtags(text: str) -> set[str]:
    return {f"#{tag.lower()}" for tag in _HASHTAG_RE.findall(text)}


def _extract_domains(text: str) -> set[str]:
    domains = set()
    for host in _URL_RE.findall(text):
        netloc = urlparse(f"http://{host}").netloc or host
        domains.add(netloc.lower().removeprefix("www."))
    return domains


# --- locations -----------------------------------------------------------------
# Heuristic, not gazetteer-complete. Three independent signals, any one of which
# is enough to record a location candidate:
#   (a) an explicit US state name/abbreviation, anywhere in the text;
#   (b) a "City, ST"/"City, State" pattern;
#   (c) a bracketed NWS-style code, e.g. "[Larimer Co, CO]" or "[CO]" -- this is
#       the ACTUAL format the live radar's own weather-alert bot posts use (see
#       diagnostic sample data), so it is worth matching directly rather than
#       only via the generic capitalized-phrase heuristic below.
# This intentionally only needs to be good enough to catch two DIFFERENT,
# EXPLICIT locations in two posts -- not to enumerate every location precisely.

US_STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}
US_STATE_ABBREVIATIONS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}
# Bug caught by diagnose_radar_cluster_quality.py's real-data run (2026-09-07): without
# this normalization, one real cluster's own representative post ("...by NWS Denver CO...
# Northeastern Larimer County in north central Colorado...") extracted the state as the
# lowercase full name "colorado", while a different member of the SAME cluster ("BOU
# issues Flood Advisory for Larimer [CO]...") extracted the bracketed abbreviation "CO" --
# two different-looking strings for the identical state, so the disjointness check in
# event_clustering_v2.classify_cluster_merge/is_cluster_claim_ambiguous saw them as a
# conflict where there was none. Every state mention is now normalized to its 2-letter
# UPPERCASE abbreviation before being added to `locations`, whichever form matched.
_US_STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY",
}
assert set(_US_STATE_NAME_TO_ABBR) == US_STATE_NAMES
assert set(_US_STATE_NAME_TO_ABBR.values()) == US_STATE_ABBREVIATIONS

_BRACKETED_LOCATION_RE = re.compile(r"\[([A-Za-z .]+?,\s*)?([A-Z]{2})\]")
_CITY_COMMA_STATE_RE = re.compile(
    r"\b([A-Z][a-zA-Z.]+(?:\s[A-Z][a-zA-Z.]+){0,2}),\s*([A-Z]{2}|" +
    "|".join(sorted((s.title() for s in US_STATE_NAMES), key=len, reverse=True)) + r")\b"
)


def _extract_locations(text: str) -> set[str]:
    locations: set[str] = set()
    lower_text = text.lower()
    for state in US_STATE_NAMES:
        if re.search(rf"\b{re.escape(state)}\b", lower_text):
            locations.add(_US_STATE_NAME_TO_ABBR[state])
    for match in _BRACKETED_LOCATION_RE.finditer(text):
        county, abbrev = match.groups()
        if abbrev in US_STATE_ABBREVIATIONS:
            locations.add(abbrev)
    for match in _CITY_COMMA_STATE_RE.finditer(text):
        city, state = match.groups()
        state_abbr = state if len(state) == 2 else _US_STATE_NAME_TO_ABBR[state.lower()]
        if state_abbr in US_STATE_ABBREVIATIONS:
            locations.add(city.strip())
            locations.add(state_abbr)
    return locations


# --- named entities (persons/organizations, combined -- see EventSignature) --
# Two or more consecutive Capitalized words, NOT already claimed as a location
# above and not the very first word of the text (reduces sentence-initial-
# capitalization false positives). No attempt to separately tag "person" vs
# "organization" -- that split needs real NER, out of scope here (see module
# docstring). Distinctive multi-word phrases (locations landmarks, incident
# names like "Payson Fire") also come from this same extraction, so callers
# needing "a shared distinctive phrase" can reuse `distinctive_phrases`.

_CAPITALIZED_RUN_RE = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3})\b")
_LEADING_STOPWORDS = {"The", "A", "An", "This", "That", "It", "Breaking", "Mass"}


def _extract_named_entities_and_phrases(text: str, locations: set[str]) -> tuple[set[str], set[str]]:
    location_lower = {loc.lower() for loc in locations}
    entities: set[str] = set()
    for match in _CAPITALIZED_RUN_RE.finditer(text):
        phrase = match.group(1).strip()
        first_word = phrase.split()[0]
        if match.start() == 0 and first_word in _LEADING_STOPWORDS:
            continue
        if phrase.lower() in location_lower:
            continue
        entities.add(phrase)
    return entities, set(entities)  # (named_entities, distinctive_phrases) -- same set today, kept as two return values so callers don't have to assume they'll always be identical


@dataclass
class EventSignature:
    """See module docstring for extraction-method caveats (heuristic, not NER).

    `named_entities`: capitalized multi-word phrases not already claimed as a
    location -- covers BOTH persons and organizations without distinguishing
    them (no NER available). Used for both "shared distinctive entity" (a
    reason to allow a merge) and "conflicting persons/organizations" (a
    reason to refuse one) -- see event_clustering_v2.py.
    """
    incident_types: set[str] = field(default_factory=set)
    locations: set[str] = field(default_factory=set)
    named_entities: set[str] = field(default_factory=set)
    casualty_expressions: set[str] = field(default_factory=set)
    hashtags: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)
    distinctive_phrases: set[str] = field(default_factory=set)

    def is_empty(self) -> bool:
        return not (
            self.incident_types or self.locations or self.named_entities
            or self.casualty_expressions or self.hashtags or self.domains
        )


def extract_event_signature(text: str) -> EventSignature:
    text = text or ""
    locations = _extract_locations(text)
    named_entities, distinctive_phrases = _extract_named_entities_and_phrases(text, locations)
    return EventSignature(
        incident_types=_detect_incident_types(text.lower()),
        locations=locations,
        named_entities=named_entities,
        casualty_expressions=_extract_casualty_expressions(text),
        hashtags=_extract_hashtags(text),
        domains=_extract_domains(text),
        distinctive_phrases=distinctive_phrases,
    )


# --- query expansion -----------------------------------------------------------

MAX_EXPANSION_QUERIES = 5


def build_expansion_queries(signature: EventSignature, max_queries: int = MAX_EXPANSION_QUERIES) -> list[str]:
    """2-5 specific Bluesky search queries meant to re-find OTHER source
    posts about the same real-world event this signature was extracted
    from. Pure and deterministic -- issuing the queries and enforcing
    per-seed limits/cooldown is radar_query_expansion.py's job, not this
    function's.

    Ordering is most-to-least specific: an incident type + a named
    location/entity is the most discriminating query this module can build
    without NER; a single distinctive phrase or hashtag alone is broader
    but still meaningfully narrower than the original single-word keyword
    search that found the seed post in the first place.
    """
    queries: list[str] = []

    def _add(query: str) -> None:
        query = query.strip()
        if query and query not in queries:
            queries.append(query)

    incident = next(iter(sorted(signature.incident_types)), None)
    locations = sorted(signature.locations, key=len, reverse=True)
    entities = sorted(signature.named_entities, key=len, reverse=True)

    if incident and locations:
        _add(f"{incident} {locations[0]}")
    if incident and entities:
        _add(f"{incident} {entities[0]}")
    # Fixed per review: Python set iteration order is not guaranteed stable across
    # processes/runs (hash randomization) -- iterating these sets directly made
    # build_expansion_queries's own "pure and deterministic" claim false for any signature
    # with 2+ distinctive_phrases/casualty_expressions/hashtags. Sorted explicitly (by
    # length descending, tie-broken alphabetically, matching the locations/entities
    # ordering above) so the SAME signature always produces the SAME query list.
    for phrase in sorted(signature.distinctive_phrases, key=lambda p: (-len(p), p)):
        if len(queries) >= max_queries:
            break
        _add(f'"{phrase}"')
    for casualty in sorted(signature.casualty_expressions):
        if len(queries) >= max_queries:
            break
        if locations:
            _add(f'"{casualty}" {locations[0]}')
        elif incident:
            _add(f'"{casualty}" {incident}')
    for tag in sorted(signature.hashtags):
        if len(queries) >= max_queries:
            break
        _add(tag)
    if not queries and incident:
        _add(incident)

    return queries[:max_queries]
