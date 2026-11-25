"""Facet logic: promotion thresholds, editable breadcrumb chain, query
composition and anti-drift guards for progressive faceted search sessions.
Pure Python.
"""
from __future__ import annotations

from .config import Settings

CONFIDENCE_ORDER = ("object", "style", "material", "era", "color")


def promote_facets(
    state: dict,
    extracted: dict[str, dict],
    threshold: float | None = None,
    settings: Settings | None = None,
) -> dict:
    """Promote facets above the confidence threshold into the session.

    Facets are visible always (returned in state['facets']) but only become
    accepted search constraints above threshold; the user can accept/reject
    any facet explicitly, which always wins.
    """
    threshold = threshold if threshold is not None else (
        settings.facet_confidence_threshold if settings else 0.55
    )
    facets = state.setdefault("facets", {})
    for key in CONFIDENCE_ORDER:
        cand = extracted.get(key)
        if not cand:
            continue
        existing = facets.get(key)
        if existing and existing.get("value") == cand["value"]:
            existing["confidence"] = max(existing["confidence"], cand["confidence"])
            continue
        facets.setdefault(key, cand)
    for key in CONFIDENCE_ORDER:
        f = facets.get(key)
        if f and key not in state["accepted"] and f["confidence"] >= threshold:
            state["accepted"].append(key)
    return state


def accepted_facets(state: dict, settings: Settings | None = None) -> list[dict]:
    """Accepted facets in canonical order, values only — the breadcrumb."""
    out = []
    for key in CONFIDENCE_ORDER:
        if key in state.get("accepted", []):
            f = state.get("facets", {}).get(key)
            if f and f.get("value"):
                out.append({"facet": key, "value": f["value"], "confidence": f["confidence"]})
    return out


def breadcrumb(state: dict, settings: Settings | None = None) -> str:
    return " · ".join(f["value"] for f in accepted_facets(state, settings))


def compose_query(state: dict, settings: Settings | None = None) -> str:
    """Web-search query from the accepted facet chain with anti-drift guards:
    only accepted facets, capped count, capped word length."""
    s = settings
    facets = accepted_facets(state, s)
    if s:
        facets = facets[: s.facet_query_max_facets]
    query = " ".join(f["value"] for f in facets)
    if s:
        words = query.split()
        if len(words) > s.max_query_words:
            query = " ".join(words[: s.max_query_words])
    return query.strip()


def apply_user_edit(state: dict, facet: str, value: str | None, accept: bool | None = None) -> dict:
    """Human edits always win: set/remove a facet value, accept or reject it."""
    if value is None:
        state["facets"].pop(facet, None)
        if facet in state["accepted"]:
            state["accepted"].remove(facet)
    else:
        state["facets"][facet] = {"value": value.strip().lower(), "confidence": 1.0, "user": True}
        if accept is not False and facet not in state["accepted"]:
            state["accepted"].append(facet)
    if accept is False and facet in state["accepted"]:
        state["accepted"].remove(facet)
    return state
