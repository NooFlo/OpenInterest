"""Facet convergence engine: recency-weighted voting over the selection order.

The session accumulates evidence from every selected image (and from the
board the session started from). Each facet's value is re-estimated from all
extractions, with later selections weighted more (like an exponential-decay
bandit), so the query converges on the user's trajectory rather than sticking
to the first image's metadata. Dismissed (skipped) images cast negative votes.

Hard guarantees kept from the original design:
  - user edits always win over model votes (state["overrides"])
  - promotion to accepted constraints is threshold-gated and human-editable
  - the web query is composed only from accepted facets, with word caps
"""
from __future__ import annotations

import time

from .config import Settings

CONFIDENCE_ORDER = ("object", "style", "material", "era", "color")


def record_extraction(
    state: dict,
    extracted: dict[str, dict],
    decay: float = 0.75,
    ts: float | None = None,
) -> dict:
    """Append one facet extraction (from a selected or boarded image).
    Each selection adds evidence; facet values are re-estimated from position on every
    subsequent recompute, so later selections can revise earlier hypotheses."""
    state.setdefault("extractions", []).append({"facets": extracted or {}, "ts": ts or time.time()})
    state.setdefault("_weighted", {})["decay"] = decay
    recompute_facets(state, decay)
    return state


def record_dismissal(state: dict, extracted: dict[str, dict] | None = None) -> dict:
    """A skipped/dismissed image is evidence against its facet values."""
    state.setdefault("dismissals", []).append({"facets": extracted or {}, "ts": time.time()})
    recompute_facets(state)
    return state


def recompute_facets(state: dict, decay: float | None = None) -> dict:
    """Re-estimate facet values from all extractions with positional decay,
    minus dismissals, then apply user overrides on top."""
    decay = decay if decay is not None else state.get("_weighted", {}).get("decay", 0.75)
    extractions = state.get("extractions", [])
    dismissals = state.get("dismissals", [])
    n = max(len(extractions), 1)

    weights = [decay ** (n - 1 - i) for i in range(len(extractions))]
    scores: dict[str, dict[str, float]] = {}
    for w, ext in zip(weights, extractions):
        for key, cand in (ext.get("facets") or {}).items():
            if key not in CONFIDENCE_ORDER:
                continue
            value = (cand.get("value") or "").strip().lower()
            conf = float(cand.get("confidence", 0.5))
            if not value:
                continue
            scores.setdefault(key, {})
            scores[key][value] = scores[key].get(value, 0.0) + w * conf
    for d in dismissals:
        for key, cand in (d.get("facets") or {}).items():
            value = (cand.get("value") or "").strip().lower()
            conf = float(cand.get("confidence", 0.5))
            if key in CONFIDENCE_ORDER and value and value in scores.get(key, {}):
                # one dismissal cancels roughly one vote of evidence for that value
                scores[key][value] -= 1.0 * conf

    facets = state.setdefault("facets", {})
    for key in CONFIDENCE_ORDER:
        if key in state.get("overrides", {}):
            continue  # host override below
        vals = scores.get(key)
        if vals:
            best = max(vals.items(), key=lambda kv: kv[1])
            share = best[1] / (sum(max(0.0, v) for v in vals.values()) or 1.0)
            if best[1] > 0:
                facets[key] = {
                    "value": best[0],
                    "confidence": round(min(1.0, share), 3),
                    "votes": round(best[1], 3),
                }
            elif key in facets:
                del facets[key]
        elif key in facets:
            del facets[key]

    # promotion with hysteresis: initial promotion needs majority share plus
    # a minimum weighted vote (roughly one solid selection); once accepted, a
    # facet stays accepted while its value still has positive votes — decay
    # alone never silently drops it from the query, only contradictory
    # evidence (dismissals, an overtaking value) does
    MIN_VOTES = 0.5
    # color is the noisiest facet for taste queries: require stronger
    # evidence before it enters the web query
    EXTRA_BAR = {"color": 1.5}
    accepted = (state.setdefault("accepted", []) if isinstance(state.get("accepted"), list) else [])
    for key in CONFIDENCE_ORDER:
        f = facets.get(key)
        if not f:
            continue
        if key in state.get("overrides", {}):
            continue
        votes = f.get("votes", 0.0)
        dominant = f.get("confidence", 1.0) >= _threshold(state) and votes > 0
        if dominant and key not in accepted:
            if votes >= MIN_VOTES * EXTRA_BAR.get(key, 1.0):
                accepted.append(key)
        elif not dominant and key in accepted:
            accepted.remove(key)
    return state


def _threshold(state: dict) -> float:
    return state.get("_threshold", 0.55)


def accepted_facets(state: dict, settings: Settings | None = None) -> list[dict]:
    out = []
    for key in CONFIDENCE_ORDER:
        if key in state.get("accepted", []):
            f = state.get("facets", {}).get(key)
            if f and f.get("value"):
                entry = {"facet": key, "value": f["value"], "confidence": f.get("confidence", 1.0)}
                if key in state.get("overrides", {}):
                    entry["user"] = True
                out.append(entry)
    return out


def breadcrumb(state: dict, settings: Settings | None = None) -> str:
    return " · ".join(f["value"] for f in accepted_facets(state, settings))


# Backwards-compatible name used by callers/tests of the old API.
promote_facets = record_extraction


def apply_user_edit(state: dict, facet: str, value: str | None, accept: bool | None = None,
                    settings: Settings | None = None) -> dict:
    """Human edits always win: set/remove a facet value, accept or reject it."""
    overrides = state.setdefault("overrides", {})
    accepted = state.setdefault("accepted", [])
    if value is None:
        state["facets"].pop(facet, None)
        overrides.pop(facet, None)
        if facet in accepted:
            accepted.remove(facet)
    else:
        state["facets"][facet] = {"value": value.strip().lower(), "confidence": 1.0, "user": True}
        overrides[facet] = value.strip().lower()
        if accept is not False and facet not in accepted:
            accepted.append(facet)
    if accept is False and facet in accepted:
        accepted.remove(facet)
    return state


def compose_query(state: dict, settings: Settings | None = None) -> str:
    """Deterministic anti-drift query: accepted facets only, capped count/words."""
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
