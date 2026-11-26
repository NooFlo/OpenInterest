"""Progressive faceted search sessions with convergence-by-selection.

The user starts vague (often from a board like "chairs"); every image they
select or dismiss is evidence. Facet values converge via recency-weighted
voting (facets.py); the system also *proposes* terms the user hasn't thought
of (chips) mined from the current neighborhood. The live query preview
(search bar) always shows the current best web-query assumption.

Tiers: local fast tier (own index) and web breadth tier (composed query →
open search APIs → embed → re-rank against session seeds).
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict

from .config import Settings
from .db import Database
from .facets import (
    CONFIDENCE_ORDER,
    accepted_facets,
    apply_user_edit,
    breadcrumb,
    compose_query,
    record_dismissal,
    record_extraction,
)
from .vectorstore import VectorStore, cosine

CHIP_FACETS = CONFIDENCE_ORDER


class SessionEngine:
    def __init__(self, db: Database, store: VectorStore, settings: Settings | None = None) -> None:
        self.db = db
        self.store = store
        self.settings = settings or Settings()

    # -- lifecycle -------------------------------------------------------------
    def create(self, seed_pin_ids: list[str], board_id: str | None = None,
               initial_query: str | None = None) -> dict:
        """Start a session. A board id bootstraps the first-level query: the
        board name seeds the object facet and member pins' tags/captions cast
        the initial votes, so the neighborhood starts in the user's folder.
        A broad free-text term (e.g. 'chairs') seeds the object facet directly
        for swipe-style exploration without a folder."""
        seed_pin_ids = list(dict.fromkeys(seed_pin_ids))
        seed_pin_ids = [pid for pid in seed_pin_ids if self.db.get_pin(pid)]
        if board_id:
            seed_pin_ids += [p["id"] for p in self.db.pins_in_board(board_id)]
            seed_pin_ids = list(dict.fromkeys(seed_pin_ids))
        session = self.db.create_session()
        state = session["state"]
        state["seed_pins"] = seed_pin_ids
        state["board_id"] = board_id
        state["history"] = []
        if board_id:
            self._bootstrap_from_board(state, board_id)
        elif initial_query:
            apply_user_edit(state, "object", initial_query.strip().lower(), accept=True)
        self.db.save_session(session["id"], state)
        return session

    def _bootstrap_from_board(self, state: dict, board_id: str) -> None:
        boards = {b["id"]: b for b in self.db.list_boards()}
        board = boards.get(board_id)
        if board:
            apply_user_edit(state, "object", board["name"].strip().lower(), accept=True)
        pins = self.db.pins_in_board(board_id)
        for pin in pins:
            ext = tags_to_extraction(pin)
            if ext:
                record_extraction(state, ext)

    # -- evidence ---------------------------------------------------------------
    def get(self, session_id: str) -> dict | None:
        return self.db.get_session(session_id)

    def require(self, session_id: str) -> dict:
        session = self.get(session_id)
        if not session:
            raise KeyError(f"session '{session_id}' not found")
        return session

    def select_image(self, session_id: str, pin_id: str, extracted: dict[str, dict] | None = None,
                     describe=None) -> dict:
        session = self.require(session_id)
        state = session["state"]
        pin = self.db.get_pin(pin_id)
        if not pin:
            pin = {"id": pin_id, "image_url": "", "tags": [], "caption": ""}
        if pin_id not in state["seed_pins"]:
            state["seed_pins"].append(pin_id)
        state.setdefault("history", []).append({"pin_id": pin_id, "ts": time.time(), "kind": "select"})
        if extracted is None and describe is not None:
            try:
                extracted = describe(pin.get("image_url", ""), pin).get("facets") or None
            except Exception:
                extracted = None
        elif extracted is None and pin.get("tags"):
            extracted = tags_to_extraction(pin)
        if extracted:
            record_extraction(state, extracted)
        state["dismissed"] = [p for p in state.get("dismissed", []) if p != pin_id]
        self.db.save_session(session_id, state)
        return self.snapshot(session_id)

    def dismiss_image(self, session_id: str, pin_id: str,
                      extracted: dict[str, dict] | None = None) -> dict:
        """Negative signal: the user skipped this candidate — record evidence
        against its tags and exclude it from future local/web results."""
        session = self.require(session_id)
        state = session["state"]
        pin = self.db.get_pin(pin_id)
        state.setdefault("dismissed", []).append(pin_id)
        state.setdefault("history", []).append({"pin_id": pin_id, "ts": time.time(), "kind": "dismiss"})
        ext = extracted or (tags_to_extraction(pin) if pin else None)
        if ext:
            record_dismissal(state, ext)
        self.db.save_session(session_id, state)
        return self.snapshot(session_id)

    def edit_facet(self, session_id: str, facet: str, value: str | None,
                   accept: bool | None = None) -> dict:
        session = self.require(session_id)
        state = apply_user_edit(session["state"], facet, value, accept, self.settings)
        self.db.save_session(session_id, state)
        return self.snapshot(session_id)

    # -- read model ----------------------------------------------------------------
    def snapshot(self, session_id: str) -> dict:
        state = self.require(session_id)["state"]
        return {
            "session_id": session_id,
            "breadcrumb": breadcrumb(state, self.settings),
            "accepted": state.get("accepted", []),
            "facets": state.get("facets", {}),
            "seed_pins": state.get("seed_pins", []),
            "dismissed": state.get("dismissed", []),
            "query": self.query_preview(session_id)["query"],
        }

    def query_preview(self, session_id: str) -> dict:
        """Live search-bar content: the current web-query assumption."""
        state = self.require(session_id)["state"]
        return {
            "session_id": session_id,
            "query": compose_query(state, self.settings),
            "breadcrumb": breadcrumb(state, self.settings),
            "facets": state.get("facets", {}),
        }

    # -- candidates ---------------------------------------------------------------
    def local_candidates(self, session_id: str, limit: int = 30) -> dict:
        """Fast tier: neighborhood of the seed set, minus dismissed pins."""
        session = self.require(session_id)
        state = session["state"]
        dismissed = set(state.get("dismissed", []))
        seeds = list(dict.fromkeys(state["seed_pins"]))
        later_bias = state.get("_weighted", {}).get("decay", 0.75)
        max_w = later_bias ** (max(len(seeds) - 1, 0)) or 1.0
        weights = [
            later_bias ** (max(len(seeds) - 1 - i, 0)) / max_w if seeds else 1.0
            for i in range(len(seeds))
        ] or [1.0]
        scores: dict[str, float] = defaultdict(float)
        for seed, w in zip(seeds, weights):
            pt = self.store.get("clip", seed)
            if not pt or not pt.get("vector"):
                continue
            for hit in self.store.search("clip", pt["vector"], limit=limit * 2, exclude_ids=[seed]):
                pid = hit["payload"].get("pin_id", hit["id"])
                if pid in seeds or pid in dismissed:
                    continue
                scores[pid] = max(scores[pid], w * hit["score"])
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
        return {
            "session_id": session_id,
            "breadcrumb": breadcrumb(state, self.settings),
            "tier": "local",
            "query": self.query_preview(session_id)["query"],
            "results": [{"pin_id": pid, "score": round(s, 6)} for pid, s in ranked],
        }

    def suggest_chips(self, session_id: str, limit: int = 8) -> dict:
        """Terms to *help the user discover what to search for*: facet values
        and tags mined from the current unselected neighborhood, ranked by how
        much they distinguish that neighborhood, minus what the session already
        accepts. Accepting a chip is a normal facet edit (confidence 1.0)."""
        state = self.require(session_id)["state"]
        current = {f["facet"]: f["value"] for f in accepted_facets(state, self.settings)}
        seen_vals = {v.lower() for f in state.get("facets", {}).values() for v in [f.get("value", "")]}
        cand = self.local_candidates(session_id, limit=40)
        counter: Counter = Counter()
        pin_by_id = {}
        chain_terms = {_norm(v) for v in current.values()}
        for r in cand["results"]:
            pin = self.db.get_pin(r["pin_id"])
            if not pin:
                continue
            pin_by_id[r["pin_id"]] = pin
            # affinity: candidates that also match accepted facet values back
            # their terms more strongly, so chips surface the agreed vocabulary
            affinity = 1.0 + sum(
                1 for t in chain_terms if t and t in {_norm(x) for x in _terms(pin)}
            )
            for tag in (pin.get("tags") or []) + _caption_words(pin.get("caption", "")):
                tag = tag.strip().lower()
                if len(tag) >= 3 and not _is_known(tag, seen_vals):
                    counter[tag] += affinity
        chips = [
            {"value": val, "count": n, "neighbor_pins": [p for p in pin_by_id if val in _terms(pin_by_id[p])][:3]}
            for val, n in counter.most_common(limit)
        ]
        return {
            "session_id": session_id,
            "breadcrumb": cand["breadcrumb"],
            "accepted": current,
            "chips": chips,
        }

    def adopt_chip(self, session_id: str, facet: str, value: str) -> dict:
        return self.edit_facet(session_id, facet, value, accept=True)

    # -- web branch ------------------------------------------------------------
    def web_query(self, session_id: str) -> dict:
        return self.query_preview(session_id)

    def rerank_web_results(self, session_id: str, candidates: list[dict]) -> list[dict]:
        """Anti-noise: web candidates are ranked by similarity to the session's
        seed images, not to the text query. Dismissed pins are removed."""
        state = self.require(session_id)["state"]
        dismissed = set(state.get("dismissed", []))
        for cand in candidates:
            best = 0.0
            for seed in state["seed_pins"]:
                for signal in ("clip", "dino"):
                    seed_vec = (self.store.get(signal, seed) or {}).get("vector")
                    cand_vec = cand.get(signal)
                    if seed_vec and cand_vec:
                        best = max(best, cosine(seed_vec, cand_vec))
            cand["score"] = best
            cand.pop("clip", None)
            cand.pop("dino", None)
        return sorted(
            (c for c in candidates if c.get("pin_id") not in dismissed and not c.get("dismissed")),
            key=lambda c: -c["score"],
        )


def tags_to_extraction(pin: dict) -> dict[str, dict]:
    """Turn a pin's stored tags into a weak facet vote. The first tag votes for
    'object' (it is usually the dominant noun); others vote for 'style'."""
    tags = [str(t).strip().lower() for t in (pin.get("tags") or []) if str(t).strip()]
    if not tags:
        caption = pin.get("caption") or ""
        tags = _caption_words(caption)
    if not tags:
        return {}
    ext: dict[str, dict] = {"object": {"value": tags[0], "confidence": 0.45}}
    if len(tags) > 1:
        ext["style"] = {"value": tags[1], "confidence": 0.35}
    return ext


def _caption_words(caption: str) -> list[str]:
    import re

    return [w for w in re.findall(r"[a-z][a-z-]{2,}", (caption or "").lower())][:6]


def _norm(term: str) -> str:
    return term.strip().lower().removesuffix("s")


def _is_known(tag: str, seen_vals: set[str]) -> bool:
    """Skip suggestions the user already has in the chain — including
    singular/plural variants of accepted values."""
    n = _norm(tag)
    return any(n == _norm(v) or n in _norm(v) or _norm(v) in n for v in seen_vals)


def _terms(pin: dict) -> set[str]:
    return {t.strip().lower() for t in (pin.get("tags") or [])} | set(_caption_words(pin.get("caption", "")))
