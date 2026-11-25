"""Progressive faceted search sessions (the differentiating feature).

Fast tier: query the local index with accepted facet constraints and seed-image
vectors. Breadth tier (web branch): compose a text query from accepted facets,
fetch candidates from the open web/corpus, embed, and re-rank against the
session's seed images to filter text-search noise. Every web image the user
pins from a session gets indexed into the local store, growing the universe
along the walked path.
"""
from __future__ import annotations

import time

from .config import Settings
from .db import Database
from .facets import apply_user_edit, breadcrumb, compose_query, promote_facets
from .vectorstore import VectorStore, cosine


class SessionEngine:
    def __init__(self, db: Database, store: VectorStore, settings: Settings | None = None) -> None:
        self.db = db
        self.store = store
        self.settings = settings or Settings()

    def create(self, seed_pin_ids: list[str]) -> dict:
        session = self.db.create_session()
        state = session["state"]
        state["seed_pins"] = list(dict.fromkeys(seed_pin_ids))
        self.db.save_session(session["id"], state)
        return session

    def get(self, session_id: str) -> dict | None:
        return self.db.get_session(session_id)

    def select_image(self, session_id: str, pin_id: str, extracted: dict[str, dict] | None = None) -> dict:
        """Add a pin to the session seed; promotes facets from VLM extraction."""
        session = self.require(session_id)
        state = session["state"]
        if pin_id not in state["seed_pins"]:
            state["seed_pins"].append(pin_id)
        entry = {"pin_id": pin_id, "ts": time.time()}
        state.setdefault("history", []).append(entry)
        if extracted:
            promote_facets(state, extracted, settings=self.settings)
        self.db.save_session(session_id, state)
        return {
            "session_id": session_id,
            "breadcrumb": breadcrumb(state, self.settings),
            "accepted": state["accepted"],
            "facets": state["facets"],
            "seed_pins": state["seed_pins"],
        }

    def edit_facet(self, session_id: str, facet: str, value: str | None, accept: bool | None = None) -> dict:
        session = self.require(session_id)
        state = apply_user_edit(session["state"], facet, value, accept)
        self.db.save_session(session_id, state)
        return {
            "session_id": session_id,
            "breadcrumb": breadcrumb(state, self.settings),
            "accepted": state["accepted"],
            "facets": state["facets"],
        }

    def local_candidates(self, session_id: str, limit: int = 30) -> dict:
        """Fast tier: local index lookup seeded by the session seed images."""
        session = self.require(session_id)
        state = session["state"]
        scores: dict[str, float] = {}
        for seed in state["seed_pins"]:
            pt = self.store.get("clip", seed)
            if not pt or not pt.get("vector"):
                continue
            for hit in self.store.search("clip", pt["vector"], limit=limit * 2,
                                        exclude_ids=[seed]):
                pid = hit["payload"].get("pin_id", hit["id"])
                if pid in state["seed_pins"]:
                    continue
                scores[pid] = max(scores.get(pid, 0.0), hit["score"])
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
        return {
            "session_id": session_id,
            "breadcrumb": breadcrumb(state, self.settings),
            "tier": "local",
            "results": [{"pin_id": pid, "score": round(s, 6)} for pid, s in ranked],
        }

    def web_query(self, session_id: str) -> dict:
        """The composed, anti-drift-guarded web query for the breadth tier."""
        session = self.require(session_id)
        state = session["state"]
        return {
            "session_id": session_id,
            "query": compose_query(state, self.settings),
            "breadcrumb": breadcrumb(state, self.settings),
        }

    def rerank_web_results(self, session_id: str, candidates: list[dict]) -> list[dict]:
        """Re-rank web-branch candidates against the session seed images'
        CLIP/DINOv2 vectors (anti-drift: text-search noise gets demoted).

        Each candidate must have {pin_id, 'clip': [vector], 'dino': [vector], ...};
        embedding happens in the fetch layer (webbranch.corpus_* helpers).
        """
        session = self.require(session_id)
        state = session["state"]
        for cand in candidates:
            best = 0.0
            for seed in state["seed_pins"]:
                for signal in ("clip", "dino"):
                    seed_pt = self.store.get(signal, seed)
                    seed_vec = (seed_pt or {}).get("vector")
                    cand_vec = cand.get(signal)
                    if seed_vec and cand_vec:
                        best = max(best, cosine(seed_vec, cand_vec))
            cand["score"] = best
            cand.pop("clip", None)
            cand.pop("dino", None)
        return sorted(candidates, key=lambda c: -c["score"])

    def pin_from_web(self, pin: dict) -> str:
        """A web image the user pinned during a session: it extends the index."""
        return pin["id"]

    def require(self, session_id: str) -> dict:
        session = self.get(session_id)
        if not session:
            raise KeyError(f"session '{session_id}' not found")
        return session
