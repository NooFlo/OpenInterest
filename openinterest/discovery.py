"""Candidate generation + fusion: the 'more like this' pipeline.

Signals (all optional; fusion tolerates absence):
  clip    — whole-image semantic neighbors (also enables text search)
  dino    — whole-image structural neighbors
  object  — per-object neighbors (Phase 2, tap-the-lamp)
  cosave  — co-save graph neighbors (Phase 3, pure SQL)
"""
from __future__ import annotations

from .cosave import cosave_candidates
from .db import Database
from .fusion import DEFAULT_WEIGHTS, fuse_and_diversify
from .vectorstore import VectorStore, cosine


def _store_ranking(store: VectorStore, collection: str, signal: str, seed_id: str,
                   excl: list[str], per_signal_limit: int) -> tuple[str, list[str]] | None:
    pt = store.get(collection, seed_id)
    if not pt or not pt.get("vector"):
        return None
    hits = store.search(collection, pt["vector"], limit=per_signal_limit, exclude_ids=excl)
    ids = [h["payload"].get("pin_id", h["id"]) for h in hits]
    ids = [i for i in dict.fromkeys(ids) if i not in excl]
    return (signal, ids) if ids else None


def find_similar(
    db: Database,
    store: VectorStore,
    seed_pin: dict,
    weights: dict[str, float] | None = None,
    limit: int = 30,
    per_signal_limit: int = 60,
    exclude_pins: list[str] | None = None,
) -> dict:
    seed_id = seed_pin["id"]
    excl = list({seed_id, *(exclude_pins or [])})

    rankings: dict[str, list[str]] = {}
    for collection, signal in (("clip", "clip"), ("dino", "dino"), ("objects", "object")):
        pair = _store_ranking(store, collection, signal, seed_id, excl, per_signal_limit)
        if pair:
            rankings[pair[0]] = pair[1]

    cosave = cosave_candidates(db, seed_id, per_signal_limit)
    cosave = [pid for pid in cosave if pid not in excl]
    if cosave:
        rankings["cosave"] = cosave

    def similarity(a: str, b: str) -> float:
        va = store.get("clip", a)
        vb = store.get("clip", b)
        if not va or not vb or not va.get("vector") or not vb.get("vector"):
            return 0.0
        return cosine(va["vector"], vb["vector"])

    fused = fuse_and_diversify(
        rankings,
        similarity,
        weights=weights or DEFAULT_WEIGHTS,
        limit=limit,
    )
    return {
        "seed_pin_id": seed_id,
        "signals": rankings,
        "results": [
            {
                "pin_id": c.pin_id,
                "score": round(c.per_signal_score.get("__fused__", 0.0), 6),
                "ranked_by": c.per_signal_rank,
            }
            for c in fused
        ],
    }


def find_by_text(store: VectorStore, text_vector: list[float], limit: int = 30) -> list[dict]:
    """Natural-language pin search via the CLIP text encoder vector."""
    hits = store.search("clip", text_vector, limit=limit)
    return [
        {
            "pin_id": h["payload"].get("pin_id", h["id"]),
            "score": h["score"],
            "image_url": h["payload"].get("image_url"),
        }
        for h in hits
    ]


def find_by_object(store: VectorStore, object_point_id: str, limit: int = 30) -> dict:
    """Phase 2: 'tap an object in a pin' -> similar objects -> their pins."""
    obj = store.get("objects", object_point_id)
    if not obj or not obj.get("vector"):
        return {"object_point_id": object_point_id, "results": []}
    hits = store.search("objects", obj["vector"], limit=limit, exclude_ids=[object_point_id])
    return {
        "object_point_id": object_point_id,
        "results": [
            {
                "pin_id": h["payload"].get("pin_id"),
                "object_label": h["payload"].get("object_label"),
                "bbox": h["payload"].get("bbox"),
                "score": h["score"],
            }
            for h in hits
        ],
    }
