"""Vector store abstraction: Qdrant in production, in-memory cosine fallback
for local/test runs without any server.

Two collections ('clip' whole-image semantic, 'dino' whole-image structural)
plus 'objects' for per-object embeddings. Falls back gracefully — anything not
stored is simply absent from that signal, and fusion tolerates missing signals.
"""
from __future__ import annotations

import math
from typing import Protocol, Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


class VectorStore(Protocol):
    def upsert(self, collection: str, point_id: str, vector: Sequence[float], payload: dict) -> None: ...
    def search(self, collection: str, vector: Sequence[float], limit: int = 20,
               exclude_ids: Sequence[str] = ()) -> list[dict]: ...
    def delete(self, collection: str, point_id: str) -> None: ...
    def get(self, collection: str, point_id: str) -> dict | None: ...
    def count(self, collection: str) -> int: ...


class InMemoryStore:
    """Brute-force cosine store. Fine for thousands of pins (personal scale)."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, dict]] = {}

    def upsert(self, collection: str, point_id: str, vector: Sequence[float], payload: dict) -> None:
        self._data.setdefault(collection, {})[point_id] = {
            "id": point_id, "vector": list(vector), "payload": payload,
        }

    def search(self, collection: str, vector: Sequence[float], limit: int = 20,
               exclude_ids: Sequence[str] = ()) -> list[dict]:
        excl = set(exclude_ids)
        scored = [
            (cosine(vector, p["vector"]), p)
            for pid, p in self._data.get(collection, {}).items()
            if pid not in excl
        ]
        scored.sort(key=lambda t: -t[0])
        return [
            {"id": p["id"], "score": s, "payload": p["payload"]}
            for s, p in scored[:limit]
        ]

    def delete(self, collection: str, point_id: str) -> None:
        self._data.get(collection, {}).pop(point_id, None)

    def get(self, collection: str, point_id: str) -> dict | None:
        p = self._data.get(collection, {}).get(point_id)
        return {"id": p["id"], "vector": p["vector"], "payload": p["payload"]} if p else None

    def count(self, collection: str) -> int:
        return len(self._data.get(collection, {}))


def make_store(qdrant_url: str, api_key: str = "") -> VectorStore:
    """Return a Qdrant-backed store if reachable; else the in-memory fallback."""
    if qdrant_url:
        try:
            from .qdrant_store import QdrantStore

            return QdrantStore(qdrant_url, api_key)
        except Exception:
            pass
    return InMemoryStore()
