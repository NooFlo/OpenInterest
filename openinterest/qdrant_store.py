"""Qdrant-backed VectorStore implementation (optional dependency)."""
from __future__ import annotations

from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

_COLLECTIONS = {"clip": 512, "dino": 384, "objects": 512}
# Real vector sizes are set on first upsert per collection; the presets are
# only defaults for common CPU-light models (ViT-B/32 -> 512, dinov2 vitb14 -> 384).


class QdrantStore:
    def __init__(self, url: str, api_key: str = "") -> None:
        self.client = QdrantClient(url=url, api_key=api_key or None, timeout=5)
        self._seen_dims: dict[str, VectorParams] = {}

    def _ensure_collection(self, name: str, dim: int) -> None:
        if name in self._seen_dims:
            return
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        else:
            actual = self.client.get_collection(name).config.params.vectors
            if getattr(actual, "size", None) not in (None, dim):
                raise ValueError(
                    f"Qdrant collection '{name}' has dim {getattr(actual, 'size')}, expected {dim}"
                )
        self._seen_dims[name] = VectorParams(size=dim, distance=Distance.COSINE)

    def upsert(self, collection: str, point_id: str, vector: Sequence[float], payload: dict) -> None:
        self._ensure_collection(collection, len(vector))
        self.client.upsert(
            collection_name=collection,
            points=[PointStruct(id=_point_uuid(point_id), vector=list(vector), payload=payload)],
        )

    def search(self, collection: str, vector: Sequence[float], limit: int = 20,
               exclude_ids: Sequence[str] = ()) -> list[dict]:
        if not self.client.collection_exists(collection):
            return []
        excl = list({_point_uuid(x) for x in exclude_ids})
        hits = self.client.query_points(
            collection_name=collection,
            query=list(vector),
            limit=limit + len(excl),
        ).points
        return [
            {"id": h.payload.get("pin_id", h.id), "score": h.score, "payload": h.payload}
            for h in hits if h.id not in excl
        ][:limit]

    def delete(self, collection: str, point_id: str) -> None:
        from qdrant_client.models import PointIdsList

        if self.client.collection_exists(collection):
            self.client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=[_point_uuid(point_id)]),
            )

    def get(self, collection: str, point_id: str) -> dict | None:
        if not self.client.collection_exists(collection):
            return None
        pts = self.client.retrieve(collection_name=collection, ids=[_point_uuid(point_id)], with_payload=True)
        return pts[0].payload if pts else None

    def count(self, collection: str) -> int:
        if not self.client.collection_exists(collection):
            return 0
        return self.client.count(collection_name=collection).count


def _point_uuid(point_id: str) -> str:
    """Qdrant accepts UUIDs or integers; derive a stable UUID-style string.
    Qdrant accepts only true UUIDs, so we map ids deterministically to UUID5.
    """
    import uuid

    try:
        return str(uuid.UUID(point_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"openinterest:{point_id}"))
