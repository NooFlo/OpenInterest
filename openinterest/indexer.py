"""Indexing worker: pin image -> vectors + caption/facets -> vector store + db.

Phase 1: CLIP whole-image + DINOv2 whole-image vectors + caption/tags.
Phase 2 (optional, ENABLE_OBJECT_EMBEDDINGS=1): per-object vectors with
bbox + label via sam-clip.
"""
from __future__ import annotations

import time
from .caption import make_captioner
from .config import Settings
from .db import Database
from .embeddings import ImageEmbedder, SamClipObjectEmbedder, fetch_image
from .vectorstore import VectorStore


class Indexer:
    def __init__(
        self,
        db: Database,
        store: VectorStore,
        captioner=None,
        settings: Settings | None = None,
        object_embedder: SamClipObjectEmbedder | None = None,
    ) -> None:
        self.db = db
        self.store = store
        self.settings = settings or Settings()
        self.captioner = captioner or make_captioner(self.settings)
        self.object_embedder = object_embedder

    def index_pin(self, pin: dict) -> dict:
        """Index one pin; returns the stored metadata document."""
        image_url = pin["image_url"]
        try:
            img = fetch_image(image_url)
        except Exception as exc:
            return {"pin_id": pin["id"], "error": f"fetch failed: {exc}"}

        payload_base = {
            "pin_id": pin["id"],
            "image_url": image_url,
            "source_url": pin.get("source_url", ""),
            "origin": pin.get("origin", "user"),
            "boards": pin.get("boards", []),
        }

        for signal, name in (("clip", "clip"), ("dino", "dino")):
            embedder: ImageEmbedder | None = getattr(self, f"_{name}_embedder", None)
            if embedder is None:
                continue
            vec = embedder.embed_image(img)
            self.store.upsert(signal, pin["id"], vec, dict(payload_base, signal=name))

        if self.object_embedder is not None:
            for obj in self.embed_objects(img):
                self.store.upsert(
                    "objects",
                    f"{pin['id']}::{obj['label']}::{obj['index']}",
                    obj["vector"],
                    dict(payload_base, signal="object", object_label=obj["label"], bbox=obj["bbox"]),
                )

        try:
            described = self.describe(image_url, pin)
        except Exception as exc:
            described = {"caption": "", "tags": [], "facets": {}, "caption_error": str(exc)}

        tags = list(dict.fromkeys(list(pin.get("tags", [])) + described.get("tags", [])))
        description = pin.get("description", "") or ""
        self.db.update_pin(
            pin["id"],
            title=pin.get("title", ""),
            description=description,
            caption=described.get("caption", ""),
            tags=tags,
            indexed_at=time.time(),
        )
        return {"pin_id": pin["id"], "caption": described.get("caption", ""), "tags": tags,
                "facets": described.get("facets", {})}

    def embed_objects(self, img) -> list[dict]:
        try:
            results = self.object_embedder.embed_objects(img)
        except Exception:
            return []
        out = []
        for i, (vector, label, bbox) in enumerate(results):
            vec = list(map(float, vector))
            if not vec:
                continue
            out.append({"vector": vec, "label": str(label), "bbox": list(map(float, bbox or [])), "index": i})
        return out

    def describe(self, image_url: str, pin: dict) -> dict:
        if hasattr(self.captioner, "describe"):
            try:
                return self.captioner.describe(image_url)
            except TypeError:
                return self.captioner.describe(image_url, pin.get("title", ""), pin.get("description", ""))
        return {"caption": "", "tags": [], "facets": {}}

    def index_pending(self, limit: int = 100) -> list[dict]:
        from .db import pins_missing_index

        results = []
        for pin in pins_missing_index(self.db):
            if len(results) >= limit:
                break
            results.append(self.index_pin(pin))
        return results


def attach_embedders(indexer: Indexer, embedders: dict[str, ImageEmbedder]) -> Indexer:
    """Wire embedder instances onto an Indexer (kept separate for testability)."""
    if "clip" in embedders:
        indexer._clip_embedder = embedders["clip"]
    if "dino" in embedders:
        indexer._dino_embedder = embedders["dino"]
    return indexer
