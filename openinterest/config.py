"""Configuration from environment with sensible CPU-friendly defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    db_path: str = field(default_factory=lambda: _env("OPENINTEREST_DB", "data/openinterest.db"))
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str = field(default_factory=lambda: _env("QDRANT_API_KEY"))
    mistral_api_key: str = field(default_factory=lambda: _env("MISTRAL_API_KEY"))
    agent_key: str = field(default_factory=lambda: _env("OPENINTEREST_AGENT_KEY"))
    openclip_model: str = field(default_factory=lambda: _env("OPENCLIP_MODEL", "ViT-B-32"))
    openclip_pretrained: str = field(default_factory=lambda: _env("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k"))
    dinov2_model: str = field(default_factory=lambda: _env("DINOV2_MODEL", "vitb14"))
    enable_object_embeddings: bool = field(
        default_factory=lambda: _env("ENABLE_OBJECT_EMBEDDINGS", "0") in ("1", "true", "yes")
    )
    facet_confidence_threshold: float = 0.55
    facet_query_max_facets: int = 4
    max_query_words: int = 8

    @property
    def has_mistral(self) -> bool:
        return bool(self.mistral_api_key)


settings = Settings()
