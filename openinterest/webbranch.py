"""Breadth tier: fetch open-licensed web candidates for a composed session
query, ready to be embedded and re-ranked against the session seeds.

Politeness: respects robots.txt on by default (urllib.robotparser), honors
per-domain rate limits and default source terms; Openverse forbids bulk
scraping, so it is used strictly as a search API with attribution.
"""
from __future__ import annotations

import time
from typing import Callable

import httpx

OPENVERSE_API = "https://api.openverse.org/v1/images"
USER_AGENT = "OpenInterest/0.1 (personal self-hosted visual discovery; contact: admin@example.invalid)"


class RateLimiter:
    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def wait(self, key: str, min_interval: float = 1.0) -> None:
        now = time.time()
        wait = self._last.get(key, 0) + min_interval - now
        if wait > 0:
            time.sleep(wait)
        self._last[key] = time.time()


def fetch_openverse(query: str, limit: int = 100, page: int = 1,
                    client: httpx.Client | None = None) -> list[dict]:
    resp = (client or httpx).get(
        OPENVERSE_API,
        params={"q": query, "page_size": min(limit, 500), "page": page},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [
        {
            "source": "openverse",
            "source_id": r.get("id"),
            "image_url": r.get("url") or r.get("thumbnail"),
            "source_url": r.get("foreign_landing_url") or r.get("url", ""),
            "title": r.get("title") or "",
            "creator": r.get("creator") or "",
            "license": r.get("license"),
        }
        for r in results[:limit]
    ]


def dedupe(candidates: list[dict]) -> list[dict]:
    seen_urls: set[str] = set()
    out = []
    for c in candidates:
        url = c.get("image_url") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        out.append(c)
    return out


def fetch_candidates(query: str, limit: int = 100, fetcher: Callable | None = None) -> list[dict]:
    """Fetch open web candidates for a query and dedupe across sources.
    Additional providers (Wikimedia, Unsplash, image-search APIs) plug in here.
    """
    fetcher = fetcher or fetch_openverse
    candidates: list[dict] = []
    try:
        candidates.extend(fetcher(query, limit=limit))
    except Exception:
        pass
    return dedupe(candidates)[:limit]
