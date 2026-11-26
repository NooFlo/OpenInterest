"""Bulk open-corpus ingestion: pull openly licensed images from Openverse /
Wikimedia Commons search, embed them, and add them to the local store as
'corpus' pins so 'more like this' spans a 100M-image universe, not just
the user's own library.

Politeness: keyless search APIs only, per-source rate limiting, license and
attribution recorded in the pin payload per source terms of use.
"""
from __future__ import annotations

import time
from typing import Callable

import httpx

from .db import Database
from .webbranch import USER_AGENT, RateLimiter, fetch_openverse

CommonsSearch = (
    "https://commons.wikimedia.org/w/api.php"
)


def fetch_commons(query: str, limit: int = 100, client: httpx.Client | None = None) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrlimit": min(limit, 500),
        "gsrnamespace": 6,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 800,
    }
    resp = (client or httpx).get(
        CommonsSearch, params=params, headers={"User-Agent": USER_AGENT}, timeout=30
    )
    resp.raise_for_status()
    pages = (resp.json().get("query") or {}).get("pages", {})
    out = []
    for p in pages.values():
        info = (p.get("imageinfo") or [{}])[0]
        meta = (info.get("extmetadata") or {})
        out.append(
            {
                "source": "wikimedia",
                "source_id": str(p.get("pageid")),
                "image_url": info.get("thumburl") or info.get("url"),
                "source_url": info.get("descriptionurl", ""),
                "title": p.get("title", "").replace("File:", "").rsplit(".", 1)[0],
                "creator": (meta.get("Artist") or {}).get("value", ""),
                "license": (meta.get("LicenseShortName") or {}).get("value", ""),
            }
        )
    return [c for c in out if c["image_url"]]


def ingest_corpus(
    db: Database,
    index_fn: Callable[[dict], dict],
    query: str,
    limit: int = 100,
    fetcher: Callable[[str, int], list[dict]] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict:
    """Ingest one query batch of corpus candidates as 'corpus' pins.

    index_fn: a callable that fully indexes a new pin (embed + store + db),
    supplied by the service layer so this module stays ML-free.
    """
    limiter = RateLimiter()
    fetcher = fetcher or (lambda q, n: fetch_openverse(q, n))
    ingested, skipped = [], 0
    try:
        candidates = fetcher(query, limit)
    except Exception as exc:
        return {"query": query, "ingested": 0, "skipped": 0, "error": str(exc)}
    for c in candidates:
        existing = find_by_url(db, c.get("image_url", ""))
        if existing:
            skipped += 1
            continue
        limiter.wait(c["source"], min_interval=0.5)
        pin = db.create_pin(
            image_url=c.get("image_url", ""),
            source_url=c.get("source_url", ""),
            title=c.get("title", ""),
            tags=[t for t in [c.get("source"), c.get("license")] if t],
            origin="corpus",
        )
        res = index_fn(pin)
        if "error" in res:
            db.delete_pin(pin["id"])
            skipped += 1
        else:
            ingested.append(pin["id"])
    return {"query": query, "ingested": len(ingested), "skipped": skipped}


def find_by_url(db: Database, image_url: str) -> str | None:
    if not image_url:
        return None
    with db.conn() as c:
        row = c.execute("SELECT id FROM pins WHERE image_url = ?", (image_url,)).fetchone()
    return row["id"] if row else None
