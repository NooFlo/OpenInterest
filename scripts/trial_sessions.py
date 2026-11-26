"""Trial run: watch the session query refine as a simulated user swipes.

No ML dependencies: deterministic unit-vector "embeddings" cluster pins into
topics the way CLIP/DINOv2 signal would, and the simulated VLM facet
extraction is what Pixtral would return for these images. The session engine,
voting, dismissal, chips and query composition are the real production code.

Usage:  python scripts/trial_sessions.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from openinterest.db import Database
from openinterest.facets import compose_query
from openinterest.sessions import SessionEngine
from openinterest.vectorstore import InMemoryStore

DIM = 16
TOPICS = {
    # topic -> one-hot base + tag vocabulary, three aesthetic clusters
    "mcm_walnut": {"base": 0, "era": "1960s",
                   "tags": ["chair", "mid-century", "walnut", "teak", "1960s", "danish"]},
    "victorian": {"base": 1, "era": "antique",
                  "tags": ["chair", "victorian", "velvet", "ornate", "mahogany", "antique"]},
    "minimal_metal": {"base": 2, "era": "contemporary",
                      "tags": ["chair", "minimalist", "steel", "modern", "black", "industrial"]},
}


class FakeEmbedder:
    """Unit vectors per topic with small per-pin jitter -> cosine clusters."""

    def __init__(self, seed: int = 0):
        import random

        self._rng = random.Random(seed)

    def embed(self, topic: str, pin_key: str) -> list[float]:
        import hashlib

        h = int(hashlib.md5(pin_key.encode()).hexdigest(), 16) >> 8
        base = [0.0] * DIM
        base[TOPICS[topic]["base"]] = 1.0
        for i in range(DIM):
            base[i] += ((h >> i) & 1) * 0.08
        n = sum(x * x for x in base) ** 0.5
        return [round(x / n, 5) for x in base]


def vlm_extract(topic: str, pin_key: str, real: bool = True) -> dict:
    """Simulated Pixtral output for a pin from this topic, with realistic noise."""
    import hashlib

    h = int(hashlib.md5(pin_key.encode()).hexdigest(), 16)
    tags = TOPICS[topic]["tags"]
    obj, style, material = tags[0], tags[1], tags[2]
    noise = 0.15 * ((h % 100) / 100.0)
    out = {
        "object": {"value": obj, "confidence": round(0.85 + noise, 2)},
        "style": {"value": style, "confidence": round(0.55 + noise, 2)},
        "material": {"value": material, "confidence": round(0.55 + noise, 2)},
    }
    if h % 3 == 0:  # VLM sometimes misses the era
        out["era"] = {"value": TOPICS[topic]["era"], "confidence": round(0.5 + noise, 2)}
    return out


def build_library(db: Database, store: InMemoryStore, emb: FakeEmbedder) -> dict[str, list[str]]:
    pins = {}
    board = db.create_board("chairs")
    for topic, spec in TOPICS.items():
        ids = []
        for i in range(1, 6):
            pin = db.create_pin(
                image_url=f"https://example.test/{topic}-{i}.jpg",
                title=f"{topic} chair {i}",
                tags=spec["tags"][:3] + [f"sample{i}"],
                origin="user",
            )
            db.add_pin_to_board(pin["id"], board)
            store.upsert("clip", pin["id"], emb.embed(topic, f"{topic}{i}"), {"pin_id": pin["id"], "boards": [board]})
            ids.append(pin["id"])
        pins[topic] = ids
    return pins


def show(step: str, engine: SessionEngine, sid: str, extra: str = ""):
    q = engine.query_preview(sid)
    print(f"  [{step}] query = {q['query']!r:60}  breadcrumb = {q['breadcrumb']!r}{extra}")


def main() -> None:
    import tempfile
    import os

    tmp = tempfile.mkdtemp()
    db = Database(os.path.join(tmp, "trial.db"))
    store = InMemoryStore()
    engine = SessionEngine(db, store)
    emb = FakeEmbedder()

    pins = build_library(db, store, emb)
    print("== TRIAL 1: broad term 'chairs', user swipes right on mid-century/walnut, left on victorian ==\n")
    sid = engine.create([], initial_query="chairs")["id"]
    show("seed", engine, sid)

    deck = [
        (pins["victorian"][0], "left"),   # user hates ornate
        (pins["mcm_walnut"][0], "right"),
        (pins["mcm_walnut"][2], "right"),
        (pins["victorian"][2], "left"),
        (pins["mcm_walnut"][4], "right"),
        (pins["minimal_metal"][1], "left"),
    ]
    for pin_id, direction in deck:
        topic = next(t for t, ids in pins.items() if pin_id in ids)
        if direction == "right":
            engine.select_image(sid, pin_id, vlm_extract(topic, pin_id))
        else:
            engine.dismiss_image(sid, pin_id, vlm_extract(topic, pin_id))
        show(f"swipe {direction} ({topic})", engine, sid)

    chips = engine.suggest_chips(sid)
    print(f"  [suggested chips] {[c['value'] for c in chips['chips'][:6]]}")
    local = engine.local_candidates(sid)
    top = [(r['pin_id'], r['score']) for r in local['results'][:5]]
    print("  [top local candidates after refinement]")
    for pid, score in top:
        t = next((t for t, ids in pins.items() if pid in ids), "?")
        print(f"      {t:16} score={score:.3f}")

    print()
    print("== TRIAL 2: user starts WRONG (picks a victorian chair first), then converges on minimal ==\n")
    sid2 = engine.create([pins["victorian"][1]])["id"]
    engine.select_image(sid2, pins["victorian"][1], vlm_extract("victorian", "v2"))
    show("select victorian", engine, sid2)
    for pid, topic in [
        (pins["minimal_metal"][1], "right"),
        (pins["minimal_metal"][3], "right"),
        (pins["mcm_walnut"][1], "left"),
        (pins["minimal_metal"][2], "right"),
    ]:
        for direction_id, topic in [("placeholder", topic)]:
            pass
    # redo properly (left/right per plan):
    engine.select_image(sid2, pins["minimal_metal"][1], vlm_extract("minimal_metal", "m2"))
    show("select minimal", engine, sid2)
    engine.dismiss_image(sid2, pins["mcm_walnut"][1], vlm_extract("mcm_walnut", "w2"))
    show("dismiss walnut", engine, sid2)
    engine.select_image(sid2, pins["minimal_metal"][3], vlm_extract("minimal_metal", "m4"))
    show("select minimal #2", engine, sid2)

    print()
    print("== TRIAL 3: board-seeded session ('chairs' folder), then 3 right-swipes on walnut ==\n")
    sid3 = engine.create([], board_id=db.list_boards()[0]["id"])["id"]
    show("board seed", engine, sid3)
    for i, pid in enumerate(pins["mcm_walnut"][:3]):
        engine.select_image(sid3, pid, vlm_extract("mcm_walnut", f"w{i}"))
        show(f"right-swipe walnut #{i+1}", engine, sid3)
    q = compose_query(engine.get(sid3)["state"])
    print(f"\n  -> final composed web query (would be sent to Openverse/Commons): {q!r}")


if __name__ == "__main__":
    main()
