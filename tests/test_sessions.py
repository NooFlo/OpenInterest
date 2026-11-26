import pytest

from openinterest.db import Database
from openinterest.sessions import SessionEngine
from openinterest.vectorstore import InMemoryStore
from openinterest.config import Settings


@pytest.fixture
def env(tmp_path):
    s = Settings()
    s.db_path = str(tmp_path / "s.db")
    s.agent_key = ""
    db = Database(s.db_path)
    store = InMemoryStore()
    engine = SessionEngine(db, store, s)
    return db, store, engine


def test_session_flow_select_promote_local(env):
    db, store, engine = env
    p1 = db.create_pin("http://x/1.jpg")["id"]
    p2 = db.create_pin("http://x/2.jpg")["id"]
    p3 = db.create_pin("http://x/3.jpg")["id"]
    store.upsert("clip", p1, [1.0, 0.0], {"pin_id": p1})
    store.upsert("clip", p2, [0.9, 0.1], {"pin_id": p2})
    store.upsert("clip", p3, [0.95, 0.05], {"pin_id": p3})

    session = engine.create([p1])
    res = engine.select_image(
        session["id"],
        p2,
        {"object": {"value": "chair", "confidence": 0.9},
         "material": {"value": "walnut", "confidence": 0.4}},
    )
    assert res["breadcrumb"] == "chair"
    assert res["accepted"] == ["object"]

    local = engine.local_candidates(session["id"])
    assert local["tier"] == "local"
    assert [r["pin_id"] for r in local["results"]] == [p3]  # selected seeds excluded


def test_web_rerank_uses_seed_similarity(env):
    db, store, engine = env
    p1 = db.create_pin("http://x/1.jpg")["id"]
    store.upsert("clip", p1, [1.0, 0.0], {"pin_id": p1})
    session = engine.create([p1])
    candidates = [
        {"pin_id": "w-near", "clip": [0.99, 0.01]},
        {"pin_id": "w-far", "clip": [0.0, 1.0]},
    ]
    out = engine.rerank_web_results(session["id"], candidates)
    assert out[0]["pin_id"] == "w-near"
    assert out[0]["score"] > out[1]["score"]
    assert "clip" not in out[0]  # vectors stripped


def test_missing_session_raises(env):
    _, _, engine = env
    with pytest.raises(KeyError):
        engine.require("nope")
