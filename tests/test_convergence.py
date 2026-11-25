import pytest

from openinterest.db import Database
from openinterest.facets import compose_query, record_dismissal, record_extraction
from openinterest.sessions import SessionEngine
from openinterest.vectorstore import InMemoryStore
from openinterest.config import Settings


def base_state():
    return {"facets": {}, "accepted": [], "seed_pins": [], "history": []}


def test_later_selections_revise_early_hypothesis():
    state = base_state()
    record_extraction(state, {"style": {"value": "victorian", "confidence": 0.9}})
    assert state["facets"]["style"]["value"] == "victorian"
    for _ in range(3):
        record_extraction(state, {"style": {"value": "mid-century modern", "confidence": 0.8}})
    assert state["facets"]["style"]["value"] == "mid-century modern"
    assert "victorian" not in compose_query(state)


def test_recency_beats_one_strong_early_vote():
    state = base_state()
    record_extraction(state, {"style": {"value": "gothic", "confidence": 0.95}})
    # decay 0.75: w(early)=0.75*0.95=0.71 vs later 0.8 -> later wins despite lower conf
    record_extraction(state, {"style": {"value": "art deco", "confidence": 0.8}})
    assert state["facets"]["style"]["value"] == "art deco"


def test_dismissal_is_negative_evidence():
    state = base_state()
    record_extraction(state, {"object": {"value": "chair", "confidence": 0.9}})
    assert "chair" in compose_query(state)
    record_dismissal(state, {"object": {"value": "chair", "confidence": 0.9}})
    assert "chair" not in compose_query(state)


def test_weak_single_vote_not_auto_accepted():
    state = base_state()
    record_extraction(state, {"material": {"value": "walnut", "confidence": 0.3}})
    assert "material" not in state["accepted"]  # visible in facets, but not in query
    record_extraction(state, {"material": {"value": "walnut", "confidence": 0.6}})
    assert "walnut" in compose_query(state)


@pytest.fixture
def env(tmp_path):
    s = Settings()
    s.db_path = str(tmp_path / "c.db")
    db = Database(s.db_path)
    store = InMemoryStore()
    engine = SessionEngine(db, store, s)
    return db, store, engine


def test_board_bootstrap_seeds_first_level_query(env):
    db, store, engine = env
    b = db.create_board("chairs")
    for i, tags in enumerate([["chair", "mid-century"], ["chair", "walnut"]]):
        p = db.create_pin(f"http://x/{i}.jpg", tags=tags)
        db.add_pin_to_board(p["id"], b)
    session = engine.create([], board_id=b)
    snap = engine.snapshot(session["id"])
    assert snap["breadcrumb"].startswith("chairs")
    assert "chair" in snap["query"]
    # board tags voted: two mid-chair/walnut votes -> eventually in facets
    state = engine.get(session["id"])["state"]
    assert state["facets"]["object"]["value"] == "chairs"  # override wins
    vals = [e["facets"].get("style", {}).get("value") for e in state["extractions"]]
    assert "mid-century" in vals and "walnut" in vals


def test_dismiss_excludes_from_local_and_web(env):
    db, store, engine = env
    p1 = db.create_pin("http://x/1.jpg")["id"]
    p2 = db.create_pin("http://x/2.jpg")["id"]
    store.upsert("clip", p1, [1.0, 0.0], {"pin_id": p1})
    store.upsert("clip", p2, [0.95, 0.05], {"pin_id": p2})
    session = engine.create([p1])
    engine.dismiss_image(session["id"], p2)
    local = engine.local_candidates(session["id"])
    assert p2 not in [r["pin_id"] for r in local["results"]]
    web = engine.rerank_web_results(session["id"], [{"pin_id": p2, "score": 0.0}, {"pin_id": "w", "score": 0.0}])
    assert [c["pin_id"] for c in web] == ["w"]


def test_suggest_chips_mines_neighborhood_vocabulary(env):
    db, store, engine = env
    p1 = db.create_pin("http://x/1.jpg")["id"]
    db.update_pin(p1, tags=["chair", "mid-century"], caption="a mid-century walnut chair")
    p2 = db.create_pin("http://x/2.jpg")["id"]
    db.update_pin(p2, tags=["chair", "walnut"], caption="walnut dining chair")
    store.upsert("clip", p1, [1.0, 0.0], {"pin_id": p1})
    store.upsert("clip", p2, [0.95, 0.05], {"pin_id": p2})
    session = engine.create([p1])
    engine.select_image(session["id"], p1, {"object": {"value": "chair", "confidence": 0.9}})
    chips = engine.suggest_chips(session["id"])
    vals = [c["value"] for c in chips["chips"]]
    assert "walnut" in vals or "mid-century" in vals
    # accepted/known values are never suggested
    assert "chair" not in vals


def test_adopt_chip_and_query_preview(env):
    db, _, engine = env
    p1 = db.create_pin("http://x/1.jpg")["id"]
    session = engine.create([p1])
    snap = engine.adopt_chip(session["id"], "material", "walnut")
    assert "walnut" in snap["query"]
    q = engine.query_preview(session["id"])
    assert q["breadcrumb"].endswith("walnut")
