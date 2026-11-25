import pytest
from fastapi.testclient import TestClient

import openinterest.api as api
from openinterest.config import Settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    s = Settings()
    s.db_path = str(tmp_path / "api.db")
    s.qdrant_url = ""  # force in-memory store
    s.agent_key = "test-agent-key"
    s.mistral_api_key = ""
    api.init_state(s)
    return TestClient(api.app)


def test_board_pin_roundtrip(client):
    r = client.post("/boards", json={"name": "chairs"})
    assert r.status_code == 200
    board_id = r.json()["board_id"]
    r = client.post(
        "/pins",
        json={
            "image_url": "http://example.com/a.jpg",
            "title": "A chair",
            "board_ids": [board_id],
            "index_now": False,
        },
    )
    assert r.status_code == 200
    pin_id = r.json()["id"]
    assert client.get(f"/pins/{pin_id}").json()["boards"] == [board_id]
    pins = client.get(f"/boards/{board_id}/pins").json()
    assert len(pins) == 1


def test_agent_auth_required(client):
    r = client.get("/agent/pins")
    assert r.status_code == 401
    r = client.get("/agent/pins", headers={"X-Agent-Key": "wrong"})
    assert r.status_code == 401
    r = client.get("/agent/pins", headers={"X-Agent-Key": "test-agent-key"})
    assert r.status_code == 200


def test_search_text_requires_embeddings(client):
    r = client.post("/search/text", json={"query": "chair"})
    assert r.status_code == 503


def test_session_crud_and_facets(client):
    r = client.post("/pins", json={"image_url": "http://e/1.jpg", "index_now": False})
    pin_id = r.json()["id"]
    r = client.post("/sessions", json={"seed_pin_ids": [pin_id]})
    assert r.status_code == 200
    sid = r.json()["id"]
    r = client.post(
        f"/sessions/{sid}/select",
        params={"pin_id": pin_id},
        json={
            "extracted_facets": {
                "object": {"value": "chair", "confidence": 0.9},
                "style": {"value": "mid-century modern", "confidence": 0.8},
            }
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["breadcrumb"] == "chair · mid-century modern"
    r = client.patch(
        f"/sessions/{sid}/facets/color", json={"value": "walnut brown", "accept": True}
    )
    assert "walnut brown" in r.json()["breadcrumb"]
    r = client.post(f"/sessions/{sid}/search")
    assert r.status_code == 200 and r.json()["tier"] == "local"


def test_session_404(client):
    assert client.post("/sessions/nope/search").status_code == 404
