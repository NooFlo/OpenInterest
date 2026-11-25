"""OpenInterest HTTP API.

Boards/pins + 'more like this' + text search + object-level search +
faceted search sessions + web branch + corpus ingest + agent API.
The agent API (X-Agent-Key header) lets an agent or external service run a
full search session on the user's behalf.
"""
from __future__ import annotations

import secrets
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field

from .caption import make_captioner
from .config import Settings, settings as default_settings
from .corpus import ingest_corpus
from .db import Database
from .discovery import find_by_object, find_by_text, find_similar
from .embeddings import fetch_image
from .indexer import Indexer, attach_embedders
from .sessions import SessionEngine
from .vectorstore import make_store
from .webbranch import fetch_candidates

app = FastAPI(title="OpenInterest", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

STATE: dict = {}

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


def init_state(settings: Settings | None = None) -> dict:
    s = settings or default_settings
    db = Database(s.db_path)
    store = make_store(s.qdrant_url, s.qdrant_api_key)
    indexer = Indexer(db, store, make_captioner(s), settings=s)
    embedders: dict = {}
    try:
        from .embeddings import load_embedders

        embedders = load_embedders(s)
        attach_embedders(indexer, embedders)
    except Exception:
        pass
    if s.enable_object_embeddings:
        try:
            from .embeddings import SamClipObjectEmbedder

            indexer.object_embedder = SamClipObjectEmbedder()
        except Exception:
            pass
    STATE.clear()
    STATE.update(
        settings=s,
        db=db,
        store=store,
        indexer=indexer,
        embedders=embedders,
        sessions=SessionEngine(db, store, s),
    )
    return STATE


def state() -> dict:
    return STATE if STATE else init_state()


def require_agent_key(
    x_agent_key: Optional[str] = Header(default=None, alias="X-Agent-Key"),
) -> None:
    s = state()["settings"]
    if not s.agent_key:
        raise HTTPException(503, "agent API disabled: set OPENINTEREST_AGENT_KEY")
    if not x_agent_key or not secrets.compare_digest(x_agent_key, s.agent_key):
        raise HTTPException(401, "invalid or missing X-Agent-Key header")


def require_embeddings() -> dict:
    embedders = state()["embedders"]
    if "clip" not in embedders:
        raise HTTPException(503, "embeddings unavailable: install the 'ml' extras and load models")
    return embedders


# --------------------------------------------------------------------------
# Boards
# --------------------------------------------------------------------------
class BoardIn(BaseModel):
    name: str


@app.post("/boards")
def create_board(body: BoardIn):
    st = state()
    board_id = st["db"].create_board(body.name)
    return {"board_id": board_id}


@app.get("/boards")
def list_boards():
    return state()["db"].list_boards()


@app.get("/boards/{board_id}/pins")
def board_pins(board_id: str):
    return state()["db"].pins_in_board(board_id)


# --------------------------------------------------------------------------
# Pins
# --------------------------------------------------------------------------
class PinIn(BaseModel):
    image_url: str
    source_url: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = []
    board_ids: list[str] = []
    index_now: bool = True


@app.post("/pins")
def create_pin(body: PinIn):
    st = state()
    pin = st["db"].create_pin(
        image_url=body.image_url,
        source_url=body.source_url,
        title=body.title,
        description=body.description,
        tags=body.tags,
    )
    for b in body.board_ids:
        st["db"].add_pin_to_board(pin["id"], b)
    if body.index_now:
        st["indexer"].index_pin(dict(pin, boards=body.board_ids))
    return st["db"].get_pin(pin["id"])


@app.get("/pins")
def list_pins(limit: int = 50, offset: int = 0):
    return state()["db"].list_pins(limit, offset)


@app.get("/pins/{pin_id}")
def get_pin(pin_id: str):
    pin = state()["db"].get_pin(pin_id)
    if not pin:
        raise HTTPException(404, "pin not found")
    return pin


@app.delete("/pins/{pin_id}")
def delete_pin(pin_id: str):
    st = state()
    if not st["db"].get_pin(pin_id):
        raise HTTPException(404, "pin not found")
    for collection in ("clip", "dino", "objects"):
        st["store"].delete(collection, pin_id)
    st["db"].delete_pin(pin_id)
    return {"deleted": pin_id}


@app.post("/pins/{pin_id}/index")
def index_pin(pin_id: str):
    st = state()
    pin = st["db"].get_pin(pin_id)
    if not pin:
        raise HTTPException(404, "pin not found")
    return st["indexer"].index_pin(pin)


# --------------------------------------------------------------------------
# Discovery: more like this / text / object
# --------------------------------------------------------------------------
@app.get("/pins/{pin_id}/similar")
def similar_pins(pin_id: str, limit: int = 30, w_clip: float = 1.0, w_dino: float = 1.0,
                 w_object: float = 1.0, w_cosave: float = 0.7):
    st = state()
    pin = st["db"].get_pin(pin_id)
    if not pin:
        raise HTTPException(404, "pin not found")
    return find_similar(
        st["db"], st["store"], pin,
        weights={"clip": w_clip, "dino": w_dino, "object": w_object, "cosave": w_cosave},
        limit=limit,
    )


class TextQuery(BaseModel):
    query: str
    limit: int = 30


@app.post("/search/text")
def search_text(body: TextQuery):
    st = state()
    embedders = require_embeddings()
    vec = embedders["clip"].embed_text(body.query)
    return {"results": find_by_text(st["store"], vec, limit=body.limit)}


@app.post("/pins/{pin_id}/similar_by_object/{point_suffix}")
def similar_by_object(pin_id: str, point_suffix: str, limit: int = 30):
    return find_by_object(state()["store"], f"{pin_id}::{point_suffix}", limit)


# --------------------------------------------------------------------------
# Search sessions (progressive faceted search)
# --------------------------------------------------------------------------
class SessionIn(BaseModel):
    seed_pin_ids: list[str] = []
    board_id: str | None = Field(
        default=None,
        description="Start from a board (e.g. 'chairs'): seeds pins and bootstraps the first-level query from the board name + member tags",
    )


@app.post("/sessions")
def create_session(body: SessionIn):
    st = state()
    for pid in body.seed_pin_ids:
        if not st["db"].get_pin(pid):
            raise HTTPException(404, f"seed pin {pid} not found")
    if body.board_id and not any(b["id"] == body.board_id for b in st["db"].list_boards()):
        raise HTTPException(404, f"board {body.board_id} not found")
    return st["sessions"].create(body.seed_pin_ids, body.board_id)


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    sess = state()["sessions"].get(session_id)
    if not sess:
        raise HTTPException(404, "session not found")
    return sess


class SelectIn(BaseModel):
    extracted_facets: dict[str, dict] | None = Field(
        default=None,
        description="VLM facet extraction {object: {value, confidence}, style: ...}; omit to auto-derive (VLM if configured, else the pin's tags)",
    )


def _describe_fn(st: dict):
    def describe(image_url: str, pin: dict) -> dict:
        return st["indexer"].describe(image_url, pin)

    return describe


@app.post("/sessions/{session_id}/select")
def session_select(session_id: str, pin_id: str, body: SelectIn | None = None):
    st = state()
    extracted = body.extracted_facets if body and body.extracted_facets else None
    try:
        return st["sessions"].select_image(
            session_id, pin_id, extracted, describe=_describe_fn(st)
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc))


class DismissIn(BaseModel):
    pin_id: str | None = None
    pin_ids: list[str] = []


@app.post("/sessions/{session_id}/dismiss")
def session_dismiss(session_id: str, body: DismissIn):
    """Negative signal: this candidate is not interesting — evidence against
    its terms and removed from future results."""
    st = state()
    ids = body.pin_ids + ([body.pin_id] if body.pin_id else [])
    try:
        for pid in ids:
            st["sessions"].dismiss_image(session_id, pid)
        return st["sessions"].snapshot(session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/sessions/{session_id}/chips")
def session_chips(session_id: str, limit: int = 8):
    """Suggested search terms mined from the current neighborhood — helps
    the user discover vocabulary for what they are looking for."""
    try:
        return state()["sessions"].suggest_chips(session_id, limit)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


class ChipIn(BaseModel):
    facet: str
    value: str


@app.post("/sessions/{session_id}/chips")
def session_adopt_chip(session_id: str, body: ChipIn):
    try:
        return state()["sessions"].adopt_chip(session_id, body.facet, body.value)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.get("/sessions/{session_id}/query")
def session_query(session_id: str):
    """Live query assumption for the top search bar."""
    try:
        return state()["sessions"].query_preview(session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


class FacetEditIn(BaseModel):
    value: str | None
    accept: bool | None = None


@app.patch("/sessions/{session_id}/facets/{facet}")
def session_edit_facet(session_id: str, facet: str, body: FacetEditIn):
    st = state()
    try:
        return st["sessions"].edit_facet(session_id, facet, body.value, body.accept)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/sessions/{session_id}/search")
def session_search(session_id: str, limit: int = 30):
    st = state()
    try:
        return st["sessions"].local_candidates(session_id, limit)
    except KeyError as exc:
        raise HTTPException(404, str(exc))


class WebBranchIn(BaseModel):
    limit: int = 100


@app.post("/sessions/{session_id}/web")
def session_web(session_id: str, body: WebBranchIn | None = None):
    """Breadth tier: compose query from accepted facets, fetch open web
    candidates, embed them, re-rank against session seeds."""
    st = state()
    embedders = require_embeddings()
    try:
        wq = st["sessions"].web_query(session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    if not wq["query"]:
        return {**wq, "tier": "web", "results": []}
    candidates = fetch_candidates(wq["query"], limit=(body or WebBranchIn()).limit)
    enriched = []
    for c in candidates:
        try:
            img = fetch_image(c["image_url"])
            cand = dict(c)
            cand["clip"] = embedders["clip"].embed_image(img)
            if "dino" in embedders:
                cand["dino"] = embedders["dino"].embed_image(img)
            enriched.append(cand)
        except Exception:
            continue
    reranked = st["sessions"].rerank_web_results(session_id, enriched)
    return {**wq, "tier": "web", "results": reranked[:30]}


# --------------------------------------------------------------------------
# Corpus ingestion
# --------------------------------------------------------------------------
class CorpusIn(BaseModel):
    query: str
    limit: int = 100


@app.post("/corpus/ingest")
def corpus_ingest(body: CorpusIn):
    st = state()
    return ingest_corpus(
        st["db"],
        st["indexer"].index_pin,
        body.query,
        limit=body.limit,
    )


# --------------------------------------------------------------------------
# Agent API (delegation)
# --------------------------------------------------------------------------
class AgentSessionStepIn(BaseModel):
    query: str = Field(description="Free-text intent, e.g. 'walnut mid-century modern chair'")
    seed_pin_ids: list[str] = []
    deep: bool = True


@app.post("/agent/search-session", dependencies=[Depends(require_agent_key)])
def agent_search_session(body: AgentSessionStepIn):
    """Run one complete delegated search step: build/reuse a session, search
    the local tier, optionally fetch + re-rank the web branch, return results
    with the facet breadcrumb. The agent never needs the browser extension."""
    st = state()
    embedders = require_embeddings()
    session = st["sessions"].create(body.seed_pin_ids)
    state_doc = session["state"]

    vec = embedders["clip"].embed_text(body.query)
    local_hits = find_by_text(st["store"], vec, limit=30)
    seeds = state_doc["seed_pins"] + [h["pin_id"] for h in local_hits[:5]]
    for pid in seeds[:5]:
        st["sessions"].select_image(session["id"], pid, None)
    try:
        st["sessions"].edit_facet(session["id"], "object", body.query)
    except KeyError:
        pass
    web_results: list[dict] = []
    if body.deep:
        wq = st["sessions"].web_query(session["id"])
        if wq["query"]:
            candidates = fetch_candidates(wq["query"], limit=60)
            enriched = []
            for c in candidates:
                try:
                    img = fetch_image(c["image_url"])
                    cand = dict(c)
                    cand["clip"] = embedders["clip"].embed_image(img)
                    if "dino" in embedders:
                        cand["dino"] = embedders["dino"].embed_image(img)
                    enriched.append(cand)
                except Exception:
                    continue
            web_results = st["sessions"].rerank_web_results(session["id"], enriched)[:20]
    return {
        "session_id": session["id"],
        "breadcrumb": st["sessions"].web_query(session["id"])["breadcrumb"],
        "local": local_hits,
        "web": web_results,
    }


@app.get("/agent/pins", dependencies=[Depends(require_agent_key)])
def agent_list_pins(limit: int = 50):
    return state()["db"].list_pins(limit)


class AgentPinIn(BaseModel):
    image_url: str
    source_url: str = ""
    title: str = ""
    description: str = ""
    tags: list[str] = []
    board_id: str | None = None


@app.post("/agent/pins", dependencies=[Depends(require_agent_key)])
def agent_create_pin(body: AgentPinIn):
    st = state()
    pin = st["db"].create_pin(
        image_url=body.image_url,
        source_url=body.source_url,
        title=body.title,
        description=body.description,
        tags=body.tags,
    )
    if body.board_id:
        st["db"].add_pin_to_board(pin["id"], body.board_id)
    st["indexer"].index_pin(dict(pin, boards=[body.board_id] if body.board_id else []))
    return st["db"].get_pin(pin["id"])
