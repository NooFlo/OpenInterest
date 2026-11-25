# OpenInterest

Self-hosted, ad-free, Pinterest-grade visual discovery. Pin images from the web into boards, get "more like this" at Pinterest quality, drill down with progressive faceted search sessions, and hand search sessions to an agent via a keyed API.

## Why it can match Pinterest at personal scale

Pinterest's related-pins pipeline is multi-signal candidate generation → a blended ranking → diversity demotion. Their advantage is data volume, not architecture. OpenInterest replaces their learned signals with open pretrained models plus your own data and simple fusion math:

| Pinterest signal | OpenInterest equivalent |
|---|---|
| Unified visual embeddings (whole-image CNN) | OpenCLIP ViT-B/32 (or SigLIP) — semantic image + free-text search |
| Per-object embeddings ("tap the lamp") | sam-clip (Grounding DINO + SAM + CLIP), optional |
| PinSage co-save graph (Pixie random walks) | Co-save SQL over your `pin × board` tables, zero ML |
| Learned RankSVM blend | Weighted reciprocal-rank fusion (WRRF) + MMR diversity pass |
| Web-scale corpus | Bulk-indexed open corpora (Openverse, Wikimedia Commons) via search APIs |

Signals are independent and optional — fusion degrades gracefully when a signal is absent (no GPU, no API key, empty boards).

## Features

- **Boards & pins**: masonry-ready CRUD API; pins proxy-display the source image with attribution recorded.
- **"More like this"**: `GET /pins/{id}/similar` — parallel candidate generation (CLIP neighbors, DINOv2 neighbors, object neighbors, co-save board-mates), weighted RRF fusion, diversity demotion.
- **Text-to-image search**: `POST /search/text` via the CLIP text encoder.
- **Object-level search** (Phase 2, opt-in): per-object vectors with bbox + label; tap a region in a pin → similar objects → their pins.
- **Progressive faceted search sessions** (the differentiating feature), built for the user who doesn't yet know what they're looking for:
  - **Convergence by selection order**: every selected image is facet-extracted by a VLM into `{object, style, material, era, color}`; facet values are re-estimated from the whole trajectory with exponential recency decay, so later selections can *revise* earlier hypotheses instead of the first image locking the query.
  - **Negative evidence**: dismissing a candidate records evidence against its terms and removes it from future local and web results.
  - **Suggested search terms (chips)**: `GET /sessions/{id}/chips` mines term candidates from the current unselected neighborhood — helping the user discover the vocabulary of what they're circling around; adopting a chip is a normal (confidence-1.0) facet edit.
  - **Folders seed the first query**: start a session from a board ("chairs") and the board name seeds the object facet while member pins' tags/captions cast the initial votes — the neighborhood opens onto the user's folder, and selecting interesting images shows more of that board's style/tags.
  - **Live search bar**: `GET /sessions/{id}/query` always exposes the current web-query assumption as it refines; the bundled UI fills it into the top search bar after every interaction (editable, removable via breadcrumb chips).
  - **Tiered handoff**: local fast tier first; breadth tier composes an anti-drift-capped query from accepted facets, fetches open-licensed candidates, embeds and re-ranks them against your session seed images. Every web image you pin from a session is indexed locally — each search path physically grows your universe.
- **Swipe mode (mobile)**: `POST /sessions {initial_query}` seeds a session from a broad term ("chairs"), and the UI's ☰ Swipe mode shows one fullscreen image at a time — swipe right = interesting (refines the query), left = not interesting (negative evidence). Local candidates feed the deck first; when it runs dry the deck auto-fills from the web tier, and right-swiping a web image pins it so the universe grows along the search.
- **Web UI**: `GET /` serves a dependency-free single page — top search bar with the live query assumption, breadcrumb chips, suggested-term chips, image grid where a click selects ("this is interesting") and ✕ dismisses.
- **Corpus ingestion**: `POST /corpus/ingest {query}` pulls, embeds and indexes Openverse/Wikimedia search results as `corpus` pins (respects source rate limits; Openverse used strictly as a search API per its terms).
- **Agent API**: everything under `/agent/*` with `X-Agent-Key` — an agent (MCP-style, like community `bowrd-mcp`) can run full search sessions, list pins, and pin results on your behalf.

## Fast start

```bash
cp .env.example .env          # set MISTRAL_API_KEY (captions/facets) and OPENINTEREST_AGENT_KEY
docker compose up -d --build  # Qdrant + API bound to localhost only
# open http://localhost:8000/docs
```

Without Docker (CPU-only, no Qdrant — falls back to a built-in in-memory vector store):

```bash
pip install -e ".[ml,dev]"    # [ml] pulls torch/open_clip; core also runs without it
export MISTRAL_API_KEY=...   # optional; falls back to title/description tags
uvicorn openinterest.api:app --host 127.0.0.1 --port 8000
```

### Configuration

See `.env.example`: vector store (`QDRANT_URL`, falls back to in-memory), caption/facet backend (`MISTRAL_API_KEY` → Pixtral; unset → heuristic), models (`OPENCLIP_MODEL`, `DINOV2_MODEL`), and `ENABLE_OBJECT_EMBEDDINGS=1` for Phase 2 (slow on CPU; use a GPU).

## Architecture

```
save pin ──▶ Indexer ──▶ CLIP vec ──▶ Qdrant 'clip'
                      ──▶ DINOv2 vec ─► Qdrant 'dino'
                      ──▶ sam-clip region vecs ──▶ Qdrant 'objects'   (optional)
                      ──▶ Pixtral caption/tags/facets ──▶ SQLite pins

GET /pins/{id}/similar ──▶ 4 ranked lists (clip, dino, object, cosave SQL)
                        ──▶ weighted RRF fusion ──▶ MMR diversity pass ──▶ ranked results

session select ──▶ VLM facets {object,style,material,era,color} + confidence
              ──▶ promote ≥ threshold ──▶ editable breadcrumb
              ──▶ local tier: index lookup by seed vectors
              ──▶ web tier: facet-capped query → open search APIs → embed → re-rank vs seeds
```

- `openinterest/db.py` — SQLite schema: pins, boards, pin_boards (co-save edges), sessions
- `openinterest/embeddings.py` — OpenCLIP / DINOv2 / sam-clip backends (lazy heavy imports)
- `openinterest/indexer.py` — indexing pipeline
- `openinterest/vectorstore.py` / `qdrant_store.py` — store protocol, Qdrant + in-memory backends
- `openinterest/fusion.py` — WRRF + diversity (pure Python, heavily tested)
- `openinterest/cosave.py` — co-save candidate SQL
- `openinterest/caption.py` / `facets.py` — Pixtral facet extraction, promotion thresholds, breadcrumb, query caps
- `openinterest/sessions.py` / `webbranch.py` / `corpus.py` — session engine, web branch, corpus ingest
- `openinterest/api.py` — FastAPI surface + agent auth

## Product choices per the research handoff

- **No fine-tuning** of CLIP/DINOv2 on synthetic triplets (retrieved-real-beats-synthetic; model collapse). The only learned component is the fusion weights, tunable from your own vetted judgments.
- **No synthetic engagement logs**; the co-save graph is your genuine data (Pinterest export imports encouraged — Bowrd's Pinterest importer can seed a frontend).
- **Politeness**: keyless search APIs only, per-source rate limits, attribution stored per pin, default bindings to localhost.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Pure-Python tests cover fusion, diversity, co-save, facet promotion/edits/caps, session flow (local tier, web re-ranking anti-drift), and the API incl. agent auth — no models or services needed.
