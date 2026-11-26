"""SQLite persistence: pins, boards, co-save edges, search sessions.

Pure stdlib (sqlite3) so the core runs anywhere; embeddings live in the
vector store, metadata here.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS boards (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pins (
    id TEXT PRIMARY KEY,
    image_url TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    caption TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    origin TEXT NOT NULL DEFAULT 'user',          -- user | web | corpus
    indexed_at REAL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pin_boards (
    pin_id TEXT NOT NULL REFERENCES pins(id) ON DELETE CASCADE,
    board_id TEXT NOT NULL REFERENCES boards(id) ON DELETE CASCADE,
    PRIMARY KEY (pin_id, board_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    state TEXT NOT NULL DEFAULT '{}'             -- facets, accepted chain, seed pins
);
CREATE INDEX IF NOT EXISTS idx_pin_boards_board ON pin_boards(board_id);
CREATE INDEX IF NOT EXISTS idx_pin_boards_pin ON pin_boards(pin_id);
"""

DEFAULT_STATE: dict = {"facets": {}, "accepted": [], "seed_pins": [], "history": []}


class Database:
    def __init__(self, path: str = "data/openinterest.db"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = self._connect()
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # -- boards ---------------------------------------------------------------
    def create_board(self, name: str) -> str:
        bid = uuid.uuid4().hex[:12]
        with self.conn() as c:
            c.execute(
                "INSERT INTO boards (id, name, created_at) VALUES (?,?,?)",
                (bid, name, time.time()),
            )
        return bid

    def list_boards(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM boards ORDER BY created_at DESC")]

    # -- pins -----------------------------------------------------------------
    def create_pin(
        self,
        image_url: str,
        source_url: str = "",
        title: str = "",
        description: str = "",
        tags: list[str] | None = None,
        origin: str = "user",
        pin_id: str | None = None,
    ) -> dict:
        pid = pin_id or uuid.uuid4().hex[:12]
        now = time.time()
        with self.conn() as c:
            c.execute(
                "INSERT INTO pins (id, image_url, source_url, title, description, tags, origin, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (pid, image_url, source_url, title, description, json.dumps(tags or []), origin, now),
            )
        return self.get_pin(pid)

    def get_pin(self, pin_id: str) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM pins WHERE id = ?", (pin_id,)).fetchone()
            if not row:
                return None
            pin = dict(row)
            pin["tags"] = json.loads(pin["tags"])
            pin["boards"] = [
                r["board_id"] for r in c.execute(
                    "SELECT board_id FROM pin_boards WHERE pin_id = ?", (pin_id,)
                )
            ]
            return pin

    def list_pins(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT id FROM pins ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            return [p for r in rows if (p := self.get_pin(r["id"])) is not None]

    def update_pin(self, pin_id: str, **fields) -> None:
        allowed = {"title", "description", "caption", "tags", "indexed_at"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k} = ?")
                vals.append(json.dumps(v) if k == "tags" else v)
        if not sets:
            return
        vals.append(pin_id)
        with self.conn() as c:
            c.execute(f"UPDATE pins SET {', '.join(sets)} WHERE id = ?", vals)

    def delete_pin(self, pin_id: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM pins WHERE id = ?", (pin_id,))

    def add_pin_to_board(self, pin_id: str, board_id: str) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO pin_boards (pin_id, board_id) VALUES (?,?)",
                (pin_id, board_id),
            )

    def remove_pin_from_board(self, pin_id: str, board_id: str) -> None:
        with self.conn() as c:
            c.execute(
                "DELETE FROM pin_boards WHERE pin_id = ? AND board_id = ?",
                (pin_id, board_id),
            )

    def pins_in_board(self, board_id: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT p.id FROM pins p JOIN pin_boards pb ON pb.pin_id = p.id"
                " WHERE pb.board_id = ? ORDER BY p.created_at DESC",
                (board_id,),
            ).fetchall()
            return [p for r in rows if (p := self.get_pin(r["id"])) is not None]

    # -- sessions ---------------------------------------------------------------
    def create_session(self, session_id: str | None = None) -> dict:
        sid = session_id or uuid.uuid4().hex[:12]
        now = time.time()
        with self.conn() as c:
            c.execute(
                "INSERT INTO sessions (id, created_at, updated_at, state) VALUES (?,?,?,?)",
                (sid, now, now, json.dumps(DEFAULT_STATE)),
            )
        return self.get_session(sid)

    def get_session(self, session_id: str) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                return None
            return {"id": row["id"], "state": json.loads(row["state"])}

    def save_session(self, session_id: str, state: dict) -> None:
        with self.conn() as c:
            c.execute(
                "UPDATE sessions SET state = ?, updated_at = ? WHERE id = ?",
                (json.dumps(state), time.time(), session_id),
            )


def pins_missing_index(db: Database) -> Iterable[dict]:
    with db.conn() as c:
        rows = c.execute("SELECT id FROM pins WHERE indexed_at IS NULL").fetchall()
    for r in rows:
        pin = db.get_pin(r["id"])
        if pin:
            yield pin
