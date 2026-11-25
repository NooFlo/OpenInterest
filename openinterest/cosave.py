"""Co-save graph candidate generation: pure SQL over the user's own pin x board
tables, mirroring Pinterest's board co-occurrence signal (Pixie-lite). No ML.
"""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .db import Database


def cosave_candidates(db: "Database", seed_pin_id: str, limit: int = 50) -> list[str]:
    """Pins sharing boards with the seed, ranked by weighted co-occurrence.

    Weight = sum over shared boards of 1/board_size (small, focused boards count
    more), which is the standard co-save normalization on a pin-board bipartite
    graph.
    """
    with db.conn() as c:
        try:
            rows = c.execute(
                """
                SELECT pb2.pin_id AS pid, SUM(1.0 / (SELECT COUNT(*) FROM pin_boards WHERE board_id = pb2.board_id)) AS w
                FROM pin_boards pb1
                JOIN pin_boards pb2 ON pb1.board_id = pb2.board_id
                WHERE pb1.pin_id = ?
                  AND pb2.pin_id != ?
                GROUP BY pb2.pin_id
                ORDER BY w DESC
                LIMIT ?
                """,
                (seed_pin_id, seed_pin_id, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [r["pid"] for r in rows]
