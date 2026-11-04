"""Research history: every run's question and events, kept on this computer only."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Status = Literal["complete", "error", "interrupted"]
_ENDINGS: dict[str, Status] = {"done": "complete", "error": "error"}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS research (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event (
    research_id INTEGER NOT NULL REFERENCES research(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    name TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (research_id, position)
);
"""


class SavedEvent(BaseModel):
    event: str
    data: str  # the SSE payload exactly as it was streamed (JSON text)


class HistoryEntry(BaseModel):
    id: int
    question: str
    created_at: str
    status: Status


class SQLiteHistory:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(self._connect()) as db:
            db.executescript(_SCHEMA)
        path.chmod(0o600)  # questions are personal: readable by this user only

    def save(self, question: str, events: list[SavedEvent]) -> int:
        status = next((_ENDINGS[e.event] for e in events if e.event in _ENDINGS), "interrupted")
        created = datetime.now(UTC).isoformat(timespec="seconds")
        with closing(self._connect()) as db, db:
            cursor = db.execute(
                "INSERT INTO research (question, created_at, status) VALUES (?, ?, ?)",
                (question, created, status),
            )
            research = cursor.lastrowid
            assert research is not None
            db.executemany(
                "INSERT INTO event VALUES (?, ?, ?, ?)",
                [(research, i, e.event, e.data) for i, e in enumerate(events)],
            )
        return research

    def entries(self, limit: int = 100) -> list[HistoryEntry]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT id, question, created_at, status FROM research ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [HistoryEntry(id=r[0], question=r[1], created_at=r[2], status=r[3]) for r in rows]

    def events(self, research: int) -> list[SavedEvent] | None:
        with closing(self._connect()) as db:
            if db.execute("SELECT 1 FROM research WHERE id = ?", (research,)).fetchone() is None:
                return None
            rows = db.execute(
                "SELECT name, data FROM event WHERE research_id = ? ORDER BY position",
                (research,),
            ).fetchall()
        return [SavedEvent(event=name, data=data) for name, data in rows]

    def delete(self, research: int) -> bool:
        with closing(self._connect()) as db, db:
            return db.execute("DELETE FROM research WHERE id = ?", (research,)).rowcount > 0

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path)
        db.execute("PRAGMA foreign_keys = ON")  # so deleting a research deletes its events
        return db
