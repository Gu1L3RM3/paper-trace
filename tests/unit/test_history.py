import stat
from pathlib import Path

from papertrace.history import SavedEvent, SQLiteHistory

EVENTS = [
    SavedEvent(event="plan", data='{"queries": []}'),
    SavedEvent(event="done", data="{}"),
]


def test_saved_research_is_listed_newest_first_and_replayable(tmp_path: Path) -> None:
    history = SQLiteHistory(tmp_path / "h.sqlite3")
    first = history.save("Does fasting help memory?", EVENTS)
    second = history.save("Is coffee good?", [SavedEvent(event="error", data='{"message": "x"}')])

    saved = history.entries()

    assert [(e.id, e.question, e.status) for e in saved] == [
        (second, "Is coffee good?", "error"),
        (first, "Does fasting help memory?", "complete"),
    ]
    assert saved[0].created_at
    assert history.events(first) == EVENTS


def test_research_without_an_ending_is_interrupted(tmp_path: Path) -> None:
    history = SQLiteHistory(tmp_path / "h.sqlite3")
    history.save("q", EVENTS[:1])
    assert history.entries()[0].status == "interrupted"


def test_unknown_entry_and_delete(tmp_path: Path) -> None:
    history = SQLiteHistory(tmp_path / "h.sqlite3")
    entry = history.save("q", EVENTS)

    assert history.events(entry + 1) is None
    assert history.delete(entry) is True
    assert history.delete(entry) is False
    assert history.entries() == [] and history.events(entry) is None


def test_history_file_is_private_and_survives_restarts(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "h.sqlite3"
    SQLiteHistory(path).save("q", EVENTS)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [e.question for e in SQLiteHistory(path).entries()] == ["q"]
