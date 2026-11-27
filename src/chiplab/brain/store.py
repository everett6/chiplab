"""Project Brain: persistent research memory with automatic recall.

Five kinds of memory:
  project       goals, constraints, decisions, rejected ideas
  architecture  designs and their lineage (see chiplab.science.genealogy)
  experiment    runs, measurements, review verdicts, known failures
  science       papers, equations, physical limits (each with a citation)
  tool          what each EDA tool / model / component can and cannot do

recall(question) scores every memory with SQLite FTS5 BM25 over title, body and tags, after
expanding the question with a small domain thesaurus (e.g. "DRAM compute" also searches
ambit, rowclone, charge sharing, pim). Results are grouped by kind so a question about
combining DRAM with computation returns past DRAM experiments, relevant papers, components
that exist, and known limitations together.

Memories carry a status: active, decision, rejected, limitation, superseded. Rejected ideas
are kept on purpose so agents stop rediscovering them.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

KINDS = ("project", "architecture", "experiment", "science", "tool")
STATUSES = ("active", "decision", "rejected", "limitation", "superseded")

_THESAURUS: dict[str, tuple[str, ...]] = {
    "dram": ("dram", "ambit", "rowclone", "charge", "sharing", "sense", "amplifier", "row",
             "activation", "capacitor", "refresh", "hbm", "pim", "computedram"),
    "sram": ("sram", "bitline", "cache", "nmc", "near", "compute", "neural", "bank"),
    "compute": ("compute", "computation", "pim", "cim", "imc", "logic", "processing", "mac",
                "bitwise", "near", "in-memory"),
    "local": ("near", "local", "bank", "in-memory", "pim"),
    "memristor": ("memristor", "reram", "rram", "crossbar", "analog", "conductance"),
    "crossbar": ("crossbar", "analog", "resistor", "memristor", "isaac", "mvm"),
    "analog": ("analog", "crossbar", "adc", "dac", "noise", "variation"),
    "energy": ("energy", "power", "pj", "landauer", "joule"),
    "movement": ("movement", "transfer", "bandwidth", "read", "write", "von", "neumann"),
    "ferrofluid": ("ferrofluid", "magnetic", "nanoparticle", "fe3o4"),
    "tapeout": ("tapeout", "tiny", "shuttle", "sky130", "gds", "fabricate", "fabrication"),
    "test": ("test", "bist", "scan", "dft", "fault", "yield"),
    "reliability": ("reliability", "disturb", "rowhammer", "retention", "endurance", "error"),
}
_WORD = re.compile(r"[a-z0-9][a-z0-9_\-]*")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "we", "can", "is",
         "are", "for", "it", "be", "how", "what", "do", "does", "this", "that", "our", "use"}


def default_db() -> Path:
    env = os.environ.get("CHIPLAB_BRAIN_DB")
    return Path(env) if env else Path.home() / ".local/share/chiplab/brain.db"


def _terms(text: str) -> list[str]:
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]
    out: list[str] = []
    for w in words:
        for t in (w, *_THESAURUS.get(w, ())):
            if t not in out:
                out.append(t)
    return out


def _fts_query(terms: Iterable[str]) -> str:
    return " OR ".join('"' + t.replace('"', "") + '"' for t in terms)


class Brain:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else default_db()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, body TEXT,
                tags TEXT, status TEXT NOT NULL DEFAULT 'active', source TEXT,
                source_key TEXT UNIQUE, data_json TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED, title, body, tags, tokenize='porter unicode61');
        """)
        self._db.commit()

    # ---- write -----------------------------------------------------------------------
    def remember(self, kind: str, title: str, body: str = "", *, tags: Iterable[str] = (),
                 status: str = "active", source: str = "", source_key: str | None = None,
                 data: dict[str, Any] | None = None) -> str:
        """Add a memory, or update it in place when source_key already exists (ingest is
        idempotent). Returns the memory id."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if not title.strip():
            raise ValueError("title is required")
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        tag_s = " ".join(sorted({t.strip().lower() for t in tags if t.strip()}))
        row = self._db.execute("SELECT id FROM memories WHERE source_key = ?",
                               (source_key,)).fetchone() if source_key else None
        mid = row["id"] if row else uuid.uuid4().hex[:12]
        dj = json.dumps(data, default=str) if data is not None else None
        if row:
            self._db.execute(
                "UPDATE memories SET kind=?, title=?, body=?, tags=?, status=?, source=?, "
                "data_json=?, updated_at=? WHERE id=?",
                (kind, title, body, tag_s, status, source, dj, now, mid))
            self._db.execute("DELETE FROM memories_fts WHERE id = ?", (mid,))
        else:
            self._db.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (mid, kind, title, body, tag_s, status, source, source_key, dj, now, now))
        self._db.execute("INSERT INTO memories_fts VALUES (?,?,?,?)", (mid, title, body, tag_s))
        self._db.commit()
        return mid

    def decide(self, title: str, rationale: str, *, rejected: bool = False,
               evidence: Iterable[str] = (), tags: Iterable[str] = ()) -> str:
        """Record a project decision or a rejected idea, with its reasons and evidence ids."""
        ev = list(evidence)
        body = rationale + (f"\nEvidence: {', '.join(ev)}" if ev else "")
        return self.remember("project", title, body, tags=tags,
                             status="rejected" if rejected else "decision",
                             source="decision", data={"evidence": ev})

    def set_status(self, memory_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        self._db.execute("UPDATE memories SET status=?, updated_at=? WHERE id=?",
                         (status, time.strftime("%Y-%m-%dT%H:%M:%S"), memory_id))
        self._db.commit()

    # ---- read ------------------------------------------------------------------------
    def get(self, memory_id: str) -> dict[str, Any] | None:
        r = self._db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _row(r) if r else None

    def recall(self, question: str, per_kind: int = 4, kinds: Iterable[str] = KINDS,
               include_superseded: bool = False) -> dict[str, Any]:
        """Return the memories most relevant to a question, grouped by kind. Limitations and
        rejected ideas are always surfaced in their own lists."""
        terms = _terms(question)
        if not terms:
            return {"question": question, "terms": [], "results": {}, "limitations": [],
                    "rejected": []}
        q = _fts_query(terms)
        rows = self._db.execute(
            "SELECT m.*, bm25(memories_fts, 0.0, 3.0, 1.0, 2.0) AS score FROM memories_fts f "
            "JOIN memories m ON m.id = f.id WHERE memories_fts MATCH ? ORDER BY score LIMIT 400",
            (q,)).fetchall()
        kinds = tuple(kinds)
        out: dict[str, list[dict[str, Any]]] = {k: [] for k in kinds}
        lim, rej = [], []
        for r in rows:
            if r["status"] == "superseded" and not include_superseded:
                continue
            item = _row(r) | {"score": round(-float(r["score"]), 3)}
            if r["status"] == "limitation" and len(lim) < per_kind * 2:
                lim.append(item)
            elif r["status"] == "rejected" and len(rej) < per_kind * 2:
                rej.append(item)
            if r["kind"] in out and len(out[r["kind"]]) < per_kind:
                out[r["kind"]].append(item)
        return {"question": question, "terms": terms,
                "results": {k: v for k, v in out.items() if v},
                "limitations": lim, "rejected": rej}

    def stats(self) -> dict[str, Any]:
        rows = self._db.execute(
            "SELECT kind, status, COUNT(*) n FROM memories GROUP BY kind, status").fetchall()
        by: dict[str, dict[str, int]] = {}
        for r in rows:
            by.setdefault(r["kind"], {})[r["status"]] = r["n"]
        return {"db": str(self.db_path), "by_kind": by,
                "total": sum(sum(v.values()) for v in by.values())}


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = {k: r[k] for k in ("id", "kind", "title", "body", "status", "source", "created_at")}
    d["tags"] = (r["tags"] or "").split()
    d["data"] = json.loads(r["data_json"]) if r["data_json"] else None
    return d
