"""Activity log: every MCP tool call from any agent, stored in the shared SQLite database.

Each client (Claude Code, Codex, Gemini, ChatGPT) runs its own server process, so the log
lives in the database they all share; the ChipLab app reads it to show agents working live.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

MAX_RESULT_CHARS = 200_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    client TEXT,
    session TEXT,
    tool TEXT NOT NULL,
    args TEXT,
    ok INTEGER NOT NULL,
    result TEXT,
    error TEXT,
    duration_ms REAL,
    design_id TEXT,
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS activity_design ON activity(design_id, id);
"""


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def _design_ref(args: dict[str, Any], result: Any) -> tuple[str | None, str | None]:
    design = run = None
    if isinstance(result, dict):
        design = result.get("design_id")
        run = result.get("run_id")
    if not design and isinstance(args, dict):
        design = args.get("design_id_or_name")
    return (str(design) if design else None, str(run) if run else None)


@dataclass
class ActivityLog:
    db_path: str

    def __post_init__(self) -> None:
        self._conn = connect(self.db_path)

    def record(self, tool: str, args: dict[str, Any], result: Any = None,
               error: str | None = None, duration_ms: float = 0.0,
               client: str | None = None, session: str | None = None) -> int:
        design_id, run_id = _design_ref(args, result)
        text = json.dumps(result, default=str) if result is not None else None
        if text and len(text) > MAX_RESULT_CHARS:
            text = json.dumps({"truncated": True, "preview": text[:2000]})
        cur = self._conn.execute(
            "INSERT INTO activity (ts, client, session, tool, args, ok, result, error, "
            "duration_ms, design_id, run_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), client, session, tool, json.dumps(args, default=str),
             0 if error else 1, text, error, duration_ms, design_id, run_id))
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def since(self, after_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM activity WHERE id > ? ORDER BY id LIMIT ?", (after_id, limit))
        return [_row(r) for r in rows]

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,))
        return [_row(r) for r in rows][::-1]

    def latest_result(self, design_id: str, tools: tuple[str, ...]) -> dict[str, Any] | None:
        q = ",".join("?" * len(tools))
        row = self._conn.execute(
            f"SELECT * FROM activity WHERE design_id = ? AND ok = 1 AND tool IN ({q}) "
            "ORDER BY id DESC LIMIT 1", (design_id, *tools)).fetchone()
        return _row(row) if row else None

    def evaluations(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM activity WHERE tool = 'chip_evaluate' AND ok = 1 "
            "ORDER BY id DESC LIMIT ?", (limit,))
        return [_row(r) for r in rows]


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    for k in ("args", "result"):
        if d.get(k):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d


def _plain(obj: Any) -> Any:
    """Turn an MCP CallToolResult (or dict) into plain JSON-able data."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(obj, dict) and ("content" in obj or "structuredContent" in obj):
        sc = obj.get("structuredContent")
        if isinstance(sc, dict):
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        texts = [c.get("text", "") for c in obj.get("content") or [] if c.get("type") == "text"]
        joined = "\n".join(texts)
        try:
            return json.loads(joined)
        except (json.JSONDecodeError, TypeError):
            return {"text": joined[:5000]}
    return obj


class ActivityMiddleware:
    """MCP server middleware that records initialize (client name) and every tools/call."""

    def __init__(self, log: ActivityLog) -> None:
        self.log = log
        self._clients: dict[str | None, str] = {}

    async def __call__(self, ctx: Any, call_next: Any) -> Any:
        method = getattr(ctx, "method", None)
        session = getattr(ctx, "session_id", None)
        params = getattr(ctx, "params", None) or {}
        if hasattr(params, "model_dump"):
            params = params.model_dump(mode="json", by_alias=True, exclude_none=True)
        if method == "initialize":
            info = params.get("clientInfo") or {}
            self._clients[session] = str(info.get("title") or info.get("name") or "unknown")
            return await call_next(ctx)
        if method != "tools/call":
            return await call_next(ctx)
        tool = str(params.get("name", "?"))
        args = params.get("arguments") or {}
        client = self._clients.get(session) or next(iter(self._clients.values()), None)
        start = time.perf_counter()
        try:
            result = await call_next(ctx)
        except Exception as exc:
            self._safe_record(tool, args, None, f"{type(exc).__name__}: {exc}", start, client,
                              session)
            raise
        plain = _plain(result)
        is_error = bool(result.get("isError") or result.get("is_error")) \
            if isinstance(result, dict) else bool(getattr(result, "is_error", False)
                                                 or getattr(result, "isError", False))
        err = None
        if is_error:
            err = plain.get("text") if isinstance(plain, dict) else str(plain)
            plain = None
        self._safe_record(tool, args, plain, err, start, client, session)
        return result

    def _safe_record(self, tool: str, args: dict[str, Any], result: Any, error: str | None,
                     start: float, client: str | None, session: str | None) -> None:
        try:
            from chiplab.library.private_pdk import redact_args

            self.log.record(tool, redact_args(args), result, error,
                            (time.perf_counter() - start) * 1000, client, session)
        except sqlite3.Error:
            pass  # never let logging break a tool call
