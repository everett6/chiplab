"""Architecture genealogy: every architecture has a parent, and every change records what
changed, why, what improved, what got worse, which assumptions changed, and which experiments
support it. Rejected branches stay in the tree, and `check_novelty` warns before an agent
re-proposes a design that was already rejected.

Stored in the Project Brain database (tables arch_nodes, arch_changes) and mirrored into the
brain as "architecture" memories so recall finds them.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Iterable

from chiplab.brain.store import Brain

STATUSES = ("active", "rejected", "superseded", "built")


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:24]


class Genealogy:
    def __init__(self, brain: Brain | None = None):
        self.brain = brain or Brain()
        self._db = self.brain._db
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS arch_nodes (
                id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, parent_id TEXT,
                status TEXT NOT NULL, spec_json TEXT, spec_hash TEXT, technology TEXT,
                created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS arch_changes (
                id TEXT PRIMARY KEY, arch_id TEXT NOT NULL, what_changed TEXT, why TEXT,
                improved_json TEXT, worsened_json TEXT, assumptions_json TEXT,
                evidence_json TEXT, created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_arch_hash ON arch_nodes(spec_hash);
        """)
        self._db.commit()

    def _node(self, key: str) -> dict[str, Any]:
        r = self._db.execute("SELECT * FROM arch_nodes WHERE id = ? OR name = ?",
                             (key, key)).fetchone()
        if r is None:
            raise KeyError(f"no architecture {key!r}")
        return dict(r) | {"spec": json.loads(r["spec_json"] or "{}")}

    def create(self, name: str, spec: dict[str, Any], *, technology: str = "",
               why: str = "root architecture") -> str:
        return self._add(name, None, spec, technology, what="created", why=why)

    def derive(self, parent: str, name: str, spec: dict[str, Any], *, why: str,
               improved: dict[str, Any] | None = None, worsened: dict[str, Any] | None = None,
               assumptions_changed: Iterable[str] = (), evidence: Iterable[str] = (),
               technology: str | None = None) -> dict[str, Any]:
        """Add a child. `why` is required; if neither improved nor worsened is given the
        change is recorded but flagged as unjustified by evidence."""
        if not why.strip():
            raise ValueError("why is required")
        p = self._node(parent)
        novelty = self.check_novelty(spec)
        diff = _diff(p["spec"], spec)
        aid = self._add(name, p["id"], spec, technology if technology is not None
                        else p["technology"], what=json.dumps(diff), why=why,
                        improved=improved, worsened=worsened,
                        assumptions=list(assumptions_changed), evidence=list(evidence))
        warn = []
        if not (improved or worsened):
            warn.append("no measured improvement or regression recorded")
        if not list(evidence):
            warn.append("no supporting experiment ids")
        return {"id": aid, "diff": diff, "warnings": warn, "novelty": novelty}

    def _add(self, name: str, parent_id: str | None, spec: dict[str, Any], tech: str, *,
             what: str, why: str, improved: Any = None, worsened: Any = None,
             assumptions: Any = None, evidence: Any = None) -> str:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        aid = uuid.uuid4().hex[:10]
        self._db.execute("INSERT INTO arch_nodes VALUES (?,?,?,?,?,?,?,?)",
                         (aid, name, parent_id, "active", json.dumps(spec, default=str),
                          spec_hash(spec), tech, now))
        self._db.execute("INSERT INTO arch_changes VALUES (?,?,?,?,?,?,?,?,?)",
                         (uuid.uuid4().hex[:10], aid, what, why, json.dumps(improved),
                          json.dumps(worsened), json.dumps(assumptions), json.dumps(evidence),
                          now))
        self._db.commit()
        self.brain.remember("architecture", f"Architecture {name}",
                            f"{why}. Spec: {json.dumps(spec, default=str)[:1500]}",
                            tags=("genealogy", tech, name), source="genealogy",
                            source_key=f"arch:{aid}", data={"arch_id": aid,
                                                            "parent": parent_id})
        return aid

    def reject(self, key: str, reason: str, evidence: Iterable[str] = ()) -> None:
        n = self._node(key)
        self._db.execute("UPDATE arch_nodes SET status='rejected' WHERE id=?", (n["id"],))
        self._db.execute("INSERT INTO arch_changes VALUES (?,?,?,?,?,?,?,?,?)",
                         (uuid.uuid4().hex[:10], n["id"], "rejected", reason, "null", "null",
                          "null", json.dumps(list(evidence)),
                          time.strftime("%Y-%m-%dT%H:%M:%S")))
        self._db.commit()
        self.brain.remember("architecture", f"Architecture {n['name']}",
                            f"REJECTED: {reason}. Spec: {json.dumps(n['spec'])[:1500]}",
                            tags=("genealogy", n["technology"], n["name"], "rejected"),
                            status="rejected", source="genealogy",
                            source_key=f"arch:{n['id']}")

    def set_status(self, key: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        n = self._node(key)
        self._db.execute("UPDATE arch_nodes SET status=? WHERE id=?", (status, n["id"]))
        self._db.commit()

    def check_novelty(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Has this exact spec been tried? Returns prior nodes with their status and the
        reason if rejected."""
        rows = self._db.execute("SELECT * FROM arch_nodes WHERE spec_hash = ?",
                                (spec_hash(spec),)).fetchall()
        prior = []
        for r in rows:
            reason = self._db.execute(
                "SELECT why FROM arch_changes WHERE arch_id=? AND what_changed='rejected' "
                "ORDER BY created_at DESC LIMIT 1", (r["id"],)).fetchone()
            prior.append({"id": r["id"], "name": r["name"], "status": r["status"],
                          "rejected_because": reason["why"] if reason else None})
        return {"novel": not prior, "previously_tried": prior,
                "warning": ("this exact design was rejected before: "
                            + "; ".join(p["rejected_because"] or "" for p in prior
                                        if p["status"] == "rejected"))
                if any(p["status"] == "rejected" for p in prior) else None}

    def lineage(self, key: str) -> list[dict[str, Any]]:
        out = []
        n: dict[str, Any] | None = self._node(key)
        while n:
            ch = [dict(r) for r in self._db.execute(
                "SELECT what_changed, why, improved_json, worsened_json, assumptions_json, "
                "evidence_json, created_at FROM arch_changes WHERE arch_id=? ORDER BY "
                "created_at", (n["id"],))]
            out.append({"id": n["id"], "name": n["name"], "status": n["status"],
                        "changes": [{"what": c["what_changed"], "why": c["why"],
                                     "improved": json.loads(c["improved_json"] or "null"),
                                     "worsened": json.loads(c["worsened_json"] or "null"),
                                     "assumptions": json.loads(c["assumptions_json"] or "null"),
                                     "evidence": json.loads(c["evidence_json"] or "null")}
                                    for c in ch]})
            n = self._node(n["parent_id"]) if n["parent_id"] else None
        return out[::-1]

    def tree(self) -> dict[str, Any]:
        rows = [dict(r) for r in self._db.execute(
            "SELECT id, name, parent_id, status, technology FROM arch_nodes ORDER BY created_at")]
        kids: dict[str | None, list[dict[str, Any]]] = {}
        for r in rows:
            kids.setdefault(r["parent_id"], []).append(r)

        def build(r: dict[str, Any]) -> dict[str, Any]:
            return {"name": r["name"], "status": r["status"], "technology": r["technology"],
                    "children": [build(c) for c in kids.get(r["id"], [])]}

        roots = [build(r) for r in kids.get(None, [])]
        lines: list[str] = []

        def draw(node: dict[str, Any], prefix: str) -> None:
            mark = " [rejected]" if node["status"] == "rejected" else ""
            lines.append(f"{prefix}{node['name']}{mark}")
            for c in node["children"]:
                draw(c, prefix + "  ")

        for r in roots:
            draw(r, "")
        return {"roots": roots, "text": "\n".join(lines), "count": len(rows)}


def _diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = set(a) | set(b)
    return {k: {"from": a.get(k), "to": b.get(k)} for k in sorted(keys) if a.get(k) != b.get(k)}
