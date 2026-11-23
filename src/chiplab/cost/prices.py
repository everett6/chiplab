"""Price lookup: built-in references plus dated, sourced overrides ("live" prices).

Overrides live in a JSON file (``CHIPLAB_PRICE_FILE``, default ``~/.chiplab/prices.json``).
An AI client with web search can look a price up and store it with ``set_override``;
an optional JSON feed (``CHIPLAB_PRICE_FEED_URL``) can be pulled with ``refresh_from_feed``.
Overrides older than ``STALE_DAYS`` are flagged so estimates never silently rot.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .reference import BUILTIN_REFERENCE

STALE_DAYS = 180


def price_file() -> Path:
    return Path(os.environ.get("CHIPLAB_PRICE_FILE", Path.home() / ".chiplab" / "prices.json"))


def load_overrides() -> dict[str, dict[str, Any]]:
    path = price_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_overrides(data: dict[str, dict[str, Any]]) -> None:
    path = price_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def _age_days(as_of: str) -> int | None:
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return (date.today() - datetime.strptime(as_of, fmt).date()).days
        except (ValueError, TypeError):
            continue
    return None


def get(key: str) -> dict[str, Any]:
    """Return the effective entry for ``key`` (override wins), with age and staleness."""
    base = BUILTIN_REFERENCE.get(key)
    override = load_overrides().get(key)
    if base is None and override is None:
        raise KeyError(f"Unknown price key '{key}'. Use chip_cost_references to list keys.")
    entry = dict(base or {})
    if override:
        entry.update(override)
        # A single sourced number keeps the built-in spread around it unless a range was given.
        if "low" not in override and base:
            ratio_lo = base["low"] / base["value"] if base["value"] else 1.0
            ratio_hi = base["high"] / base["value"] if base["value"] else 1.0
            entry["low"] = override["value"] * ratio_lo
            entry["high"] = override["value"] * ratio_hi
        entry["origin"] = "override"
    else:
        entry["origin"] = "builtin"
    age = _age_days(entry.get("as_of", ""))
    entry["age_days"] = age
    entry["stale"] = age is None or age > STALE_DAYS
    return entry


def value(key: str, scenario: str = "value") -> float:
    """Return the number for ``key``; scenario is 'value', 'low' or 'high'."""
    return float(get(key)[scenario])


def set_override(key: str, new_value: float, source: str, as_of: str | None = None,
                 low: float | None = None, high: float | None = None,
                 unit: str | None = None) -> dict[str, Any]:
    """Store a sourced price. Unknown keys are allowed only with a unit (new item)."""
    if not source or not source.strip():
        raise ValueError("source is required (URL or citation) so the number stays traceable.")
    if new_value <= 0:
        raise ValueError("value must be positive.")
    if key not in BUILTIN_REFERENCE and not unit:
        raise ValueError(f"'{key}' is not a built-in key; pass unit= to add a new item.")
    if low is not None and high is not None and not (low <= new_value <= high):
        raise ValueError("Expected low <= value <= high.")
    as_of = as_of or date.today().isoformat()
    if _age_days(as_of) is None:
        raise ValueError("as_of must be YYYY-MM-DD or YYYY-MM.")
    rec: dict[str, Any] = {"value": float(new_value), "source": source.strip(),
                           "as_of": as_of, "provenance": "published"}
    if low is not None:
        rec["low"] = float(low)
    if high is not None:
        rec["high"] = float(high)
    if unit:
        rec["unit"] = unit
    data = load_overrides()
    data[key] = rec
    _save_overrides(data)
    return get(key)


def clear_override(key: str) -> bool:
    data = load_overrides()
    if key in data:
        del data[key]
        _save_overrides(data)
        return True
    return False


def list_entries(prefix: str = "") -> dict[str, dict[str, Any]]:
    keys = set(BUILTIN_REFERENCE) | set(load_overrides())
    return {k: get(k) for k in sorted(keys) if k.startswith(prefix)}


def lookup_plan(prefix: str = "", only_stale: bool = True) -> list[dict[str, Any]]:
    """Web-search queries the AI client should run to refresh prices, then call set_override."""
    plan = []
    for k, e in list_entries(prefix).items():
        if not e.get("lookup_query") or (only_stale and not e["stale"]):
            continue
        plan.append({"key": k, "search_query": e["lookup_query"], "unit": e.get("unit"),
                     "current_value": e["value"], "as_of": e.get("as_of"),
                     "origin": e["origin"]})
    return plan


def refresh_from_feed(url: str | None = None, timeout: float = 10.0) -> dict[str, Any]:
    """Pull overrides from a JSON feed: {key: {value, source, as_of, [low, high, unit]}}."""
    url = url or os.environ.get("CHIPLAB_PRICE_FEED_URL")
    if not url:
        return {"updated": [], "skipped": [], "error":
                "No feed configured. Set CHIPLAB_PRICE_FEED_URL or pass url, or use web "
                "search with chip_price_lookup_plan + chip_set_price."}
    if not url.startswith("https://"):
        return {"updated": [], "skipped": [], "error": "Feed URL must use https."}
    req = urllib.request.Request(url, headers={"User-Agent": "chiplab-price-feed/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https enforced)
        feed = json.loads(resp.read(2_000_000))
    updated, skipped = [], []
    for key, rec in (feed.items() if isinstance(feed, dict) else []):
        try:
            set_override(key, float(rec["value"]), str(rec.get("source") or url),
                         rec.get("as_of"), rec.get("low"), rec.get("high"), rec.get("unit"))
            updated.append(key)
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append({"key": key, "reason": str(exc)})
    return {"updated": updated, "skipped": skipped, "error": None}
