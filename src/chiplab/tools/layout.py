"""Layout Studio: read-only GDS/OASIS viewing through KLayout's own renderer.

Users upload a layout file; ChipLab stores it under a content-hash name (never the client's
file name), opens it with the ``klayout`` Python package (KLayout's standalone LayoutView,
no GUI), lists its layers and renders PNG images with the chosen layers visible. Files are
never modified or written back.
"""

from __future__ import annotations

import base64
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any

MAX_BYTES = 64 * 1024 * 1024
EXTS = {".gds": "GDSII", ".gds2": "GDSII", ".gdsii": "GDSII", ".oas": "OASIS",
        ".oasis": "OASIS"}
_GDS_MAGIC = b"\x00\x06\x00\x02"          # HEADER record
_OAS_MAGIC = b"%SEMI-OASIS\r\n"
_ID = re.compile(r"^[0-9a-f]{20}\.(gds|oas)$")


class LayoutError(ValueError):
    pass


def available() -> bool:
    try:
        import klayout.db  # noqa: F401
        import klayout.lay  # noqa: F401
    except ImportError:
        return False
    return True


def store_upload(root: Path, filename: str, data: bytes) -> str:
    """Validate and store an uploaded layout; returns its id (hash + canonical extension)."""
    if not data:
        raise LayoutError("empty file")
    if len(data) > MAX_BYTES:
        raise LayoutError(f"layout larger than {MAX_BYTES // 2**20} MB")
    ext = Path(filename or "").suffix.lower()
    if data.startswith(_OAS_MAGIC):
        kind = "oas"
    elif data[:4] == _GDS_MAGIC:
        kind = "gds"
    else:
        raise LayoutError("not a GDSII or OASIS file (bad header)")
    if ext and ext not in EXTS:
        raise LayoutError(f"unsupported extension {ext!r}; use .gds or .oas")
    lid = f"{hashlib.sha256(data).hexdigest()[:20]}.{kind}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / lid
    if not path.exists():
        path.write_bytes(data)
    return lid


def _path(root: Path, lid: str) -> Path:
    if not _ID.match(lid or ""):
        raise LayoutError("bad layout id")
    p = root / lid
    if not p.is_file():
        raise LayoutError("layout not found; upload it again")
    return p


def info(root: Path, lid: str) -> dict[str, Any]:
    import klayout.db as kdb
    ly = kdb.Layout()
    ly.read(str(_path(root, lid)))
    tops = [c.name for c in ly.top_cells()]
    top = ly.top_cells()[0] if tops else None
    layers = []
    for li in ly.layer_indexes():
        lp = ly.get_info(li)
        n = 0
        if top is not None:
            it = top.begin_shapes_rec(li)
            while not it.at_end() and n < 200000:
                n += 1
                it.next()
        layers.append({"index": li, "layer": lp.layer, "datatype": lp.datatype,
                       "name": lp.name or "", "shapes": n, "capped": n >= 200000})
    bbox = top.dbbox() if top is not None else None
    return {"id": lid, "format": "OASIS" if lid.endswith(".oas") else "GDSII",
            "dbu_um": ly.dbu, "top_cells": tops[:20], "cells": ly.cells(),
            "bbox_um": [bbox.left, bbox.bottom, bbox.right, bbox.top] if bbox else None,
            "layers": layers}


def render(root: Path, lid: str, layers: list[str] | None = None, width: int = 1200,
           height: int = 800) -> dict[str, Any]:
    """PNG of the layout from KLayout's LayoutView; ``layers`` like ["1/0", "2/0"]."""
    import klayout.lay as lay
    width = max(200, min(int(width), 2400))
    height = max(200, min(int(height), 2400))
    want = None
    if layers:
        want = set()
        for s in layers:
            if not re.fullmatch(r"\d{1,5}/\d{1,5}", s):
                raise LayoutError(f"bad layer spec {s!r}; use layer/datatype like 1/0")
            want.add(s)
    lv = lay.LayoutView()
    lv.load_layout(str(_path(root, lid)), True)
    lv.max_hier()
    it = lv.begin_layers()
    while not it.at_end():
        node = it.current()
        src = node.source.split("@")[0]
        if want is not None:
            node.visible = src in want
        it.next()
    lv.zoom_fit()
    with tempfile.TemporaryDirectory(prefix="chiplab-klayout-") as tmp:
        png = Path(tmp) / "layout.png"
        lv.save_image(str(png), width, height)
        data = png.read_bytes()
    return {"png_b64": base64.b64encode(data).decode(), "width": width, "height": height,
            "layers": sorted(want) if want is not None else "all"}


# Layer map for the floorplan export: package-level shapes, not mask layers.
FLOORPLAN_LAYERS = {"interposer": (1, 0), "compute_die": (10, 0), "hbm_stack": (20, 0),
                    "io_die": (30, 0), "other": (40, 0), "link": (50, 0), "label": (63, 0)}


def floorplan_gds(root: Path, scene: dict[str, Any]) -> str:
    """Write the studio package floorplan (interposer, dies, links) as GDSII via klayout.db.

    The result goes through the same upload path, so it is viewed exactly like a user file.
    Coordinates are millimetres in the scene, written in micrometres (dbu 1 nm).
    """
    import klayout.db as kdb
    ly = kdb.Layout()
    ly.dbu = 0.001
    top = ly.create_cell(re.sub(r"[^A-Za-z0-9_]", "_", scene["design"]["name"])[:60] or "TOP")
    li = {k: ly.layer(kdb.LayerInfo(a, b, k)) for k, (a, b) in FLOORPLAN_LAYERS.items()}
    ip = scene["package"]["interposer"]
    top.shapes(li["interposer"]).insert(kdb.DBox(
        (ip["x"] - ip["w"] / 2) * 1e3, (ip["y"] - ip["d"] / 2) * 1e3,
        (ip["x"] + ip["w"] / 2) * 1e3, (ip["y"] + ip["d"] / 2) * 1e3))
    pos = {}
    for c in scene["components"]:
        layer = li.get(c["kind"], li["other"])
        x0, y0 = (c["x"] - c["w"] / 2) * 1e3, (c["y"] - c["d"] / 2) * 1e3
        top.shapes(layer).insert(kdb.DBox(x0, y0, x0 + c["w"] * 1e3, y0 + c["d"] * 1e3))
        top.shapes(li["label"]).insert(kdb.DText(c["id"], kdb.DTrans(c["x"] * 1e3, c["y"] * 1e3)))
        pos[c["id"]] = (c["x"] * 1e3, c["y"] * 1e3)
    for link in scene.get("links", []):
        a, b = pos.get(link.get("from")), pos.get(link.get("to"))
        if a and b:
            top.shapes(li["link"]).insert(kdb.DPath([kdb.DPoint(*a), kdb.DPoint(*b)], 300.0))
    with tempfile.TemporaryDirectory(prefix="chiplab-gds-") as tmp:
        f = Path(tmp) / "floorplan.gds"
        ly.write(str(f))
        return store_upload(root, "floorplan.gds", f.read_bytes())
