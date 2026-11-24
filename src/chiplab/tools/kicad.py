"""Board Studio: export the studio PCB as a real ``.kicad_pcb`` and let KiCad render it.

The board, every part and every trace come from ``scene["pcb"]`` (built by
``studio/scene.py`` from the design: VRM phases from package power and core voltage, PCIe
lanes, 12 V connectors). The file uses the KiCad 7 s-expression format (version 20221018),
which KiCad 9 opens and upgrades itself. Footprints are written inline (no library lookup),
so the file is self-contained. With ``kicad-cli`` present ChipLab asks KiCad for:

* an SVG plot of the copper, silkscreen and outline layers (``pcb export svg``),
* a 3D render to PNG (``pcb render``),
* a DRC report (``pcb drc --format json``), shown as-is.

Placement is the studio's coarse automatic placement, not a routed board: expect KiCad's DRC
to report unconnected items and clearance issues. Those counts are KiCad's, not ChipLab's.
"""

from __future__ import annotations

import base64
import json
import math
import tempfile
import uuid
from pathlib import Path
from typing import Any

from chiplab.tools import registry

ORIGIN = 50.0   # mm offset so the board sits on the sheet

LAYERS = """  (layers
    (0 "F.Cu" signal)
    (1 "In1.Cu" power)
    (2 "In2.Cu" power)
    (31 "B.Cu" signal)
    (32 "B.Adhes" user "B.Adhesive")
    (33 "F.Adhes" user "F.Adhesive")
    (34 "B.Paste" user)
    (35 "F.Paste" user)
    (36 "B.SilkS" user "B.Silkscreen")
    (37 "F.SilkS" user "F.Silkscreen")
    (38 "B.Mask" user)
    (39 "F.Mask" user)
    (40 "Dwgs.User" user "User.Drawings")
    (41 "Cmts.User" user "User.Comments")
    (44 "Edge.Cuts" user)
    (45 "Margin" user)
    (46 "B.CrtYd" user "B.Courtyard")
    (47 "F.CrtYd" user "F.Courtyard")
    (48 "B.Fab" user)
    (49 "F.Fab" user)
  )"""

SMD = '(layers "F.Cu" "F.Paste" "F.Mask")'
THT = '(layers "*.Cu" "*.Mask")'


def _u() -> str:
    return str(uuid.uuid4())


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


class _Nets:
    def __init__(self) -> None:
        self.ids: dict[str, int] = {"": 0}

    def __call__(self, name: str) -> str:
        if name not in self.ids:
            self.ids[name] = len(self.ids)
        return f"(net {self.ids[name]} {_q(name)})"

    def decl(self) -> str:
        return "\n".join(f"  (net {i} {_q(n)})" for n, i in self.ids.items())


def _pad(num: str | int, x: float, y: float, w: float, h: float, net: str,
         shape: str = "rect") -> str:
    return (f'    (pad {_q(str(num))} smd {shape} (at {x:.3f} {y:.3f}) (size {w:.3f} {h:.3f}) '
            f'{SMD} {net} (tstamp {_u()}))')


def _footprint(lib: str, ref: str, value: str, x: float, y: float, w: float, d: float,
               pads: list[str]) -> str:
    hw, hd = w / 2, d / 2
    return "\n".join([
        f'  (footprint {_q("ChipLab:" + lib)} (layer "F.Cu") (tstamp {_u()}) '
        f'(at {x:.3f} {y:.3f})',
        f'    (fp_text reference {_q(ref)} (at 0 {-hd - 1.2:.3f}) (layer "F.SilkS") '
        f'(effects (font (size 1 1) (thickness 0.15))) (tstamp {_u()}))',
        f'    (fp_text value {_q(value)} (at 0 {hd + 1.2:.3f}) (layer "F.Fab") '
        f'(effects (font (size 1 1) (thickness 0.15))) (tstamp {_u()}))',
        f'    (fp_rect (start {-hw:.3f} {-hd:.3f}) (end {hw:.3f} {hd:.3f}) '
        f'(stroke (width 0.12) (type solid)) (fill none) (layer "F.SilkS") (tstamp {_u()}))',
        f'    (fp_rect (start {-hw - 0.25:.3f} {-hd - 0.25:.3f}) (end {hw + 0.25:.3f} '
        f'{hd + 0.25:.3f}) (stroke (width 0.05) (type solid)) (fill none) (layer "F.CrtYd") '
        f'(tstamp {_u()}))',
        *pads, "  )"])


def _part(p: dict[str, Any], net: _Nets, refs: dict[str, int]) -> str:
    kind, w, d = p["kind"], float(p["w"]), float(p["d"])
    x, y = ORIGIN + float(p["x"]), ORIGIN + float(p["y"])
    prefix = {"power_stage": "U", "inductor": "L", "capacitor": "C", "power_connector": "J",
              "edge_connector": "J"}.get(kind, "X")
    refs[prefix] = refs.get(prefix, 0) + 1
    ref = f"{prefix}{refs[prefix] + (1 if prefix == 'U' else 0)}"
    phase = (p.get("metrics") or {}).get("phase")
    pads: list[str] = []
    if kind == "power_stage":
        pads = [_pad(1, -w / 2 + 0.6, -d / 4, 0.8, d / 3, net("+12V")),
                _pad(2, -w / 2 + 0.6, d / 4, 0.8, d / 3, net("GND")),
                _pad(3, w / 2 - 0.6, 0, 0.8, d * 0.7, net(f"SW{phase}")),
                _pad(4, 0, 0, w * 0.4, d * 0.5, net("GND"))]
    elif kind == "inductor":
        pads = [_pad(1, -w / 2 + 1.2, 0, 2.0, d * 0.8, net(f"SW{phase}")),
                _pad(2, w / 2 - 1.2, 0, 2.0, d * 0.8, net("VCORE"))]
    elif kind == "capacitor":
        pads = [_pad(1, -w / 4, 0, w / 2.6, d * 0.8, net("VCORE")),
                _pad(2, w / 4, 0, w / 2.6, d * 0.8, net("GND"))]
    elif kind == "power_connector":
        for i in range(8):
            cx, cy = -w / 2 + 2.5 + (i % 4) * (w - 5) / 3, -d / 4 + (i // 4) * d / 2
            n = net("+12V") if i // 4 == 0 else net("GND")
            pads.append(f'    (pad "{i + 1}" thru_hole circle (at {cx:.3f} {cy:.3f}) '
                        f'(size 2.2 2.2) (drill 1.2) {THT} {n} (tstamp {_u()}))')
    elif kind == "edge_connector":
        lanes = int((p.get("metrics") or {}).get("lanes", 16))
        for i in range(lanes):
            pads.append(_pad(i + 1, -w / 2 + 2 + i * 2.2 + 0.5, 0, 1.4, d * 0.9,
                             net(f"PCIE_L{i}")))
    return _footprint(kind, ref, kind.replace("_", " "), x, y, w, d, pads)


def _package(pcb: dict[str, Any], pkg: dict[str, Any], net: _Nets, design_name: str,
             lanes: int) -> str:
    w, d = float(pkg["w"]), float(pkg["d"])
    x, y = ORIGIN + float(pcb["package_x"]), ORIGIN
    pitch = max(1.0, round(max(w, d) / 40, 2))
    nx, ny = int(w // pitch) - 1, int(d // pitch) - 1
    pads = []
    k = 0
    for j in range(ny):
        for i in range(nx):
            px, py = -w / 2 + pitch * (i + 1), -d / 2 + pitch * (j + 1)
            edge = j == 0 and nx // 4 <= i < nx // 4 + lanes
            n = (net(f"PCIE_L{i - nx // 4}") if edge
                 else net("VCORE") if (i + j) % 2 else net("GND"))
            k += 1
            pads.append(f'    (pad "{k}" smd circle (at {px:.3f} {py:.3f}) '
                        f'(size {pitch * 0.5:.3f} {pitch * 0.5:.3f}) {SMD} {n} (tstamp {_u()}))')
    return _footprint("gpu_package_bga", "U1", design_name[:40], x, y, w, d, pads)


def kicad_pcb(scene: dict[str, Any]) -> str:
    """Write ``scene["pcb"]`` (and the package substrate) as a .kicad_pcb file."""
    pcb = scene["pcb"]
    pkg = scene["package"]["substrate"]
    net = _Nets()
    for n in ("GND", "VCORE", "+12V"):
        net(n)
    bw, bd = float(pcb["w"]), float(pcb["d"])
    lanes = int(next((p["metrics"].get("lanes", 16) for p in pcb["parts"]
                      if p["kind"] == "edge_connector"), 16))
    body: list[str] = [_package(pcb, pkg, net, scene["design"]["name"], lanes)]
    refs: dict[str, int] = {}
    for p in pcb["parts"]:
        body.append(_part(p, net, refs))
    lane = 0
    for t in pcb["traces"]:
        pts = t["points"]
        if t["kind"] == "pcie":           # traces are emitted lane by lane
            name, lane = f"PCIE_L{lane}", lane + 1
        else:
            name = "VCORE"
        net(name)
        nid = net.ids[name]               # segments take the bare net number
        for a, b in zip(pts, pts[1:]):
            body.append(f'  (segment (start {ORIGIN + a[0]:.3f} {ORIGIN + a[1]:.3f}) '
                        f'(end {ORIGIN + b[0]:.3f} {ORIGIN + b[1]:.3f}) (width {t["width"]:g}) '
                        f'(layer "F.Cu") (net {nid}) (tstamp {_u()}))')
    x0, y0, x1, y1 = ORIGIN - bw / 2, ORIGIN - bd / 2, ORIGIN + bw / 2, ORIGIN + bd / 2
    body.append(f'  (gr_rect (start {x0:.3f} {y0:.3f}) (end {x1:.3f} {y1:.3f}) '
                f'(stroke (width 0.1) (type default)) (fill none) (layer "Edge.Cuts") '
                f'(tstamp {_u()}))')
    gnd = net.ids["GND"]
    for layer, nid, name in (("In1.Cu", gnd, "GND"), ("In2.Cu", net.ids["VCORE"], "VCORE")):
        body.append(
            f'  (zone (net {nid}) (net_name {_q(name)}) (layer {_q(layer)}) (tstamp {_u()}) '
            f'(hatch edge 0.5) (connect_pads (clearance 0.2)) (min_thickness 0.25) '
            f'(fill (thermal_gap 0.5) (thermal_bridge_width 0.5)) '
            f'(polygon (pts (xy {x0 + 1:.3f} {y0 + 1:.3f}) (xy {x1 - 1:.3f} {y0 + 1:.3f}) '
            f'(xy {x1 - 1:.3f} {y1 - 1:.3f}) (xy {x0 + 1:.3f} {y1 - 1:.3f}))))')
    head = [
        '(kicad_pcb (version 20221018) (generator chiplab)',
        f'  (general (thickness {float(pcb.get("t", 1.6)):g}))',
        '  (paper "A3")',
        '  (title_block (title ' + _q(f"{scene['design']['name']} board (ChipLab studio)")
        + ') (comment 1 "Generated from ChipLab scene[pcb]; coarse placement, not routed"))',
        LAYERS,
        '  (setup (pad_to_mask_clearance 0))',
        net.decl(),
    ]
    return "\n".join(head + body) + "\n)\n"


def summary(scene: dict[str, Any]) -> dict[str, Any]:
    pcb = scene["pcb"]
    kinds: dict[str, int] = {}
    for p in pcb["parts"]:
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    length = sum(math.dist(a, b) for t in pcb["traces"] for a, b in zip(t["points"],
                                                                         t["points"][1:]))
    return {"board_mm": [round(pcb["w"], 1), round(pcb["d"], 1), pcb.get("t", 1.6)],
            "layers": 4, "parts": kinds, "traces": len(pcb["traces"]),
            "trace_length_mm": round(length, 1), "vrm": pcb["vrm"]}


def render(scene: dict[str, Any], *, png: bool = True, drc: bool = True,
           timeout: float = 120) -> dict[str, Any]:
    """Run kicad-cli on the exported board: SVG plot, optional 3D PNG and DRC report."""
    tool = registry.get("kicad")
    if not tool.available:
        raise registry.ToolError("kicad-cli is not installed")
    out: dict[str, Any] = {"kicad_version": tool.version, "summary": summary(scene),
                           "errors": []}
    text = kicad_pcb(scene)
    with tempfile.TemporaryDirectory(prefix="chiplab-kicad-") as tmp:
        board = Path(tmp) / "board.kicad_pcb"
        board.write_text(text)
        env = {"HOME": tmp}
        svg = Path(tmp) / "board.svg"
        r = registry.run(tool, ["pcb", "export", "svg", "--output", str(svg), "--layers",
                                "F.Cu,F.SilkS,Edge.Cuts,F.Mask", "--page-size-mode", "2",
                                "--exclude-drawing-sheet", "--mode-single", str(board)],
                         cwd=tmp, timeout=timeout, extra_env=env)
        if svg.is_file():
            out["svg"] = svg.read_text(errors="replace")
        else:
            out["errors"].append("svg: " + (r.stdout + r.stderr)[-400:])
        if png:
            img = Path(tmp) / "board.png"
            r = registry.run(tool, ["pcb", "render", "--output", str(img), "--width", "1400",
                                    "--height", "900", "--side", "top", "--rotate", "-30,0,0",
                                    "--zoom", "1.3",
                                    "--perspective", "--quality", "basic", str(board)],
                             cwd=tmp, timeout=timeout, extra_env=env)
            if img.is_file() and img.stat().st_size > 0:
                out["png_b64"] = base64.b64encode(img.read_bytes()).decode()
            else:
                out["errors"].append("3D render: " + (r.stdout + r.stderr).strip()[-400:])
        if drc:
            rep = Path(tmp) / "drc.json"
            registry.run(tool, ["pcb", "drc", "--format", "json", "--output", str(rep),
                                str(board)], cwd=tmp, timeout=timeout, extra_env=env)
            if rep.is_file():
                data = json.loads(rep.read_text() or "{}")
                vio = data.get("violations", [])
                kinds: dict[str, int] = {}
                for v in vio:
                    kinds[v.get("type", "?")] = kinds.get(v.get("type", "?"), 0) + 1
                out["drc"] = {"violations": len(vio), "by_type": kinds,
                              "unconnected": len(data.get("unconnected_items", [])),
                              "schematic_parity": len(data.get("schematic_parity", []))}
    return out
