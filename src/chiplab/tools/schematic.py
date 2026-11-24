"""Schematics for Circuit Studio from the same device list ngspice simulates.

With Xschem installed, ChipLab writes a real ``.sch`` (stock ``devices`` library symbols,
every pin tied to a net label) and lets Xschem export the SVG headlessly
(``xschem -x -q --svg``). Without it, a plain SVG is generated here with the same placement.
Either way every net label carries ``data-net="<net>"`` so the studio can colour nodes by
their simulated voltage.
"""

from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path
from typing import Any

from chiplab.tools import registry

DX, DY, X0, Y0, STUB = 140, 120, 80, 70, 20

# symbol -> pin -> (x, y, stub dx, stub dy)
PINS: dict[str, dict[str, tuple[int, int, int, int]]] = {
    "nmos": {"d": (20, -30, 0, -STUB), "g": (-20, 0, -STUB, 0), "s": (20, 30, 0, STUB),
             "b": (20, 0, STUB, 0)},
    "pmos": {"d": (20, 30, 0, STUB), "g": (-20, 0, -STUB, 0), "s": (20, -30, 0, -STUB),
             "b": (20, 0, STUB, 0)},
    "cap": {"p": (0, -30, 0, -STUB), "m": (0, 30, 0, STUB)},
    "res": {"p": (0, -30, 0, -STUB), "m": (0, 30, 0, STUB)},
    "vsrc": {"p": (0, -30, 0, -STUB), "m": (0, 30, 0, STUB)},
}
SYM = {"nmos": "nmos4.sym", "pmos": "pmos4.sym", "cap": "capa.sym", "res": "res.sym",
       "vsrc": "vsource.sym"}


def _origin(d: dict[str, Any]) -> tuple[int, int]:
    c, r = d["pos"]
    return int(X0 + c * DX), int(Y0 + r * DY)


def xschem_sch(devices: list[dict[str, Any]]) -> str:
    out = ["v {xschem version=3.4.4 file_version=1.2}", "G {}", "K {}", "V {}", "S {}", "E {}"]
    k = 0
    for d in devices:
        x, y = _origin(d)
        kind = d["kind"]
        if kind in ("nmos", "pmos"):
            props = f"name={d['name']} model={'nch' if kind == 'nmos' else 'pch'} " \
                    f"w={d['w_um']:g}u l={d['l_um']:g}u del=0 m=1"
        else:
            props = f"name={d['name']} value=\"{d['value']}\""
        out.append(f"C {{{SYM[kind]}}} {x} {y} 0 0 {{{props}}}")
        for pin, net in d["pins"].items():
            px, py, sx, sy = PINS[kind][pin]
            ax, ay, bx, by = x + px, y + py, x + px + sx, y + py + sy
            out.append(f"N {ax} {ay} {bx} {by} {{lab={net}}}")
            k += 1
            if net == "0":
                out.append(f"C {{gnd.sym}} {bx} {by} 0 0 {{name=g{k} lab=0}}")
            else:
                flip = 1 if sx > 0 else 0
                out.append(f"C {{lab_pin.sym}} {bx} {by} 0 {flip} {{name=p{k} lab={net}}}")
    return "\n".join(out) + "\n"


def _tag_nets(svg: str, nets: list[str]) -> str:
    wanted = set(nets)

    def sub(m: re.Match[str]) -> str:
        name = html.unescape(m.group(2))
        if name in wanted:
            return f'{m.group(1)} class="net" data-net="{html.escape(name)}">{m.group(2)}</text>'
        return m.group(0)
    svg = re.sub(r"(<text[^>]*)>([^<]+)</text>", sub, svg)
    m = re.search(r'<svg[^>]*width="(\d+)"[^>]*height="(\d+)"', svg)
    if m and "viewBox" not in svg[: m.end()]:
        svg = svg.replace("<svg ", f'<svg viewBox="0 0 {m.group(1)} {m.group(2)}" ', 1)
    return svg


def xschem_svg(devices: list[dict[str, Any]], nets: list[str], timeout: float = 30) -> str:
    tool = registry.get("xschem")
    if not tool.available:
        raise registry.ToolError("xschem is not installed")
    share = tool.env.get("XSCHEM_SHAREDIR") or "/usr/share/xschem"
    with tempfile.TemporaryDirectory(prefix="chiplab-xschem-") as tmp:
        sch, svg, rc = Path(tmp) / "circuit.sch", Path(tmp) / "circuit.svg", Path(tmp) / "rc"
        sch.write_text(xschem_sch(devices))
        (Path(tmp) / ".xschem").mkdir()   # xschem segfaults if $HOME/.xschem is missing
        rc.write_text(f"set XSCHEM_SHAREDIR {{{share}}}\n"
                      f"set XSCHEM_LIBRARY_PATH {{{share}/xschem_library/devices}}\n")
        registry.run(tool, ["--rcfile", str(rc), "-x", "-q", "-r", "--svg", "--plotfile",
                            str(svg), str(sch)], cwd=tmp, timeout=timeout,
                     extra_env={"HOME": tmp})
        if not svg.is_file() or svg.stat().st_size < 200:
            raise registry.ToolError("xschem did not export an SVG")
        return _tag_nets(svg.read_text(errors="replace"), nets)


# ---------------------------------------------------------------- fallback SVG
def _mos(x: int, y: int, pmos: bool) -> str:
    s = [f"M{x - 20},{y} L{x - 6},{y}", f"M{x - 6},{y - 16} L{x - 6},{y + 16}",
         f"M{x},{y - 18} L{x},{y + 18}", f"M{x},{y - 14} L{x + 20},{y - 14} L{x + 20},{y - 30}",
         f"M{x},{y + 14} L{x + 20},{y + 14} L{x + 20},{y + 30}", f"M{x},{y} L{x + 20},{y}"]
    bubble = f'<circle cx="{x - 9}" cy="{y}" r="3" class="sym"/>' if pmos else ""
    return f'<path class="sym" d="{" ".join(s)}"/>{bubble}'


def _two(x: int, y: int, kind: str) -> str:
    if kind == "cap":
        d = (f"M{x},{y - 30} L{x},{y - 5} M{x - 14},{y - 5} L{x + 14},{y - 5} "
             f"M{x - 14},{y + 5} L{x + 14},{y + 5} M{x},{y + 5} L{x},{y + 30}")
        return f'<path class="sym" d="{d}"/>'
    if kind == "res":
        zz = " ".join(f"L{x + (8 if i % 2 else -8)},{y - 18 + i * 6}" for i in range(7))
        return f'<path class="sym" d="M{x},{y - 30} L{x},{y - 21} {zz} L{x},{y + 30}"/>'
    return (f'<path class="sym" d="M{x},{y - 30} L{x},{y - 16} M{x},{y + 16} L{x},{y + 30}"/>'
            f'<circle class="sym" cx="{x}" cy="{y}" r="16"/>'
            f'<text class="pm" x="{x - 4}" y="{y - 3}">+</text>'
            f'<text class="pm" x="{x - 3}" y="{y + 11}">−</text>')


def generated_svg(devices: list[dict[str, Any]]) -> str:
    cols = max(d["pos"][0] for d in devices) + 1
    rows = max(d["pos"][1] for d in devices) + 1
    w, h = int(X0 * 2 + (cols - 1) * DX), int(Y0 * 2 + (rows - 1) * DY)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
             f'width="{w}" height="{h}" font-family="monospace">',
             "<style>.sym{fill:none;stroke:currentColor;stroke-width:2}"
             ".wire{stroke:#4a90c2;stroke-width:2}.net{font-size:13px;fill:#2f7fb8}"
             ".nm{font-size:11px;fill:currentColor;opacity:.75}.pm{font-size:11px;"
             "fill:currentColor}</style>"]
    for d in devices:
        x, y = _origin(d)
        kind = d["kind"]
        parts.append(_mos(x, y, kind == "pmos") if kind in ("nmos", "pmos") else _two(x, y, kind))
        val = (f"{d['w_um']:g}/{d['l_um']:g}" if kind in ("nmos", "pmos")
               else str(d["value"])[:18])
        parts.append(f'<text class="nm" x="{x + 26}" y="{y - 36}">{html.escape(d["name"])}'
                     f' {html.escape(val)}</text>')
        for pin, net in d["pins"].items():
            px, py, sx, sy = PINS[kind][pin]
            ax, ay, bx, by = x + px, y + py, x + px + sx, y + py + sy
            parts.append(f'<line class="wire" data-net="{html.escape(net)}" x1="{ax}" y1="{ay}" '
                         f'x2="{bx}" y2="{by}"/>')
            if net == "0":
                gnd = (f"M{bx - 9},{by} L{bx + 9},{by} M{bx - 5},{by + 4} L{bx + 5},{by + 4} "
                       f"M{bx - 2},{by + 8} L{bx + 2},{by + 8}")
                parts.append(f'<path class="sym" d="{gnd}"/>')
                continue
            tx = bx + (4 if sx > 0 else -4 if sx < 0 else 3)
            ty = by + (-4 if sy < 0 else 14 if sy > 0 else 4)
            anchor = "start" if sx > 0 or sx == 0 else "end"
            parts.append(f'<text class="net" data-net="{html.escape(net)}" x="{tx}" y="{ty}" '
                         f'text-anchor="{anchor}">{html.escape(net)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def schematic(devices: list[dict[str, Any]], nets: list[str]) -> dict[str, Any]:
    """Xschem SVG when available, else the generated one. Never raises for a missing tool."""
    note = ""
    if registry.get("xschem").available:
        try:
            return {"source": "xschem", "svg": xschem_svg(devices, nets),
                    "sch": xschem_sch(devices)}
        except (registry.ToolError, OSError) as exc:
            note = f"Xschem export failed ({exc}); showing the generated schematic."
    return {"source": "generated", "svg": generated_svg(devices),
            "sch": xschem_sch(devices), "note": note}
