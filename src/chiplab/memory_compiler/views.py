"""SVG views of a compiled memory, one per abstraction level:

  logical_svg    - contents and address organization (what is stored)
  circuit_svg    - transistor-level column / cell schematic for the chosen cell architecture
  physical_svg   - the manufactured array: one mark per bit site, programmed sites shown
                   (mask ROM), column-mux interleaving, decoder and sense-amp periphery drawn to
                   the estimated areas (a floorplan, not a GDS)
  block_svg      - hierarchical block diagram of the macro

All views are generated from the same compiler result, so they stay consistent with it.
"""

from __future__ import annotations

import math
from html import escape
from typing import Any

from chiplab.memory_compiler.compiler import CELLS, MemSpec

CSS = ("<style>text{font:11px ui-monospace,Menlo,monospace;fill:#cfd8e3}.t{font-size:13px;font-weight:600}"
       ".m{fill:#8b97a4}.w{stroke:#cfd8e3;stroke-width:1.4;fill:none}.g{stroke:#e6c35c}.b{stroke:#6ea0ff}"
       ".r{stroke:#f2555f}</style>")


def _svg(w: float, h: float, body: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" '
            f'height="{h:.0f}" style="background:#0e141b">{CSS}{body}</svg>')


def site_map(spec: MemSpec, cand: dict[str, Any]) -> list[list[int]]:
    """Physical bit at each array site (bank 0): word a, bit b -> row a // mux,
    column b * mux + a % mux. Values: stored bit (1/0)."""
    rows, mux, w = cand["rows"], cand["column_mux"], spec.width
    data = list(spec.contents or [])
    grid = [[0] * (w * mux) for _ in range(rows)]
    for a in range(min(spec.words, rows * mux)):
        v = data[a] if a < len(data) else 0
        r, m = divmod(a, mux)
        for b in range(w):
            grid[r][b * mux + m] = v >> b & 1
    return grid


def logical_svg(spec: MemSpec, max_words: int = 32) -> str:
    rows = []
    data = list(spec.contents or [])
    n = min(spec.words, max_words)
    hexw = max(1, math.ceil(spec.width / 4))
    for a in range(n):
        v = data[a] if a < len(data) else None
        bits = "".join("1" if v is not None and v >> b & 1 else "0" if v is not None else "·"
                       for b in range(spec.width - 1, -1, -1))[:64]
        rows.append((a, "--" if v is None else f"{v:0{hexw}x}", bits))
    h = 60 + 16 * len(rows)
    body = [f'<text class="t" x="12" y="22">Logical view · {escape(spec.name)}: {spec.words} x {spec.width} '
            f'({spec.addr_bits}-bit address)</text>',
            '<text class="m" x="12" y="42">addr      data   bits (MSB..LSB)</text>']
    for i, (a, hx, bits) in enumerate(rows):
        body.append(f'<text x="12" y="{60 + 16 * i}">{a:0{max(1, math.ceil(spec.addr_bits / 4))}x}'
                    f'{"":6}{hx:>{hexw}}   {bits}</text>')
    if spec.words > n:
        body.append(f'<text class="m" x="12" y="{h - 4}">… {spec.words - n} more words</text>')
    return _svg(max(520, 200 + 8 * min(spec.width, 64)), h + 8, "".join(body))


def _nmos(x: float, y: float, gate_label: str = "", present: bool = True, cls: str = "w") -> str:
    """nMOS symbol, drain at top (x, y), source at (x, y + 40), gate on the left."""
    if not present:
        return (f'<line class="w" x1="{x}" y1="{y}" x2="{x}" y2="{y + 8}" stroke-dasharray="2 3"/>'
                f'<text class="m" x="{x - 30}" y="{y + 24}">{escape(gate_label)} ∅</text>')
    return (f'<path class="{cls}" d="M{x} {y} v10 h-8 v20 h8 v10 M{x - 12} {y + 10} v20 M{x - 12} {y + 20} h-14"/>'
            f'<text x="{x - 60}" y="{y + 24}">{escape(gate_label)}</text>')


def circuit_svg(spec: MemSpec, cand: dict[str, Any], column: int = 0, nrows: int = 6) -> str:
    cell = cand["cell"]
    c = CELLS[cell]
    body = [f'<text class="t" x="12" y="22">Circuit view · {escape(cell)}: {escape(c["desc"])}</text>']
    grid = site_map(spec, cand) if spec.kind == "mask_rom" else None
    if cell in ("rom_nor_contact", "rom_nor_active"):
        x = 260
        body.append(f'<line class="w b" x1="{x}" y1="50" x2="{x}" y2="{70 + nrows * 52}"/>'
                    f'<text x="{x + 8}" y="62">BL{column} (precharged high)</text>')
        for r in range(nrows):
            y = 70 + r * 52
            bit = grid[r][column] if grid and r < len(grid) else 0
            present = bit == 0                      # NOR ROM: a transistor pulls BL low -> reads 0
            body.append(f'<line class="w" x1="{x}" y1="{y}" x2="{x + 40}" y2="{y}"/>')
            if cell == "rom_nor_contact":
                body.append(_nmos(x + 40, y, f"WL{r}", True, "w" if present else "w r"))
                body.append(f'<text class="m" x="{x + 60}" y="{y + 24}">{"contact" if present else "no contact"} -> {bit}</text>')
            else:
                body.append(_nmos(x + 40, y, f"WL{r}", present))
                body.append(f'<text class="m" x="{x + 60}" y="{y + 24}">stores {bit}</text>')
            body.append(f'<text class="m" x="{x + 36}" y="{y + 50}">⏚</text>')
        body.append(f'<text class="m" x="12" y="{90 + nrows * 52}">Selected word line turns its row on; a present (contacted) '
                    f'transistor discharges the bit line, read as 0 by the sense amp.</text>')
        h = 110 + nrows * 52
    elif cell == "rom_nand":
        x, stack = 260, min(nrows, c["stack"])
        body.append(f'<text x="{x + 8}" y="56">BL{column}</text>')
        for r in range(stack):
            y = 60 + r * 46
            bit = grid[r][column] if grid and r < len(grid) else 0
            body.append(_nmos(x, y, f"WL{r}", True, "w g" if bit else "w"))
            body.append(f'<text class="m" x="{x + 18}" y="{y + 24}">{"depletion (always on)" if bit else "enhancement"} -> {bit}</text>')
        body.append(f'<text class="m" x="{x - 4}" y="{66 + stack * 46}">⏚</text>'
                    f'<text class="m" x="12" y="{90 + stack * 46}">Series stack: unselected word lines high; the selected one low. '
                    f'A depletion device conducts anyway, so the stack discharges BL.</text>')
        h = 110 + stack * 46
    elif cell in ("sram6t", "sram8t"):
        body.append('<path class="w" d="M180 70 h60 M180 150 h60 M200 70 v80 M260 70 v80 M330 70 v80 M390 70 v80 '
                    'M330 70 h60 M330 150 h60"/>'
                    '<text x="196" y="64">P1</text><text x="376" y="64">P2</text>'
                    '<text x="196" y="170">N1</text><text x="376" y="170">N2</text>'
                    '<path class="w b" d="M120 50 v150 M450 50 v150"/><text x="96" y="44">BL</text><text x="436" y="44">BLB</text>'
                    '<path class="w g" d="M100 110 h40 M420 110 h50"/><text x="60" y="114">WL→A1</text><text x="470" y="114">A2</text>'
                    '<text class="m" x="12" y="230">Cross-coupled inverters (P1/N1, P2/N2) hold the bit; access devices A1/A2 '
                    'connect it to BL/BLB when WL is high.</text>')
        if cell == "sram8t":
            body.append('<path class="w r" d="M500 90 v80 M480 130 h40"/><text x="526" y="134">read stack (RWL, RBL)</text>')
        h = 250
    elif cell in ("gc2t", "gc3t"):
        body.append('<path class="w" d="M200 80 h80 M240 80 v50 M240 130 h80"/><text x="190" y="74">write device (WWL, WBL)</text>'
                    '<circle cx="240" cy="130" r="4" fill="#e6c35c"/><text x="250" y="150">storage node (gate cap)</text>'
                    '<path class="w b" d="M320 110 v60"/><text x="330" y="140">read device (RWL, RBL)</text>'
                    '<text class="m" x="12" y="200">Charge on the read transistor gate stores the bit; leakage sets retention, so rows '
                    'must be refreshed.</text>')
        h = 220
    elif cell == "dram1t1c":
        body.append('<path class="w b" d="M200 50 v150"/><text x="186" y="44">BL</text>'
                    + _nmos(260, 80, "WL") + '<path class="w" d="M260 120 v20 M244 140 h32 M244 148 h32 M260 148 v20"/>'
                    '<text x="284" y="146">Cs</text><text class="m" x="12" y="220">Access transistor shares the capacitor charge with the '
                    'bit line; the read is destructive and needs a write-back and refresh.</text>')
        h = 240
    else:
        body.append(f'<text x="12" y="60">{escape(c["desc"])}: per bit {c.get("gates_per_bit", "?")} gate equivalents, '
                    f'{c.get("tx", "?")} transistors; read through a {spec.words}:1 mux per port.</text>')
        h = 90
    return _svg(720, h, "".join(body))


def physical_svg(spec: MemSpec, cand: dict[str, Any], max_sites: int = 96) -> str:
    rows, cols = cand["rows"], cand["cols"]
    c = CELLS[cand["cell"]]
    std = bool(c.get("std_cell"))
    W, H = 860, 420
    title = (f'<text class="t" x="12" y="20">Physical view · {escape(cand["cell"])}: {cand["banks"]} bank(s) x '
             f'{rows} rows x {cols} cols, column mux {cand["column_mux"]} · {cand["area_um2"]:.0f} µm² (estimate)</text>')
    if std:
        return _svg(W, 120, title + '<text class="m" x="12" y="50">Standard-cell implementation: the layout is produced by '
                    'place and route, so there is no regular bit-cell array to draw.</text>')
    # floorplan: decoder strip left, column periphery bottom, array to scale
    arr = cand["cell_array_um2"] / cand["banks"]
    per = cand["periphery_um2"] / cand["banks"]
    aspect = cols / rows if rows else 1
    ax = 120
    aw = min(W - ax - 20, 420)
    ah = min(H - 140, aw / max(aspect, 0.25))
    aw = min(aw, ah * aspect) if aspect < 4 else aw
    dec_w = max(18, min(90, aw * per / max(arr, 1) * 0.5))
    sa_h = max(14, min(60, ah * per / max(arr, 1) * 0.5))
    y0 = 40
    body = [title,
            f'<rect x="{ax - dec_w - 4}" y="{y0}" width="{dec_w}" height="{ah}" fill="#3a2f22" stroke="#8a6a3a"/>'
            f'<text x="{ax - dec_w - 2}" y="{y0 + ah + 14}">row dec</text>',
            f'<rect x="{ax}" y="{y0 + ah + 4}" width="{aw}" height="{sa_h}" fill="#1f3a2c" stroke="#4a8f6a"/>'
            f'<text x="{ax + 4}" y="{y0 + ah + sa_h / 2 + 8}">col mux {cand["column_mux"]}:1 · sense amps · drivers</text>',
            f'<rect x="{ax}" y="{y0}" width="{aw}" height="{ah}" fill="#2a2140" stroke="#5a4f8e"/>']
    sr, sc = min(rows, max_sites), min(cols, max_sites)
    cw, chh = aw / sc, ah / sr
    grid = site_map(spec, cand) if spec.kind == "mask_rom" else None
    prog_color = "#e6c35c"
    for r in range(sr):
        for k in range(sc):
            rr, kk = r * rows // sr, k * cols // sc
            if grid is not None:
                bit = grid[rr][kk]
                programmed = (bit == 0) if cand["cell"].startswith("rom_nor") else bit == 1
                if programmed:
                    body.append(f'<rect x="{ax + k * cw:.2f}" y="{y0 + r * chh:.2f}" width="{max(cw - 0.4, 0.4):.2f}" '
                                f'height="{max(chh - 0.4, 0.4):.2f}" fill="{prog_color}"/>')
            elif (r + k) % 2 == 0 and cw > 2:
                body.append(f'<rect x="{ax + k * cw:.2f}" y="{y0 + r * chh:.2f}" width="{cw - 0.4:.2f}" '
                            f'height="{chh - 0.4:.2f}" fill="#9c5470" opacity=".5"/>')
    legend = ""
    if grid is not None:
        what = {"rom_nor_contact": "contact present (reads 0)", "rom_nor_active": "transistor present (reads 0)",
                "rom_nand": "depletion implant (reads 1)"}[cand["cell"]]
        n_prog = sum(((b == 0) if cand["cell"].startswith("rom_nor") else b == 1) for row in grid for b in row)
        legend = (f'<rect x="{ax}" y="{H - 40}" width="10" height="10" fill="{prog_color}"/>'
                  f'<text x="{ax + 16}" y="{H - 31}">programmed site: {what} · {n_prog} of {rows * cols} sites '
                  f'({escape(c["programmed_by"])})</text>')
    note = (f'<text class="m" x="12" y="{H - 12}">Floorplan to the estimated areas, not a GDS. '
            f'{"Showing " + str(sr) + "x" + str(sc) + " of " + str(rows) + "x" + str(cols) + " sites. " if (sr < rows or sc < cols) else ""}'
            f'Word a, bit b sits at row a/{cand["column_mux"]}, column b·{cand["column_mux"]} + a mod {cand["column_mux"]}.</text>')
    return _svg(W, H, "".join(body) + legend + note)


def block_svg(spec: MemSpec, cand: dict[str, Any]) -> str:
    blocks = [("address", f"{spec.addr_bits} b"), ("row decoder", cand["decoder"]),
              ("bit-cell array", f"{cand['rows']}x{cand['cols']} {cand['cell']}"),
              ("column mux", f"{cand['column_mux']}:1"), ("sense / drivers", ""), ("data", f"{spec.width} b")]
    if cand.get("refresh"):
        blocks.insert(3, ("refresh ctrl", f"{cand['refresh']['retention_us']:.0f} µs"))
    body = [f'<text class="t" x="12" y="20">Block diagram · {escape(spec.name)} ({spec.kind})</text>']
    x = 12
    for i, (n, sub) in enumerate(blocks):
        body.append(f'<rect x="{x}" y="40" width="104" height="54" rx="6" fill="#18202a" stroke="#6ea0ff"/>'
                    f'<text x="{x + 8}" y="62">{escape(n)}</text><text class="m" x="{x + 8}" y="80">{escape(sub)[:16]}</text>')
        if i < len(blocks) - 1:
            body.append(f'<path class="w" d="M{x + 104} 67 h14"/>')
        x += 118
    return _svg(x + 10, 110, "".join(body))


def all_views(spec: MemSpec, result: dict[str, Any]) -> dict[str, str]:
    b = result["best"]
    return {"logical": logical_svg(spec), "circuit": circuit_svg(spec, b), "physical": physical_svg(spec, b),
            "block": block_svg(spec, b)}
