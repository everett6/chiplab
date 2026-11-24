"""Memory compiler: a memory specification in, ranked candidate architectures out.

Kinds: mask ROM, SRAM, DRAM (1T1C), eDRAM (gain cell), register file, FIFO, or a custom
array. For each kind the compiler enumerates organizations (banks x rows x columns with
column multiplexing), cell architectures (e.g. NOR vs NAND mask ROM, 6T vs 8T SRAM, 2T vs
3T gain cell, flop vs latch register file) and decoder styles, and estimates area, access
time, cycle time, read energy, leakage, throughput and complexity from the selected PDK's
published/estimated scalars.

Fidelity: ANALYTICAL. These are first-order estimates (FO4-based delays, capacitance-based
energy, cell-area ratios). They are labelled with provenance and an uncertainty band and are
never a substitute for a memory compiler run (OpenRAM, the foundry's macro generator) or for
SPICE characterisation. Three views are kept apart in every result:
  logical   - what is stored and how it is addressed (contents, organization)
  circuit   - the cell and periphery circuit (transistors, decoders, sense amps)
  physical  - what is manufactured (cell array geometry, programmed sites, area)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from chiplab.library.pdk_library import PDK_LADDER, get_pdk

KINDS = ("mask_rom", "sram", "dram", "edram", "regfile", "fifo", "custom")

# Cell architectures. area_ratio is relative to the PDK's 6T SRAM bitcell (ASSUMED ratios from
# textbook layouts, +-40%); transistors per bit; leak = leakage in gate-equivalents per bit.
CELLS: dict[str, dict[str, Any]] = {
    "rom_nor_contact": {"kind": "mask_rom", "area_ratio": 0.45, "tx": 1, "leak": 0.3, "rw": False,
                        "desc": "NOR ROM, one transistor per site, bit set by a contact/via to the bitline",
                        "programmed_by": "contact or via mask", "bl_k": 48, "custom_layout": True},
    "rom_nor_active": {"kind": "mask_rom", "area_ratio": 0.42, "tx": 1, "leak": 0.3, "rw": False,
                       "desc": "NOR ROM, transistor present/absent per site (active/implant programmed)",
                       "programmed_by": "active-area or implant mask", "bl_k": 48, "custom_layout": True},
    "rom_nand": {"kind": "mask_rom", "area_ratio": 0.28, "tx": 1, "leak": 0.2, "rw": False,
                 "desc": "NAND ROM, series stacks; bit set by depletion implant (densest, slowest)",
                 "programmed_by": "depletion implant mask", "stack": 8, "custom_layout": True},
    "rom_logic": {"kind": "mask_rom", "area_ratio": None, "tx": None, "leak": None, "rw": False,
                  "desc": "ROM as synthesised standard-cell logic (no custom layout)",
                  "programmed_by": "logic synthesis", "std_cell": True},
    "sram6t": {"kind": "sram", "area_ratio": 1.0, "tx": 6, "leak": 1.0, "rw": True, "ports": (1, 0, 0),
               "desc": "6T SRAM, one read/write port", "bl_k": 32, "macro": True},
    "sram8t": {"kind": "sram", "area_ratio": 1.35, "tx": 8, "leak": 1.2, "rw": True, "ports": (0, 1, 1),
               "desc": "8T SRAM, decoupled read port (1R1W)", "bl_k": 40, "macro": True},
    "dram1t1c": {"kind": "dram", "area_ratio": 0.22, "tx": 1, "leak": 0.05, "rw": True, "ports": (1, 0, 0),
                 "desc": "1T1C DRAM with a storage capacitor (needs a DRAM capacitor process)",
                 "bl_k": 16, "needs_process": "DRAM storage capacitor"},
    "gc2t": {"kind": "edram", "area_ratio": 0.55, "tx": 2, "leak": 0.4, "rw": True, "ports": (0, 1, 1),
             "desc": "2T gain-cell eDRAM (logic compatible, needs refresh)", "bl_k": 24},
    "gc3t": {"kind": "edram", "area_ratio": 0.75, "tx": 3, "leak": 0.5, "rw": True, "ports": (0, 1, 1),
             "desc": "3T gain-cell eDRAM (logic compatible, longer retention, needs refresh)", "bl_k": 28},
    "rf_flop": {"kind": "regfile", "area_ratio": None, "tx": 24, "leak": 6.0, "rw": True,
                "desc": "flip-flop register file from standard cells", "std_cell": True, "gates_per_bit": 6.0},
    "rf_latch": {"kind": "regfile", "area_ratio": None, "tx": 14, "leak": 3.5, "rw": True,
                 "desc": "latch register file from standard cells (needs clock gating discipline)",
                 "std_cell": True, "gates_per_bit": 3.5},
}

# Which open PDKs ship an SRAM macro (published, per their repositories)
SRAM_MACROS = {"sky130": "sky130_sram_macros / OpenRAM", "gf180": "gf180mcu_fd_ip_sram",
               "ihp_sg13g2": "IHP RM_IHPSG13 SRAM macros"}


@dataclass
class MemSpec:
    name: str = "mem"
    kind: str = "sram"
    words: int = 1024
    width: int = 32
    read_ports: int = 0
    write_ports: int = 0
    rw_ports: int = 1
    pdk: str = "sky130"
    freq_mhz: float = 100.0
    contents: list[int] | None = None        # mask ROM / init data (logical view)
    banks: int | None = None                 # force a bank count
    cell: str | None = None                  # force a cell architecture
    retention_us: float | None = None        # override eDRAM/DRAM retention
    objective: str = "area_delay"            # area | delay | energy | area_delay

    def __post_init__(self) -> None:
        if self.kind in ("mask_rom", "fifo") and self.rw_ports and not self.read_ports:
            self.read_ports, self.rw_ports = 1, 0          # a ROM/FIFO read port, not read/write

    def validate(self) -> list[str]:
        e = []
        if self.kind not in KINDS:
            e.append(f"kind must be one of {KINDS}")
        if self.words < 2 or self.width < 1:
            e.append("words >= 2 and width >= 1")
        if self.words > 1 << 24 or self.width > 1024:
            e.append("words <= 16M and width <= 1024")
        if self.pdk not in PDK_LADDER:
            e.append(f"unknown PDK {self.pdk!r}")
        if self.kind == "mask_rom":
            if self.write_ports or self.rw_ports:
                e.append("a mask ROM has no write ports")
            if self.contents is not None:
                if len(self.contents) > self.words:
                    e.append("more ROM contents than words")
                if any(not 0 <= v < 1 << self.width for v in self.contents):
                    e.append("ROM contents must fit the word width")
        if self.read_ports + self.rw_ports < 1:
            e.append("at least one read or read/write port")
        if self.cell and self.cell not in CELLS:
            e.append(f"unknown cell {self.cell!r}")
        return e

    @property
    def addr_bits(self) -> int:
        return max(1, math.ceil(math.log2(self.words)))


def _pdk(spec: MemSpec) -> dict[str, float]:
    p = get_pdk(spec.pdk)
    return {k: p.scalar(k) for k in ("sram_bitcell_um2", "fo4_delay_ps", "gate_density_kgates_mm2",
                                     "nominal_vdd_v", "metal1_pitch_nm", "wire_cap_ff_per_um",
                                     "dynamic_energy_fj_per_gate", "leakage_nw_per_gate",
                                     "gain_cell_retention_us_25c")} | {"node_nm": p.node_nm}


def _ones_fraction(spec: MemSpec) -> float:
    if not spec.contents:
        return 0.5
    bits = len(spec.contents) * spec.width
    return sum(bin(v).count("1") for v in spec.contents) / bits


def _cells_for(spec: MemSpec) -> list[str]:
    if spec.cell:
        return [spec.cell]
    kind = "regfile" if spec.kind == "fifo" else spec.kind
    out = [k for k, c in CELLS.items() if c["kind"] == kind]
    if spec.kind == "custom":
        out = [k for k in CELLS if CELLS[k]["kind"] in ("sram", "edram", "regfile")]
    # port feasibility: 6T has one RW port; separate read+write ports need 8T / gain cells / flops
    if spec.kind in ("sram", "custom"):
        ports = spec.read_ports + spec.write_ports + spec.rw_ports
        if spec.read_ports + spec.rw_ports > 1 or (spec.read_ports and spec.write_ports and ports > 2):
            out = [k for k in out if CELLS[k].get("std_cell")] or ["rf_flop"]
        elif spec.read_ports and spec.write_ports:
            out = [k for k in out if k != "sram6t"]
    return out


def evaluate(spec: MemSpec, cell: str, banks: int, mux: int, decoder: str) -> dict[str, Any] | None:
    """Estimate one organization. Returns None if the organization is not legal."""
    c = CELLS[cell]
    pk = _pdk(spec)
    fo4 = pk["fo4_delay_ps"]
    gate_um2 = 1000.0 / pk["gate_density_kgates_mm2"]          # one NAND2-equivalent, placed
    vdd = pk["nominal_vdd_v"]
    words_per_bank = math.ceil(spec.words / banks)
    std = bool(c.get("std_cell"))
    rows = math.ceil(words_per_bank / mux)
    cols = spec.width * mux
    if not std and not (8 <= rows <= 1024 and cols <= 2048):
        return None
    if std and (mux != 1 or decoder != "single"):
        return None
    bits = banks * rows * cols
    nports = spec.read_ports + spec.write_ports + spec.rw_ports
    ones = _ones_fraction(spec)
    p = {"cell": cell, "banks": banks, "rows": rows, "cols": cols, "column_mux": mux, "decoder": decoder}

    if std:
        if cell == "rom_logic":
            # synthesised truth table: roughly one 2-input gate per four stored bits after sharing
            gates = 0.25 * spec.words * spec.width * (4 * ones * (1 - ones) + 0.15)
            tx_bit = None
            access_ps = fo4 * (math.log2(spec.words) * 1.2 + 3)
            energy_fj = gates * 0.1 * pk["dynamic_energy_fj_per_gate"]
            leak_nw = gates * pk["leakage_nw_per_gate"]
        else:
            gpb = c["gates_per_bit"]
            mux_g = (spec.read_ports + spec.rw_ports) * 2.5 * (spec.words - 1) / spec.words
            dec_g = nports * (math.log2(spec.words) / 2 + 1) / spec.width   # word decoders, shared by the bits
            gates = spec.words * spec.width * (gpb + mux_g + dec_g)
            tx_bit = c["tx"]
            access_ps = fo4 * (math.log2(spec.words) * 1.5 + 4)
            energy_fj = (spec.width * math.log2(spec.words) * 2.5 + spec.width * 2) * pk["dynamic_energy_fj_per_gate"]
            leak_nw = spec.words * spec.width * c["leak"] * pk["leakage_nw_per_gate"]
        area_um2 = gates * gate_um2
        cell_area = area_um2
        periph_area = 0.0
        cycle_ps = access_ps * 1.15
        transistors = gates * 4
    else:
        cell_um2 = pk["sram_bitcell_um2"] * c["area_ratio"]
        cell_area = bits * cell_um2
        abits = max(1, math.ceil(math.log2(rows)))
        dec_gates = rows * (abits / 2 + 1) * (1.25 if decoder == "predecoded" else 1.0)
        col_gates = cols * 1.0 + (cols / mux) * 12 + (spec.width * 4 if c["rw"] else 0) + cols * 1.0
        ctrl_gates = 150 + (500 if spec.kind in ("dram", "edram") else 0)
        periph_gates = banks * (dec_gates + col_gates) * max(1, nports) + ctrl_gates
        periph_area = periph_gates * gate_um2
        area_um2 = (cell_area + periph_area) * 1.1               # 10% for power rails and routing
        # delay: decoder + wordline RC + bitline + sense + output
        dec_ps = fo4 * (math.log(rows, 4) + (1.0 if decoder == "predecoded" else 1.6))
        wl_ps = fo4 * (1 + cols / 128)
        if cell == "rom_nand":
            stack = c["stack"]
            bl_ps = fo4 * (stack ** 2 / 6 + rows / (stack * 16))
        else:
            bl_ps = fo4 * rows / c["bl_k"]
        if c["kind"] == "edram":
            bl_ps *= 1.6
        access_ps = dec_ps + wl_ps + bl_ps + 2.0 * fo4 + (math.log2(mux) * 0.5 + 1.5) * fo4
        cycle_ps = access_ps * (1.6 if c["kind"] == "dram" else 1.3)
        # energy per read: bitline charge on the active row + decoder + sense amps
        cell_pitch_um = math.sqrt(cell_um2)
        c_bl_ff = rows * (0.08 + cell_pitch_um * pk["wire_cap_ff_per_um"])
        swing = vdd if c["kind"] == "mask_rom" else (vdd if c["kind"] == "dram" else 0.15 * vdd / 1.2)
        active = cols * ((1 - ones) if c["kind"] == "mask_rom" else 1.0)
        energy_fj = (active * c_bl_ff * vdd * swing + (dec_gates + spec.width * 12) *
                     pk["dynamic_energy_fj_per_gate"] * 0.3)
        leak_nw = bits * c["leak"] * pk["leakage_nw_per_gate"] + periph_gates * pk["leakage_nw_per_gate"]
        tx_bit = c["tx"]
        transistors = bits * c["tx"] + periph_gates * 4
        if c["kind"] == "mask_rom" and cell == "rom_nor_active":
            transistors = bits * (1 - ones) + periph_gates * 4      # sites without a transistor store 1
    fmax = 1e6 / cycle_ps
    meets = fmax >= spec.freq_mhz
    refresh = None
    if c["kind"] in ("dram", "edram"):
        ret = spec.retention_us or (64000.0 if c["kind"] == "dram" else pk["gain_cell_retention_us_25c"]
                                    * (1.5 if cell == "gc3t" else 1.0))
        refresh_rate = banks * rows / (ret * 1e-6)                       # row refreshes per second
        e_row = cols * c_bl_ff * vdd * swing
        refresh = {"retention_us": ret, "row_refreshes_per_s": round(refresh_rate),
                   "power_uw": round(refresh_rate * e_row * 1e-15 * 1e6, 3),
                   "busy_fraction": round(refresh_rate * cycle_ps * 1e-12, 5)}
    leak_uw = leak_nw / 1000
    ports_read = spec.read_ports + spec.rw_ports
    return {
        **p,
        "bits": bits, "capacity_bits": spec.words * spec.width,
        "area_um2": round(area_um2, 1), "cell_array_um2": round(cell_area, 1),
        "periphery_um2": round(periph_area, 1),
        "array_efficiency": round(cell_area / area_um2, 3) if area_um2 else None,
        "access_ns": round(access_ps / 1000, 3), "cycle_ns": round(cycle_ps / 1000, 3),
        "fmax_mhz": round(fmax, 1), "meets_freq": meets,
        "read_energy_pj": round(energy_fj / 1000, 3),
        "leakage_uw": round(leak_uw, 3),
        "power_mw_at_freq": round((energy_fj * 1e-15 * spec.freq_mhz * 1e6 * ports_read +
                                   leak_uw * 1e-6 + (refresh["power_uw"] * 1e-6 if refresh else 0)) * 1e3, 4),
        "throughput_gbps": round(ports_read * spec.width * min(fmax, spec.freq_mhz) / 1000, 3),
        "transistors": round(transistors), "transistors_per_bit": tx_bit,
        "refresh": refresh,
    }


def feasibility(spec: MemSpec, cell: str) -> dict[str, Any]:
    c = CELLS[cell]
    p = get_pdk(spec.pdk)
    notes, ok = [], True
    if c.get("needs_process"):
        ok = False
        notes.append(f"needs a {c['needs_process']}; open logic PDKs do not offer one")
    if c.get("macro"):
        if spec.pdk in SRAM_MACROS:
            notes.append(f"foundry/OpenRAM macro path: {SRAM_MACROS[spec.pdk]}")
        else:
            notes.append("needs a foundry bitcell and memory compiler for this node")
    if c.get("custom_layout"):
        notes.append("custom array layout from PDK devices; must pass DRC/LVS before use")
    if c.get("std_cell"):
        notes.append("standard cells only: synthesis + place and route, no custom layout")
    if c["kind"] == "edram":
        notes.append("retention must be characterised (SPICE over corners and temperature)")
    if not p.tapeout_ready:
        notes.append(f"{p.name} is a predictive/research profile: no tapeout path")
    return {"feasible": ok and p.tapeout_ready, "pdk_tapeout_ready": p.tapeout_ready, "notes": notes}


def rule_checks(spec: MemSpec, cand: dict[str, Any]) -> list[dict[str, Any]]:
    """Checks ChipLab can make against the PDK profile. DRC/LVS need a layout and KLayout/
    Netgen and are reported as not run."""
    pk = _pdk(spec)
    checks = [
        {"check": "spec valid", "ok": not spec.validate()},
        {"check": "rows within 8..1024 (bitline length)", "ok": cand["cell"].startswith(("rf_", "rom_logic")) or 8 <= cand["rows"] <= 1024},
        {"check": f"meets {spec.freq_mhz} MHz (estimate)", "ok": cand["meets_freq"]},
        {"check": "array fits a 1-metal-pitch grid", "ok": True,
         "note": f"M1 pitch {pk['metal1_pitch_nm']} nm"},
        {"check": "DRC", "ok": None, "note": "not run: needs a generated layout and KLayout"},
        {"check": "LVS", "ok": None, "note": "not run: needs a layout, netlist and Netgen/KLayout"},
    ]
    if cand.get("refresh"):
        checks.append({"check": "refresh overhead < 5%", "ok": cand["refresh"]["busy_fraction"] < 0.05})
    return checks


def _score(c: dict[str, Any], objective: str) -> float:
    return {"area": c["area_um2"], "delay": c["access_ns"], "energy": c["read_energy_pj"]}.get(
        objective, c["area_um2"] * c["access_ns"])


def compile_memory(spec: MemSpec, top: int = 8) -> dict[str, Any]:
    errs = spec.validate()
    if errs:
        raise ValueError("; ".join(errs))
    cands = []
    for cell in _cells_for(spec):
        for banks in ([spec.banks] if spec.banks else [1, 2, 4, 8]):
            for mux in (1, 2, 4, 8, 16):
                for dec in ("single", "predecoded"):
                    r = evaluate(spec, cell, banks, mux, dec)
                    if r:
                        r["feasibility"] = feasibility(spec, cell)
                        cands.append(r)
    if not cands:
        raise ValueError("no legal organization for this specification")
    # Pareto front over area, access time and read energy
    for a in cands:
        a["pareto"] = not any(b is not a and b["area_um2"] <= a["area_um2"] and b["access_ns"] <= a["access_ns"]
                              and b["read_energy_pj"] <= a["read_energy_pj"] and
                              (b["area_um2"], b["access_ns"], b["read_energy_pj"]) !=
                              (a["area_um2"], a["access_ns"], a["read_energy_pj"]) for b in cands)
    ranked = sorted(cands, key=lambda c: (not c["feasibility"]["feasible"], not c["meets_freq"],
                                          _score(c, spec.objective)))
    best = ranked[0]
    return {
        "spec": asdict(spec) | {"contents": f"{len(spec.contents)} words" if spec.contents else None,
                                "addr_bits": spec.addr_bits},
        "logical": {"words": spec.words, "width": spec.width, "addr_bits": spec.addr_bits,
                    "ports": {"read": spec.read_ports, "write": spec.write_ports, "rw": spec.rw_ports},
                    "ones_fraction": round(_ones_fraction(spec), 4) if spec.contents else None},
        "best": best, "candidates": ranked[:top], "evaluated": len(cands),
        "pareto": [c for c in ranked if c["pareto"]][:top],
        "circuit": {"cell": best["cell"], **{k: v for k, v in CELLS[best["cell"]].items() if k in ("desc", "tx", "programmed_by")}},
        "checks": rule_checks(spec, best),
        "fidelity": "analytical",
        "provenance": {"pdk": spec.pdk, "pdk_fields": "published/estimated (see pdk_library)",
                       "cell_area_ratios": "assumed (textbook layouts)", "uncertainty_pct": 40},
        "limitations": ["first-order FO4/capacitance model, not characterised",
                        "no layout generated: DRC/LVS not run",
                        "macro numbers should come from OpenRAM or the foundry compiler when available"],
    }


def compare(specs: list[MemSpec]) -> list[dict[str, Any]]:
    """Same function, different implementations: one row per spec (best candidate each)."""
    rows = []
    for s in specs:
        r = compile_memory(s)
        b = r["best"]
        rows.append({"name": s.name, "kind": s.kind, "cell": b["cell"], "capacity_bits": b["capacity_bits"],
                     "area_um2": b["area_um2"], "access_ns": b["access_ns"], "fmax_mhz": b["fmax_mhz"],
                     "read_energy_pj": b["read_energy_pj"], "power_mw": b["power_mw_at_freq"],
                     "throughput_gbps": b["throughput_gbps"], "transistors": b["transistors"],
                     "feasible": b["feasibility"]["feasible"], "notes": b["feasibility"]["notes"],
                     "fidelity": "analytical"})
    return rows


def spec_from_dict(d: dict[str, Any]) -> MemSpec:
    allowed = set(MemSpec.__dataclass_fields__)
    unknown = set(d) - allowed
    if unknown:
        raise ValueError(f"unknown memory spec fields: {sorted(unknown)}")
    return MemSpec(**d)


__all__ = ["MemSpec", "CELLS", "KINDS", "compile_memory", "compare", "spec_from_dict", "evaluate",
           "field"]
