"""Workload-to-hardware compiler.

Give a workload (matmul, attention, conv2d, reduction, sparse matrix-vector, graph traversal,
bitwise, or a custom op count); ChipLab lowers it to primitive work (MACs, adds, bitwise ops,
unique bytes, reuse) and maps it onto six implementations:

  cpu, gpu             roofline with cache/HBM traffic from a tiling bound
  sram_nmc             bank-local MACs in SRAM next to the data (weights stationary)
  dram_in_array        Ambit-style bulk bitwise logic (only bitwise/popcount work)
  dram_near_bank       SIMD units at DRAM banks (HBM-PIM / UPMEM style)
  analog_crossbar      resistive crossbar MVM tiles with ADCs (weights stationary)
  hybrid               per-phase best of the above, respecting precision needs

and compares data movement, utilisation, energy, latency, area and precision. All machine
parameters are labelled; this is a first-level (analytical) screen. Use the fidelity
scheduler to decide which candidates deserve circuit/RTL/synthesis runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DTYPE_BITS = {"int1": 1, "int4": 4, "int8": 8, "int16": 16, "fp16": 16, "bf16": 16,
              "fp32": 32}

MACHINES: dict[str, dict[str, tuple[float, str]]] = {
    "cpu": {"peak_ops_s": (2e12, "assumed: 16-core AVX-512 int8 order of magnitude"),
            "dram_bytes_s": (51.2e9, "two DDR4-3200 channels"),
            "cache_bytes": (32 * 2**20, "assumed: 32 MB LLC"),
            "pj_per_op8": (0.5, "assumed: SIMD op incl. register file, 8-bit"),
            "pj_per_byte_dram": (160.0, "derived: Horowitz 2014, ~20 pJ/bit DRAM access"),
            "pj_per_byte_cache": (2.5, "derived: Horowitz 2014, 20 pJ per 64-bit 32 KB read"),
            "area_mm2": (200.0, "assumed: server CPU die share")},
    "gpu": {"peak_ops_s": (400e12, "assumed: datacenter GPU dense int8 order of magnitude"),
            "dram_bytes_s": (2.0e12, "assumed: HBM3-class bandwidth"),
            "cache_bytes": (50 * 2**20, "assumed: L2"),
            "pj_per_op8": (0.1, "assumed: tensor-core int8 MAC energy"),
            "pj_per_byte_dram": (30.0, "assumed: HBM ~3-4 pJ/bit"),
            "pj_per_byte_cache": (1.0, "assumed"),
            "area_mm2": (800.0, "assumed: reticle-class die")},
    "sram_nmc": {"banks": (1024, "assumed: 1024 banks of 8 KB = 8 MB"),
                 "bank_bytes": (8192, "assumed"),
                 "lanes": (8, "8 lanes per bank (chiplab.pim.sram style)"),
                 "clock_hz": (500e6, "assumed"),
                 "pj_per_mac8": (0.23, "Horowitz 2014: 8b mul 0.2 + add 0.03 pJ"),
                 "pj_per_byte_local": (1.25, "Horowitz 2014: 10 pJ per 64-bit 8 KB read"),
                 "pj_per_byte_load": (160.0, "weights/activations loaded from DRAM"),
                 "dram_bytes_s": (51.2e9, "host DRAM feeding the SRAM"),
                 "area_mm2": (40.0, "assumed: 8 MB SRAM + ALUs at an older node")},
    "dram_near_bank": {"banks": (16 * 8, "16 banks x 8 chips"),
                       "bank_bytes_s": (9.6e9, "256 b x 300 MHz per bank (assumed)"),
                       "pj_per_byte": (8.0, "assumed: 1 pJ/bit internal column access"),
                       "pj_per_op8": (0.23, "Horowitz 2014 MAC, logic in DRAM process"),
                       "ops_per_bank_s": (4.8e9, "assumed: 16 lanes x 300 MHz"),
                       "area_mm2": (32.0, "assumed: 16 x 0.25 mm2 x 8 chips")},
    "dram_in_array": {"row_bytes_s": (44.6e9, "chiplab.pim.dram_designer.schedule('and')"),
                      "pj_per_byte": (0.5, "chiplab.pim.dram.timing_energy('and'), "
                                           "upper-bound optimistic"),
                      "area_mm2": (5.0, "~1 % of 8 chips x 60 mm2")},
    "analog_crossbar": {"tile_rows": (128, "assumed: 128x128 tiles (ISAAC-like)"),
                        "tile_cols": (128, "assumed"),
                        "tiles": (1024, "assumed"),
                        "mvm_s": (100e-9, "assumed: one tile MVM incl. ADC"),
                        "pj_per_adc": (2.0, "assumed: 8-bit ADC conversion"),
                        "pj_per_cell": (0.01, "assumed: read energy per cell"),
                        "pj_per_cell_write": (100.0, "assumed: programming a cell"),
                        "effective_bits": (6.0, "assumed: device noise + ADC; see imc_lab"),
                        "area_mm2": (60.0, "assumed")},
}


class WorkloadError(ValueError):
    pass


@dataclass
class Work:
    name: str
    macs: float = 0.0
    adds: float = 0.0
    bitwise_bytes: float = 0.0       # bulk bitwise work, in bytes of operand
    unique_bytes: float = 0.0        # compulsory traffic (inputs + outputs once)
    weight_bytes: float = 0.0        # stationary operand (matrix) if any
    out_bytes: float = 0.0
    bits: int = 8
    reuse_kind: str = "none"         # gemm | stream | irregular
    irregular: bool = False
    phases: tuple[tuple[str, float], ...] = ()   # (phase, fraction of macs)

    @property
    def ops(self) -> float:
        return 2 * self.macs + self.adds + self.bitwise_bytes


def lower(spec: dict[str, Any]) -> Work:
    """Turn a workload description into primitive work."""
    kind = spec.get("kind")
    dt = spec.get("dtype", "int8")
    if dt not in DTYPE_BITS:
        raise WorkloadError(f"dtype must be one of {list(DTYPE_BITS)}")
    b = DTYPE_BITS[dt]
    B = b / 8

    def pos(k: str, default: float | None = None) -> float:
        v = spec.get(k, default)
        if not isinstance(v, (int, float)) or v <= 0 or v > 1e12:
            raise WorkloadError(f"{k} must be a positive number")
        return float(v)

    if kind == "matmul":
        m, n, k = pos("m"), pos("n", spec.get("m")), pos("k", spec.get("m"))
        return Work("matmul", macs=m * n * k, unique_bytes=(m * k + k * n + m * n) * B,
                    weight_bytes=k * n * B, out_bytes=m * n * B, bits=b, reuse_kind="gemm")
    if kind == "attention":
        s, d, h = pos("seq"), pos("head_dim", 64), pos("heads", 1)
        qk = h * s * s * d
        return Work("attention", macs=2 * qk, adds=h * s * s * 5,
                    unique_bytes=(4 * h * s * d) * B + h * s * s * B,
                    weight_bytes=0, out_bytes=h * s * d * B, bits=b, reuse_kind="gemm",
                    phases=(("qk_t", 0.5), ("softmax", 0.0), ("pv", 0.5)))
    if kind == "conv2d":
        hh, ww, ci, co, kk = pos("h"), pos("w"), pos("cin"), pos("cout"), pos("k", 3)
        macs = hh * ww * ci * co * kk * kk
        return Work("conv2d", macs=macs, unique_bytes=(hh * ww * (ci + co) + kk * kk * ci * co)
                    * B, weight_bytes=kk * kk * ci * co * B, out_bytes=hh * ww * co * B,
                    bits=b, reuse_kind="gemm")
    if kind == "reduction":
        n = pos("n")
        return Work("reduction", adds=n, unique_bytes=n * B, out_bytes=B, bits=b,
                    reuse_kind="stream")
    if kind == "sparse_mv":
        rows, cols, nnz = pos("rows"), pos("cols"), pos("nnz")
        return Work("sparse_mv", macs=nnz, unique_bytes=nnz * (B + 4) + cols * B + rows * B,
                    weight_bytes=nnz * (B + 4), out_bytes=rows * B, bits=b,
                    reuse_kind="irregular", irregular=True)
    if kind == "graph_bfs":
        v, e = pos("vertices"), pos("edges")
        return Work("graph_bfs", adds=e, unique_bytes=e * 4 + v * 8, out_bytes=v * 4, bits=32,
                    reuse_kind="irregular", irregular=True)
    if kind == "bitwise":
        n = pos("n_bytes")
        return Work("bitwise", bitwise_bytes=n, unique_bytes=3 * n, out_bytes=n, bits=1,
                    reuse_kind="stream")
    if kind == "custom":
        return Work("custom", macs=float(spec.get("macs", 0)), adds=float(spec.get("adds", 0)),
                    bitwise_bytes=float(spec.get("bitwise_bytes", 0)),
                    unique_bytes=pos("bytes"), bits=b, reuse_kind=spec.get("reuse", "stream"))
    raise WorkloadError("kind must be matmul, attention, conv2d, reduction, sparse_mv, "
                        "graph_bfs, bitwise or custom")


def _m(name: str) -> dict[str, float]:
    return {k: v for k, (v, _) in MACHINES[name].items()}


def _traffic(w: Work, cache: float) -> float:
    """Off-chip bytes: compulsory traffic, plus re-reads for GEMM when the working set
    exceeds the cache (Hong-Kung style bound ~ 2*MACs*B/sqrt(S/B))."""
    if w.reuse_kind == "gemm" and w.macs:
        B = w.bits / 8
        elems = cache / max(B, 1 / 8)
        return max(w.unique_bytes, 2 * w.macs * B / math.sqrt(elems))
    if w.irregular:
        return w.unique_bytes * 4          # poor locality: ~4x line over-fetch (assumed)
    return w.unique_bytes


def _roofline(w: Work, name: str) -> dict[str, Any]:
    m = _m(name)
    traffic = _traffic(w, m["cache_bytes"])
    ops = w.ops * max(1, w.bits / 8)
    t = max(ops / m["peak_ops_s"], traffic / m["dram_bytes_s"])
    e = (ops * m["pj_per_op8"] + traffic * m["pj_per_byte_dram"] + 3 * w.unique_bytes * m[
        "pj_per_byte_cache"]) * 1e-12
    return {"supported": True, "latency_s": t, "energy_j": e, "data_movement_bytes": traffic,
            "utilization": (ops / m["peak_ops_s"]) / t, "area_mm2": m["area_mm2"],
            "precision": f"exact {w.bits}-bit" if w.bits <= 32 else "fp",
            "bound": "compute" if ops / m["peak_ops_s"] >= traffic / m["dram_bytes_s"]
            else "memory"}


def _sram(w: Work) -> dict[str, Any]:
    m = _m("sram_nmc")
    cap = m["banks"] * m["bank_bytes"]
    if w.bitwise_bytes and not w.macs:
        rate = m["banks"] * m["lanes"] * m["clock_hz"]
    else:
        rate = m["banks"] * m["lanes"] * m["clock_hz"] / max(1, w.bits / 8)
    stationary = w.weight_bytes or w.unique_bytes
    passes = max(1.0, math.ceil(stationary / cap))
    load = w.unique_bytes + (passes - 1) * (w.unique_bytes - stationary)
    work = w.macs + w.adds + w.bitwise_bytes
    t = max(work / rate, load / m["dram_bytes_s"])
    e = (w.macs * m["pj_per_mac8"] * (w.bits / 8) ** 2 + (w.adds + w.bitwise_bytes) * 0.03 +
         2 * work * m["pj_per_byte_local"] * w.bits / 8 + load * m["pj_per_byte_load"]) * 1e-12
    return {"supported": True, "latency_s": t, "energy_j": e, "data_movement_bytes": load,
            "utilization": (work / rate) / t, "area_mm2": m["area_mm2"],
            "precision": f"exact {w.bits}-bit", "fits_on_chip": stationary <= cap,
            "bound": "compute" if work / rate >= load / m["dram_bytes_s"] else "load"}


def _near_bank(w: Work) -> dict[str, Any]:
    m = _m("dram_near_bank")
    bw = m["banks"] * m["bank_bytes_s"]
    ops_rate = m["banks"] * m["ops_per_bank_s"] / max(1, w.bits / 8)
    stream = _traffic(w, 64 * 1024 * m["banks"])        # small per-bank buffers
    work = w.macs + w.adds + w.bitwise_bytes
    t = max(work / ops_rate, stream / bw)
    e = (stream * m["pj_per_byte"] + work * m["pj_per_op8"] * (w.bits / 8)) * 1e-12
    note = None
    if w.irregular:
        note = "irregular access crosses banks: inter-bank traffic is not modelled (optimistic)"
    return {"supported": True, "latency_s": t, "energy_j": e, "data_movement_bytes": 0,
            "internal_bytes": stream, "utilization": (work / ops_rate) / t,
            "area_mm2": m["area_mm2"], "precision": f"exact {w.bits}-bit",
            "bound": "compute" if work / ops_rate >= stream / bw else "bank bandwidth",
            "note": note}


def _in_array(w: Work) -> dict[str, Any]:
    m = _m("dram_in_array")
    if w.macs or w.adds or not w.bitwise_bytes:
        return {"supported": False,
                "why": "in-array DRAM does bulk bitwise logic and copy; arithmetic needs "
                       "bit-serial majority sequences (SIMDRAM-like), not modelled"}
    t = w.bitwise_bytes / m["row_bytes_s"]
    return {"supported": True, "latency_s": t, "energy_j": w.bitwise_bytes * m["pj_per_byte"]
            * 1e-12, "data_movement_bytes": 0, "utilization": 1.0, "area_mm2": m["area_mm2"],
            "precision": "exact bitwise; BER from chiplab.pim.dram.monte_carlo",
            "bound": "activation rate (tFAW)"}


def _analog(w: Work) -> dict[str, Any]:
    m = _m("analog_crossbar")
    if not w.macs or w.irregular or w.reuse_kind != "gemm":
        return {"supported": False,
                "why": "crossbars need a dense stationary matrix (MVM/GEMM-like work)"}
    rows, cols = m["tile_rows"], m["tile_cols"]
    weight_elems = w.weight_bytes / (w.bits / 8) if w.weight_bytes else w.macs ** (2 / 3)
    tiles_needed = math.ceil(weight_elems / (rows * cols))
    mvms = w.macs / (rows * cols)
    rounds = math.ceil(tiles_needed / m["tiles"])
    t = mvms / min(tiles_needed, m["tiles"]) * m["mvm_s"]
    reprogram = (rounds - 1) * m["tiles"] * rows * cols if rounds > 1 else 0
    adc = mvms * cols
    e = (adc * m["pj_per_adc"] + w.macs * m["pj_per_cell"] + (weight_elems + reprogram) * m[
        "pj_per_cell_write"]) * 1e-12
    return {"supported": True, "latency_s": t, "energy_j": e,
            "data_movement_bytes": w.unique_bytes - w.weight_bytes,
            "utilization": min(1.0, tiles_needed / m["tiles"]), "area_mm2": m["area_mm2"],
            "precision": f"~{m['effective_bits']:.0f} effective bits (analog)",
            "effective_bits": m["effective_bits"], "tiles_needed": tiles_needed,
            "weights_reprogrammed": rounds > 1, "bound": "ADC conversions"}


def compile_workload(spec: dict[str, Any], min_precision_bits: float | None = None,
                     objective: str = "energy_j") -> dict[str, Any]:
    """Lower a workload and evaluate every implementation. min_precision_bits removes
    implementations that cannot meet the precision (analog)."""
    w = lower(spec)
    impl = {"cpu": _roofline(w, "cpu"), "gpu": _roofline(w, "gpu"), "sram_nmc": _sram(w),
            "dram_in_array": _in_array(w), "dram_near_bank": _near_bank(w),
            "analog_crossbar": _analog(w)}
    need = min_precision_bits if min_precision_bits is not None else w.bits
    for k, v in impl.items():
        if v.get("supported") and v.get("effective_bits") and v["effective_bits"] < need:
            v["meets_precision"] = False
        elif v.get("supported"):
            v["meets_precision"] = True
    ok = {k: v for k, v in impl.items() if v.get("supported") and v.get("meets_precision")}
    if objective not in ("energy_j", "latency_s", "area_mm2", "data_movement_bytes"):
        raise WorkloadError("objective must be energy_j, latency_s, area_mm2 or "
                            "data_movement_bytes")
    # hybrid: memory-bound phases (softmax, reductions) on near-bank, MAC phases on the best
    # MAC engine that meets precision.
    mac_engines = {k: v for k, v in ok.items() if k in ("sram_nmc", "analog_crossbar", "gpu",
                                                        "cpu")}
    hybrid = None
    if w.macs and (w.adds or w.phases) and mac_engines and "dram_near_bank" in ok:
        best_mac = min(mac_engines, key=lambda k: mac_engines[k][objective])
        mac_w = Work(w.name, macs=w.macs, unique_bytes=w.unique_bytes, weight_bytes=w
                     .weight_bytes, bits=w.bits, reuse_kind=w.reuse_kind)
        mem_w = Work(w.name, adds=w.adds, unique_bytes=w.adds * w.bits / 8, bits=w.bits,
                     reuse_kind="stream")
        a = {"cpu": lambda x: _roofline(x, "cpu"), "gpu": lambda x: _roofline(x, "gpu"),
             "sram_nmc": _sram, "analog_crossbar": _analog}[best_mac](mac_w)
        b = _near_bank(mem_w)
        hybrid = {"supported": True, "meets_precision": True, "composition":
                  {"mac_phases": best_mac, "elementwise_phases": "dram_near_bank"},
                  "latency_s": a["latency_s"] + b["latency_s"],
                  "energy_j": a["energy_j"] + b["energy_j"],
                  "data_movement_bytes": a["data_movement_bytes"],
                  "utilization": a["utilization"], "area_mm2": a["area_mm2"] + b["area_mm2"],
                  "precision": a["precision"]}
    impl["hybrid"] = hybrid or {"supported": False,
                                "why": "workload has no separable elementwise phase"}
    if hybrid:
        ok["hybrid"] = hybrid
    ranking = sorted(ok, key=lambda k: ok[k][objective])
    return {"workload": w.name, "lowered": {"macs": w.macs, "adds": w.adds,
                                            "bitwise_bytes": w.bitwise_bytes,
                                            "unique_bytes": w.unique_bytes, "bits": w.bits},
            "implementations": impl, "objective": objective, "ranking": ranking,
            "fidelity": "analytical",
            "parameters": {k: {p: {"value": v, "source": s} for p, (v, s) in m.items()}
                           for k, m in MACHINES.items()},
            "caveats": ["First-level screen with labelled assumptions; promote candidates "
                        "with chiplab.science.fidelity before trusting a ranking.",
                        "Analog energy omits DAC/peripheral power beyond ADCs."]}
