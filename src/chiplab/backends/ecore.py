"""eCore backend: Echo Lab's analog DRAM charge-domain compute-in-memory simulator.

ChipLab asks eCore for array-level numbers (accuracy, latency, energy, area) and uses them
for package-level estimates of PIM / analog-CIM chiplets. Echo Lab stays a separate,
read-only codebase.

Integration: Echo Lab's runtime dependencies (numpy, PyYAML) are already in ChipLab's venv,
so it is imported in-process from a fixed path (``~/.gemini/antigravity/scratch/Echo Lab``,
overridable only by the server operator through the ``CHIPLAB_ECHO_LAB_PATH`` environment
variable). Nothing here runs a shell, a subprocess, eval or exec, and no MCP argument can
choose a path or a command: MCP clients only pick from validated enums and bounded ints.

If Echo Lab is not importable, or its SPICE calibration file is missing (eCore would otherwise
silently fall back to analytical defaults), ``is_available()`` is False and every simulation
call raises ``EcoreUnavailableError``. No empty or fallback result is ever returned.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..schemas.common import FidelityLevel, Provenance
from ..schemas.design import ChipletSpec, DesignSpec
from ..schemas.simulation import AssumptionRecord, SimulationConfig, SimulationResult
from .base import SimulatorBackend

DEFAULT_ECHO_LAB_PATH = Path.home() / ".gemini" / "antigravity" / "scratch" / "Echo Lab"
CALIBRATION_REL = Path("results") / "calibration" / "spice_calibrated_params.json"

# precision -> (weight_bits, input_bits, ChipLab precision label, accuracy path)
PRECISIONS: dict[str, tuple[int, int, str, str]] = {
    "int4": (4, 4, "INT4", "int"),
    "int8": (8, 8, "INT8", "int"),
    "fp8_e4m3": (8, 8, "FP8", "fp"),
    "fp8_e5m2": (8, 8, "FP8", "fp"),
    "fp16": (16, 16, "FP16", "fp"),
}
ORGANIZATIONS = ("org_a", "org_b")
ADC_BITS = (4, 6, 8, 10, 12)
MAX_DIM = 1024
MIN_DIM = 8
ACCURACY_VECTORS = 8

FIDELITY_DISCLAIMER = (
    "eCore array-level result (FidelityLevel.SPECIALIZED): behavioral charge-domain model "
    "calibrated against ngspice simulations with GENERIC Level-1 MOSFET models, not a foundry "
    "PDK and not silicon measurements. ADC, periphery, area and refresh terms are analytical "
    "assumptions (each number carries its Echo Lab source tag).")

# Echo Lab source tag -> ChipLab provenance. Simulated numbers are estimates, never "measured".
_SOURCE_TO_PROVENANCE = {
    "ngspice_derived": Provenance.ESTIMATED,
    "analytical_assumption": Provenance.ASSUMED,
}

_import_lock = threading.Lock()


class EcoreUnavailableError(RuntimeError):
    """Echo Lab / eCore cannot be used (not installed, or not SPICE-calibrated)."""


def echo_lab_path() -> Path:
    return Path(os.environ.get("CHIPLAB_ECHO_LAB_PATH") or DEFAULT_ECHO_LAB_PATH).expanduser()


@lru_cache(maxsize=1)
def _load_echo_lab() -> dict[str, Any]:
    """Import Echo Lab in-process. Raises EcoreUnavailableError with the reason."""
    root = echo_lab_path().resolve()
    if not (root / "simulator" / "ecore" / "ecore.py").is_file():
        raise EcoreUnavailableError(f"Echo Lab not found at {root} (simulator/ecore/ecore.py).")
    cal_path = root / CALIBRATION_REL
    if not cal_path.is_file():
        raise EcoreUnavailableError(
            f"eCore SPICE calibration missing at {cal_path}; refusing to use eCore's "
            "uncalibrated analytical fallback.")
    with _import_lock:
        if str(root) not in sys.path:
            # Appended, not prepended, so Echo Lab never shadows ChipLab's own imports.
            sys.path.append(str(root))
        # Echo Lab is read-only: never write __pycache__ files into its tree.
        prev_dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            import simulator as _sim_pkg
            import workloads as _wl_pkg
            from simulator.ecore.ecore import ECore, ECoreConfig
            from workloads.matvec.analog_fp_matvec import AnalogFPConfig, AnalogFPMatVec
        except Exception as e:  # ImportError or a broken Echo Lab module
            raise EcoreUnavailableError(f"Echo Lab import failed: {type(e).__name__}: {e}") from e
        finally:
            sys.dont_write_bytecode = prev_dwb
    for pkg in (_sim_pkg, _wl_pkg):
        if not Path(pkg.__file__ or "").resolve().is_relative_to(root):
            raise EcoreUnavailableError(
                f"Module '{pkg.__name__}' resolved to {pkg.__file__}, not Echo Lab at {root}.")
    raw = cal_path.read_bytes()
    mtime = _dt.datetime.fromtimestamp(cal_path.stat().st_mtime, tz=_dt.timezone.utc)
    return {
        "root": root, "cal_path": cal_path, "ECore": ECore, "ECoreConfig": ECoreConfig,
        "AnalogFPConfig": AnalogFPConfig, "AnalogFPMatVec": AnalogFPMatVec,
        "cal_sha256": hashlib.sha256(raw).hexdigest(),
        "cal_date": mtime.isoformat(timespec="seconds"),
    }


def _tagged(value: float, unit: str, src: str) -> dict[str, Any]:
    return {"value": float(value), "unit": unit, "source": f"echo_lab:{src}",
            "provenance": _SOURCE_TO_PROVENANCE.get(src, Provenance.ASSUMED).value}


def _flatten(breakdown: dict[str, Any], unit: str) -> dict[str, Any]:
    """Turn Echo Lab's nested {"x": {"value", "source"}} breakdowns into tagged ChipLab dicts."""
    out: dict[str, Any] = {}
    for key, v in breakdown.items():
        if isinstance(v, dict) and "value" in v and "source" in v:
            u = unit if key.endswith(("_ns", "_fJ", "_um2")) else ""
            out[key] = _tagged(v["value"], u, v["source"])
        elif isinstance(v, dict):
            for sub, sv in _flatten(v, unit).items():
                out[f"{key}.{sub}"] = sv
    return out


def _error_metrics(y, ref) -> dict[str, float | None]:
    import numpy as np
    y = np.asarray(y, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    err = y - ref
    ref_norm = float(np.linalg.norm(ref))
    err_norm = float(np.linalg.norm(err))
    # Non-finite values become None so MCP results stay valid JSON.
    return {k: (v if math.isfinite(v) else None) for k, v in {
        "snr_db_vs_reference": (20 * math.log10(ref_norm / err_norm) if err_norm > 0
                                else float("inf")),
        "rel_l2_error": err_norm / ref_norm if ref_norm > 0 else float("nan"),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error": float(np.max(np.abs(err))),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "pearson_r": float(np.corrcoef(y, ref)[0, 1]) if np.std(y) > 0 else 0.0,
    }.items()}


def validate_args(rows: int, cols: int, precision: str, adc_bits: int,
                  organization: str) -> None:
    for name, v in (("rows", rows), ("cols", cols)):
        if not isinstance(v, int) or isinstance(v, bool) or not MIN_DIM <= v <= MAX_DIM:
            raise ValueError(f"{name} must be an int in [{MIN_DIM}, {MAX_DIM}], got {v!r}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
    if adc_bits not in ADC_BITS:
        raise ValueError(f"adc_bits must be one of {ADC_BITS}, got {adc_bits!r}")
    if organization not in ORGANIZATIONS:
        raise ValueError(f"organization must be one of {ORGANIZATIONS}, got {organization!r}")


@lru_cache(maxsize=64)
def _simulate_cached(rows: int, cols: int, precision: str, adc_bits: int,
                     organization: str, seed: int) -> dict[str, Any]:
    import numpy as np
    el = _load_echo_lab()
    wb, ib, label, path = PRECISIONS[precision]
    cfg = el["ECoreConfig"](rows=rows, cols=cols, org=organization, adc_bits=adc_bits,
                            subtile_rows=min(64, rows), seed=seed,
                            calibration_path=str(el["cal_path"]))
    core = el["ECore"](cfg)
    if not core.cal.get("calibrated", False):
        raise EcoreUnavailableError(
            "eCore loaded uncalibrated analytical fallback parameters "
            f"({core.cal.get('_fallback_warning', 'no reason given')}); refusing to report them.")

    rng = np.random.default_rng(seed)
    if path == "int":
        lo, hi = -(2 ** (wb - 1)), 2 ** (wb - 1)
        w = rng.integers(lo, hi, size=(rows, cols), dtype=np.int32)
        ys, refs = [], []
        for _ in range(ACCURACY_VECTORS):
            x = rng.integers(lo, hi, size=(cols,), dtype=np.int32)
            y, _ = core.matvec_signed_int(w, x, weight_bits=wb, input_bits=ib)
            ys.append(y)
            refs.append(w.astype(np.float64) @ x.astype(np.float64))
        accuracy = {"method": "ecore_bitlevel_signed_int_matvec",
                    "description": "Bit-sliced weights x bit-serial inputs through eCore's "
                                   "charge-domain tile model, ADC and shift-add; compared "
                                   "with exact integer W@x.",
                    **_error_metrics(np.concatenate(ys), np.concatenate(refs))}
    else:
        fp = el["AnalogFPMatVec"](el["AnalogFPConfig"](
            format=precision, organization=organization, adc_bits=adc_bits, rows=rows,
            cols=cols, subtile_rows=min(64, rows), seed=seed,
            calibration_path=str(el["cal_path"])))
        W = rng.standard_normal((rows, cols)).astype(np.float32)
        X = rng.standard_normal((cols, ACCURACY_VECTORS)).astype(np.float32)
        y, info = fp.matvec(W, X)
        ref = W.astype(np.float64) @ X.astype(np.float64)
        accuracy = {"method": "calibrated_statistical_fp_model",
                    "description": "Echo Lab's FP pipeline applies SPICE-calibrated noise, "
                                   "mismatch and nonlinearity statistics to the quantized "
                                   "digital reference; it does not run the bit-level eCore "
                                   f"tile model (mode={info.get('mode')}, "
                                   f"ecore_calls={info.get('ecore_calls')}). Compared with "
                                   "FP64 W@X on unquantized inputs.",
                    **_error_metrics(y, ref)}
    accuracy["vectors"] = ACCURACY_VECTORS
    accuracy["seed"] = seed

    hw = core.estimate_hardware_metrics(weight_bits=wb, input_bits=ib)
    meta = core.cal["metadata"]
    t_ns = float(hw["latency_ns"]["t_total_ns"]["value"])
    e_fj = float(hw["energy_fJ"]["e_total_fJ"]["value"])
    a_um2 = float(hw["area_um2"]["a_total_um2"]["value"])
    macs = rows * cols
    ops = 2 * macs
    derived = {
        "macs_per_matvec": macs,
        "ops_per_matvec": ops,
        "array_tops": _tagged(ops / t_ns / 1e3, "TOPS", "ngspice_derived"),
        "energy_pj_per_op": _tagged(e_fj / 1e3 / ops, "pJ/op", "ngspice_derived"),
        "array_power_w": _tagged(e_fj * 1e-15 / (t_ns * 1e-9), "W", "ngspice_derived"),
        "array_area_mm2": _tagged(a_um2 * 1e-6, "mm2", "analytical_assumption"),
        "weight_capacity_bytes": float(hw["weight_capacity"]["weight_capacity_bytes"]["value"]),
    }
    return {
        "success": True,
        "backend": "ecore",
        "fidelity": FidelityLevel.SPECIALIZED.value,
        "fidelity_disclaimer": FIDELITY_DISCLAIMER,
        "inputs": {"rows": rows, "cols": cols, "precision": precision, "adc_bits": adc_bits,
                   "organization": organization, "seed": seed, "weight_bits": wb,
                   "input_bits": ib, "chiplab_precision": label},
        "accuracy": accuracy,
        "per_mac_snr_db": {
            **_tagged(hw["snr_db"]["value"], "dB", hw["snr_db"]["source"]),
            "note": "Single-MAC signal swing / calibrated noise rms (eCore estimate_hardware_"
                    "metrics). Not an end-to-end matvec accuracy; see 'accuracy'."},
        "latency_ns": {"total": _tagged(t_ns, "ns", hw["latency_ns"]["t_total_ns"]["source"]),
                       "breakdown": _flatten(hw["latency_ns"], "ns")},
        "energy_fj": {"total": _tagged(e_fj, "fJ", hw["energy_fJ"]["e_total_fJ"]["source"]),
                      "breakdown": _flatten(hw["energy_fJ"], "fJ")},
        "area_um2": {"total": _tagged(a_um2, "um2", hw["area_um2"]["a_total_um2"]["source"]),
                     "breakdown": _flatten(hw["area_um2"], "um2")},
        "derived": derived,
        "calibration": {
            "kind": "spice",
            "measured_silicon": False,
            "simulator": meta.get("spice_simulator"),
            "model_type": meta.get("model_type"),
            "warning": meta.get("silicon_feasibility_warning"),
            "file": str(el["cal_path"]),
            "sha256": el["cal_sha256"],
            "date": el["cal_date"],
            "date_source": "calibration file modification time (the JSON carries no date)",
            "temp_c": meta.get("temp_c"),
            "vdd_v": meta.get("vdd_v"),
        },
        "echo_lab": {"path": str(el["root"])},
        "limitations": _limitations(precision, organization, accuracy),
    }


def _limitations(precision: str, organization: str, accuracy: dict[str, Any]) -> list[str]:
    lim = [
        "SPICE calibration uses generic Level-1 MOSFET models, not a foundry PDK; no silicon "
        "measurements back any number.",
        "ADC energy/time, digital control, sense-amp/ADC/cell area and refresh overhead are "
        "Echo Lab analytical assumptions (cell area assumes ~45 nm 1T1C); no process-node "
        "scaling is applied.",
        "Latency serializes ADC conversions per output column; a single array is modeled, "
        "with a fixed 0.5 ns interconnect stub and no multi-array network.",
        "Energy is one matvec with all cells read; idle, leakage and periphery (I/O, "
        "controllers) power outside the array are not modeled.",
    ]
    if accuracy["method"] == "calibrated_statistical_fp_model":
        lim.append(
            "FP accuracy comes from a statistical noise model over the digital reference, "
            "not from the bit-level eCore tile path; treat it as an optimistic bound.")
    snr = accuracy.get("snr_db_vs_reference")
    if snr is not None and snr < 10.0:
        lim.append(
            f"End-to-end accuracy is poor (SNR {accuracy['snr_db_vs_reference']:.1f} dB, "
            f"rel L2 {accuracy['rel_l2_error']:.2f}): this eCore configuration does not "
            "reproduce W@x. Echo Lab's own Stage 3 sweep (snr_db_vs_exact ~0 dB) shows the "
            "same; throughput/energy are for an array that is not yet numerically usable.")
    if organization == "org_a":
        lim.append("Org A reads destructively; restore energy is included, wear is not.")
    return lim


class EcoreBackend(SimulatorBackend):
    """Array-level analog DRAM CIM simulator (Echo Lab eCore)."""

    name: str = "ecore"
    display_name: str = "Echo Lab eCore"
    fidelity: FidelityLevel = FidelityLevel.SPECIALIZED
    integrated: bool = True

    def unavailable_reason(self) -> str | None:
        try:
            _load_echo_lab()
        except EcoreUnavailableError as e:
            return str(e)
        return None

    def is_available(self) -> bool:
        return self.unavailable_reason() is None

    def status(self) -> dict[str, Any]:
        reason = self.unavailable_reason()
        st: dict[str, Any] = {"name": self.name, "installed": reason is None,
                              "integrated": self.integrated, "usable": reason is None,
                              "scope": "array-level analog DRAM CIM (PIM chiplets only)"}
        if reason:
            st["reason"] = reason
        return st

    def simulate(self, rows: int = 64, cols: int = 64, precision: str = "int8",
                 adc_bits: int = 8, organization: str = "org_b",
                 seed: int = 42) -> dict[str, Any]:
        """Simulate one rows x cols eCore array; returns a fresh deep copy of the result."""
        import copy
        validate_args(rows, cols, precision, adc_bits, organization)
        if not isinstance(seed, int) or not 0 <= seed < 2 ** 31:
            raise ValueError(f"seed must be an int in [0, 2**31), got {seed!r}")
        reason = self.unavailable_reason()
        if reason:
            raise EcoreUnavailableError(reason)
        return copy.deepcopy(_simulate_cached(rows, cols, precision, adc_bits,
                                              organization, seed))

    def run(self, design: DesignSpec, workload: Any, config: SimulationConfig
            ) -> SimulationResult:
        """Resolve the design's eCore-backed chiplets, then run ChipLab's analytical model.

        The package-level result stays labeled analytical; the array numbers it used are
        listed as eCore assumptions. Raises if no chiplet declares ``compute.ecore``.
        """
        from ..engine import run_analytical_simulation
        if not design_uses_ecore(design):
            raise ValueError("Design has no chiplet with a compute.ecore block; nothing for "
                             "the eCore backend to simulate.")
        reason = self.unavailable_reason()
        if reason:
            raise EcoreUnavailableError(reason)
        # The analytical engine applies apply_ecore_to_design and records the provenance.
        return run_analytical_simulation(design, workload, config)


# --- Chiplet integration ---------------------------------------------------------------------

def design_uses_ecore(design: DesignSpec) -> bool:
    return any(c.compute is not None and c.compute.ecore is not None for c in design.chiplets)


def _resolve_chiplet(ch: ChipletSpec, result: dict[str, Any]
                     ) -> tuple[dict[str, Any], list[AssumptionRecord]]:
    spec = ch.compute.ecore  # type: ignore[union-attr]
    d = result["derived"]
    arr_area = d["array_area_mm2"]["value"]
    arr_tops = d["array_tops"]["value"]
    arr_w = d["array_power_w"]["value"]
    frac = float(spec.array_area_fraction)
    if spec.arrays is not None:
        n = int(spec.arrays)
        area = n * arr_area / frac
        area_src = f"eCore: {n} arrays x {arr_area:.4g} mm2 / array_area_fraction {frac}"
    elif ch.area_mm2:
        area = float(getattr(ch.area_mm2, "value", ch.area_mm2))
        n = int(area * frac // arr_area)
        area_src = "chiplet_spec (array count derived from it)"
        if n < 1:
            raise ValueError(f"Chiplet '{ch.id}': area_mm2 {area} x {frac} fits no "
                             f"{arr_area:.4g} mm2 eCore array.")
    else:
        raise ValueError(f"Chiplet '{ch.id}': compute.ecore needs 'arrays' or chiplet area_mm2.")
    peak = n * arr_tops
    power = n * arr_w
    tag = (f"ecore:{result['inputs']['rows']}x{result['inputs']['cols']}/"
           f"{result['inputs']['precision']}/{result['inputs']['organization']}/"
           f"adc{result['inputs']['adc_bits']}")
    cal = result["calibration"]
    note = (f"Echo Lab eCore, SPICE ({cal['simulator']}, generic models) calibration dated "
            f"{cal['date']}, sha256 {cal['sha256'][:12]}")
    prov = {
        "chiplet": ch.id, "arrays": n, "array_area_fraction": frac,
        "peak_tops": peak, "tdp_watts": power, "area_mm2": area, "area_source": area_src,
        "energy_pj_per_op": d["energy_pj_per_op"]["value"],
        "accuracy": {k: result["accuracy"][k] for k in
                     ("method", "snr_db_vs_reference", "rel_l2_error")},
        "inputs": result["inputs"], "calibration": cal, "limitations": result["limitations"],
        "source": tag,
    }
    recs = [
        AssumptionRecord(parameter=f"{ch.id}.peak_tops", value=peak, unit="TOPS",
                         provenance=Provenance.ESTIMATED, source=tag,
                         notes=f"{n} arrays x {arr_tops:.4g} TOPS; {note}"),
        AssumptionRecord(parameter=f"{ch.id}.energy_pj_per_op",
                         value=d["energy_pj_per_op"]["value"], unit="pJ/op",
                         provenance=Provenance.ESTIMATED, source=tag, notes=note),
        AssumptionRecord(parameter=f"{ch.id}.tdp_watts", value=power, unit="W",
                         provenance=Provenance.ESTIMATED, source=tag,
                         notes="All arrays active at eCore energy/op; periphery not included."),
        AssumptionRecord(parameter=f"{ch.id}.area_mm2", value=area, unit="mm2",
                         provenance=(Provenance.ASSUMED if spec.arrays is not None
                                     else get_area_provenance(ch)),
                         source=tag if spec.arrays is not None else "chiplet_spec",
                         notes=area_src),
        AssumptionRecord(parameter=f"{ch.id}.array_area_fraction", value=frac,
                         unit="fraction", provenance=Provenance.ASSUMED, source="ecore_spec"),
        AssumptionRecord(parameter=f"{ch.id}.ecore_accuracy_snr_db",
                         value=result["accuracy"]["snr_db_vs_reference"], unit="dB",
                         provenance=Provenance.ESTIMATED, source=tag,
                         notes=result["accuracy"]["method"]),
    ]
    return prov, recs


def get_area_provenance(ch: ChipletSpec) -> Provenance:
    from ..schemas.common import get_provenance
    return get_provenance(ch.area_mm2) if ch.area_mm2 is not None else Provenance.ASSUMED


def apply_ecore_to_design(design: DesignSpec, backend: EcoreBackend | None = None
                          ) -> tuple[DesignSpec, list[AssumptionRecord], list[str]]:
    """Return a copy of ``design`` whose ``compute.ecore`` chiplets take peak_tops, tdp_watts
    and (when ``arrays`` is set) area_mm2 from eCore, with provenance in
    ``metadata['ecore_provenance']``. Designs without eCore chiplets come back unchanged.
    Raises EcoreUnavailableError if a chiplet needs eCore and it is not available."""
    if not design_uses_ecore(design):
        return design, [], []
    be = backend or EcoreBackend()
    out = design.model_copy(deep=True)
    records: list[AssumptionRecord] = []
    warns: list[str] = []
    provenance: dict[str, Any] = {}
    for idx, ch in enumerate(out.chiplets):
        if ch.compute is None or ch.compute.ecore is None:
            continue
        spec = ch.compute.ecore
        ch.id = ch.id or f"chiplet_{idx}"
        res = be.simulate(rows=spec.rows, cols=spec.cols, precision=spec.precision,
                          adc_bits=spec.adc_bits, organization=spec.organization,
                          seed=spec.seed)
        prov, recs = _resolve_chiplet(ch, res)
        ch.compute.peak_tops = prov["peak_tops"]
        ch.compute.precision = res["inputs"]["chiplab_precision"]
        ch.tdp_watts = prov["tdp_watts"]
        ch.area_mm2 = prov["area_mm2"]
        provenance[ch.id] = prov
        records.extend(recs)
        acc = res["accuracy"]
        snr = acc["snr_db_vs_reference"]
        warns.append(
            f"{ch.id}: compute rate, energy/op and power from eCore ({prov['source']}); "
            f"array accuracy SNR {'exact' if snr is None else f'{snr:.1f} dB'} "
            f"({acc['method']}). {FIDELITY_DISCLAIMER}")
        if snr is not None and snr < 10.0:
            warns.append(f"{ch.id}: eCore array does not reproduce W@x at this configuration "
                         f"(rel L2 {acc['rel_l2_error']:.2f}); its TOPS are not usable "
                         "compute until the accuracy problem is fixed.")
    out.metadata = {**out.metadata, "ecore_provenance": provenance}
    return out, records, warns
