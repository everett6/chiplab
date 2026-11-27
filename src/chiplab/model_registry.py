"""Scientific model registry: what each physics model computes and how far to trust it.

A new material model plugs in by calling ``register(ModelRecord(...))`` from its own
module; ``load(model_id)`` then returns its entry point.  The registry is a catalogue: the
design engines do not dispatch through it, and registering a model does not let it into an
architecture-level design (see ``chiplab.credibility``).

The equations below were transcribed from the code.  "not validated against measurements"
means exactly that.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

FIDELITY = ("heuristic", "analytical", "reduced_order", "particle_resolved",
            "phenomenological_state_space")
UNVALIDATED = "not validated against measurements"
_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelRecord:
    id: str
    technology: str
    title: str
    fidelity: str
    entry_points: tuple[str, ...]  # "package.module:function"
    governing_equations: tuple[str, ...]
    supported_effects: tuple[str, ...]
    inputs: tuple[tuple[str, str], ...]  # (name, unit)
    calibration_datasets: tuple[str, ...]  # what it can be fitted to, or why not
    validation_tests: tuple[str, ...]  # repository paths
    validation_status: str
    known_limitations: tuple[str, ...]
    key_results: tuple[tuple[str, str], ...]  # (result, provenance)


_REGISTRY: dict[str, ModelRecord] = {}


def register(record: ModelRecord, replace: bool = False) -> None:
    empty = [name for name, value in asdict(record).items() if not value]
    if empty:
        raise ValueError(f"model record {record.id!r}: empty field(s) {empty}")
    if record.fidelity not in FIDELITY:
        raise ValueError(f"fidelity must be one of {FIDELITY}")
    if any(":" not in e for e in record.entry_points):
        raise ValueError("entry points look like 'package.module:function'")
    if any(len(pair) != 2 or not all(pair) for pair in (*record.inputs, *record.key_results)):
        raise ValueError("inputs are (name, unit); key_results are (result, provenance)")
    if record.id in _REGISTRY and not replace:
        raise ValueError(f"model {record.id!r} is already registered")
    _REGISTRY[record.id] = record


def get(model_id: str) -> ModelRecord:
    if model_id not in _REGISTRY:
        raise ValueError(f"unknown model {model_id!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[model_id]


def load(model_id: str, index: int = 0) -> Any:
    """The callable behind a registered entry point."""
    module, name = get(model_id).entry_points[index].split(":")
    return getattr(importlib.import_module(module), name)


def check(record: ModelRecord) -> dict[str, Any]:
    """Do the recorded entry points import and the recorded test files exist?"""
    broken = []
    for entry in record.entry_points:
        module, name = entry.split(":")
        try:
            getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError) as error:
            broken.append(f"{entry}: {error}")
    missing = [p for p in record.validation_tests if not (_ROOT / p).is_file()]
    return {"entry_points_import": not broken, "broken_entry_points": broken,
            "missing_test_files": missing,
            "note": "existence only: this does not run the tests"}


def describe(technology: str | None = None, model_id: str | None = None) -> dict[str, Any]:
    if model_id:
        records = [get(model_id)]
    else:
        records = [r for r in _REGISTRY.values()
                   if technology is None or r.technology == technology.lower()]
    out = []
    for r in records:
        data = asdict(r)
        data["inputs"] = [{"name": n, "unit": u} for n, u in r.inputs]
        data["key_results"] = [{"result": a, "provenance": b} for a, b in r.key_results]
        data["checks"] = check(r)
        out.append(data)
    return {"models": out, "technologies": sorted({r.technology for r in _REGISTRY.values()}),
            "scope": "only the models listed are registered; ChipLab contains other models "
                     "with no record yet"}


_M = "chiplab.models.materials."
_HARD_CODED = "constant written in the code, no source recorded"

for _record in (
    ModelRecord(
        "ferrofluid_electrical_statespace", "ferrofluid",
        "Electrically programmed N-port state-space network", "phenomenological_state_space",
        (_M + "ferro_statespace:simulate", _M + "ferro_experiment:fit",
         _M + "ferro_reference:reproduce"),
        ("dz/dt = (cosh(v/Vw) - 1)/tau_w (tanh(v/Vs) - z) - z/tau_r  (fast and slow states)",
         "dw/dt = (cosh(v/Vf) - 1)/tau_f (1 - w) - w/tau_rec  (reversible fatigue)",
         "tau(T) = tau exp(Ea/kB (1/T - 1/Tref))",
         "Z_branch = R_inf + dR/(1 + (j omega tau)^alpha), in parallel with C; each of "
         "R_inf, dR, C scales as exp((1 - depth w) (g_fast z_fast + g_slow z_slow))",
         "Z(omega) = Y(omega)^-1 of the shared port network"),
        ("bipolar and pulse-width dependent writes", "hysteresis", "volatile and slowly "
         "relaxing states", "frequency-dependent N x N impedance", "temperature dependence",
         "cross-port interaction and simultaneous writes", "reversible shrinking dynamics"),
        (("port voltages", "V"), ("duration", "s"), ("temperature", "K"),
         ("frequency", "Hz"), ("branch resistances", "ohm"), ("branch capacitance", "F")),
        ("impedance rows", "summed impedance", "DC current", "Touchstone S-parameters",
         "no dataset is bundled"),
        ("tests/test_ferro_statespace.py",),
        UNVALIDATED,
        ("parameters have no physical-material meaning and no defaults", "reciprocal: "
         "cannot represent Z12 != Z21", "coupling between many ports is extrapolated unless "
         "fitted branch by branch", "no irreversible degradation or device variation"),
        (("reproduction of arXiv:2211.08152", "not reproduced: raw data not supplied"),)),
    ModelRecord(
        "ferrofluid_magnetic_particle", "ferrofluid",
        "Magnetic particle model with conductor ports", "reduced_order",
        (_M + "ferro_device:simulate_sequence", _M + "ferro_calibration:fit_parameters"),
        ("M = phi M_d L(xi), first-order mean-field correction",
         "1/tau = 1/tau_Brown + 1/tau_Neel; dM/dt = -(M - M_eq)/tau",
         "Z_ij = R delta_ij + j omega (L0_ij + mu0 integral h_i chi* h_j dV)"),
        ("magnetic write and inductive read at several ports", "relaxation", "concentration "
         "drift (single mode)"),
        (("volume fraction", "1"), ("core diameter", "nm"), ("viscosity", "Pa s"),
         ("temperature", "K"), ("port voltages", "V")),
        ("magnetization curve", "susceptibility spectrum", "magnetization decay",
         "no dataset is bundled"),
        ("tests/test_ferrofluid_physics.py", "tests/test_material_credibility.py"),
        UNVALIDATED,
        ("no electric-field effects: cannot represent an electrically programmed "
         "experiment", "see docs/ferrofluid_model.md"),
        (("default device read signal below thermal noise", "model output with assumed "
          "parameters"),)),
    ModelRecord(
        "memristor_basic", "memristor", "Linear-mixing resistive switch", "heuristic",
        (_M + "memristor:memristor_iv_step", _M + "memristor:memristor_write_pulse",
         _M + "memristor:memristor_crossbar_vmm"),
        ("R = R_off - (R_off - R_on) w", "dw/dt = eta |I| (1 - w) (iv step)",
         "write pulse: dw proportional to |V| duration with fixed coefficients"),
        ("state-dependent resistance", "crossbar vector-matrix product"),
        (("voltage", "V"), ("time", "s")),
        ("none: no fitting routine",), ("tests/test_memristor_pcm.py",), UNVALIDATED,
        ("write-pulse step sizes are arbitrary", "no temperature, noise or variability "
         "model recorded"),
        (("switching rates", _HARD_CODED),)),
    ModelRecord(
        "pcm_basic", "pcm", "Phase-change cell", "heuristic",
        (_M + "pcm:pcm_crystallization_step", _M + "pcm:pcm_melt_quench_pulse",
         _M + "pcm:pcm_resistance_drift"),
        ("k = k0 exp(-Ea/kB T), Avrami exponent 2",
         "R = R_amorphous (1 - X) + R_crystalline X",
         "drift: R (1 + S ln(t/t0))", "melt-quench: dX proportional to |V| duration"),
        ("crystallization", "pulse programming", "resistance drift"),
        (("temperature", "K"), ("voltage", "V"), ("time", "s")),
        ("none: no fitting routine",), ("tests/test_memristor_pcm.py",), UNVALIDATED,
        ("no electro-thermal solution: pulses change phase by a fixed rule",),
        (("k0 and Avrami exponent", _HARD_CODED),)),
    ModelRecord(
        "mram_basic", "spintronics", "STT/SOT magnetic tunnel junction", "heuristic",
        (_M + "spintronic:stt_switching_step", _M + "spintronic:sot_switching_step",
         _M + "spintronic:thermal_stability_factor"),
        ("Jc = 2 e alpha Ms t / (hbar eta)", "switching probability logistic in J/Jc",
         "Delta = K V / kB T; retention tau0 exp(Delta)"),
        ("current-induced switching", "tunnel magnetoresistance read", "thermal stability"),
        (("current density", "A/m^2"), ("pulse duration", "s"), ("temperature", "K")),
        ("none: no fitting routine",), ("tests/test_ferro_spintronic.py",), UNVALIDATED,
        ("no magnetization dynamics in these functions", "logistic switching probability "
         "is not derived from a physical model"),
        (("Ms, Hk and anisotropy energy density", _HARD_CODED),)),
    ModelRecord(
        "photonic_basic", "photonics", "Ideal MZI mesh, microring weights, link budget",
        "analytical",
        (_M + "photonic:mzi_mesh_vmm", _M + "photonic:microring_weight_bank",
         _M + "photonic:photonic_link_budget"),
        ("2x2 rotations composing an N-port unitary", "Lorentzian ring weight",
         "link budget: sum of losses in dB; thermal noise floor -174 dBm/Hz"),
        ("matrix-vector product", "ring weighting", "optical power budget"),
        (("phases", "rad"), ("wavelength", "m"), ("power", "dBm")),
        ("none: no fitting routine",), ("tests/test_photonic.py",), UNVALIDATED,
        ("ideal components in the mesh: no fabrication error or crosstalk recorded",),
        (("connector, splice and waveguide losses", _HARD_CODED),)),
):
    register(_record)
