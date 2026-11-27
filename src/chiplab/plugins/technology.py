"""Technology plugin SDK: add a hardware technology without touching ChipLab's core.

A plugin is a `TechnologyPlugin` with:

  material_properties        {name: {"value", "unit", "source"}}
  equations                  governing equations as text (documentation, shown to agents)
  device_model(inputs)       -> {metric: value}; inputs carry units in their names
  circuit_interface          ports: {name: {"direction", "kind": "analog|digital", "unit"}}
  memory_behavior            optional {"read": fn(state, addr), "write": fn(state, addr, v)}
  reliability_model(inputs)  optional -> {"endurance_cycles", "retention_s", ...}
  thermal_model(inputs)      optional -> {"power_w", "temperature_k", ...}
  manufacturing_constraints  {"process": ..., "digital_shuttle": bool, ...}
  validation_tests           [fn() -> (name, passed, detail)]
  default_inputs             inputs used by the review gate

`register()` admits a plugin only if: required fields are present, every property has a
recognised unit and a source, its own validation tests pass, and its device model passes the
review gate (finite, physical bounds, reproducible, responds to inputs). Admission never
marks a model "validated" against reality: that needs measurements (learning loop).

Plugins are Python objects registered by code the user runs; there is no loader that executes
files by path from an MCP or web request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from chiplab.science.review import review, unit_ok

TestFn = Callable[[], tuple[str, bool, str]]


@dataclass
class TechnologyPlugin:
    name: str
    version: str
    material_properties: dict[str, dict[str, Any]]
    equations: list[str]
    device_model: Callable[[dict[str, Any]], dict[str, Any]]
    circuit_interface: dict[str, dict[str, str]]
    manufacturing_constraints: dict[str, Any]
    validation_tests: list[TestFn]
    default_inputs: dict[str, Any]
    memory_behavior: dict[str, Callable[..., Any]] | None = None
    reliability_model: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    thermal_model: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    fidelity: str = "analytical"
    authors: str = ""
    notes: list[str] = field(default_factory=list)


_REGISTRY: dict[str, dict[str, Any]] = {}


def check(plugin: TechnologyPlugin) -> dict[str, Any]:
    """Run every admission check; returns problems and the review verdict."""
    problems: list[str] = []
    if not plugin.name.isidentifier():
        problems.append("name must be an identifier")
    for k, prop in plugin.material_properties.items():
        if not isinstance(prop, dict) or "value" not in prop:
            problems.append(f"property {k}: needs value")
            continue
        if not unit_ok(str(prop.get("unit", ""))):
            problems.append(f"property {k}: unit {prop.get('unit')!r} not recognised")
        if not str(prop.get("source", "")).strip():
            problems.append(f"property {k}: no source (use 'assumed' if it is a guess)")
    if not plugin.equations:
        problems.append("no governing equations given")
    for port, info in plugin.circuit_interface.items():
        if info.get("direction") not in ("in", "out", "inout"):
            problems.append(f"port {port}: direction must be in/out/inout")
        if info.get("kind") not in ("analog", "digital"):
            problems.append(f"port {port}: kind must be analog/digital")
    if "digital_shuttle" not in plugin.manufacturing_constraints:
        problems.append("manufacturing_constraints must say whether a digital shuttle can "
                        "build it (digital_shuttle: bool)")
    tests = []
    if not plugin.validation_tests:
        problems.append("at least one validation test is required")
    for fn in plugin.validation_tests:
        try:
            name, ok, detail = fn()
        except Exception as e:  # noqa: BLE001 - a crashing test is a failed test
            name, ok, detail = getattr(fn, "__name__", "test"), False, f"raised {e!r}"
        tests.append({"test": name, "passed": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"validation test {name} failed: {detail}")
    try:
        out = plugin.device_model(dict(plugin.default_inputs))
        rv = review(f"plugin:{plugin.name}", out, fidelity=plugin.fidelity,
                    provenance={"tool": f"plugin:{plugin.name}@{plugin.version}"},
                    evaluate=plugin.device_model, inputs=plugin.default_inputs).as_dict()
    except Exception as e:  # noqa: BLE001
        rv = {"verdict": "rejected", "failed": [{"check": "device_model", "why": repr(e)}]}
    if rv["verdict"] == "rejected":
        problems.append("device model failed review: " + "; ".join(
            f["why"] for f in rv.get("failed", [])))
    optional = {k: getattr(plugin, k) is not None for k in
                ("memory_behavior", "reliability_model", "thermal_model")}
    return {"plugin": plugin.name, "version": plugin.version, "admissible": not problems,
            "problems": problems, "tests": tests, "review": rv, "provides": optional}


def register(plugin: TechnologyPlugin, replace: bool = False) -> dict[str, Any]:
    key = f"{plugin.name}@{plugin.version}"
    if key in _REGISTRY and not replace:
        raise ValueError(f"{key} already registered")
    res = check(plugin)
    if res["admissible"]:
        _REGISTRY[key] = {"plugin": plugin, "check": res}
    return res


def get(name: str, version: str | None = None) -> TechnologyPlugin:
    keys = [k for k in _REGISTRY if k.split("@")[0] == name and (version is None or
                                                                k.endswith("@" + version))]
    if not keys:
        raise KeyError(f"no plugin {name}")
    return _REGISTRY[sorted(keys)[-1]]["plugin"]


def describe() -> list[dict[str, Any]]:
    out = []
    for key, entry in sorted(_REGISTRY.items()):
        p: TechnologyPlugin = entry["plugin"]
        out.append({"key": key, "fidelity": p.fidelity, "equations": p.equations,
                    "properties": p.material_properties, "ports": p.circuit_interface,
                    "manufacturing": p.manufacturing_constraints,
                    "review_verdict": entry["check"]["review"]["verdict"],
                    "provides": entry["check"]["provides"],
                    "validated_against_measurement": False})
    return out


def simulate(name: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    p = get(name)
    x = dict(p.default_inputs) | (inputs or {})
    out: dict[str, Any] = {"device": p.device_model(x)}
    if p.reliability_model:
        out["reliability"] = p.reliability_model(x)
    if p.thermal_model:
        out["thermal"] = p.thermal_model(x)
    return out | {"plugin": f"{p.name}@{p.version}", "fidelity": p.fidelity}


def example_resistive_plugin() -> TechnologyPlugin:
    """A worked example: a linear resistive memory cell (Ohm's law read, Joule heating)."""
    def model(x: dict[str, Any]) -> dict[str, Any]:
        r = x["r_on_ohm"] if x["state"] >= 0.5 else x["r_off_ohm"]
        i = x["v_read_v"] / r
        return {"read_current_a": i, "read_power_w": x["v_read_v"] * i,
                "on_off_ratio": x["r_off_ohm"] / x["r_on_ohm"]}

    def t_ohm() -> tuple[str, bool, str]:
        o = model({"r_on_ohm": 1e4, "r_off_ohm": 1e6, "v_read_v": 0.2, "state": 1})
        return "ohms_law", abs(o["read_current_a"] - 2e-5) < 1e-12, "I = V/R"

    def t_off() -> tuple[str, bool, str]:
        o = model({"r_on_ohm": 1e4, "r_off_ohm": 1e6, "v_read_v": 0.2, "state": 0})
        return "off_state", o["read_current_a"] < 1e-6, "off current below 1 uA"

    return TechnologyPlugin(
        name="example_resistive", version="0.1",
        material_properties={
            "r_on": {"value": 1e4, "unit": "ohm", "source": "assumed"},
            "r_off": {"value": 1e6, "unit": "ohm", "source": "assumed"}},
        equations=["I = V / R(state)", "P = V I"],
        device_model=model,
        circuit_interface={"bl": {"direction": "inout", "kind": "analog", "unit": "V"},
                           "wl": {"direction": "in", "kind": "digital", "unit": "V"}},
        manufacturing_constraints={"process": "BEOL resistive memory", "digital_shuttle": False},
        validation_tests=[t_ohm, t_off],
        default_inputs={"r_on_ohm": 1e4, "r_off_ohm": 1e6, "v_read_v": 0.2, "state": 1.0},
        reliability_model=lambda x: {"endurance_cycles": 1e6, "source": "assumed"},
    )
