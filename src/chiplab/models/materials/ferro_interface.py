"""JSON-friendly adapter between the MCP tools and the ferrofluid particle model.

Every input that the caller did not give is listed under ``assumed_inputs``; unknown
keys are rejected instead of being silently ignored.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from chiplab.credibility import CredibilityError, is_ferrofluid, mentions_ferrofluid
from chiplab.models.materials.ferro_device import (
    NOT_MODELLED,
    CoilPort,
    FerrofluidDevice,
    FluidRegion,
    ReadEvent,
    WritePulse,
    complex_dict,
    simulate_sequence,
)
from chiplab.models.materials.ferro_particles import KB, FerrofluidParams

DEFAULT_PITCH_UM = 100.0
DEFAULT_RADIUS_UM = 20.0
DEFAULT_WRITE_NS = 1000.0
DEFAULT_READ_HZ = 1e6

GEOMETRY_KEYS = ("conductor", "radius_um", "turns", "wire_radius_um", "series_resistance_ohm",
                 "axis", "start_um", "end_um", "electrode_radius_um")
PORT_DEFAULT_NOTES = {
    "conductor": "loop", "radius_um": DEFAULT_RADIUS_UM, "turns": 1, "wire_radius_um": 1.0,
    "series_resistance_ohm": 50.0, "axis": [0, 0, 1],
}
DEVICE_KEYS = ("fluid_region", "concentration_length_um", "conductivity_s_per_m",
               "relative_permittivity", "double_layer_f_per_m2")


ROUTING_KEYS = ("technology", "material", "type", "name")


def routes_to_ferrofluid(device_spec: dict | None, electrode_array: dict | None = None
                         ) -> bool:
    """True if a device spec or array declares a ferrofluid under any of its names.

    A spec that names another technology and also names a ferrofluid is refused rather
    than guessed at.
    """
    spec = device_spec or {}
    declared = [spec.get(k) for k in ROUTING_KEYS]
    declared += [r.get("material") for r in spec.get("regions") or [] if isinstance(r, dict)]
    declared.append((electrode_array or {}).get("technology"))
    nested = [spec.get("parameters"), spec.get("ferrofluid")]
    if not (any(mentions_ferrofluid(d) for d in declared)
            or any(mentions_ferrofluid(d) for d in nested) or "ferrofluid" in spec):
        return False
    technology = spec.get("technology")
    if technology and not is_ferrofluid(technology) and str(technology).lower() != "generic":
        raise CredibilityError(
            f"ambiguous input: technology is {technology!r} but a ferrofluid is also named; "
            "set technology to 'ferrofluid' or remove the ferrofluid reference")
    return True


def particle_params(*sources: dict | None) -> FerrofluidParams:
    """Collect particle/fluid parameters from top-level keys, 'parameters' or 'ferrofluid'.

    A 'calibration' entry (the output of the ferrofluid fit) supplies fitted values;
    explicit values override it.
    """
    values: dict[str, Any] = {}
    labels: dict[str, str] = {}
    names = set(FerrofluidParams.names())
    for source in sources:
        if not source:
            continue
        containers = [c for c in (source.get("parameters"), source.get("ferrofluid"), source)
                      if isinstance(c, dict)]
        for calibration in [c["calibration"] for c in containers if c.get("calibration")]:
            if calibration.get("status") not in ("fitted_to_user_data",
                                                 "fitted_with_reservations"):
                raise ValueError("'calibration' must be the result of a converged ferrofluid fit")
            origin = calibration.get("provenance", {}).get("source", "unknown source")
            for name, value in calibration.get("parameters_for_simulation", {}).items():
                values[name] = value
                fitted = calibration.get("parameters", {}).get(name, {}).get("identifiable")
                labels[name] = (f"fitted to user data: {origin}" if fitted
                                else "user_supplied (held fixed during fit)")
        for nested in containers:
            for name in names & set(nested):
                values[name] = nested[name]
                labels[name] = "user_supplied"
    return FerrofluidParams(**values, provenance=labels)


def check_keys(data: dict, allowed: set[str], what: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {what} key(s) {unknown}; accepted: {sorted(allowed)}")


ARRANGEMENT_ALIASES = {"line": "linear", "square": "grid", "2x2": "grid_2x2",
                       "4x4": "grid_4x4", "ring": "circular", "hex": "hexagonal"}


def canonical_arrangement(arrangement: str) -> str:
    kind = str(arrangement).strip().lower()
    return ARRANGEMENT_ALIASES.get(kind, kind)


def layout_positions(n_ports: int, arrangement: str, pitch_um: float,
                     custom: list | None = None) -> list[tuple[float, float]]:
    """Port centres (x, y) in micrometres for a named arrangement."""
    if isinstance(n_ports, bool) or int(n_ports) != n_ports or not 1 <= n_ports <= 1024:
        raise ValueError("n_ports must be an integer between 1 and 1024")
    n_ports = int(n_ports)
    if not (math.isfinite(pitch_um) and pitch_um > 0):
        raise ValueError("pitch_um must be a positive number")
    kind = canonical_arrangement(arrangement)
    if kind == "linear":
        return [(i * pitch_um, 0.0) for i in range(n_ports)]
    if kind in ("grid", "grid_2x2", "grid_4x4"):
        side = {"grid_2x2": 2, "grid_4x4": 4}.get(kind, math.ceil(math.sqrt(n_ports)))
        if n_ports > side * side:
            raise ValueError(f"{kind} holds at most {side * side} ports, got {n_ports}")
        return [((i % side) * pitch_um, (i // side) * pitch_um) for i in range(n_ports)]
    if kind == "circular":
        # circumference = n_ports * pitch
        radius = pitch_um * n_ports / (2 * math.pi)
        return [(radius * math.cos(2 * math.pi * i / n_ports),
                 radius * math.sin(2 * math.pi * i / n_ports)) for i in range(n_ports)]
    if kind == "hexagonal":
        side = math.ceil(math.sqrt(n_ports))
        return [((i % side + 0.5 * ((i // side) % 2)) * pitch_um,
                 (i // side) * pitch_um * math.sqrt(3) / 2) for i in range(n_ports)]
    if kind == "custom":
        if not custom or len(custom) != n_ports:
            raise ValueError("custom arrangement needs port_config['positions'] with exactly "
                             f"{n_ports} [x_um, y_um] entries")
        return [(float(p[0]), float(p[1])) for p in custom]
    raise ValueError(f"unknown arrangement {arrangement!r}; use linear, grid, grid_2x2, "
                     "grid_4x4, circular, hexagonal or custom")


def _position(entry: dict, fallback: tuple[float, float, float]
              ) -> tuple[tuple[float, float, float], bool]:
    if "position_um" in entry:
        value = entry["position_um"]
    elif "position" in entry:
        value = entry["position"]
    else:
        return fallback, True
    if isinstance(value, dict):
        return (float(value["x_um"]), float(value["y_um"]), float(value.get("z_um", 0.0))), False
    if isinstance(value, (int, float)):
        return (float(value), 0.0, 0.0), False
    xyz = [float(v) for v in value] + [0.0]
    return (xyz[0], xyz[1], xyz[2] if len(value) > 2 else 0.0), False


def make_port(port_id: str, entry: dict, role: str, fallback: tuple[float, float, float]
              ) -> CoilPort:
    position, assumed_position = _position(entry, fallback)
    geometry = {k: entry[k] for k in GEOMETRY_KEYS if entry.get(k) is not None}
    assumed = [f"{k}={v}" for k, v in PORT_DEFAULT_NOTES.items() if k not in geometry
               and not (geometry.get("conductor") == "line" and k in ("radius_um", "axis"))]
    if assumed_position and geometry.get("conductor") != "line":
        assumed.append(f"position_um={list(position)}")
    return CoilPort(id=port_id, role=role, position_um=position, assumed=assumed, **geometry)


def _role_of(name: str, entry: dict) -> str:
    role = str(entry.get("role", "")).lower()
    if role:
        return role
    for text in (str(entry.get("port", "")).lower(), name.lower()):
        if text.startswith("write") or text in ("w",) or (text[:1] == "w" and text[1:].isdigit()):
            return "write"
        if text.startswith("read") or text in ("r",) or (text[:1] == "r" and text[1:].isdigit()):
            return "read"
    writes, reads = "write_mechanism" in entry, "read_mechanism" in entry
    if writes != reads:
        return "write" if writes else "read"
    return "both"


ARRAY_PORT_KEYS = set(GEOMETRY_KEYS) | {
    "port_index", "id", "role", "position", "position_um", "bias_voltage", "impedance_ohm",
    "impedance_source", "functional_group"}


def ports_from_array(electrode_array: dict) -> list[CoilPort]:
    """Ports from a configure_material_array result (or an equivalent dict)."""
    entries = electrode_array.get("port_config") or []
    n_ports = electrode_array.get("n_ports", len(entries))
    if not n_ports:
        raise ValueError("electrode_array needs n_ports or a port_config list")
    if entries and len(entries) != n_ports:
        raise ValueError(f"port_config has {len(entries)} entries but n_ports is {n_ports}")
    positions = None
    ports = []
    for i in range(int(n_ports)):
        entry = dict(entries[i]) if entries else {}
        check_keys(entry, ARRAY_PORT_KEYS, f"port {i}")
        if "position" not in entry and "position_um" not in entry:
            if positions is None:
                positions = layout_positions(
                    int(n_ports), electrode_array.get("arrangement", "linear"),
                    float(electrode_array.get("pitch_um", DEFAULT_PITCH_UM)))
            fallback = (positions[i][0], positions[i][1], 0.0)
        else:
            fallback = (0.0, 0.0, 0.0)
        port_id = str(entry.get("id", f"port_{i}"))
        ports.append(make_port(port_id, entry, str(entry.get("role", "both")), fallback))
    return ports


ELECTRODE_KEYS = ARRAY_PORT_KEYS | {"port", "write_mechanism", "read_mechanism", "voltage_v"}


def ports_from_electrodes(electrodes: dict) -> list[CoilPort]:
    """Ports from a custom device 'electrodes' mapping {name: {...}}."""
    if not isinstance(electrodes, dict) or not electrodes:
        raise ValueError("a ferrofluid device needs a non-empty 'electrodes' mapping")
    ports = []
    for i, (name, entry) in enumerate(electrodes.items()):
        entry = dict(entry or {})
        check_keys(entry, ELECTRODE_KEYS, f"electrode {name!r}")
        ports.append(make_port(str(name), entry, _role_of(str(name), entry),
                               (i * DEFAULT_PITCH_UM, 0.0, 0.0)))
    return ports


def build_device(ports: list[CoilPort], *sources: dict | None) -> FerrofluidDevice:
    params = particle_params(*sources)
    options: dict[str, Any] = {}
    for source in sources:
        if source:
            for container in (source.get("parameters"), source.get("ferrofluid"), source):
                if isinstance(container, dict):
                    options.update({k: container[k] for k in DEVICE_KEYS if k in container})
    region = options.pop("fluid_region", None)
    if region is not None:
        check_keys(region, {"center_um", "size_um", "grid", "height_um", "standoff_um"},
                   "fluid_region")
        if "size_um" in region or "center_um" in region:
            if not {"size_um", "center_um"} <= set(region):
                raise ValueError("fluid_region needs both center_um and size_um")
            assumed = [] if "grid" in region else ["fluid grid 24 x 24 x 8 cells"]
            region = FluidRegion(region["center_um"], region["size_um"],
                                 region.get("grid", (24, 24, 8)), assumed=assumed)
        else:
            default = FluidRegion.around(ports, float(region.get("height_um", 10.0)),
                                         float(region.get("standoff_um", 1.0)))
            if "grid" in region:
                default = FluidRegion(default.center_um, default.size_um, region["grid"],
                                      assumed=default.assumed)
            region = default
    return FerrofluidDevice(ports, params, region, **options)


def _port_ref(item: dict, device: FerrofluidDevice, candidates: list[int], n: int) -> int:
    for key in ("port", "port_id", "port_index", "channel"):
        if key in item:
            return device.index(item[key])
    if n >= len(candidates):
        raise ValueError("more sequence entries without a 'port' than there are ports")
    return candidates[n]


def _candidates(device: FerrofluidDevice, role: str) -> list[int]:
    explicit = [i for i, p in enumerate(device.ports) if p.role == role]
    if explicit:
        return explicit
    return [i for i, p in enumerate(device.ports) if p.role == "both"]


WRITE_KEYS = {"port", "port_id", "port_index", "channel", "amplitude", "voltage_v", "voltage",
              "duration_ns", "start_ns", "pattern"}
READ_KEYS = {"port", "port_id", "port_index", "channel", "frequency_hz", "time_ns", "delay_ns",
             "amplitude_v", "bandwidth_hz", "integration_time_ns", "pattern", "sensitivity"}


def parse_writes(sequence: list, device: FerrofluidDevice, assumed: list[str],
                 default_voltage: float | None = None) -> list[WritePulse]:
    candidates = _candidates(device, "write")
    writes = []
    for n, item in enumerate(sequence or []):
        check_keys(item, WRITE_KEYS, "write_sequence")
        if str(item.get("pattern", "pulse")).lower() != "pulse":
            raise ValueError(f"write pattern {item['pattern']!r} is not modelled; only "
                             "rectangular voltage 'pulse' is")
        voltage = next((item[k] for k in ("voltage_v", "amplitude", "voltage") if k in item),
                       None)
        if voltage is None:
            voltage = 1.0 if default_voltage is None else default_voltage
            assumed.append(f"write {n}: voltage_v={voltage}")
        if "duration_ns" not in item:
            assumed.append(f"write {n}: duration_ns={DEFAULT_WRITE_NS}")
        writes.append(WritePulse(
            _port_ref(item, device, candidates, n), float(voltage),
            float(item.get("duration_ns", DEFAULT_WRITE_NS)) * 1e-9,
            float(item.get("start_ns", 0.0)) * 1e-9))
    return writes


def parse_reads(sequence: list, device: FerrofluidDevice, assumed: list[str],
                warnings: list[str], default_frequency: float | None = None
                ) -> list[ReadEvent]:
    candidates = _candidates(device, "read")
    if not sequence:
        sequence = [{"port_index": i} for i in candidates]
        assumed.append("read_sequence: every readable port")
    reads = []
    for n, item in enumerate(sequence):
        check_keys(item, READ_KEYS, "read_sequence")
        if str(item.get("pattern", "sense")).lower() != "sense":
            raise ValueError(f"read pattern {item['pattern']!r} is not modelled; only 'sense' is")
        if "sensitivity" in item:
            warnings.append("read 'sensitivity' ignored: no detector model exists")
        frequency = item.get("frequency_hz", default_frequency)
        if frequency is None:
            frequency = DEFAULT_READ_HZ
            assumed.append(f"read {n}: frequency_hz={DEFAULT_READ_HZ}")
        if "bandwidth_hz" in item:
            bandwidth = float(item["bandwidth_hz"])
        elif "integration_time_ns" in item:
            bandwidth = 1.0 / (2.0 * float(item["integration_time_ns"]) * 1e-9)
        else:
            bandwidth = 1e6
            assumed.append(f"read {n}: bandwidth_hz=1e6")
        if "time_ns" not in item and "delay_ns" not in item:
            assumed.append(f"read {n}: at the instant the last write ends (bias off)")
        reads.append(ReadEvent(
            _port_ref(item, device, candidates, n), float(frequency),
            None if "time_ns" not in item else float(item["time_ns"]) * 1e-9,
            float(item.get("delay_ns", 0.0)) * 1e-9, item.get("amplitude_v"), bandwidth))
    return reads


def simulate(device: FerrofluidDevice, write_sequence: list, read_sequence: list,
             default_voltage: float | None = None, default_frequency: float | None = None,
             spectrum_hz: list[float] | None = None) -> dict[str, Any]:
    assumed: list[str] = []
    warnings: list[str] = []
    writes = parse_writes(write_sequence, device, assumed, default_voltage)
    reads = parse_reads(read_sequence, device, assumed, warnings, default_frequency)
    result = simulate_sequence(device, writes, reads)
    result["warnings"] = warnings + result["warnings"]
    change = device.grid_convergence()
    result["fluid_region"]["grid_convergence_relative_change"] = change
    if change > 0.1:
        result["warnings"].append(
            f"fluid grid not converged: port fluid inductance changes by {change:.0%} on a "
            "1.5x finer grid; set fluid_region.grid")
    if spectrum_hz is not None:
        freqs = np.asarray(spectrum_hz, dtype=float)
        # spectrum with every write port held at its drive voltage (last pulse per port)
        held = {device.index(w.port): w.voltage_v for w in writes}
        jw = 2j * np.pi * freqs[:, None, None]
        air = np.diag(device.resistance)[None] + jw * device.inductance_air[None]
        fluid = jw * device.fluid_inductance(freqs, np.zeros(len(device.ports)))
        rest = air + fluid
        biased = air + jw * device.fluid_inductance(freqs, device.current(held))
        result["impedance_spectrum"] = [{
            "frequency_hz": float(f),
            "ports": {p.id: {
                "unwritten_ohm": complex_dict(rest[k, i, i]),
                "with_write_bias_ohm": complex_dict(biased[k, i, i]),
                "fluid_part_unwritten_ohm": complex_dict(fluid[k, i, i]),
            } for i, p in enumerate(device.ports)},
        } for k, f in enumerate(freqs)]
        result["impedance_spectrum_note"] = (
            "steady-state spectrum; 'with_write_bias' holds each written port at its pulse "
            "voltage, particles redistribution excluded")
    result["assumed_inputs"] = assumed + [
        f"port {p.id}: {', '.join(p.assumed)}" for p in device.ports if p.assumed]
    return result


def default_spectrum() -> list[float]:
    return [float(f) for f in np.logspace(3, 8, 11)]


# ---------------------------------------------------------------------------
# Experimental-technology summary
# ---------------------------------------------------------------------------

METRIC_KEYS = ("energy_per_bit_j", "latency_ns", "endurance_cycles", "retention_s",
               "area_density_gb_cm2", "thermal_sensitivity_ppm_K")
SIMPLE_KEYS = {"voltage", "voltage_v", "frequency_hz", "read_frequency_hz", "duration_ns",
               "read_delay_ns", "ports", "electrodes"} | set(DEVICE_KEYS) | set(
    GEOMETRY_KEYS) | {"separation_um", "calibration", "ferrofluid"}


def technology_summary(params: dict | None, pulse_schedule: list | None) -> dict[str, Any]:
    """Physics-based metrics for chiplab_simulate_experimental_tech("ferrofluid").

    Without explicit ports the device is an assumed pair: a write loop and a read loop.
    """
    params = dict(params or {})
    check_keys(params, SIMPLE_KEYS | set(FerrofluidParams.names()) | set(METRIC_KEYS),
               "ferrofluid parameter")
    overrides = {k: params.pop(k) for k in METRIC_KEYS if k in params}
    assumed: list[str] = []
    if "electrodes" in params:
        ports = ports_from_electrodes(params["electrodes"])
    elif "ports" in params:
        ports = ports_from_array({"port_config": params["ports"]})
    else:
        geometry = {k: params[k] for k in GEOMETRY_KEYS if k in params}
        separation = float(params.get("separation_um", 50.0))
        if "separation_um" not in params:
            assumed.append("separation_um=50 between one write and one read loop")
        ports = [make_port("write", {**geometry, "position_um": [0, 0, 0]}, "write", (0, 0, 0)),
                 make_port("read", {**geometry, "position_um": [separation, 0, 0]}, "read",
                           (separation, 0, 0))]
    device = build_device(ports, params)
    voltage = params.get("voltage_v", params.get("voltage"))
    sequence = []
    if pulse_schedule:
        clock = 0.0
        for step in pulse_schedule:
            item = dict(step)
            if "start_ns" not in item:
                item["start_ns"] = clock
            clock = max(clock, item["start_ns"] + item.get("duration_ns", DEFAULT_WRITE_NS))
            sequence.append(item)
        for item in sequence:
            if not any(k in item for k in ("port", "port_id", "port_index", "channel")):
                item["port_index"] = _candidates(device, "write")[0]
    else:
        if voltage is None:
            voltage = 1.0
            assumed.append("voltage_v=1.0")
        item = {"voltage_v": voltage, "port_index": _candidates(device, "write")[0]}
        if "duration_ns" in params:
            item["duration_ns"] = params["duration_ns"]
        sequence = [item]
    frequency = params.get("read_frequency_hz", params.get("frequency_hz"))
    end = max(s.get("start_ns", 0.0) + s.get("duration_ns", DEFAULT_WRITE_NS) for s in sequence)
    delay = params.get("read_delay_ns")
    reads = []
    for i in _candidates(device, "read"):
        if delay is not None:
            reads.append({"port_index": i, "delay_ns": delay})
        else:
            reads.append({"port_index": i, "time_ns": end * (1 - 1e-6)})
    if delay is None:
        assumed.append("read just before the last write ends (write field still applied)")
    result = simulate(device, sequence, reads, voltage, frequency, default_spectrum())
    result["assumed_inputs"] = assumed + result["assumed_inputs"]

    metrics: dict[str, Any] = {k: None for k in METRIC_KEYS}
    status = {k: "not_modelled" for k in METRIC_KEYS}
    for key, value in overrides.items():
        metrics[key] = value
        status[key] = "user_supplied (echoed, not computed)"
    first_read = result["reads"][0]
    first_write = result["writes"][0]
    decay = result["retention"]["per_port"].get(first_read["port"], {})
    computed = {
        "write_current_a": first_write["current_a"],
        "write_peak_field_a_m": first_write["peak_field_in_fluid_a_m"],
        "write_drive_energy_j": float(sum(w["drive_energy_j"] for w in result["writes"])),
        "read_impedance_magnitude_ohm": first_read["impedance_magnitude_ohm"],
        "read_impedance_signal_ohm": float(math.hypot(
            first_read["signal_vs_unwritten_ohm"]["real"],
            first_read["signal_vs_unwritten_ohm"]["imag"])),
        "read_flux_linkage_wb": first_read["magnetization_flux_linkage_wb"],
        "magnetization_decay_1_over_e_s": decay.get("time_to_1_over_e_s"),
        "thermal_noise_v_rms": first_read["thermal_noise_v_rms"],
    }
    metrics.update(computed)
    status.update({k: "computed (uncalibrated physics model)" for k in computed})
    return {"metrics": metrics, "metric_status": status, "simulation": result,
            "not_modelled": NOT_MODELLED}


# ---------------------------------------------------------------------------
# Equivalent-circuit analyses
# ---------------------------------------------------------------------------


def circuit_analysis(device: FerrofluidDevice, write_circuit: dict, read_circuit: dict,
                     analysis: str) -> dict[str, Any]:
    assumed: list[str] = []

    def pick(circuit: dict, role: str) -> int:
        for key in ("port", "port_index"):
            if key in circuit:
                return device.index(circuit[key])
        i = _candidates(device, role)[0]
        assumed.append(f"{role} port: {device.ports[i].id}")
        return i

    w_keys = {"port", "port_index", "voltage", "voltage_v", "duration_ns", "frequency_range",
              "n_freqs"}
    r_keys = {"port", "port_index", "bandwidth_hz", "noise_floor_v/√Hz",
              "noise_floor_v_sqrt_hz", "sensitivity", "n_samples"}
    check_keys(write_circuit, w_keys, "write_circuit")
    check_keys(read_circuit, r_keys, "read_circuit")
    kind = analysis.lower()
    if kind not in ("dc", "ac", "noise", "transient"):
        raise ValueError("analysis must be dc, ac, noise or transient")
    voltage = write_circuit.get("voltage_v", write_circuit.get("voltage"))
    if voltage is None:
        voltage = 1.0
        assumed.append("voltage_v=1.0")
    i = pick(write_circuit, "write")
    current = device.current({i: voltage})
    out: dict[str, Any] = {"analysis": kind, "write_port": device.ports[i].id,
                           "voltage_v": float(voltage)}
    resistance = float(device.resistance[i])
    inductance = float(device.inductance_air[i, i])
    if kind == "dc":
        out.update(resistance_ohm=resistance, current_a=float(current[i]),
                   power_w=float(voltage * current[i]),
                   peak_field_in_fluid_a_m=float(
                       np.linalg.norm(device.field(current), axis=1).max()))
    elif kind == "ac":
        j = pick(read_circuit, "read")
        span = write_circuit.get("frequency_range", {"min_hz": 1e3, "max_hz": 1e8})
        n = int(write_circuit.get("n_freqs", 11))
        if n < 1 or n > 500:
            raise ValueError("n_freqs must be between 1 and 500")
        freqs = np.logspace(np.log10(span["min_hz"]), np.log10(span["max_hz"]), n)
        z = device.impedance_matrix(freqs, current)
        out["read_port"] = device.ports[j].id
        out["bias"] = "write port held at voltage_v"
        out["impedance_response"] = [{
            "frequency_hz": float(f),
            "self_impedance_ohm": complex_dict(z[k, j, j]),
            "transfer_impedance_write_to_read_ohm": complex_dict(z[k, j, i]),
            "magnitude": float(abs(z[k, j, j])),
            "phase_deg": float(np.degrees(np.angle(z[k, j, j]))),
        } for k, f in enumerate(freqs)]
    elif kind == "noise":
        j = pick(read_circuit, "read")
        bandwidth = float(read_circuit.get("bandwidth_hz", 1e6))
        if "bandwidth_hz" not in read_circuit:
            assumed.append("bandwidth_hz=1e6 starting at 0 Hz")
        freqs = np.linspace(bandwidth / 200, bandwidth, 100)
        z = device.impedance_matrix(freqs)
        psd = 4 * KB * device.params.temperature_k * z[:, j, j].real
        thermal = float(np.sqrt(np.mean(psd) * bandwidth))
        floor = read_circuit.get("noise_floor_v_sqrt_hz", read_circuit.get("noise_floor_v/√Hz"))
        total = thermal if floor is None else float(np.hypot(thermal, floor * np.sqrt(bandwidth)))
        out.update(read_port=device.ports[j].id, bandwidth_hz=bandwidth,
                   thermal_noise_v_rms=thermal, rms_noise_v=total,
                   amplifier_noise="not modelled (no noise_floor supplied)" if floor is None
                   else "user_supplied floor added in quadrature",
                   snr_db=None,
                   snr_note="needs a signal: use run_multiphysics_simulation with amplitude_v")
    else:
        duration = float(write_circuit.get("duration_ns", DEFAULT_WRITE_NS)) * 1e-9
        if "duration_ns" not in write_circuit:
            assumed.append(f"duration_ns={DEFAULT_WRITE_NS}")
        j = pick(read_circuit, "read")
        pulse = WritePulse(i, voltage, duration)
        times = np.concatenate([np.linspace(0, duration, 26),
                                np.linspace(duration, 2 * duration, 26)[1:]])
        state = device.initial_state()
        previous = 0.0
        rows = []
        zero = np.zeros(len(device.ports))
        for t in times:
            state = device.evolve(state, current if previous < pulse.end_s else zero,
                                  float(t - previous))
            previous = float(t)
            active = current if t < pulse.end_s else zero
            rows.append({
                "time_ns": float(t * 1e9),
                "write_voltage_v": float(voltage) if t < pulse.end_s else 0.0,
                "write_current_a": float(active[i]),
                "read_flux_linkage_wb": float(device.flux_linkage(state)[j]),
                "read_magnetization_emf_v": float(device.emf(state, active)[j]),
            })
        out.update(read_port=device.ports[j].id, write_duration_ns=duration * 1e9,
                   n_points=len(rows), waveforms=rows,
                   current_rise_time_constant_s=inductance / resistance,
                   current_model="quasi-static I = V/R (L/R rise not resolved)")
    out["assumed_inputs"] = assumed + [
        f"port {p.id}: {', '.join(p.assumed)}" for p in device.ports if p.assumed]
    out["warnings"] = device.warnings + device.ensemble.validity_warnings()
    out["not_modelled"] = NOT_MODELLED
    return out


# ---------------------------------------------------------------------------
# MCP tool back ends
# ---------------------------------------------------------------------------

DEVICE_SPEC_KEYS = (set(FerrofluidParams.names()) | set(DEVICE_KEYS) | {
    "technology", "material", "type", "name", "description", "parameters", "ferrofluid",
    "calibration",
    "spectrum_hz", "fidelity", "electrodes", "regions", "write_sequence", "read_sequence",
    "write_duration_ns", "write_voltage_v", "impedance_ohm", "energy_pj"})


def validate_device_spec(spec: dict) -> None:
    check_keys(spec, DEVICE_SPEC_KEYS, "ferrofluid device_spec")
    for key in ("parameters", "ferrofluid"):
        if key in spec:
            check_keys(spec[key], set(FerrofluidParams.names()) | set(DEVICE_KEYS)
                       | {"calibration"}, f"device_spec[{key!r}]")


def custom_device(device_spec: dict, write_voltage_v: float, read_freq_hz: float
                  ) -> dict[str, Any]:
    """Back end of chiplab_custom_device_simulate for ferrofluid regions."""
    validate_device_spec(device_spec)
    spec = dict(device_spec)
    warnings: list[str] = []
    assumed: list[str] = []
    for region in spec.get("regions") or []:
        if not is_ferrofluid(region.get("material", "")):
            warnings.append(f"region material {region.get('material')!r} is not modelled "
                            "in the ferrofluid simulation")
            continue
        geometry = region.get("geometry")
        if isinstance(geometry, dict):
            if "fluid_region" in spec:
                raise ValueError("give the fluid geometry once: one ferrofluid region or "
                                 "'fluid_region'")
            spec["fluid_region"] = geometry
        elif geometry is not None:
            warnings.append(f"region geometry {geometry!r} is a label only; give a dict with "
                            "center_um and size_um to set the fluid geometry")
    electrodes = spec.get("electrodes") or {}
    device = build_device(ports_from_electrodes(electrodes), spec)
    writers = _candidates(device, "write")
    duration = spec.get("write_duration_ns", DEFAULT_WRITE_NS)
    if "write_duration_ns" not in spec:
        assumed.append(f"write_duration_ns={DEFAULT_WRITE_NS}")
    writes = spec.get("write_sequence")
    if writes is None:
        writes = []
        for i in writers:
            entry = electrodes[device.ports[i].id] or {}
            voltage = entry.get("voltage_v", write_voltage_v)
            writes.append({"port_index": i, "voltage_v": voltage, "duration_ns": duration})
        assumed.append("every write port pulsed simultaneously")
    end = max((w.get("start_ns", 0.0) + w.get("duration_ns", DEFAULT_WRITE_NS)
               for w in writes), default=0.0)
    reads = spec.get("read_sequence")
    if reads is None:
        reads = []
        for i in _candidates(device, "read"):
            reads.append({"port_index": i, "time_ns": end * (1 - 1e-6)})
            reads.append({"port_index": i, "time_ns": end})
        assumed.append("each read port sampled just before the writes end (field on) and "
                       "at the end (field off)")
    result = simulate(device, writes, reads, write_voltage_v, read_freq_hz,
                      spec.get("spectrum_hz", default_spectrum()))
    result["assumed_inputs"] = assumed + result["assumed_inputs"]
    result["warnings"] = warnings + result["warnings"]
    pairs = result["electrical_electrode_pairs"]
    metrics: dict[str, Any] = {
        "conductance_s": None, "capacitance_f": None, "f3d_hz": None,
        "write_voltage_v": write_voltage_v, "read_freq_hz": read_freq_hz,
        "reads": [{k: r[k] for k in (
            "port", "time_s", "frequency_hz", "impedance_ohm", "signal_vs_unwritten_ohm",
            "magnetization_flux_linkage_wb", "magnetization_emf_v", "thermal_noise_v_rms")}
            for r in result["reads"]],
    }
    status = {
        "reads": "computed (uncalibrated physics model)",
        "conductance_s": "not modelled without conductivity_s_per_m, relative_permittivity "
                         "and electrode_radius_um",
        "f3d_hz": "not modelled: no circuit topology is defined",
    }
    status["capacitance_f"] = status["conductance_s"]
    if isinstance(pairs, list):
        first = pairs[0]["ports"]
        freq = np.array([read_freq_hz])
        z = device.electrical_pair_impedance(first[0], first[1], freq)
        bulk = device.electrical_pair_impedance(first[0], first[1], freq,
                                                include_double_layer=False)
        y = 1.0 / bulk[0]
        metrics["conductance_s"] = float(y.real)
        metrics["capacitance_f"] = float(y.imag / (2 * np.pi * read_freq_hz))
        metrics["electrode_pair"] = first
        metrics["electrode_pair_impedance_ohm"] = complex_dict(z[0])
        status["conductance_s"] = status["capacitance_f"] = (
            "computed for the first electrode pair, bulk fluid only (user-supplied "
            "conductivity and permittivity)")
    return {
        "device_name": spec.get("name", "custom_device"),
        "description": spec.get("description", ""),
        "write_voltage_v": write_voltage_v,
        "read_freq_hz": read_freq_hz,
        "metrics": metrics,
        "metric_status": status,
        "region_count": len(spec.get("regions") or []),
        "electrode_count": len(electrodes),
        "simulation": result,
        "provenance": {
            "model": "chiplab ferrofluid particle model (docs/ferrofluid_model.md)",
            "note": "physics-based and uncalibrated unless parameters are labelled fitted",
        },
    }


def multiphysics(device_spec: dict, electrode_array: dict, write_sequence: list,
                 read_sequence: list) -> dict[str, Any]:
    """Back end of run_multiphysics_simulation for ferrofluid."""
    validate_device_spec(device_spec)
    device = build_device(ports_from_array(electrode_array), electrode_array, device_spec)
    result = simulate(device, write_sequence, read_sequence,
                      device_spec.get("write_voltage_v"), None,
                      device_spec.get("spectrum_hz", default_spectrum()))
    unused = sorted({"electrodes", "regions", "write_sequence", "read_sequence",
                     "write_duration_ns"} & set(device_spec))
    if unused:
        result["warnings"].append(
            f"device_spec key(s) {unused} ignored: this tool takes ports from electrode_array "
            "and pulses from its sequence arguments")
    energy = sum(w["drive_energy_j"] for w in result["writes"])
    return {
        "device_technology": "ferrofluid",
        "electrode_array": {
            "arrangement": electrode_array.get("arrangement"),
            "n_ports": len(device.ports),
            "pitch_um": electrode_array.get("pitch_um"),
        },
        "total_energy_pj": float(energy * 1e12),
        "energy_definition": "resistive drive energy V^2/R x duration of the write pulses",
        **result,
    }


SWEEP_OBJECTIVES = {
    # objective: (metric, True if larger is better)
    "write_energy": ("write_drive_energy_j", False),
    "power": ("write_drive_power_w", False),
    "read_signal": ("read_impedance_signal_ohm", True),
    "read_flux": ("read_flux_linkage_abs_wb", True),
    "decay_time": ("magnetization_decay_1_over_e_s", True),
    "peak_field": ("write_peak_field_a_m", True),
}


def sweep(device_spec: dict, combos: list[dict], objective: str) -> list[dict[str, Any]]:
    """Evaluate parameter combinations; each point is a full physics simulation."""
    validate_device_spec(device_spec)
    base: dict[str, Any] = {}
    for source in (device_spec.get("parameters"), device_spec.get("ferrofluid"), device_spec):
        if source:
            base.update({k: v for k, v in source.items() if k in SIMPLE_KEYS
                         or k in FerrofluidParams.names()})
    if "write_voltage_v" in device_spec:
        base["voltage_v"] = device_spec["write_voltage_v"]
    results = []
    for combo in combos:
        summary = technology_summary({**base, **combo}, None)
        metrics = summary["metrics"]
        writes = summary["simulation"]["writes"]
        metrics["write_drive_power_w"] = float(sum(
            w["drive_energy_j"] / w["duration_s"] for w in writes))
        metrics["read_flux_linkage_abs_wb"] = abs(metrics["read_flux_linkage_wb"])
        key = SWEEP_OBJECTIVES.get(objective, (None, False))[0]
        results.append({
            "parameters": combo,
            "objective_value": None if key is None else metrics.get(key),
            "metrics": metrics,
            "warnings": summary["simulation"]["warnings"],
            "success": True,
        })
    return results


ANALYSIS_OPTIONS = {
    "sensitivity": {"relative_step", "parameters"},
    "read_disturbance": {"port", "amplitude_v", "frequency_hz", "cycles", "steps_per_cycle"},
    "retention": {"port", "state_b_write_sequence", "bandwidth_hz", "snr_threshold"},
    "scaling": {"small_shape", "large_shape", "voltage_v", "pitch_um", "geometry"},
}


def analyze(analysis: str, device_spec: dict, electrode_array: dict,
            write_sequence: list | None, read_sequence: list | None, options: dict | None
            ) -> dict[str, Any]:
    """Back end of chiplab_ferrofluid_analyze."""
    from chiplab.models.materials import ferro_analysis

    if analysis not in ANALYSIS_OPTIONS:
        raise ValueError(f"analysis must be one of {sorted(ANALYSIS_OPTIONS)}")
    options = dict(options or {})
    check_keys(options, ANALYSIS_OPTIONS[analysis], f"{analysis} option")
    if not routes_to_ferrofluid(device_spec, electrode_array):
        raise CredibilityError("these analyses exist only for ferrofluid devices")
    validate_device_spec(device_spec)
    assumed: list[str] = []
    ignored: list[str] = []

    def need(name: str) -> Any:
        if name not in options:
            raise ValueError(f"{analysis} needs options[{name!r}]")
        return options.pop(name)

    if analysis == "scaling":
        pitch = options.pop("pitch_um", (electrode_array or {}).get("pitch_um"))
        if pitch is None:
            raise ValueError("scaling needs pitch_um")
        geometry = dict(options.pop("geometry", {}))
        check_keys(geometry, set(GEOMETRY_KEYS) - {"start_um", "end_um"}, "geometry")
        if geometry.get("conductor", "loop") != "loop":
            raise ValueError("scaling is implemented for loop ports only")
        if not geometry:
            assumed.append(f"port geometry: {PORT_DEFAULT_NOTES}")
        result = ferro_analysis.array_scaling(
            particle_params(device_spec), geometry, float(pitch),
            tuple(options.pop("small_shape", (2, 2))),
            tuple(options.pop("large_shape", (4, 4))), float(need("voltage_v")))
        return {"analysis": analysis, **result, "assumed_inputs": assumed}

    device = build_device(ports_from_array(electrode_array), electrode_array, device_spec)
    default_voltage = device_spec.get("write_voltage_v")
    writes = parse_writes(write_sequence or [], device, assumed, default_voltage)
    if not writes:
        raise ValueError(f"{analysis} needs a write_sequence")
    if analysis == "sensitivity":
        reads = parse_reads(read_sequence or [], device, assumed, ignored)
        result = ferro_analysis.parameter_sensitivity(
            device, writes, reads, float(options.get("relative_step", 0.05)),
            options.get("parameters"))
    elif analysis == "read_disturbance":
        port = options.get("port", _candidates(device, "read")[0])
        result = ferro_analysis.read_disturbance(
            device, writes, port, float(need("amplitude_v")), float(need("frequency_hz")),
            int(options.get("cycles", 10)), int(options.get("steps_per_cycle", 32)))
    else:
        port = options.get("port", _candidates(device, "read")[0])
        other = parse_writes(options.get("state_b_write_sequence") or [], device, assumed,
                             default_voltage)
        if "bandwidth_hz" not in options:
            assumed.append("bandwidth_hz=1e6")
        result = ferro_analysis.state_distinguishability(
            device, writes, other, port, float(options.get("bandwidth_hz", 1e6)),
            float(options.get("snr_threshold", 3.0)))
    return {
        "analysis": analysis, **result,
        "assumed_inputs": assumed + [
            f"port {p.id}: {', '.join(p.assumed)}" for p in device.ports if p.assumed],
        "warnings": result.get("warnings", []) + ignored + device.warnings
        + device.ensemble.validity_warnings(),
    }
