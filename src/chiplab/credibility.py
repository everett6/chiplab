"""Scientific credibility gate between material devices and architecture-level designs.

A chiplet that is a material device is accepted by the design store, the validator, the
performance engines and the cost model only if it carries its device record (geometry,
parameters, operating conditions and the user's measurements) and that record passes
every check in ``models.materials.ferro_validation``.  The checks are recomputed from the
record each time; a stored "passed" flag is never read.  Only ferrofluid has checks, so
every other material device is rejected.

What this cannot do: ChipLab has no way to know whether measurements are authentic, or
that a chiplet is a material device if nothing in it says so.
"""

from __future__ import annotations

import re
from typing import Any

FERROFLUID_NAMES = (
    "ferrofluid", "magneticfluid", "magneticnanofluid", "magneticnanoparticle",
    "ferronanofluid", "magneticcolloid")

# Other material technologies ChipLab lists as experimental. A hand-written chiplet naming
# one of these is flagged by the validator but not rejected (existing designs use them).
UNGATED_EXPERIMENTAL = (
    "memristor", "memristive", "reram", "phasechange", "pcm", "ferroelectric", "fefet",
    "ftj", "spintronic", "photonic", "domainwall", "skyrmion", "ncfet", "tfet", "tmdfet",
    "graphene", "transmon", "fluxonium", "majorana", "sfq")


class CredibilityError(ValueError):
    """A material device was refused entry to an architecture-level design."""


def _normalise(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def is_ferrofluid(value: Any) -> bool:
    """True for any spelling of ferrofluid / magnetic fluid inside a string."""
    return isinstance(value, str) and any(n in _normalise(value) for n in FERROFLUID_NAMES)


def mentions_ferrofluid(data: Any, _depth: int = 0) -> bool:
    """True if any key or string anywhere inside nested data names a ferrofluid."""
    if _depth > 12:
        return False
    if isinstance(data, str):
        return is_ferrofluid(data)
    if isinstance(data, dict):
        return any(mentions_ferrofluid(k, _depth + 1) or mentions_ferrofluid(v, _depth + 1)
                   for k, v in data.items())
    if isinstance(data, (list, tuple, set)):
        return any(mentions_ferrofluid(v, _depth + 1) for v in data)
    return False


def refuse_ferrofluid(where: str, *data: Any) -> None:
    """Used by generic models that must not return numbers for a ferrofluid."""
    if any(mentions_ferrofluid(d) for d in data):
        raise CredibilityError(
            f"{where} has no ferrofluid physics and will not return numbers for one. Use "
            "run_multiphysics_simulation, simulate_material_circuit or "
            "chiplab_ferrofluid_analyze (docs/ferrofluid_model.md).")


def is_material_chiplet(chiplet: Any) -> bool:
    """A chiplet that declares itself (or names itself) as a simulated material device."""
    if getattr(chiplet, "material_device", None) is not None:
        return True
    ident = str(chiplet.id or "").lower()
    if _normalise(chiplet.type) in ("material", "materialdevice") or ident.startswith(
            "material_"):
        return True
    names = [chiplet.type, chiplet.id]
    if chiplet.memory is not None:
        names.append(chiplet.memory.technology)
    if chiplet.compute is not None:
        names.extend([chiplet.compute.architecture, chiplet.compute.precision])
    return any(is_ferrofluid(n) for n in names if n)


def _chiplet_problems(chiplet: Any) -> list[str]:
    label = f"chiplet '{chiplet.id or chiplet.type}'"
    record = getattr(chiplet, "material_device", None)
    if record is None:
        return [f"{label} is a material device without a validation record: create it with "
                "material_device_to_chiplet using your measurements"]
    from chiplab.models.materials.ferro_validation import chiplet_problems

    return [f"{label}: {p}" for p in chiplet_problems(chiplet, record)]


def material_problems(design: Any) -> list[str]:
    """Every reason the material chiplets in a design are not acceptable (empty if fine)."""
    problems: list[str] = []
    for chiplet in design.chiplets:
        if is_material_chiplet(chiplet):
            problems.extend(_chiplet_problems(chiplet))
    return problems


def material_warnings(design: Any) -> list[str]:
    out = []
    for chiplet in design.chiplets:
        if is_material_chiplet(chiplet):
            continue
        names = [chiplet.type]
        if chiplet.memory is not None:
            names.append(chiplet.memory.technology)
        if chiplet.compute is not None:
            names.append(chiplet.compute.architecture)
        text = " ".join(_normalise(n) for n in names if n)
        found = next((n for n in UNGATED_EXPERIMENTAL if n in text), None)
        if found:
            out.append(
                f"chiplet '{chiplet.id or chiplet.type}' names an experimental material "
                f"technology ({found}); its numbers are not physics-validated by ChipLab")
    return out


def require_credible_design(design: Any) -> None:
    """Raise CredibilityError unless every material chiplet in the design is accepted."""
    problems = material_problems(design)
    if problems:
        raise CredibilityError(
            "material device rejected by the physics validation gate: " + "; ".join(problems))
