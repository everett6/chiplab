"""CFD Studio (optional): a coarse OpenFOAM case for one heatsink fin channel.

The studio's airflow model (``studio/thermal.py``) turns fan flow into a laminar flat-plate
heat-transfer coefficient. This module writes an OpenFOAM case for one fin-to-fin channel of
the same heatsink (3D box: fin gap x fin height x heatsink depth, mesh about 12 x 24 x 60),
solves the laminar flow with ``simpleFoam`` and then the air temperature with
``scalarTransportFoam`` (walls at a fixed fin temperature, inlet at ambient). The outlet
flow-weighted mean temperature gives the channel's heat-transfer coefficient through the
log-mean temperature difference, which is compared against the airflow model's ``h``.

Limits: forced convection in a single channel with isothermal walls (fin conduction is not
solved; the airflow model's fin efficiency is kept), constant air properties, no bypass flow.
Only runs when OpenFOAM's ``blockMesh``, ``simpleFoam`` and ``scalarTransportFoam`` are on
the path; otherwise the studio shows the case and the airflow model numbers only.
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path
from typing import Any

from chiplab.studio import thermal as th
from chiplab.tools import registry

HDR = """FoamFile
{{
    version 2.0;
    format ascii;
    class {cls};
    object {obj};
}}
"""


def channel(scene: dict[str, Any]) -> dict[str, Any]:
    """Channel geometry and boundary conditions from the scene's heatsink + airflow result."""
    sink = scene["package"]["heatsink"]
    air = scene["thermal"]["airflow"]
    pitch = sink["w"] / max(int(sink["fins"]), 1)
    gap = max(pitch - sink["fin_t"], 0.2)
    amb = float(scene["thermal"].get("ambient_c", 25.0))
    # heatsink base temperature implied by the airflow model: ambient + P * R_heatsink
    t_wall = amb + float(scene["thermal"]["power_in_w"]) * float(
        scene["thermal"]["r_heatsink_c_per_w"])
    return {"gap_mm": gap, "height_mm": float(sink["fin_h"]), "length_mm": float(sink["d"]),
            "velocity_ms": float(air["velocity_ms"]), "ambient_c": amb,
            "wall_c": max(t_wall, amb + 5.0), "model_h_w_m2k": float(air["h_w_m2k"]),
            "reynolds": float(air["reynolds"])}


def _files(ch: dict[str, Any]) -> dict[str, str]:
    g, h, length = ch["gap_mm"] * 1e-3, ch["height_mm"] * 1e-3, ch["length_mm"] * 1e-3
    nu = th.AIR_NU
    alpha = th.AIR_K / (th.AIR_RHO * th.AIR_CP)
    v = ch["velocity_ms"]
    t_in, t_w = ch["ambient_c"] + 273.15, ch["wall_c"] + 273.15
    end_time = 400
    verts = " ".join(f"({x:g} {y:g} {z:g})" for z in (0, h) for x, y in
                     ((0, 0), (length, 0), (length, g), (0, g)))
    block_mesh = HDR.format(cls="dictionary", obj="blockMeshDict") + f"""
convertToMeters 1;
vertices ( {verts} );
blocks ( hex (0 1 2 3 4 5 6 7) (60 12 24) simpleGrading (1 1 1) );
boundary
(
    inlet  {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    fins   {{ type wall;  faces ((0 1 5 4) (3 7 6 2)); }}
    base   {{ type wall;  faces ((0 3 2 1)); }}
    shroud {{ type wall;  faces ((4 5 6 7)); }}
);
"""
    walls_u = "fins { type noSlip; } base { type noSlip; } shroud { type noSlip; }"
    u = HDR.format(cls="volVectorField", obj="U") + f"""
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({v:g} 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform ({v:g} 0 0); }}
    outlet {{ type zeroGradient; }}
    {walls_u}
}}
"""
    p = HDR.format(cls="volScalarField", obj="p") + """
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    fins { type zeroGradient; } base { type zeroGradient; } shroud { type zeroGradient; }
}
"""
    t = HDR.format(cls="volScalarField", obj="T") + f"""
dimensions [0 0 0 1 0 0 0];
internalField uniform {t_in:g};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {t_in:g}; }}
    outlet {{ type zeroGradient; }}
    fins {{ type fixedValue; value uniform {t_w:g}; }}
    base {{ type fixedValue; value uniform {t_w:g}; }}
    shroud {{ type zeroGradient; }}
}}
"""
    transport = HDR.format(cls="dictionary", obj="transportProperties") + f"""
transportModel Newtonian;
nu {nu:g};
DT {alpha:g};
"""
    momentum = HDR.format(cls="dictionary", obj="momentumTransport") + "simulationType laminar;\n"
    turbulence = HDR.format(cls="dictionary", obj="turbulenceProperties") + \
        "simulationType laminar;\n"
    control = HDR.format(cls="dictionary", obj="controlDict") + f"""
application simpleFoam;
startFrom latestTime;
stopAt endTime;
endTime {end_time};
deltaT 1;
writeControl timeStep;
writeInterval {end_time};
writeFormat ascii;
functions
{{
    outletT
    {{
        type surfaceFieldValue;
        libs ("libfieldFunctionObjects.so");
        writeControl writeTime;
        regionType patch;
        name outlet;
        operation weightedAverage;
        weightField phi;
        fields (T);
        writeFields false;
    }}
}}
"""
    schemes = HDR.format(cls="dictionary", obj="fvSchemes") + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U);
             div(phi,T) bounded Gauss linearUpwind grad(T);
             div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
    solution = HDR.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; }
    "(U|T)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE { nNonOrthogonalCorrectors 0; consistent yes; residualControl { p 1e-5; U 1e-6; } }
relaxationFactors { equations { U 0.9; ".*" 0.9; } }
"""
    return {"system/blockMeshDict": block_mesh, "system/controlDict": control,
            "system/fvSchemes": schemes, "system/fvSolution": solution,
            "constant/transportProperties": transport,
            "constant/momentumTransport": momentum,
            "constant/turbulenceProperties": turbulence,
            "0/U": u, "0/p": p, "0/T": t}


def write_case(root: Path, scene: dict[str, Any]) -> dict[str, Any]:
    ch = channel(scene)
    for rel, text in _files(ch).items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return ch


def h_from_outlet(ch: dict[str, Any], t_out_c: float) -> float:
    """Channel h from the outlet mean temperature (log-mean temperature difference)."""
    g, h, length = ch["gap_mm"] * 1e-3, ch["height_mm"] * 1e-3, ch["length_mm"] * 1e-3
    mdot = th.AIR_RHO * ch["velocity_ms"] * g * h
    area = (2 * h + g) * length           # two fin faces + base
    dt_in, dt_out = ch["wall_c"] - ch["ambient_c"], ch["wall_c"] - t_out_c
    if dt_out <= 0 or dt_in <= dt_out:
        return float("nan")
    return mdot * th.AIR_CP * math.log(dt_in / dt_out) / area


def _sibling(name: str) -> registry.Tool:
    base = registry.get("openfoam")
    path = str(Path(str(base.path)).with_name(name)) if base.path else None
    t = registry.Tool("openfoam", path if path and Path(path).is_file() else None, base.version,
                      dict(base.env))
    return t


def run(scene: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
    if not registry.get("openfoam").available:
        raise registry.ToolError("OpenFOAM is not installed")
    with tempfile.TemporaryDirectory(prefix="chiplab-foam-") as tmp:
        ch = write_case(Path(tmp), scene)
        logs: dict[str, str] = {}
        for app in ("blockMesh", "simpleFoam", "scalarTransportFoam"):
            tool = _sibling(app)
            if not tool.available:
                raise registry.ToolError(f"OpenFOAM {app} not found next to blockMesh")
            if app == "scalarTransportFoam":
                ctl = Path(tmp) / "system" / "controlDict"
                ctl.write_text(ctl.read_text().replace("application simpleFoam;",
                                                       "application scalarTransportFoam;"))
            r = registry.run(tool, ["-case", tmp], cwd=tmp, timeout=timeout)
            logs[app] = (r.stdout + r.stderr)[-1500:]
            if r.returncode != 0:
                raise registry.ToolError(f"{app} failed: {logs[app][-600:]}")
        dat = sorted(Path(tmp).glob("postProcessing/outletT/*/surfaceFieldValue.dat"))
        if not dat:
            raise registry.ToolError("OpenFOAM wrote no outlet temperature")
        rows = [ln.split() for ln in dat[-1].read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
        t_out = float(rows[-1][-1]) - 273.15
    h_cfd = h_from_outlet(ch, t_out)
    return {"channel": ch, "outlet_c": round(t_out, 3), "h_cfd_w_m2k": round(h_cfd, 2),
            "h_model_w_m2k": round(ch["model_h_w_m2k"], 2),
            "ratio_cfd_over_model": round(h_cfd / ch["model_h_w_m2k"], 3)
            if ch["model_h_w_m2k"] > 0 else None,
            "logs": {k: re.sub(r"\s+\n", "\n", v)[-600:] for k, v in logs.items()}}


def preview(scene: dict[str, Any]) -> dict[str, Any]:
    """Case files + airflow-model numbers, without running OpenFOAM."""
    ch = channel(scene)
    return {"channel": ch, "files": _files(ch),
            "airflow_model": scene["thermal"]["airflow"]}
