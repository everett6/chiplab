"""Package thermal and airflow model for GPU Studio (numpy only, deterministic).

Two stacked finite-volume layers on one grid over the heat-spreader footprint:

* layer 0, die plane: silicon dies where components sit (k = 150 W/mK x die thickness), the
  interposer/mold elsewhere; component power is injected here (interconnect power spreads
  over the interposer);
* layer 1, heat spreader (IHS): copper, 3 mm, plus the heatsink base's thermal mass.

Layer 0 couples to layer 1 through TIM1 under dies (mold elsewhere) and to the board through
the substrate; layer 1 couples to ambient through the heatsink resistance, spread uniformly
over the IHS. The heatsink resistance comes from the airflow model: fan flow through the fin
channels -> air velocity -> laminar flat-plate convection coefficient -> fin efficiency ->
R = 1 / (h * (A_base + eta * A_fins)). A design's ``constraints.thermal_resistance_c_per_w``
overrides it. Steady state and a backward-Euler power-on transient are solved with Jacobi-
preconditioned conjugate gradient, matrix-free. At steady state, heat out equals power in
(checked in tests).

First-order engineering model: no CFD (air temperature rises uniformly along the fins),
no temperature-dependent leakage, no micro-channel / liquid cooling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

K_SI, K_CU, K_AL = 150.0, 390.0, 205.0          # W/mK
RHOC_SI, RHOC_CU, RHOC_AL = 1.66e6, 3.45e6, 2.42e6  # J/m^3K
AIR_K, AIR_NU, AIR_PR, AIR_RHO, AIR_CP = 0.026, 1.6e-5, 0.71, 1.16, 1007.0
CFM_TO_M3S = 4.719e-4


@dataclass
class Cooling:
    fans: int = 2
    fan_cfm: float = 60.0           # free-air flow per fan
    fan_diameter_mm: float = 90.0
    fin_count: int = 60
    fin_height_mm: float = 40.0
    fin_thickness_mm: float = 0.5
    base_thickness_mm: float = 6.0
    ihs_thickness_mm: float = 3.0
    tim1_w_mk: float = 5.0
    tim1_um: float = 50.0
    board_r_c_per_w: float = 3.0    # package -> board -> ambient (parallel path)

    @classmethod
    def from_meta(cls, meta: dict[str, Any] | None) -> Cooling:
        c = cls()
        for k, v in (meta or {}).items():
            if hasattr(c, k) and isinstance(v, (int, float)) and v > 0:
                setattr(c, k, type(getattr(c, k))(v))
        return c


def airflow(c: Cooling, sink_w_mm: float, sink_d_mm: float, power_w: float) -> dict[str, Any]:
    """Fan flow through the fin channels -> heatsink resistance and air temperature rise."""
    q = c.fans * c.fan_cfm * CFM_TO_M3S                     # m^3/s (free-air rating)
    pitch = sink_w_mm / max(c.fin_count, 1)
    gap = max(pitch - c.fin_thickness_mm, 0.2)
    a_free = (c.fin_count - 1) * gap * c.fin_height_mm * 1e-6  # m^2 open channel area
    v = q / a_free if a_free > 0 else 0.0
    length = sink_d_mm * 1e-3                                 # flow runs along the depth
    re = v * length / AIR_NU
    h = 0.664 * AIR_K * math.sqrt(max(re, 1.0)) * AIR_PR ** (1 / 3) / length
    m = math.sqrt(2 * h / (K_AL * c.fin_thickness_mm * 1e-3))
    ml = m * c.fin_height_mm * 1e-3
    eta = math.tanh(ml) / ml if ml > 0 else 1.0
    a_fins = 2 * c.fin_count * c.fin_height_mm * sink_d_mm * 1e-6
    a_base = sink_w_mm * sink_d_mm * 1e-6 - c.fin_count * c.fin_thickness_mm * sink_d_mm * 1e-6
    r_hs = 1.0 / (h * (max(a_base, 0) + eta * a_fins))
    dt_air = power_w / (AIR_RHO * AIR_CP * q) if q > 0 else float("inf")
    return {"flow_m3s": q, "flow_cfm": c.fans * c.fan_cfm, "velocity_ms": v,
            "reynolds": re, "h_w_m2k": h, "fin_efficiency": eta, "fin_area_m2": a_fins,
            "r_heatsink_c_per_w": r_hs, "air_rise_c": dt_air,
            "fan_rpm": round(2000 * c.fan_cfm / 60.0), "regime": "laminar" if re < 5e5
            else "turbulent"}


def _cg(apply, b, precond, x0, tol=1e-7, maxit=3000):
    """Preconditioned conjugate gradient; ``precond`` maps a residual to a correction."""
    x = x0.copy()
    r = b - apply(x)
    z = precond(r)
    p = z.copy()
    rz = float((r * z).sum())
    bn = float(np.sqrt((b * b).sum())) or 1.0
    for it in range(maxit):
        ap = apply(p)
        alpha = rz / float((p * ap).sum())
        x += alpha * p
        r -= alpha * ap
        if float(np.sqrt((r * r).sum())) / bn < tol:
            break
        z = precond(r)
        rz_new = float((r * z).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def _two_level(apply, diag, block: int):
    """Jacobi + coarse correction on block aggregates (Galerkin P^T A P, built by applying
    the operator to each aggregate indicator). It removes the slow near-uniform modes left by
    the weak heatsink coupling, so CG converges in tens of iterations instead of thousands."""
    L, ny, nx = diag.shape
    by, bx = -(-ny // block), -(-nx // block)
    agg = (np.arange(ny)[:, None] // block) * bx + (np.arange(nx)[None, :] // block)
    nc = by * bx

    def restrict(u):
        return np.stack([np.bincount(agg.ravel(), u[k].ravel(), nc) for k in range(L)]).ravel()

    def prolong(v):
        v = v.reshape(L, nc)
        return np.stack([v[k][agg] for k in range(L)])

    ac = np.zeros((L * nc, L * nc))
    for j in range(L * nc):
        e = np.zeros(L * nc)
        e[j] = 1.0
        ac[:, j] = restrict(apply(prolong(e)))
    ac_inv = np.linalg.inv(ac)
    inv_d = 1.0 / diag
    return lambda r: inv_d * r + prolong(ac_inv @ restrict(r))


def solve(domain: tuple[float, float, float, float], components: list[dict[str, Any]],
          interposer_power_w: float, cooling: Cooling, ambient_c: float,
          r_override: float | None = None, n: int = 72) -> dict[str, Any]:
    """domain = (x0, y0, width, depth) in mm; components carry x, y, w, d (center, mm), power_w,
    thickness_mm. Returns grids (die plane), per-component temperatures and a transient."""
    x0, y0, wd, dp = domain
    nx = n
    ny = max(8, round(n * dp / wd))
    dx, dy = wd / nx * 1e-3, dp / ny * 1e-3                  # m
    cell = dx * dy
    xs = x0 + (np.arange(nx) + 0.5) * wd / nx
    ys = y0 + (np.arange(ny) + 0.5) * dp / ny
    gx, gy = np.meshgrid(xs, ys)                              # [ny, nx]

    t0 = np.full((ny, nx), 0.15e-3)                           # interposer + mold, m
    k0 = np.full((ny, nx), 60.0)                              # mold/underfill/interposer mix
    q0 = np.zeros((ny, nx))
    under = np.zeros((ny, nx), bool)
    masks = []
    for c in components:
        m = ((np.abs(gx - c["x"]) <= c["w"] / 2) & (np.abs(gy - c["y"]) <= c["d"] / 2))
        if not m.any():  # tiny part: nearest cell
            m = np.zeros_like(under)
            m[np.abs(ys - c["y"]).argmin(), np.abs(xs - c["x"]).argmin()] = True
        masks.append(m)
        under |= m
        t0[m] = max(c.get("thickness_mm", 0.775), 0.05) * 1e-3
        k0[m] = K_SI
        q0[m] += c.get("power_w", 0.0) / m.sum()
    q0 += interposer_power_w / q0.size
    p_total = float(q0.sum())

    air = airflow(cooling, wd + 20, dp + 20, p_total)
    r_hs = r_override if r_override else air["r_heatsink_c_per_w"]
    g_tim = np.where(under, cooling.tim1_w_mk / (cooling.tim1_um * 1e-6), 200.0) * cell
    g_sink = (1.0 / r_hs) / (nx * ny)                          # W/K per cell
    g_board = (1.0 / cooling.board_r_c_per_w) / (nx * ny)
    t1 = cooling.ihs_thickness_mm * 1e-3
    c0 = RHOC_SI * t0 * cell
    sink_mass = RHOC_AL * ((wd + 20) * (dp + 20) * cooling.base_thickness_mm
                           + cooling.fin_count * cooling.fin_thickness_mm
                           * cooling.fin_height_mm * (dp + 20)) * 1e-9
    c1 = np.full((ny, nx), RHOC_CU * t1 * cell + sink_mass / (nx * ny))

    def faces(kt):
        gxf = 2 * kt[:, 1:] * kt[:, :-1] / (kt[:, 1:] + kt[:, :-1]) * dy / dx
        gyf = 2 * kt[1:, :] * kt[:-1, :] / (kt[1:, :] + kt[:-1, :]) * dx / dy
        d = np.zeros_like(kt)
        d[:, 1:] += gxf
        d[:, :-1] += gxf
        d[1:, :] += gyf
        d[:-1, :] += gyf
        return gxf, gyf, d

    kt0, kt1 = k0 * t0, np.full((ny, nx), K_CU * t1)
    f0, f1 = faces(kt0), faces(kt1)

    def lateral(T, f):
        """-div(k t grad T) integrated over a cell; insulated edges, harmonic-mean faces."""
        gxf, gyf, _ = f
        out = np.zeros_like(T)
        fx = gxf * (T[:, 1:] - T[:, :-1])
        fy = gyf * (T[1:, :] - T[:-1, :])
        out[:, 1:] += fx          # cell j+1 loses g (T[j+1] - T[j]) to cell j
        out[:, :-1] -= fx
        out[1:, :] += fy
        out[:-1, :] -= fy
        return out

    base_diag = np.stack([f0[2] + g_tim + g_board, f1[2] + g_tim + g_sink])
    block = max(4, nx // 12)

    def make_apply(cdt0=0.0, cdt1=0.0):
        def apply(u):
            a, b = u[0], u[1]
            r0 = lateral(a, f0) + g_tim * (a - b) + g_board * a + cdt0 * a
            r1 = lateral(b, f1) + g_tim * (b - a) + g_sink * b + cdt1 * b
            return np.stack([r0, r1])
        return apply

    rhs = np.stack([q0, np.zeros_like(q0)])
    ap = make_apply()
    rise = _cg(ap, rhs, _two_level(ap, base_diag, block), np.zeros_like(rhs), tol=1e-9)
    steady = rise + ambient_c
    heat_out = float((g_sink * rise[1]).sum() + (g_board * rise[0]).sum())

    # Power-on transient (backward Euler on log-spaced times)
    tau = (c0.sum() + c1.sum()) * (r_hs * cooling.board_r_c_per_w / (r_hs + cooling.board_r_c_per_w))
    times = np.unique(np.concatenate([[0.0], np.geomspace(0.01, 6 * tau, 15)]))
    u = np.zeros_like(rhs)
    frames, series = [], []
    ds = max(1, nx // 36)
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        cdt0, cdt1 = c0 / dt, c1 / dt
        d = base_diag + np.stack([cdt0, cdt1])
        ap = make_apply(cdt0, cdt1)
        u = _cg(ap, rhs + np.stack([cdt0 * u[0], cdt1 * u[1]]), _two_level(ap, d, block), u)
        die = u[0] + ambient_c
        series.append({"t_s": float(times[i]), "max_c": float(die.max()),
                       "mean_die_c": float(die[under].mean()) if under.any() else float(die.mean())})
        frames.append(np.round(die[::ds, ::ds], 2).tolist())

    per = []
    for c, m in zip(components, masks):
        per.append({"id": c["id"], "max_c": float(steady[0][m].max()),
                    "mean_c": float(steady[0][m].mean()),
                    "headroom_c": None})
    iy, ix = np.unravel_index(int(steady[0].argmax()), steady[0].shape)
    return {
        "grid": {"nx": nx, "ny": ny, "x0": x0, "y0": y0, "w": wd, "d": dp,
                 "die_c": np.round(steady[0], 2).tolist(), "ihs_c": np.round(steady[1], 2).tolist()},
        "components": per,
        "hotspot": {"x": float(xs[ix]), "y": float(ys[iy]), "temp_c": float(steady[0].max())},
        "ambient_c": ambient_c, "power_in_w": p_total, "heat_out_w": heat_out,
        "r_heatsink_c_per_w": r_hs,
        "r_source": "design constraints" if r_override else "airflow model",
        "airflow": air, "time_constant_s": tau,
        "transient": {"series": series, "frames": frames,
                      "frame_grid": {"nx": len(frames[0][0]) if frames else 0,
                                     "ny": len(frames[0]) if frames else 0}},
    }


def stack_profile(top_c: float, die_powers_w: list[float], area_mm2: float,
                  bond_um: float = 20.0, bond_k: float = 1.5) -> list[float]:
    """Temperatures of the dies in a stack, bottom (base die) first, given the temperature of
    the top die: heat from every die below an interface flows up through it."""
    a = area_mm2 * 1e-6
    r_if = bond_um * 1e-6 / (bond_k * a) if a > 0 else 0.0
    temps = [0.0] * len(die_powers_w)
    t = top_c
    temps[-1] = t
    for i in range(len(die_powers_w) - 2, -1, -1):
        t += sum(die_powers_w[: i + 1]) * r_if
        temps[i] = t
    return temps
