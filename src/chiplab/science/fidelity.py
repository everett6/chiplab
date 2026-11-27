"""Multi-fidelity simulation scheduler.

Levels, cheapest first:
  1 analytical       eliminate bad ideas quickly
  2 behavioral       check functionality (golden model, Monte Carlo)
  3 circuit_spice    electrical behaviour
  4 rtl_sim          digital implementation, bit-exact
  5 synthesis        gates and area
  6 place_route      physical estimates
  7 measured         calibrate against hardware (learning loop)

Every candidate starts at level 1. After each level, a candidate is promoted only if it can
still win: its optimistic bound (value shrunk by the level's relative uncertainty) must beat
the best candidate's pessimistic bound, and it must pass that level's functional check.
Levels a candidate type cannot run are skipped with a reason. The trace records every
decision so it can be reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

LEVELS = ("analytical", "behavioral", "circuit_spice", "rtl_sim", "synthesis", "place_route",
          "measured")
# Default relative uncertainty of a metric estimate at each level (assumed, conservative).
UNCERTAINTY = {"analytical": 0.5, "behavioral": 0.3, "circuit_spice": 0.15, "rtl_sim": 0.15,
               "synthesis": 0.10, "place_route": 0.05, "measured": 0.02}


@dataclass
class Level:
    name: str
    run: Callable[[dict[str, Any]], dict[str, Any]]
    """Returns {"ok": bool, "value": float | None, "why": str, ...}; value is the objective
    estimate at this level (None keeps the previous estimate)."""
    cost_s: float = 1.0
    rel_uncertainty: float | None = None


@dataclass
class Candidate:
    id: str
    spec: dict[str, Any]
    value: float | None = None
    level: str | None = None
    alive: bool = True
    history: list[dict[str, Any]] = field(default_factory=list)


def run(candidates: list[tuple[str, dict[str, Any]]], ladder: list[Level], *,
        minimize: bool = True, keep_at_least: int = 1, max_cost_s: float = 3600.0
        ) -> dict[str, Any]:
    """Screen candidates up the ladder. Returns survivors, eliminations and the trace."""
    if not candidates:
        raise ValueError("no candidates")
    order = {n: i for i, n in enumerate(LEVELS)}
    ladder = sorted(ladder, key=lambda lv: order.get(lv.name, 99))
    cands = [Candidate(cid, spec) for cid, spec in candidates]
    spent = 0.0
    trace: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for lv in ladder:
        live = [c for c in cands if c.alive]
        if not live:
            break
        if spent + lv.cost_s * len(live) > max_cost_s:
            trace.append({"level": lv.name, "decision": "stopped",
                          "why": f"budget: {len(live)} x {lv.cost_s}s would exceed "
                                 f"{max_cost_s}s"})
            break
        unc = lv.rel_uncertainty if lv.rel_uncertainty is not None else UNCERTAINTY.get(
            lv.name, 0.5)
        for c in live:
            try:
                r = lv.run(c.spec)
            except Exception as e:  # noqa: BLE001 - a crashing level eliminates the candidate
                r = {"ok": False, "why": f"{type(e).__name__}: {e}"}
            spent += lv.cost_s
            if r.get("skip"):
                c.history.append({"level": lv.name, "skipped": r.get("why", "")})
                continue
            if r.get("value") is not None:
                new = float(r["value"])
                if c.value is not None and c.level is not None:
                    prev_unc = UNCERTAINTY.get(c.level, 0.5)
                    rel = abs(new - c.value) / max(abs(c.value), 1e-300)
                    if rel > prev_unc:
                        violations.append({"candidate": c.id, "from": c.level, "to": lv.name,
                                           "old": c.value, "new": new, "rel_change": rel,
                                           "band": prev_unc})
                        trace.append({"level": lv.name, "candidate": c.id,
                                      "decision": "band_violation",
                                      "why": f"{c.level} estimate {c.value:.4g} is {rel:.0%} "
                                             f"off {lv.name} {new:.4g}, outside its "
                                             f"±{prev_unc:.0%} band: earlier prunes may be "
                                             f"wrong; recalibrate the {c.level} model"})
                c.value, c.level = new, lv.name
            c.history.append({"level": lv.name, "ok": bool(r.get("ok")), "value": c.value,
                              "why": r.get("why", "")})
            if not r.get("ok"):
                c.alive = False
                trace.append({"level": lv.name, "candidate": c.id, "decision": "eliminated",
                              "why": r.get("why", "failed")})
        live = [c for c in cands if c.alive and c.value is not None]
        if len(live) > keep_at_least:
            sign = 1 if minimize else -1
            best = min(live, key=lambda c: sign * c.value)
            pess_best = best.value * (1 + sign * unc)
            for c in live:
                if c is best:
                    continue
                optimistic = c.value * (1 - sign * unc)
                if sign * optimistic > sign * pess_best:
                    c.alive = False
                    trace.append({"level": lv.name, "candidate": c.id, "decision": "pruned",
                                  "why": f"even optimistic {optimistic:.4g} cannot beat "
                                         f"{best.id} pessimistic {pess_best:.4g} "
                                         f"(±{unc:.0%})"})
        for c in cands:
            if c.alive:
                trace.append({"level": lv.name, "candidate": c.id, "decision": "promoted",
                              "value": c.value})
    survivors = [c for c in cands if c.alive]
    return {"survivors": [{"id": c.id, "value": c.value, "level": c.level,
                           "history": c.history} for c in survivors],
            "eliminated": [{"id": c.id, "last_level": c.history[-1]["level"] if c.history
                            else None, "history": c.history} for c in cands if not c.alive],
            "trace": trace, "spent_s_estimate": spent, "band_violations": violations,
            "trustworthy": not violations,
            "saved_runs": sum(len(ladder) - len(c.history) for c in cands if not c.alive)}


# Cell area per flip-flop including its share of ALU logic, derived from two yosys runs on
# IHP sg13g2 (4x1x4: 155 um2/flop, 8x4x8: 199 um2/flop). Crude; the band check catches drift.
UM2_PER_FLOP = 175.0


def sram_nmc_ladder(objective: str = "energy_ratio", op_mix: dict[str, int] | None = None
                    ) -> list[Level]:
    """Concrete ladder for chiplab.pim.sram candidates. objective: "energy_ratio" (maximise
    CPU/NMC energy -> minimise its inverse) or "area" (minimise cell area)."""
    from chiplab.pim import _eda, sram

    def l1(spec: dict[str, Any]) -> dict[str, Any]:
        c = sram.compare_with_cpu(spec, op_mix)
        v = (1 / c["total"]["energy_ratio_cpu_over_nmc"] if objective == "energy_ratio"
             else sram.predict(spec)["flip_flops"]["value"] * UM2_PER_FLOP)
        return {"ok": True, "value": v, "why": "analytical"}

    def l2(spec: dict[str, Any]) -> dict[str, Any]:
        p = sram.validate(spec)
        m = sram.NmcModel(p)
        m.mem = list(range(p["rows"]))
        m.op("ADD", 1, 1, 0)
        return {"ok": m.mem[0] == sram.NmcModel(p)._pack([2 * x for x in m._lanes(1)]),
                "why": "golden model self-check"}

    def l3(spec: dict[str, Any]) -> dict[str, Any]:
        return {"skip": True, "why": "digital design: no circuit-level model needed"}

    def l4(spec: dict[str, Any]) -> dict[str, Any]:
        if not _eda.have("iverilog", "vvp"):
            return {"skip": True, "why": "iverilog not installed"}
        r = sram.simulate(spec, n_ops=40)
        return {"ok": r["bit_exact"], "why": f"{r['checked']} commands bit-exact"
                if r["bit_exact"] else f"mismatch {r['first_mismatch']}"}

    def l5(spec: dict[str, Any]) -> dict[str, Any]:
        try:
            syn = sram.synthesize(spec)
        except _eda.EdaError as e:
            return {"skip": True, "why": str(e)}
        if objective == "area":
            return {"ok": True, "value": syn["cell_area_um2"], "why": "yosys cell area"}
        return {"ok": True, "why": f"{syn['cells']} cells"}

    def l6(spec: dict[str, Any]) -> dict[str, Any]:
        return {"skip": True, "why": "place & route runs in the Tiny Tapeout GitHub Actions "
                                     "flow, not locally"}

    def l7(spec: dict[str, Any]) -> dict[str, Any]:
        return {"skip": True, "why": "no hardware measurement recorded for this design"}

    return [Level("analytical", l1, 0.01), Level("behavioral", l2, 0.01),
            Level("circuit_spice", l3, 0.0), Level("rtl_sim", l4, 2.0),
            Level("synthesis", l5, 10.0), Level("place_route", l6, 0.0),
            Level("measured", l7, 0.0)]
