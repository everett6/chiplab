"""ChipLab Live: a local web app that shows AI agents designing chips in real time.

It reads the database the MCP server writes (designs + activity log), so every agent that
uses ChipLab (Claude Code, Codex, Gemini CLI, ChatGPT) shows up here as it works.

    chiplab-app            # http://127.0.0.1:8770
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import webbrowser
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from chiplab.activity import ActivityLog
from chiplab.app import asic_routes, tools_api
from chiplab.cost import model as cost_model
from chiplab.cost import prices
from chiplab.engine.validator import validate_design
from chiplab.experiments.store import ExperimentStore

STATIC = Path(__file__).parent / "static"
ACTIVE_WINDOW_S = 300


def summarize(event: dict[str, Any]) -> str:
    """One-line human summary of a tool call for the activity feed."""
    if not event["ok"]:
        return (event.get("error") or "failed")[:160]
    r = event.get("result") if isinstance(event.get("result"), dict) else {}
    a = event.get("args") if isinstance(event.get("args"), dict) else {}
    tool = event["tool"]
    if tool == "chip_evaluate":
        perf = r.get("performance") or {}
        bits = [f"${(r.get('cost') or {}).get('unit_cost_usd', 0):,.0f}/unit"]
        if perf.get("throughput_tokens_per_sec"):
            bits.append(f"{perf['throughput_tokens_per_sec']:,.0f} tok/s")
        if perf.get("peak_compute_tops"):
            bits.append(f"{perf['peak_compute_tops']:,.0f} TOPS")
        pw = (r.get("power_thermal") or {}).get("total_power_watts")
        if pw:
            bits.append(f"{pw:,.0f} W")
        fails = [c["constraint"] for c in r.get("constraint_checks", []) if not c["pass"]]
        if fails:
            bits.append("fails " + ", ".join(fails))
        return " · ".join(bits)
    if tool == "chip_estimate_cost":
        rng = r.get("range_usd") or {}
        return f"${rng.get('optimistic', 0):,.0f} – ${rng.get('pessimistic', 0):,.0f} per unit"
    if tool in ("chip_create_design", "chip_design_from_template"):
        v = r.get("validation") or {}
        tail = "valid" if v.get("valid") else f"{len(v.get('errors', []))} errors"
        src = f"from {a['template']} · " if a.get("template") else ""
        return f"{src}{r.get('name', '')} v{r.get('version', '')} · {tail}"
    if tool == "chip_set_price":
        e = r.get("entry") or {}
        return f"{a.get('key')} = {e.get('value')} {e.get('unit', '')}"
    if tool == "chip_run_simulation":
        m = r.get("metrics") or {}
        return f"{r.get('fidelity', '')} · {m.get('estimated_latency_ms', 0):.2f} ms"
    if tool == "chip_price_lookup_plan":
        return f"{r.get('count', 0)} prices to look up"
    if tool == "run_digital_twin_search":
        top = (r.get("top") or [{}])[0]
        rob = (top.get("robustness") or {}).get("robustness_score")
        return (f"{r.get('evaluated', 0):,} designs + {r.get('mc_evaluations', 0):,} MC · "
                f"{r.get('designs_per_sec', 0):,.0f}/s · "
                f"front {r.get('pareto_size', 0)}" + (f" · best robustness {rob}" if rob else ""))
    if tool in ("calibrate_from_eda_report", "ingest_byo_pdk"):
        fs = ", ".join(f"{f['name']}={f['value']}" for f in r.get("factors", []))
        return f"{r.get('node', '')} · {fs or 'no factors'}"
    if tool == "export_pareto_handoff_bundle":
        return f"{len(r.get('files', []))} files · lint {', '.join(r.get('structural_lint', []))}"
    if tool == "design_onchip_memory_subsystem":
        m = r.get("metrics") or {}
        cfg = r.get("config") or {}
        bits = [f"{cfg.get('technology', '')} {cfg.get('capacity_kib', '')} KiB on "
                f"{(r.get('pdk') or {}).get('key', '')}", f"{m.get('area_mm2', 0):.3f} mm2",
                f"{r.get('freq_mhz', 0):,.0f} MHz", f"conflicts {m.get('conflict_rate', 0):.1%}"]
        if m.get("refresh_overhead"):
            bits.append(f"refresh {m['refresh_overhead']:.1%}")
        if r.get("rtl"):
            bits.append("RTL " + ("PASS" if all(v == ["PASS"] for v in r["rtl"]["lint"].values())
                                  else "lint issues"))
        return " · ".join(bits)
    if tool == "list_pdk_ladder":
        return f"{len(r.get('nodes', []))} PDK nodes"
    if tool == "chip_compare_designs":
        return f"compared {len(a.get('run_ids', []))} runs"
    return ""


def _light(event: dict[str, Any]) -> dict[str, Any]:
    out = {k: event.get(k) for k in ("id", "ts", "client", "tool", "ok", "duration_ms",
                                      "design_id", "run_id")}
    out["summary"] = summarize(event)
    return out


MAX_LAB_CYCLES = 50_000
MAX_LAB_TRACE = 400


class LabError(ValueError):
    pass


async def _json_body(request: Request) -> dict[str, Any]:
    """Lab endpoints only accept same-origin JSON (blocks cross-site form posts)."""
    origin = request.headers.get("origin")
    if origin and origin.split("://", 1)[-1] != request.headers.get("host", ""):
        raise LabError("cross-origin request refused")
    if "application/json" not in request.headers.get("content-type", ""):
        raise LabError("expected application/json")
    body = await request.body()
    if len(body) > 200_000:
        raise LabError("request too large")
    data = json.loads(body or b"{}")
    if not isinstance(data, dict):
        raise LabError("expected a JSON object")
    return data


def lab_memory(data: dict[str, Any]) -> dict[str, Any]:
    """Analytical model + animated cycle simulation (+ optional in-memory RTL)."""
    from chiplab.memory_gen.rtl_generator import generate
    from chiplab.models.onchip_memory import analyze, simulate
    from chiplab.schemas.digital_twin import MemWorkload, OnChipMemoryConfig

    cfg = OnChipMemoryConfig.model_validate(data.get("config") or {})
    wl = MemWorkload.model_validate(data.get("workload") or {})
    rep = analyze(cfg, wl)
    cycles = int(min(max(int(data.get("cycles", 5000)), 100), MAX_LAB_CYCLES))
    trace = int(min(max(int(data.get("trace_cycles", 200)), 0), MAX_LAB_TRACE))
    sim = simulate(cfg, wl, cycles=cycles, seed=int(data.get("seed", 1)),
                   freq_mhz=rep["freq_mhz"], trace_cycles=trace)
    out: dict[str, Any] = {"report": rep, "simulation": sim}
    if data.get("rtl"):
        try:
            g = generate(cfg, wl, int(data.get("seed", 1)))
            out["rtl"] = {"files": g["files"], "lint": g["lint"], "params": g["params"]}
        except ValueError as exc:
            out["rtl"] = {"error": str(exc)}
    return out


def lab_point(data: dict[str, Any]) -> dict[str, Any]:
    from chiplab.engine.digital_twin import NODE_MARGIN_PCT, NODE_NAMES, _node_index, \
        evaluate_point
    from chiplab.schemas.digital_twin import TwinMemory, TwinWorkload

    params = dict(data.get("params") or {})
    mem = TwinMemory.model_validate(data["memory"]) if data.get("memory") else None
    if not mem:
        for k in ("mem_tech", "mem_banks", "mem_interleave", "pim_lanes"):
            params.pop(k, None)
    wl = TwinWorkload.model_validate(data.get("workload") or {})
    m = evaluate_point(params, wl, memory=mem)
    node = NODE_NAMES[_node_index(params.get("node", "N7"))]
    return {"metrics": {k: float(v) for k, v in m.items()}, "node": node,
            "accuracy_margin_pct": NODE_MARGIN_PCT[node]}


def lab_design_sim(store: ExperimentStore, data: dict[str, Any]) -> dict[str, Any]:
    """Run an agent design through the analytical or discrete-event engine (not saved)."""
    from chiplab.engine import run_analytical_simulation, run_discrete_event_simulation
    from chiplab.schemas.common import FidelityLevel
    from chiplab.schemas.simulation import SimulationConfig

    design_id, spec = store.get_design(str(data.get("design_id", "")))
    de = data.get("fidelity") == "discrete_event"
    cfg = SimulationConfig(fidelity=FidelityLevel.DISCRETE_EVENT if de else
                           FidelityLevel.ANALYTICAL, seed=int(data.get("seed", 42)),
                           num_requests=int(min(max(int(data.get("num_requests", 32)), 1), 512)))
    fn = run_discrete_event_simulation if de else run_analytical_simulation
    wl = None
    if data.get("workload"):
        from chiplab.schemas.workload import WorkloadSpec
        wl = WorkloadSpec.model_validate(data["workload"])
    res = fn(spec, wl, cfg, design_id)
    out = res.model_dump(mode="json") if hasattr(res, "model_dump") else dict(res)
    nums = [v for v in (out.get("metrics") or {}).values() if isinstance(v, (int, float))]
    if nums and not any(nums):
        raise ValueError("This design has no compute or memory specs to simulate (all "
                         "metrics are zero); add them, or start from a template.")
    return out


STUDIO_WORKLOADS = {
    "llm8_decode": {"name": "llm8_decode", "type": "llm_decode", "model_params_b": 8,
                    "batch_size": 4},
    "llm70_decode": {"name": "llm70_decode", "type": "llm_decode", "model_params_b": 70,
                     "batch_size": 8},
    "llm8_prefill": {"name": "llm8_prefill", "type": "llm_prefill", "model_params_b": 8,
                     "batch_size": 1, "sequence_length": 2048},
    "moe_decode": {"name": "moe_decode", "type": "moe_llm_decode", "model_params_b": 47,
                   "batch_size": 8, "num_experts": 8, "active_experts": 2},
}


def studio_scene(store: ExperimentStore, design: str, workload: str = "llm8_decode") -> dict[str, Any]:
    """Scene for GPU Studio: geometry + power + thermal + simulated activity of one design."""
    from chiplab.schemas.workload import WorkloadSpec
    from chiplab.studio.scene import build_scene

    if workload not in STUDIO_WORKLOADS:
        raise ValueError(f"unknown workload {workload!r}; choose from {sorted(STUDIO_WORKLOADS)}")
    design_id, spec = store.get_design(design)
    if not spec.chiplets:
        raise ValueError("This design has no chiplets to show.")
    scene = build_scene(spec, design_id, WorkloadSpec.model_validate(STUDIO_WORKLOADS[workload]))
    scene["workloads"] = sorted(STUDIO_WORKLOADS)
    scene["workload"] = workload
    return scene


def _next_version(store: ExperimentStore, name: str, version: str) -> str:
    taken = {d["version"] for d in store.list_designs(name)}
    parts = version.split(".")
    while True:
        try:
            parts[-1] = str(int(parts[-1]) + 1)
        except ValueError:
            parts.append("1")
        cand = ".".join(parts)
        if cand not in taken:
            return cand


def studio_edit(store: ExperimentStore, data: dict[str, Any]) -> dict[str, Any]:
    """Apply a studio edit (placement, stack/memory/package/cooling parameters) to a design.

    Saves in place, or as a new version (parent_version set) when ``new_version`` is true.
    The edited design is validated first; nothing is saved if it is invalid."""
    from chiplab.engine.validator import validate_design
    from chiplab.studio.scene import apply_edit

    design_id, spec = store.get_design(str(data.get("design_id", "")))
    patch = data.get("patch") or {}
    if not isinstance(patch, dict):
        raise ValueError("patch must be an object")
    new = apply_edit(spec, patch)
    v = validate_design(new, True)
    if not v.valid:
        raise ValueError("Edit rejected: " + "; ".join(v.errors))
    if data.get("new_version"):
        new.parent_version = spec.version
        new.version = _next_version(store, spec.name, spec.version)
        design_id = store.save_design(new)
    else:
        store.save_design(new, design_id)
    return {"design_id": design_id, "version": new.version,
            **studio_scene(store, design_id, str(data.get("workload", "llm8_decode")))}


def lab_twin(store: ExperimentStore, data: dict[str, Any]) -> dict[str, Any]:
    from chiplab.engine.digital_twin import run_digital_twin
    from chiplab.schemas.digital_twin import DigitalTwinConfig, default_search_space

    memory = data.get("memory")
    cfg = DigitalTwinConfig(
        search_space=default_search_space(data.get("node_set", "tsmc"),
                                          memory=bool(memory)),
        memory=memory or None, seed=int(data.get("seed", 42)),
        population=int(min(max(int(data.get("population", 48)), 8), 128)),
        generations=int(min(max(int(data.get("generations", 20)), 1), 60)),
        mc_trials=int(min(max(int(data.get("mc_trials", 128)), 0), 512)),
        constraints=data.get("constraints") or {})
    summary, trace = run_digital_twin(cfg)
    dumped = summary.model_dump(mode="json")
    store.save_twin_run(dumped, trace)
    return {"run_id": dumped["run_id"], "evaluated": dumped["evaluated"],
            "designs_per_sec": dumped["designs_per_sec"]}


def lab_meta() -> dict[str, Any]:
    from chiplab.engine.digital_twin import DEFAULTS, NODE_MARGIN_PCT, NODE_NAMES
    from chiplab.library.pdk_library import ladder_table

    return {"ladder": ladder_table(), "twin_nodes": NODE_NAMES, "margins": NODE_MARGIN_PCT,
            "defaults": DEFAULTS}


def create_app(db_path: str) -> Starlette:
    log = ActivityLog(db_path)
    store = ExperimentStore(db_path)

    async def index(_: Request) -> FileResponse:
        return FileResponse(STATIC / "studio.html")

    async def live(_: Request) -> FileResponse:
        return FileResponse(STATIC / "index.html")

    async def studio_js(_: Request) -> FileResponse:
        return FileResponse(STATIC / "studio.js", media_type="text/javascript")

    async def toolviews_js(_: Request) -> FileResponse:
        return FileResponse(STATIC / "toolviews.js", media_type="text/javascript")

    async def layout3d_js(_: Request) -> FileResponse:
        return FileResponse(STATIC / "layout3d.js", media_type="text/javascript")

    async def scene(request: Request) -> JSONResponse:
        q = request.query_params
        try:
            data = await asyncio.to_thread(studio_scene, store, q.get("design", ""),
                                           q.get("workload", "llm8_decode"))
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": f"{exc}"[:600]}, status_code=400)
        return JSONResponse(json.loads(json.dumps(data, default=str)))

    async def events(request: Request) -> JSONResponse:
        after = int(request.query_params.get("after", 0))
        rows = log.since(after) if after else log.recent(150)
        return JSONResponse([_light(e) for e in rows])

    async def stream(request: Request) -> StreamingResponse:
        last = int(request.query_params.get("after", 0))

        async def gen():
            nonlocal last
            yield "retry: 2000\n\n"
            ticks = 0
            while not await request.is_disconnected():
                rows = log.since(last)
                for e in rows:
                    last = e["id"]
                    yield f"id: {e['id']}\nevent: activity\ndata: {json.dumps(_light(e))}\n\n"
                ticks += 1
                if ticks % 20 == 0:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.7)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    async def designs(_: Request) -> JSONResponse:
        evals: dict[str, dict[str, Any]] = {}
        for e in log.evaluations():
            evals.setdefault(e["design_id"], e)
        out = []
        for d in store.list_designs():
            ev = evals.get(d["design_id"])
            out.append({**d, "evaluated": bool(ev),
                        "summary": summarize(ev) if ev else ""})
        return JSONResponse(out)

    async def design(request: Request) -> JSONResponse:
        did = request.path_params["design_id"]
        try:
            design_id, spec = store.get_design(did)
        except (KeyError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        ev = log.latest_result(design_id, ("chip_evaluate",))
        validation = validate_design(spec)
        try:
            cost = cost_model.estimate_cost_range(
                spec, volume=int(spec.metadata.get("suggested_volume") or 100_000))
        except Exception as exc:  # cost needs reasonable inputs; show the error instead
            cost = {"error": str(exc)}
        history = store.get_design_history(spec.name)
        return JSONResponse({
            "design_id": design_id,
            "spec": spec.model_dump(mode="json", exclude_none=True),
            "validation": {"valid": validation.valid, "errors": validation.errors,
                           "warnings": validation.warnings},
            "cost": cost,
            "evaluation": ev["result"] if ev else None,
            "evaluated_at": ev["ts"] if ev else None,
            "evaluated_by": ev["client"] if ev else None,
            "history": [{"version": h.get("version"), "design_id": h.get("design_id"),
                         "created_at": h.get("created_at")} for h in history],
            "activity": [_light(e) for e in log.recent(400) if e["design_id"] == design_id][-30:],
        })

    async def pareto(_: Request) -> JSONResponse:
        seen: dict[str, dict[str, Any]] = {}
        for e in log.evaluations():
            r = e["result"] if isinstance(e["result"], dict) else {}
            if e["design_id"] in seen or not r.get("cost"):
                continue
            perf, pt = r.get("performance") or {}, r.get("power_thermal") or {}
            seen[e["design_id"]] = {
                "design_id": e["design_id"], "name": r.get("design"), "version": r.get("version"),
                "client": e["client"], "unit_cost_usd": r["cost"].get("unit_cost_usd"),
                "range_usd": r["cost"].get("range_usd"),
                "tokens_per_sec": perf.get("throughput_tokens_per_sec"),
                "peak_tops": perf.get("peak_compute_tops"),
                "power_w": pt.get("total_power_watts"), "ts": e["ts"]}
        return JSONResponse(list(seen.values()))

    async def twin_runs(_: Request) -> JSONResponse:
        out = []
        for r in store.list_twin_runs(100):
            summ = store.get_twin_run(r["run_id"])["summary"]
            out.append({**r, "evaluated": summ["evaluated"], "pareto_size": summ["pareto_size"],
                        "designs_per_sec": summ["designs_per_sec"],
                        "objectives": list(summ["top"][0]["objectives"]) if summ["top"] else []})
        return JSONResponse(out)

    async def twin_run(request: Request) -> JSONResponse:
        try:
            data = store.get_twin_run(request.path_params["run_id"], with_trace=True)
        except KeyError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        tr = data["trace"]
        front = set(tr.get("front", []))
        rows = tr["rows"]
        step = max(1, len(rows) // 4000)  # keep the scatter light
        points = [{"i": r["i"], "gen": r["gen"], "node": r["params"].get("node"),
                   "front": r["i"] in front, "feasible": r["violation"] == 0,
                   "m": r["metrics"], "p": r["params"]}
                  for k, r in enumerate(rows) if r["i"] in front or k % step == 0]
        return JSONResponse({"summary": data["summary"], "created_at": data["created_at"],
                             "points": points, "objectives": tr["config"]["objectives"],
                             "constraints": tr["config"]["constraints"]})

    async def status(_: Request) -> JSONResponse:
        recent = log.recent(500)
        now = time.time()
        active = sorted({e["client"] or "unknown" for e in recent
                         if now - e["ts"] < ACTIVE_WINDOW_S})
        entries = prices.list_entries()
        return JSONResponse({
            "db_path": os.path.abspath(db_path),
            "active_agents": active,
            "designs": len(store.list_designs()),
            "evaluations": len(log.evaluations()),
            "twin_runs": len(store.list_twin_runs(1000)),
            "prices": {"total": len(entries),
                       "stale": sum(1 for e in entries.values() if e["stale"]),
                       "overrides": sum(1 for e in entries.values()
                                        if e["origin"] == "override")},
        })

    def _lab(fn: Any) -> Any:
        async def handler(request: Request) -> JSONResponse:
            try:
                data = await _json_body(request)
                result = await asyncio.to_thread(fn, data)
            except LabError as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)
            except (ValueError, KeyError, TypeError) as exc:
                return JSONResponse({"error": f"{type(exc).__name__}: {exc}"[:600]},
                                    status_code=400)
            return JSONResponse(json.loads(json.dumps(result, default=str)))
        return handler

    async def meta(_: Request) -> JSONResponse:
        return JSONResponse(lab_meta())

    return Starlette(routes=[
        Route("/api/lab/meta", meta),
        Route("/api/lab/memory", _lab(lab_memory), methods=["POST"]),
        Route("/api/lab/point", _lab(lab_point), methods=["POST"]),
        Route("/api/lab/design_sim", _lab(lambda d: lab_design_sim(store, d)),
              methods=["POST"]),
        Route("/api/lab/twin", _lab(lambda d: lab_twin(store, d)), methods=["POST"]),
        Route("/", index),
        Route("/live", live),
        Route("/studio.js", studio_js),
        Route("/layout3d.js", layout3d_js),
        Route("/toolviews.js", toolviews_js),
        *tools_api.routes(store, db_path, studio_scene),
        *asic_routes.routes(_lab),
        Route("/api/studio/scene", scene),
        Route("/api/studio/edit", _lab(lambda d: studio_edit(store, d)), methods=["POST"]),
        Route("/api/events", events),
        Route("/api/stream", stream),
        Route("/api/designs", designs),
        Route("/api/designs/{design_id:path}", design),
        Route("/api/pareto", pareto),
        Route("/api/status", status),
        Route("/api/twin_runs", twin_runs),
        Route("/api/twin_runs/{run_id}", twin_run),
    ])


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    ap = argparse.ArgumentParser(prog="chiplab-app", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("CHIPLAB_DB_PATH", "chiplab_experiments.db"),
                    help="Database the MCP server writes (default: $CHIPLAB_DB_PATH or "
                         "./chiplab_experiments.db)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--open", action="store_true", help="Open the app in your browser")
    args = ap.parse_args(argv)
    url = f"http://{args.host}:{args.port}"
    print(f"ChipLab Live on {url}  (database: {os.path.abspath(args.db)})")
    if args.open:
        webbrowser.open(url)
    uvicorn.run(create_app(args.db), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
