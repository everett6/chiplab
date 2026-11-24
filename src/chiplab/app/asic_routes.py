"""Web routes for the ASIC workspaces (/asic): competition ASIC, memory studio, component
foundry. Thin wrappers over chiplab.asic_api; POSTs go through the app's same-origin JSON guard."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route

from chiplab import asic_api

STATIC = Path(__file__).parent / "static"


def _safe(fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
    def run(d: dict[str, Any]) -> Any:
        try:
            return fn(d)
        except RuntimeError as e:                    # tool/build failures -> client-visible error
            raise ValueError(str(e)[:500]) from e
    return run


def routes(lab: Callable[[Callable[[dict[str, Any]], Any]], Any]) -> list[Route]:
    async def page(_: Request) -> FileResponse:
        return FileResponse(STATIC / "asic.html")

    async def js(_: Request) -> FileResponse:
        return FileResponse(STATIC / "asic.js", media_type="text/javascript")

    async def overview(_: Request) -> JSONResponse:
        import asyncio
        arch, rep, lv = await asyncio.gather(asyncio.to_thread(asic_api.asic_architecture),
                                             asyncio.to_thread(asic_api.asic_report),
                                             asyncio.to_thread(asic_api.simulation_levels))
        return JSONResponse({"architecture": arch, **rep, "levels": lv})

    async def components(_: Request) -> JSONResponse:
        return JSONResponse(asic_api.component_list())

    async def tech_report(_: Request) -> PlainTextResponse:
        return PlainTextResponse(asic_api.asic_technical_report(), media_type="text/markdown")

    def post(path: str, fn: Callable[[dict[str, Any]], Any]) -> Route:
        return Route(path, lab(_safe(fn)), methods=["POST"])

    return [
        Route("/asic", page), Route("/asic.js", js),
        Route("/api/asic/overview", overview), Route("/api/asic/technical_report", tech_report),
        Route("/api/components", components),
        post("/api/asic/assemble", lambda d: asic_api.asic_assemble(str(d.get("source", "")))),
        post("/api/asic/simulate", lambda d: asic_api.asic_simulate(
            str(d.get("source", "")), d.get("lane_regs") or {}, int(d.get("cycles", 2000)),
            str(d.get("bench", "")), int(d.get("lane", 0)), int(d.get("clkdiv", 1)))),
        post("/api/asic/run", lambda d: asic_api.asic_run(str(d.get("kind", "")), int(d.get("seeds", 20)),
                                                          int(d.get("cycles", 3000)), str(d.get("backend", "rtl")))),
        post("/api/asic/export", lambda d: asic_api.asic_export(str(d.get("name", "jane_street_pem")))),
        post("/api/memory/compile", lambda d: asic_api.memory_compile(d.get("spec") or {}, bool(d.get("views", True)))),
        post("/api/memory/compare", lambda d: asic_api.memory_compare(d.get("specs") or [])),
        post("/api/memory/rtl", lambda d: asic_api.memory_rtl(
            d.get("spec") or {}, bool(d.get("run_tests", True)), int(d.get("n_ops", 300)),
            int(d.get("retention_cycles", 0)), int(d.get("refresh_interval", 0)), bool(d.get("refresh", True)))),
        post("/api/components/elaborate", lambda d: asic_api.architecture_elaborate(
            str(d.get("ref", "")), d.get("params") or {}, str(d.get("pdk", "sky130")), True)),
        post("/api/components/simulate", lambda d: asic_api.component_simulate(
            str(d.get("ref", "")), d.get("stimulus") or [], d.get("params") or {})),
        post("/api/components/create", lambda d: asic_api.component_create(d.get("definition") or {})),
        post("/api/components/verilog", lambda d: {"verilog": asic_api.architecture_verilog(
            str(d.get("ref", "")), d.get("params") or {})}),
    ]
