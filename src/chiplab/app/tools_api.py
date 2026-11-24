"""HTTP API for the studio's external-tool views (Circuit, Board, Layout, CFD).

Every endpoint maps to a fixed operation in :mod:`chiplab.tools`; clients send template keys,
numbers, design ids and layout files, never commands or paths. Missing tools give a clear
JSON error (HTTP 409) instead of failing; ``/api/tools/status`` tells the UI what to hide.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from chiplab.tools import cfd, kicad, layout, registry, schematic, spice


def _same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin.split("://", 1)[-1] != request.headers.get("host", ""):
        raise PermissionError("cross-origin request refused")


def _ok(data: Any) -> JSONResponse:
    return JSONResponse(json.loads(json.dumps(data, default=str)))


def circuit_run(data: dict[str, Any]) -> dict[str, Any]:
    template = str(data.get("template", ""))
    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    p, circ = spice.build(template, params)
    out: dict[str, Any] = {"schematic": schematic.schematic(circ["devices"], spice.nets(circ))}
    if registry.get("ngspice").available:
        out.update(spice.simulate(template, p))
    else:
        out.update({"template": template, "params": p, "netlist": spice.netlist(template, p, circ),
                    "simulator": None, "missing": "ngspice"})
    return out


def routes(store: Any, db_path: str, scene_fn: Callable[..., dict[str, Any]]) -> list[Route]:
    layouts = Path(db_path).resolve().parent / "chiplab_layouts"

    def scene_of(data: dict[str, Any]) -> dict[str, Any]:
        return scene_fn(store, str(data.get("design", "")),
                        str(data.get("workload", "llm8_decode")))

    def wrap(fn: Callable[[dict[str, Any]], Any], method: str = "POST") -> Any:
        async def handler(request: Request) -> Response:
            try:
                _same_origin(request)
                if method == "POST":
                    if "application/json" not in request.headers.get("content-type", ""):
                        raise PermissionError("expected application/json")
                    body = await request.body()
                    if len(body) > 200_000:
                        raise ValueError("request too large")
                    data = json.loads(body or b"{}")
                    if not isinstance(data, dict):
                        raise ValueError("expected a JSON object")
                else:
                    data = dict(request.query_params)
                return _ok(await asyncio.to_thread(fn, data))
            except PermissionError as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)
            except registry.ToolError as exc:
                return JSONResponse({"error": str(exc), "missing_tool": True}, status_code=409)
            except subprocess.TimeoutExpired:
                return JSONResponse({"error": "tool timed out"}, status_code=504)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                return JSONResponse({"error": f"{exc}"[:600]}, status_code=400)
        return handler

    def board(data: dict[str, Any]) -> dict[str, Any]:
        sc = scene_of(data)
        if registry.get("kicad").available:
            return kicad.render(sc, png=bool(data.get("png", True)))
        return {"summary": kicad.summary(sc), "missing": "kicad"}

    async def board_file(request: Request) -> Response:
        q = request.query_params
        try:
            sc = await asyncio.to_thread(scene_fn, store, q.get("design", ""),
                                         q.get("workload", "llm8_decode"))
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": f"{exc}"[:600]}, status_code=400)
        name = "".join(ch if ch.isalnum() or ch in "-_." else "_"
                       for ch in sc["design"]["name"])[:60] or "board"
        return Response(kicad.kicad_pcb(sc), media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{name}.kicad_pcb"'})

    async def layout_upload(request: Request) -> Response:
        try:
            _same_origin(request)
            if not layout.available():
                raise registry.ToolError("klayout Python package is not installed")
            if "application/octet-stream" not in request.headers.get("content-type", ""):
                raise PermissionError("expected application/octet-stream")
            size = int(request.headers.get("content-length") or 0)
            if size > layout.MAX_BYTES:
                raise ValueError("layout file too large")
            body = await request.body()
            lid = layout.store_upload(layouts, request.headers.get("x-filename", ""), body)
            return _ok(await asyncio.to_thread(layout.info, layouts, lid))
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)
        except registry.ToolError as exc:
            return JSONResponse({"error": str(exc), "missing_tool": True}, status_code=409)
        except (ValueError, OSError, RuntimeError) as exc:
            return JSONResponse({"error": f"{exc}"[:600]}, status_code=400)

    def need_klayout() -> None:
        if not layout.available():
            raise registry.ToolError("klayout Python package is not installed")

    def layout_floorplan(data: dict[str, Any]) -> dict[str, Any]:
        need_klayout()
        lid = layout.floorplan_gds(layouts, scene_of(data))
        return layout.info(layouts, lid)

    def layout_render(data: dict[str, Any]) -> dict[str, Any]:
        need_klayout()
        layers = data.get("layers")
        if layers is not None and not (isinstance(layers, list)
                                       and all(isinstance(s, str) for s in layers)):
            raise ValueError("layers must be a list like ['1/0']")
        return layout.render(layouts, str(data.get("id", "")), layers,
                             int(data.get("width", 1200)), int(data.get("height", 800)))

    def cfd_view(data: dict[str, Any]) -> dict[str, Any]:
        sc = scene_of(data)
        out = cfd.preview(sc)
        if data.get("run"):
            out["result"] = cfd.run(sc)
        elif not registry.get("openfoam").available:
            out["missing"] = "openfoam"
        return out

    return [
        Route("/api/tools/status", wrap(lambda _: registry.status(), "GET")),
        Route("/api/tools/circuit/templates", wrap(lambda _: spice.templates(), "GET")),
        Route("/api/tools/circuit/run", wrap(circuit_run), methods=["POST"]),
        Route("/api/tools/board", wrap(board), methods=["POST"]),
        Route("/api/tools/board/kicad_pcb", board_file),
        Route("/api/tools/layout/upload", layout_upload, methods=["POST"]),
        Route("/api/tools/layout/floorplan", wrap(layout_floorplan), methods=["POST"]),
        Route("/api/tools/layout/render", wrap(layout_render), methods=["POST"]),
        Route("/api/tools/cfd", wrap(cfd_view), methods=["POST"]),
    ]
