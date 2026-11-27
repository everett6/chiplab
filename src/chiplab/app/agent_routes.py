"""Web routes for ChipLab's built-in OpenRouter AI agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from chiplab.agent.openrouter_agent import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterAgent,
    execute_agent_tool,
    get_agent_tool_schemas,
    list_free_openrouter_models,
    list_local_llm_models,
)

STATIC = Path(__file__).parent / "static"


def routes(lab_wrap: Callable[[Callable[[dict[str, Any]], Any]], Any]) -> list[Route]:
    """Return Starlette routes for the OpenRouter AI agent UI and API."""

    async def page(_: Request) -> Response:
        html_path = STATIC / "agent.html"
        if html_path.is_file():
            return FileResponse(html_path)
        return JSONResponse(
            {
                "status": "agent_ui_not_built",
                "message": "agent.html is not present yet; use the /api/agent/* endpoints.",
                "endpoints": [
                    "/api/agent/models",
                    "/api/agent/tools",
                    "/api/agent/chat",
                ],
            }
        )

    async def models(request: Request) -> JSONResponse:
        fetch_live = request.query_params.get("live", "0").lower() in ("1", "true", "yes")
        free_models = list_free_openrouter_models(fetch_live=fetch_live)
        return JSONResponse(
            {
                "default_model": DEFAULT_OPENROUTER_MODEL,
                "free_models": free_models,
                "local_models": list_local_llm_models(),
                "has_api_key": bool(os.environ.get("OPENROUTER_API_KEY")),
            }
        )

    async def tools(_: Request) -> JSONResponse:
        return JSONResponse({"tools": get_agent_tool_schemas()})

    def chat_handler(data: dict[str, Any]) -> dict[str, Any]:
        raw_messages = data.get("messages")
        history = data.get("history")
        message = str(data.get("message") or "").strip()
        if not message and isinstance(raw_messages, list) and raw_messages:
            for item in reversed(raw_messages):
                if isinstance(item, dict) and str(item.get("role", "user")) == "user":
                    message = str(item.get("content") or "").strip()
                    break
            if not message and isinstance(raw_messages[-1], dict):
                message = str(raw_messages[-1].get("content") or "").strip()
            if history is None:
                history = [m for m in raw_messages[:-1] if isinstance(m, dict) and "role" in m]
        if not message:
            raise ValueError("message is required")
        if history is not None and not isinstance(history, list):
            raise ValueError("history must be a list of message objects")
        model = data.get("model")
        api_key = data.get("api_key")
        provider = str(data.get("provider") or "auto").strip().lower()
        agent = OpenRouterAgent(
            api_key=str(api_key) if api_key else None,
            model=str(model) if model else DEFAULT_OPENROUTER_MODEL,
            provider=provider,
        )
        return agent.chat(message=message, history=history, model=str(model) if model else None)

    def tool_handler(data: dict[str, Any]) -> dict[str, Any]:
        name = str(data.get("name") or data.get("tool") or "").strip()
        if not name:
            raise ValueError("tool name is required")
        args = data.get("arguments") if data.get("arguments") is not None else data.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        res = execute_agent_tool(name, args)
        if isinstance(res, dict):
            return {"ok": res.get("ok", not bool(res.get("error"))), "result": res, **res}
        return {"ok": True, "result": res}

    return [
        Route("/agent", page),
        Route("/api/agent/models", models),
        Route("/api/agent/tools", tools),
        Route("/api/agent/chat", lab_wrap(chat_handler), methods=["POST"]),
        Route("/api/agent/tool", lab_wrap(tool_handler), methods=["POST"]),
    ]
