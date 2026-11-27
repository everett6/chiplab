"""Built-in AI agent integrations for ChipLab."""

from __future__ import annotations

from chiplab.agent.openrouter_agent import (
    BEST_FREE_OPENROUTER_MODELS,
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterAgent,
    execute_agent_tool,
    get_agent_tool_schemas,
    list_free_openrouter_models,
    list_local_llm_models,
)

__all__ = [
    "BEST_FREE_OPENROUTER_MODELS",
    "DEFAULT_OPENROUTER_MODEL",
    "OpenRouterAgent",
    "execute_agent_tool",
    "get_agent_tool_schemas",
    "list_free_openrouter_models",
    "list_local_llm_models",
]
