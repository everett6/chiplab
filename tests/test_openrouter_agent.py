"""Tests for ChipLab's built-in OpenRouter AI agent and web API routes."""

from __future__ import annotations

import io
import json
import os
import tempfile
import urllib.error
from typing import Any

import pytest
from starlette.testclient import TestClient

from chiplab.agent import (
    BEST_FREE_OPENROUTER_MODELS,
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterAgent,
    execute_agent_tool,
    get_agent_tool_schemas,
    list_free_openrouter_models,
    list_local_llm_models,
)
from chiplab.agent.eda_frameworks import (
    chipro_open_klayout,
    chipro_rtl_lint,
    chipro_run_pnr,
    chipro_tweak_constraints,
)
from chiplab.app.server import create_app


class _MockHTTPResponse:
    """Minimal context-manager mock for urllib.request.urlopen."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _MockHTTPResponse:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def test_list_free_openrouter_models_offline_fallback() -> None:
    models = list_free_openrouter_models(fetch_live=False)
    assert len(models) == len(BEST_FREE_OPENROUTER_MODELS)
    ids = [m["id"] for m in models]
    assert ids[0] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert "openrouter/free" in ids
    assert "cohere/north-mini-code:free" in ids
    assert "thinkingmachines/inkling:free" in ids


def test_get_agent_tool_schemas_and_execute_tools() -> None:
    schemas = get_agent_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    assert {
        "chiplab_custom_circuit_simulate",
        "chiplab_scale_technology_node",
        "chiplab_simulate_experimental_tech",
        "chiplab_compare_technologies",
        "chiplab_custom_device_simulate",
        "chiplab_gpu_status",
    } <= names

    circ_res = execute_agent_tool(
        "chiplab_custom_circuit_simulate",
        {"template": "cmos_inverter", "node_nm": 28.0, "corner": "TT"},
    )
    assert "error" not in circ_res
    assert circ_res.get("node_nm") == 28.0 or "metrics" in circ_res

    scale_res = execute_agent_tool(
        "chiplab_scale_technology_node",
        json.dumps({"technology": "cmos", "nodes_nm": [28, 7, 3]}),
    )
    assert "error" not in scale_res
    assert isinstance(scale_res, dict)

    gpu_res = execute_agent_tool("chiplab_gpu_status", {})
    assert "gpu_available" in gpu_res
    assert "backend" in gpu_res

    unknown_res = execute_agent_tool("nonexistent_tool", {})
    assert unknown_res["ok"] is False


def test_openrouter_agent_chat_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = OpenRouterAgent(api_key=None)
    res = agent.chat("Simulate a 7nm CMOS inverter")
    assert res["ok"] is True
    assert res["provider"] == "chiplab"
    assert len(res["tool_calls_executed"]) >= 1


def test_openrouter_agent_chat_tool_call_and_429_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_models: list[str] = []
    call_count = 0

    def fake_urlopen(req: Any, timeout: float = 60.0) -> _MockHTTPResponse:
        nonlocal call_count
        call_count += 1
        body = json.loads(req.data.decode("utf-8"))
        model_id = body["model"]
        seen_models.append(model_id)

        # First call on primary model returns HTTP 429 to trigger fallback.
        if call_count == 1:
            raise urllib.error.HTTPError(
                url=req.full_url,
                code=429,
                msg="Too Many Requests",
                hdrs=None,  # type: ignore[arg-type]
                fp=io.BytesIO(b'{"error":"rate limited"}'),
            )

        # Second call (on fallback model) requests a tool call.
        if call_count == 2:
            return _MockHTTPResponse(
                {
                    "model": model_id,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_scale_1",
                                        "type": "function",
                                        "function": {
                                            "name": "chiplab_scale_technology_node",
                                            "arguments": json.dumps(
                                                {"technology": "cmos", "nodes_nm": [28, 7]}
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }
            )

        # Third call returns final assistant answer using the tool result.
        assert any(m.get("role") == "tool" for m in body["messages"])
        return _MockHTTPResponse(
            {
                "model": model_id,
                "usage": {"prompt_tokens": 220, "completion_tokens": 40, "total_tokens": 260},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Scaled CMOS across 28nm and 7nm successfully.",
                        }
                    }
                ],
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    agent = OpenRouterAgent(api_key="test-key-123")
    res = agent.chat("Compare CMOS at 28nm and 7nm")

    assert res["ok"] is True
    assert res["model_used"] == "openrouter/free"
    assert seen_models == [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "openrouter/free",
        "openrouter/free",
    ]
    assert len(res["tool_calls_executed"]) == 1
    assert res["tool_calls_executed"][0]["name"] == "chiplab_scale_technology_node"
    assert "28nm and 7nm" in res["reply"]
    assert res["usage"]["total_tokens"] == 385


def test_agent_web_routes(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    fd, db_path = tempfile.mkstemp(suffix=".db", dir=tmp_path)
    os.close(fd)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    app = create_app(db_path=db_path)
    client = TestClient(app)

    r_models = client.get("/api/agent/models")
    assert r_models.status_code == 200
    d_models = r_models.json()
    assert d_models["default_model"] == DEFAULT_OPENROUTER_MODEL
    assert d_models["has_api_key"] is False
    assert len(d_models["free_models"]) >= 4

    r_tools = client.get("/api/agent/tools")
    assert r_tools.status_code == 200
    d_tools = r_tools.json()
    assert len(d_tools["tools"]) == 10
    assert "local_models" in d_models

    r_page = client.get("/agent")
    assert r_page.status_code == 200

    # Without API key, /api/agent/chat falls back to built-in local ChipLab tool agent.
    r_chat_no_key = client.post(
        "/api/agent/chat",
        json={"message": "Check GPU status"},
    )
    assert r_chat_no_key.status_code == 200
    d_chat_no_key = r_chat_no_key.json()
    assert d_chat_no_key["ok"] is True
    assert d_chat_no_key["provider"] == "chiplab"
    assert len(d_chat_no_key["tool_calls_executed"]) >= 1

    # With mocked urlopen and api_key in payload, /api/agent/chat executes cleanly.
    def fake_urlopen(req: Any, timeout: float = 60.0) -> _MockHTTPResponse:
        return _MockHTTPResponse(
            {
                "model": DEFAULT_OPENROUTER_MODEL,
                "usage": {"prompt_tokens": 50, "completion_tokens": 12, "total_tokens": 62},
                "choices": [{"message": {"role": "assistant", "content": "Hello from ChipLab!"}}],
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    r_chat = client.post(
        "/api/agent/chat",
        json={"message": "Hi", "api_key": "test-key"},
    )
    assert r_chat.status_code == 200
    d_chat = r_chat.json()
    assert d_chat["ok"] is True
    assert d_chat["reply"] == "Hello from ChipLab!"


def test_list_local_llm_models_and_providers(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    # Offline / unreachable servers return empty lists gracefully.
    def fail_urlopen(req: Any, timeout: float = 1.5) -> _MockHTTPResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)
    offline = list_local_llm_models(timeout_s=0.1)
    assert offline == {"ollama": [], "lmstudio": []}

    # Mocked Ollama and LM Studio responses.
    seen_urls: list[str] = []

    def fake_urlopen(req: Any, timeout: float = 1.5) -> _MockHTTPResponse:
        seen_urls.append(req.full_url)
        if req.full_url == "http://127.0.0.1:11434/api/tags":
            return _MockHTTPResponse({"models": [{"name": "qwen2.5-coder:7b", "size": 4200000}]})
        if req.full_url == "http://127.0.0.1:1234/v1/models":
            return _MockHTTPResponse({"data": [{"id": "deepseek-coder-v2"}]})
        if req.full_url in (
            "http://127.0.0.1:11434/v1/chat/completions",
            "http://127.0.0.1:1234/v1/chat/completions",
        ):
            body = json.loads(req.data.decode("utf-8"))
            return _MockHTTPResponse(
                {
                    "model": body["model"],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "choices": [{"message": {"role": "assistant", "content": f"ok:{body['model']}"}}],
                }
            )
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    local = list_local_llm_models()
    assert [m["id"] for m in local["ollama"]] == ["qwen2.5-coder:7b"]
    assert [m["id"] for m in local["lmstudio"]] == ["deepseek-coder-v2"]

    ollama_agent = OpenRouterAgent(model="qwen2.5-coder:7b", provider="ollama")
    res_ollama = ollama_agent.chat("Hello")
    assert res_ollama["ok"] is True
    assert res_ollama["reply"] == "ok:qwen2.5-coder:7b"
    assert "http://127.0.0.1:11434/v1/chat/completions" in seen_urls

    lms_agent = OpenRouterAgent(model="deepseek-coder-v2", provider="lmstudio")
    res_lms = lms_agent.chat("Hello")
    assert res_lms["ok"] is True
    assert res_lms["reply"] == "ok:deepseek-coder-v2"
    assert "http://127.0.0.1:1234/v1/chat/completions" in seen_urls

    fd, db_path = tempfile.mkstemp(suffix=".db", dir=tmp_path)
    os.close(fd)
    app = create_app(db_path=db_path)
    client = TestClient(app)
    r_chat = client.post(
        "/api/agent/chat",
        json={"message": "Ping", "provider": "ollama", "model": "qwen2.5-coder:7b"},
    )
    assert r_chat.status_code == 200
    assert r_chat.json()["ok"] is True


def test_chipro_eda_frameworks(tmp_path: Any) -> None:
    clean_rtl = """
module top(input logic clk, input logic d, output logic q);
  always @(posedge clk) begin
    q <= d;
  end
endmodule
"""
    lint_clean = chipro_rtl_lint(clean_rtl, top_module="top")
    assert lint_clean["error_count"] == 0

    buggy_rtl = """
module top(input wire clk, input wire a, output reg y, output reg q);
  wire undriven_w;
  always @(posedge clk) begin
    q = a;
  end
  always @(*) begin
    if (a)
      y = 1'b1;
  end
endmodule
"""
    lint_buggy = chipro_rtl_lint(buggy_rtl, top_module="top")
    rules = {issue["rule"] for issue in lint_buggy["issues"]}
    assert "blocking_in_sequential" in rules
    assert "incomplete_if_latch" in rules
    assert "undriven_signal" in rules

    unbalanced_rtl = "module top(input wire a);\n"
    lint_unbal = chipro_rtl_lint(unbalanced_rtl, top_module="top")
    assert lint_unbal["ok"] is False
    assert lint_unbal["error_count"] >= 1

    pnr_130 = chipro_run_pnr(clean_rtl, top_module="top", pdk="sky130", node_nm=130.0)
    pnr_7 = chipro_run_pnr(clean_rtl, top_module="top", pdk="n7", node_nm=7.0)
    assert pnr_130["ok"] is True
    assert pnr_130["core_area_um2"] > pnr_7["core_area_um2"]
    assert pnr_130["critical_path_delay_ns"] > pnr_7["critical_path_delay_ns"]

    sdc_in = "create_clock -name clk -period 10.0 [get_ports clk]\nset_input_delay 1.0 -clock clk [all_inputs]\n"
    sdc_out = chipro_tweak_constraints(sdc_in, target_freq_mhz=250.0, input_delay_ns=0.4, output_delay_ns=0.6)
    assert sdc_out["ok"] is True
    assert sdc_out["period_ns"] == pytest.approx(4.0)
    assert "-period 4.0" in sdc_out["sdc_text"]
    assert "set_input_delay 0.4" in sdc_out["sdc_text"]
    assert "set_output_delay 0.6" in sdc_out["sdc_text"]

    gds_file = tmp_path / "chip.gds"
    gds_file.write_bytes(b"GDS2")
    kl_missing = chipro_open_klayout(str(tmp_path / "missing.gds"))
    assert kl_missing["gds_exists"] is False
    assert kl_missing["launched"] is False
    kl_exists = chipro_open_klayout(str(gds_file))
    assert kl_exists["gds_exists"] is True
    assert "klayout" in kl_exists["command"]
