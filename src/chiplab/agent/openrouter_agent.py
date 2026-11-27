"""Built-in OpenRouter AI agent with ChipLab simulation tool-calling."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/v1/chat/completions"
LMSTUDIO_MODELS_URL = "http://127.0.0.1:1234/v1/models"
LMSTUDIO_CHAT_URL = "http://127.0.0.1:1234/v1/chat/completions"

PROVIDER_CHAT_URLS: dict[str, str] = {
    "openrouter": OPENROUTER_CHAT_URL,
    "ollama": OLLAMA_CHAT_URL,
    "lmstudio": LMSTUDIO_CHAT_URL,
}

RETRYABLE_HTTP_STATUSES = (404, 429, 502, 503)

BEST_FREE_OPENROUTER_MODELS: list[dict[str, Any]] = [
    {
        "id": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "name": "NVIDIA Nemotron 3 Ultra 550B (Free)",
        "context_length": 1_000_000,
        "description": "1M context window, strong reasoning and tool-calling.",
        "is_free": True,
    },
    {
        "id": "openrouter/free",
        "name": "OpenRouter Free Auto-Router",
        "context_length": 200_000,
        "description": "Automatic router across available free OpenRouter models.",
        "is_free": True,
    },
    {
        "id": "cohere/north-mini-code:free",
        "name": "Cohere North Mini Code (Free)",
        "context_length": 128_000,
        "description": "Fast coding and structured tool-calling free model.",
        "is_free": True,
    },
    {
        "id": "thinkingmachines/inkling:free",
        "name": "Thinking Machines Inkling (Free)",
        "context_length": 128_000,
        "description": "Compact reasoning free model.",
        "is_free": True,
    },
]

DEFAULT_OPENROUTER_MODEL: str = BEST_FREE_OPENROUTER_MODELS[0]["id"]

SYSTEM_PROMPT = (
    "You are ChipLab's built-in semiconductor, circuit, and experimental device "
    "engineering assistant. Use the provided ChipLab simulation tools to run real "
    "physics, circuit, process-node scaling, and GPU status queries whenever the user "
    "asks for quantitative results or comparisons. Report exact numbers and units "
    "from tool outputs."
)


def list_free_openrouter_models(
    api_key: str | None = None,
    fetch_live: bool = True,
    timeout_s: float = 5.0,
) -> list[dict[str, Any]]:
    """Return ranked free OpenRouter models, optionally querying the live API.

    Falls back gracefully to ``BEST_FREE_OPENROUTER_MODELS`` if ``fetch_live`` is
    False or if the network request fails or times out.
    """
    fallback = [dict(m) for m in BEST_FREE_OPENROUTER_MODELS]
    if not fetch_live:
        return fallback

    key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    req = urllib.request.Request(OPENROUTER_MODELS_URL, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return fallback

    raw_models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        return fallback

    live_free_by_id: dict[str, dict[str, Any]] = {}
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "")
        if not mid:
            continue
        if mid.endswith(":free") or mid == "openrouter/free":
            live_free_by_id[mid] = {
                "id": mid,
                "name": str(item.get("name") or mid),
                "context_length": int(item.get("context_length") or 0),
                "description": str(item.get("description") or ""),
                "is_free": True,
            }

    if not live_free_by_id:
        return fallback

    # Keep curated ranked models first (merging live metadata if available), then remaining live models.
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for curated in fallback:
        cid = curated["id"]
        if cid in live_free_by_id:
            merged = dict(curated)
            merged["context_length"] = live_free_by_id[cid]["context_length"] or curated["context_length"]
            ranked.append(merged)
            seen.add(cid)
        else:
            ranked.append(curated)
            seen.add(cid)

    for mid in sorted(live_free_by_id):
        if mid not in seen:
            ranked.append(live_free_by_id[mid])
    return ranked


def list_local_llm_models(timeout_s: float = 1.5) -> dict[str, Any]:
    """Probe local Ollama and LM Studio servers for available models.

    Uses a short timeout (default 1.5s) and returns empty lists gracefully if
    either local server is not running.
    """
    ollama_models: list[dict[str, Any]] = []
    lmstudio_models: list[dict[str, Any]] = []

    req_ollama = urllib.request.Request(
        OLLAMA_TAGS_URL,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req_ollama, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        raw_ollama = payload.get("models") if isinstance(payload, dict) else None
        if isinstance(raw_ollama, list):
            for item in raw_ollama:
                if not isinstance(item, dict):
                    continue
                mid = str(item.get("name") or item.get("model") or "").strip()
                if not mid:
                    continue
                ollama_models.append(
                    {
                        "id": mid,
                        "name": mid,
                        "provider": "ollama",
                        "size": item.get("size"),
                    }
                )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass

    req_lmstudio = urllib.request.Request(
        LMSTUDIO_MODELS_URL,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req_lmstudio, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        raw_lms = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(raw_lms, list):
            for item in raw_lms:
                if not isinstance(item, dict):
                    continue
                mid = str(item.get("id") or "").strip()
                if not mid:
                    continue
                lmstudio_models.append(
                    {
                        "id": mid,
                        "name": str(item.get("name") or mid),
                        "provider": "lmstudio",
                    }
                )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass

    return {
        "ollama": ollama_models,
        "lmstudio": lmstudio_models,
    }


def get_agent_tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI-compatible function/tool schemas for ChipLab simulation tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "chiplab_custom_circuit_simulate",
                "description": (
                    "Build and simulate a custom mixed-technology circuit at any nm process "
                    "node (180 nm down to 1 nm). Provide one of template, components, or spice_netlist."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "circuit_name": {"type": "string", "default": "custom_circuit"},
                        "node_nm": {"type": "number", "default": 28.0},
                        "corner": {
                            "type": "string",
                            "enum": ["TT", "FF", "SS", "FS", "SF"],
                            "default": "TT",
                        },
                        "temperature_k": {"type": "number", "default": 300.0},
                        "template": {
                            "type": "string",
                            "enum": [
                                "cmos_inverter",
                                "ring_oscillator",
                                "sram_6t",
                                "1t1r",
                                "crossbar_column",
                            ],
                        },
                        "components": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                        "spice_netlist": {"type": "string"},
                        "analysis": {
                            "type": "string",
                            "enum": ["transient", "dc", "ac"],
                            "default": "transient",
                        },
                        "t_stop_s": {"type": "number", "default": 1e-9},
                        "dt_s": {"type": "number", "default": 2e-11},
                        "compare_nodes_nm": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "monte_carlo_runs": {"type": "integer", "default": 0},
                        "device": {"type": "string", "default": "auto"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chiplab_scale_technology_node",
                "description": (
                    "Compare or sweep CMOS or experimental device metrics across nm process nodes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "technology": {"type": "string", "default": "cmos"},
                        "nodes_nm": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "corner": {
                            "type": "string",
                            "enum": ["TT", "FF", "SS", "FS", "SF"],
                            "default": "TT",
                        },
                        "temperature_k": {"type": "number", "default": 300.0},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chiplab_simulate_experimental_tech",
                "description": (
                    "Simulate an experimental semiconductor/memory technology (e.g., memristor, "
                    "pcm, fefet, mram, ftj, photonic, ncfet, tfet, tmd_fet, ferrofluid, spintronic)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "technology": {"type": "string"},
                        "params": {"type": "object"},
                        "pulse_schedule": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                        "device": {"type": "string", "default": "auto"},
                        "node_nm": {"type": "number"},
                        "corner": {
                            "type": "string",
                            "enum": ["TT", "FF", "SS", "FS", "SF"],
                            "default": "TT",
                        },
                    },
                    "required": ["technology"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chiplab_compare_technologies",
                "description": (
                    "Compare multiple experimental technologies across energy/bit, latency, "
                    "endurance, retention, area density, and thermal sensitivity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "technologies": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "workload": {"type": "object"},
                    },
                    "required": ["technologies"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chiplab_custom_device_simulate",
                "description": (
                    "Build and simulate a custom PhysicalDevice from a device_spec dict "
                    "with name, description, regions, and electrodes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_spec": {"type": "object"},
                        "write_voltage_v": {"type": "number", "default": 1.0},
                        "read_freq_hz": {"type": "number", "default": 1e6},
                        "device": {"type": "string", "default": "auto"},
                    },
                    "required": ["device_spec"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chiplab_gpu_status",
                "description": (
                    "Return GPU availability, backend (CUDA/CuPy/Torch/Numba/CPU), and device memory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chipro_rtl_lint",
                "description": (
                    "Lint Verilog/SystemVerilog RTL using Verilator (if available) and a fast "
                    "Python structural linter checking module balance, undriven signals, blocking "
                    "assignments in sequential blocks, and potential combinational latches."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rtl_code": {"type": "string"},
                        "top_module": {"type": "string", "default": "top"},
                    },
                    "required": ["rtl_code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chipro_run_pnr",
                "description": (
                    "Run or analytically estimate Place-and-Route PPA (gate count, critical path, "
                    "WNS/TNS, core area, dynamic/leakage power) scaled by process node."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rtl_code": {"type": "string"},
                        "top_module": {"type": "string", "default": "top"},
                        "pdk": {"type": "string", "default": "sky130"},
                        "node_nm": {"type": "number", "default": 130.0},
                    },
                    "required": ["rtl_code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chipro_tweak_constraints",
                "description": (
                    "Update SDC clock period and input/output delays for a target frequency in MHz."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sdc_text": {"type": "string"},
                        "target_freq_mhz": {"type": "number"},
                        "input_delay_ns": {"type": "number", "default": 0.5},
                        "output_delay_ns": {"type": "number", "default": 0.5},
                    },
                    "required": ["sdc_text", "target_freq_mhz"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "chipro_open_klayout",
                "description": (
                    "Check KLayout and GDSII file availability and safely launch KLayout if a GUI display is present."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gds_path": {"type": "string"},
                    },
                    "required": ["gds_path"],
                },
            },
        },
    ]


def _get_tool_dispatch() -> dict[str, Callable[..., Any]]:
    from chiplab.agent.eda_frameworks import (
        chipro_open_klayout,
        chipro_rtl_lint,
        chipro_run_pnr,
        chipro_tweak_constraints,
    )
    from chiplab.core.gpu_benchmark import get_gpu_status
    from chiplab.mcp_experimental_tools import (
        chiplab_2d_beyond_cmos_simulate,
        chiplab_chia_codesign_loop,
        chiplab_chipro_eda_flow,
        chiplab_compare_technologies,
        chiplab_custom_circuit_simulate,
        chiplab_custom_device_simulate,
        chiplab_mage_rtl_pipeline,
        chiplab_quantum_device_simulate,
        chiplab_scale_technology_node,
        chiplab_simulate_experimental_tech,
        chiplab_spice_netlist_run,
    )

    return {
        "chiplab_custom_circuit_simulate": chiplab_custom_circuit_simulate,
        "chiplab_scale_technology_node": chiplab_scale_technology_node,
        "chiplab_simulate_experimental_tech": chiplab_simulate_experimental_tech,
        "chiplab_compare_technologies": chiplab_compare_technologies,
        "chiplab_custom_device_simulate": chiplab_custom_device_simulate,
        "chiplab_2d_beyond_cmos_simulate": chiplab_2d_beyond_cmos_simulate,
        "chiplab_quantum_device_simulate": chiplab_quantum_device_simulate,
        "chiplab_spice_netlist_run": chiplab_spice_netlist_run,
        "chiplab_chipro_eda_flow": chiplab_chipro_eda_flow,
        "chiplab_mage_rtl_pipeline": chiplab_mage_rtl_pipeline,
        "chiplab_chia_codesign_loop": chiplab_chia_codesign_loop,
        "chiplab_gpu_status": get_gpu_status,
        "chipro_rtl_lint": chipro_rtl_lint,
        "chipro_run_pnr": chipro_run_pnr,
        "chipro_tweak_constraints": chipro_tweak_constraints,
        "chipro_open_klayout": chipro_open_klayout,
    }


_TOOL_ALIASES: dict[str, str] = {
    "simulate_custom_circuit": "chiplab_custom_circuit_simulate",
    "chiplab_custom_circuit": "chiplab_custom_circuit_simulate",
    "simulate_custom_spice": "chiplab_custom_circuit_simulate",
    "chiplab_spice_netlist": "chiplab_custom_circuit_simulate",
    "scale_technology_node": "chiplab_scale_technology_node",
    "simulate_experimental_tech": "chiplab_simulate_experimental_tech",
    "compare_technologies": "chiplab_compare_technologies",
    "simulate_2d_beyond_cmos": "chiplab_2d_beyond_cmos_simulate",
    "simulate_quantum_device": "chiplab_quantum_device_simulate",
    "gpu_status": "chiplab_gpu_status",
    "rtl_lint": "chipro_rtl_lint",
    "run_pnr": "chipro_run_pnr",
}


def execute_agent_tool(
    name: str,
    arguments: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Execute a ChipLab agent tool by name and return a JSON-safe dict."""
    dispatch = _get_tool_dispatch()
    resolved_name = _TOOL_ALIASES.get(name, name)
    fn = dispatch.get(resolved_name)
    if fn is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}

    if arguments is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(arguments, str):
        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except ValueError as exc:
            return {"ok": False, "error": f"Invalid JSON arguments for {name}: {exc}"}
        if not isinstance(parsed, dict):
            return {"ok": False, "error": f"Arguments for {name} must be a JSON object"}
        kwargs = parsed
    elif isinstance(arguments, dict):
        kwargs = dict(arguments)
    else:
        return {"ok": False, "error": f"Unsupported arguments type for {name}"}

    if resolved_name == "chiplab_custom_circuit_simulate":
        if "spice_netlist" not in kwargs and "netlist" in kwargs:
            kwargs["spice_netlist"] = kwargs.pop("netlist")

    try:
        result = fn(**kwargs)
        safe = json.loads(json.dumps(result, default=str))
        if isinstance(safe, dict) and "ok" not in safe:
            safe["ok"] = not bool(safe.get("error"))
        return safe
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _run_builtin_chiplab_cycle(message: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Built-in zero-config local ChipLab EDA agent when no external LLM key/server is configured."""
    text = message.lower()
    tool_calls_executed: list[dict[str, Any]] = []
    sections: list[str] = []

    def call_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        res = execute_agent_tool(tool_name, args)
        tool_calls_executed.append(
            {
                "id": f"call_builtin_{len(tool_calls_executed) + 1}",
                "name": tool_name,
                "arguments": args,
                "result": res,
            }
        )
        return res

    gpu_info = call_tool("chiplab_gpu_status", {})
    gpu_line = (
        f"- **Compute Backend**: `{gpu_info.get('backend', 'cpu')}` on `{gpu_info.get('device', 'CPU')}` "
        f"(GPU Enabled: `{gpu_info.get('is_gpu', False)}`)"
        if gpu_info.get("ok")
        else "- **Compute Backend**: CPU fallback"
    )

    matched_circuit = False
    if any(k in text for k in ("ring oscillator", "ring_oscillator", "rosc", "inverter", "cmos", "sram", "6t", "1t1r", "crossbar", "spice", "netlist", "circuit")):
        matched_circuit = True
        node_nm = 28.0
        for token in ("180", "130", "65", "45", "28", "22", "16", "14", "10", "7", "5", "3", "2", "1.4", "1"):
            if f"{token}nm" in text or f"{token} nm" in text:
                node_nm = float(token)
                break
        corner = "TT"
        if "ff corner" in text or "fast corner" in text:
            corner = "FF"
        elif "ss corner" in text or "slow corner" in text:
            corner = "SS"
        elif "fs corner" in text:
            corner = "FS"
        elif "sf corner" in text:
            corner = "SF"

        template = "cmos_inverter"
        extra_args: dict[str, Any] = {}
        if "ring" in text or "rosc" in text:
            template = "ring_oscillator"
            extra_args["n_stages"] = 5
        elif "sram" in text or "6t" in text:
            template = "sram_6t"
        elif "1t1r" in text:
            template = "1t1r"
            extra_args["tech"] = "pcm" if "pcm" in text else ("mtj" if ("mtj" in text or "mram" in text) else "memristor")
        elif "crossbar" in text:
            template = "crossbar_column"
            extra_args["n_rows"] = 16 if "16" in text else 8
            extra_args["tech"] = "memristor"

        circ_res = call_tool(
            "chiplab_custom_circuit_simulate",
            {
                "template": template,
                "node_nm": node_nm,
                "corner": corner,
                "compare_nodes_nm": [180.0, 28.0, 7.0, 3.0, 2.0, 1.0],
                **extra_args,
            },
        )
        m = circ_res.get("metrics") or {}
        sections.append(
            f"### Custom Circuit & Multi-nm Scaling (`{template}` @ {node_nm:g} nm, {corner})\n"
            f"- **Supply Voltage ($V_{{DD}}$)**: {m.get('vdd_v', '–')} V\n"
            f"- **Avg / Peak Power**: {m.get('avg_power_w', '–')} W / {m.get('peak_power_w', '–')} W\n"
            f"- **Switching Energy**: {m.get('total_energy_j', '–')} J\n"
            + (f"- **Oscillation Frequency**: {m.get('fosc_ghz', '–')} GHz\n" if "fosc_ghz" in m else "")
            + (f"- **Propagation Delay**: {m.get('delay_ps', '–')} ps\n" if "delay_ps" in m else "")
        )

    matched_2d = any(k in text for k in ("2d", "tmd", "mose2", "wse2", "ncfet", "tfet", "beyond cmos", "tunneling", "graphene"))
    if matched_2d:
        tech_2d = "tmd_fet" if ("tmd" in text or "mose2" in text or "wse2" in text) else ("ncfet" if "ncfet" in text else ("tfet" if "tfet" in text else "graphene_pn"))
        res_2d = call_tool("chiplab_2d_beyond_cmos_simulate", {"technology": tech_2d, "node_nm": 2.0})
        m2d = res_2d.get("metrics") or {}
        sections.append(
            f"### 2D / Beyond-CMOS Device Evaluation (`{tech_2d}` @ 2.0 nm)\n"
            f"- **On-Current ($I_{{on}}$)**: {m2d.get('ion_uA_per_um', '–')} µA/µm\n"
            f"- **SSW / Subthreshold Swing**: {m2d.get('ssw_mV_per_dec', '–')} mV/dec\n"
            f"- **Intrinsic Delay ($\\tau$)**: {m2d.get('intrinsic_delay_ps', '–')} ps"
        )

    matched_quantum = any(k in text for k in ("quantum", "transmon", "fluxonium", "majorana", "sfq", "cryogenic", "qubit"))
    if matched_quantum:
        q_tech = "transmon" if "transmon" in text else ("fluxonium" if "fluxonium" in text else ("majorana" if "majorana" in text else ("sfq" if "sfq" in text else "adiabatic_qc")))
        res_q = call_tool("chiplab_quantum_device_simulate", {"technology": q_tech})
        mq = res_q.get("metrics") or res_q
        sections.append(
            f"### Quantum & Cryogenic Device Simulation (`{q_tech}`)\n"
            f"- **Transition Frequency**: {mq.get('freq_ghz', '–')} GHz\n"
            f"- **Anharmonicity**: {mq.get('anharmonicity_mhz', '–')} MHz\n"
            f"- **Quality Factor ($Q$)**: {mq.get('quality_factor', '–')}"
        )

    matched_mem = any(k in text for k in ("memristor", "reram", "pcm", "mram", "mtj", "fefet", "ferroelectric", "ftj", "spintronic", "photonic", "optical", "memory", "experimental"))
    if matched_mem and not matched_circuit:
        mem_map = [
            ("pcm", "pcm"),
            ("mram", "mram"),
            ("mtj", "mram"),
            ("fefet", "fefet"),
            ("ferroelectric", "fefet"),
            ("ftj", "fefet"),
            ("photonic", "photonic"),
            ("optical", "photonic"),
            ("spintronic", "spintronic"),
            ("memristor", "memristor"),
            ("reram", "memristor"),
        ]
        chosen = "memristor"
        for kw, mapped in mem_map:
            if kw in text:
                chosen = mapped
                break
        res_m = call_tool("chiplab_simulate_experimental_tech", {"technology": chosen, "node_nm": 7.0})
        mm = res_m.get("metrics") or {}
        sections.append(
            f"### Emerging Memory & Device Physics (`{chosen}` @ 7.0 nm)\n"
            f"- **Write Energy**: {mm.get('energy_per_bit_j', '–')} J/bit\n"
            f"- **Read Latency**: {mm.get('latency_ns', '–')} ns\n"
            f"- **Endurance**: {mm.get('endurance_cycles', '–')} cycles\n"
            f"- **Retention**: {mm.get('retention_s', '–')} s"
        )

    matched_node = any(k in text for k in ("node", "scaling", "nm", "cmos", "process", "pdk", "ladder"))
    if matched_node and not matched_circuit:
        res_node = call_tool(
            "chiplab_scale_technology_node",
            {"technology": "cmos", "nodes_nm": [180.0, 65.0, 28.0, 14.0, 7.0, 5.0, 3.0, 2.0, 1.0]},
        )
        sections.append(
            f"### Multi-nm CMOS Process Node Scaling Ladder (180 nm → 1.0 nm)\n"
            f"- Evaluated **{len(res_node.get('nodes', []))} technology nodes** across TT corner with GPU acceleration."
        )

    matched_rtl = any(k in text for k in ("rtl", "verilog", "lint", "pnr", "place", "route", "chipro", "mage", "chia", "hdl"))
    if matched_rtl:
        sample_rtl = "module top(input clk, input rst_n, input [7:0] d, output reg [7:0] q); always @(posedge clk or negedge rst_n) if (!rst_n) q <= 8'h0; else q <= d; endmodule"
        res_lint = call_tool("chipro_rtl_lint", {"rtl_code": sample_rtl, "top_module": "top"})
        res_pnr = call_tool("chipro_run_pnr", {"rtl_code": sample_rtl, "top_module": "top", "node_nm": 7.0})
        sections.append(
            f"### RTL Co-Design & Hardware Agent Flow (Chipro / MAGE / CHIA)\n"
            f"- **RTL Structural Lint**: {res_lint.get('summary', 'PASS')}\n"
            f"- **Synthesis & PnR @ 7 nm**: {res_pnr.get('metrics', res_pnr.get('summary', 'Completed'))}"
        )

    if not sections:
        res_cmp = call_tool(
            "chiplab_compare_technologies",
            {"technologies": ["dram", "sram", "reram", "fefet", "mram", "pcm", "optical"]},
        )
        sections.append(
            "### ChipLab Architecture & Technology Comparison\n"
            "Compared 7 memory technologies (DRAM, SRAM, ReRAM, FeFET, MRAM, PCM, Optical) across bandwidth, energy/bit, latency, and endurance:\n\n"
            f"```json\n{json.dumps(res_cmp, indent=2)[:1200]}\n```"
        )

    reply = (
        f"### ChipLab Local Engineering Agent Report\n\n"
        f"{gpu_line}\n\n"
        + "\n\n".join(sections)
        + "\n\n> [!NOTE]\n> Running locally via ChipLab's built-in zero-config EDA tool engine. To use a hosted 1M-context free model (NVIDIA Nemotron 3 Ultra 550B on OpenRouter), enter an `OPENROUTER_API_KEY` in the API Key box above."
    )

    return {
        "ok": True,
        "reply": reply,
        "model_used": "chiplab-builtin-local",
        "provider": "chiplab",
        "tool_calls_executed": tool_calls_executed,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class OpenRouterAgent:
    """Multi-turn tool-calling AI agent backed by OpenRouter, Ollama, or LM Studio."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_OPENROUTER_MODEL,
        fallback_models: list[str] | None = None,
        max_tool_rounds: int = 5,
        timeout_s: float = 60.0,
        provider: str = "auto",
    ) -> None:
        normalized_provider = (provider or "auto").strip().lower()
        if normalized_provider == "auto":
            normalized_provider = "openrouter" if api_key else "chiplab"
        if normalized_provider not in (*PROVIDER_CHAT_URLS, "chiplab"):
            raise ValueError(
                f"Unsupported provider '{provider}'. Expected one of: auto, {', '.join(PROVIDER_CHAT_URLS)}, chiplab"
            )
        self.provider: str = normalized_provider
        self.api_key: str | None = (
            api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        )
        self.model: str = model or DEFAULT_OPENROUTER_MODEL
        if fallback_models is None:
            if self.provider == "openrouter":
                self.fallback_models = [
                    m["id"] for m in BEST_FREE_OPENROUTER_MODELS if m["id"] != self.model
                ]
            else:
                self.fallback_models = []
        else:
            self.fallback_models = [m for m in fallback_models if m != self.model]
        self.max_tool_rounds = max(1, int(max_tool_rounds))
        self.timeout_s = float(timeout_s)

    def _candidate_models(self, override_model: str | None) -> list[str]:
        primary = override_model or self.model
        candidates = [primary]
        for fb in self.fallback_models:
            if fb not in candidates:
                candidates.append(fb)
        return candidates

    def _post_chat_completion(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        chat_url = PROVIDER_CHAT_URLS[self.provider]
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.provider == "openrouter":
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["HTTP-Referer"] = "https://chiplab.local"
            headers["X-Title"] = "ChipLab"
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            chat_url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _run_with_model(
        self,
        model_id: str,
        initial_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        messages = [dict(m) for m in initial_messages]
        tool_calls_executed: list[dict[str, Any]] = []
        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for _round in range(self.max_tool_rounds):
            resp = self._post_chat_completion(model_id, messages, tools=tools)
            usage = resp.get("usage") or {}
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[k] += int(usage.get(k) or 0)

            choices = resp.get("choices") or []
            if not choices:
                return {
                    "ok": False,
                    "error": f"Model {model_id} returned no choices",
                    "model": model_id,
                }

            assistant_msg = choices[0].get("message") or {}
            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                return {
                    "ok": True,
                    "reply": str(assistant_msg.get("content") or ""),
                    "model_used": str(resp.get("model") or model_id),
                    "tool_calls_executed": tool_calls_executed,
                    "usage": total_usage,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content"),
                    "tool_calls": tool_calls,
                }
            )
            for tc in tool_calls:
                fn_info = tc.get("function") or {}
                tc_name = str(fn_info.get("name") or "")
                tc_raw_args = fn_info.get("arguments")
                tc_id = str(tc.get("id") or f"call_{len(tool_calls_executed)}")
                result = execute_agent_tool(tc_name, tc_raw_args)
                tool_calls_executed.append(
                    {
                        "id": tc_id,
                        "name": tc_name,
                        "arguments": tc_raw_args,
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": tc_name,
                        "content": json.dumps(result, default=str),
                    }
                )

        # Final synthesis turn without tools after hitting max_tool_rounds.
        resp = self._post_chat_completion(model_id, messages, tools=None)
        usage = resp.get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total_usage[k] += int(usage.get(k) or 0)
        choices = resp.get("choices") or []
        reply = ""
        if choices:
            reply = str((choices[0].get("message") or {}).get("content") or "")
        return {
            "ok": True,
            "reply": reply,
            "model_used": str(resp.get("model") or model_id),
            "tool_calls_executed": tool_calls_executed,
            "usage": total_usage,
        }

    def run(self, prompt: str, model: str | None = None) -> dict[str, Any]:
        """Alias for chat(message=prompt, model=model)."""
        return self.chat(message=prompt, model=model)

    def chat(
        self,
        message: str,
        history: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Send a user message to the configured LLM provider, executing ChipLab tools as needed."""
        candidates = self._candidate_models(model)
        if self.provider == "openrouter" and not self.api_key:
            local = list_local_llm_models()
            if local.get("ollama"):
                first_ollama = local["ollama"][0]["id"]
                self.provider = "ollama"
                self.model = first_ollama
                candidates = [first_ollama]
            elif local.get("lmstudio"):
                first_lms = local["lmstudio"][0]["id"]
                self.provider = "lmstudio"
                self.model = first_lms
                candidates = [first_lms]
            else:
                return _run_builtin_chiplab_cycle(message, history)
        elif self.provider == "chiplab":
            return _run_builtin_chiplab_cycle(message, history)

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for item in history:
                if isinstance(item, dict) and "role" in item:
                    messages.append(dict(item))
        messages.append({"role": "user", "content": message})

        tools = get_agent_tool_schemas()
        errors: list[str] = []
        provider_label = "OpenRouter" if self.provider == "openrouter" else self.provider

        for candidate in candidates:
            try:
                result = self._run_with_model(candidate, messages, tools)
                result.setdefault("provider", self.provider)
                return result
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_HTTP_STATUSES:
                    errors.append(f"{candidate}: HTTP {exc.code}")
                    continue
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                return {
                    "ok": False,
                    "error": f"{provider_label} HTTP {exc.code} on {candidate}: {detail}".strip(),
                    "model": candidate,
                    "provider": self.provider,
                }
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
                continue

        return {
            "ok": False,
            "error": f"All {provider_label} models failed: " + "; ".join(errors),
            "model": candidates[0],
            "provider": self.provider,
            "tried_models": candidates,
        }
