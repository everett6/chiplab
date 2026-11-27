"""Ready-made MCP prompts: step-by-step ChipLab workflows that name the exact tools and
arguments to call. Each returns the text of one user message.

tests/test_prompts.py runs every prompt's tool chain against a temporary database, so the
calls written here are known to work.
"""

from __future__ import annotations

import json


def _j(obj: object) -> str:
    return json.dumps(obj, separators=(", ", ": "))


def moe_decode_chiplet_balance(model_params_b: float = 47.0, num_experts: int = 8,
                               active_experts: int = 2, batch_size: int = 8,
                               max_unit_cost_usd: float = 8000.0) -> str:
    """Balance compute dies vs HBM stacks for MoE LLM decode at a unit-cost cap."""
    wl = {"type": "moe_llm_decode", "model_params_b": model_params_b,
          "num_experts": num_experts, "active_experts": active_experts,
          "batch_size": batch_size}
    return f"""Goal: find the cheapest chiplet mix that serves MoE decode ({model_params_b}B
params, {active_experts}/{num_experts} experts active, batch {batch_size}) under
${max_unit_cost_usd:,.0f} per unit.

1. chip_design_from_template(template="llm_accelerator_hbm3e", name="moe_balance",
   overrides={_j({"constraints": {"max_cost_usd": max_unit_cost_usd}})})
2. chip_evaluate(design_id_or_name="moe_balance", workload={_j(wl)})
   Note the bottleneck. MoE decode reads only the active experts' weights, so it is usually
   memory-bandwidth bound.
3. chip_sweep_parameters(design_id_or_name="moe_balance",
   parameters={_j({"chiplets[0].count": [1, 2], "chiplets[1].count": [4, 6, 8]})},
   workload={_j(wl)}, objective="throughput_tokens_per_sec")
4. For the best two points, re-create them with chip_design_from_template overrides
   ({{"chiplets": [{{"id": "compute", "count": N}}, {{"id": "hbm", "count": M}}]}}) and run
   chip_evaluate on each. Compare tokens/s per $1k, TOPS/W and the constraint checks.
5. Report the winner with its cost range and the assumptions that matter most
   (chip_get_results / chip_analyze_bottleneck on its run_id). Say which numbers are
   estimated or assumed."""


def riscv_soc_power_area_tradeoff(max_power_w: float = 5.0, max_area_mm2: float = 60.0,
                                  node_set: str = "tsmc") -> str:
    """Pareto search over a RISC-V SoC (cores, vector width, caches, Vdd/f, node)."""
    if node_set not in ("tsmc", "ladder"):
        raise ValueError("node_set must be 'tsmc' or 'ladder'")
    return f"""Goal: the Pareto front of performance vs power vs area for a RISC-V SoC under
{max_power_w} W and {max_area_mm2} mm2, and how robust the winners are.

1. list_pdk_ladder() if node_set is 'ladder', to see each node's accuracy margin.
2. run_digital_twin_search(node_set="{node_set}", strategy="nsga2", population=64,
   generations=20, mc_trials=128, top_k=5, seed=1,
   config={_j({"constraints": {"max_power_w": max_power_w, "max_area_mm2": max_area_mm2},
               "objectives": [{"name": "perf", "goal": "max"},
                              {"name": "power_w", "goal": "min"},
                              {"name": "area_mm2", "goal": "min"}]})})
   It runs thousands of designs locally and returns a compact summary.
3. get_digital_twin_run(run_id=<run_id>, include_rows=5) for the top designs, their slack
   yield, thermal-violation probability and the sensitivity ranking.
4. Explain the trade-off: which knobs move power vs area most (sensitivity), and which
   winner is most robust, not just fastest.
5. Optional: export_pareto_handoff_bundle(run_id=<run_id>, rank=1) for the chosen design."""


def hbm_accelerator_cost_per_token(model_params_b: float = 70.0, batch_size: int = 8,
                                   max_power_w: float = 1200.0) -> str:
    """Minimize $ per token/s for dense LLM decode on an HBM accelerator."""
    wl = {"type": "llm_decode", "model_params_b": model_params_b, "batch_size": batch_size}
    return f"""Goal: the HBM accelerator with the best tokens/s per $1k for dense decode of a
{model_params_b}B model at batch {batch_size}, within {max_power_w} W.

1. chip_quickstart() to see the templates, nodes, memory and package types.
2. chip_design_from_template(template="llm_accelerator_hbm3e", name="hbm_cpt",
   overrides={_j({"constraints": {"max_power_watts": max_power_w}})})
3. chip_evaluate(design_id_or_name="hbm_cpt", workload={_j(wl)})
4. chip_sweep_parameters(design_id_or_name="hbm_cpt",
   parameters={_j({"chiplets[1].count": [4, 6, 8], "workload.batch_size": [1, 8, 32]})},
   workload={_j(wl)}, objective="throughput_tokens_per_sec")
5. chip_estimate_cost(design_id_or_name="hbm_cpt") for the cost breakdown. If prices are
   stale and you can search the web, follow chip_price_lookup_plan and chip_set_price.
6. Report tokens/s per $1k, the bottleneck, and which cost line dominates."""


def analog_pim_feasibility(rows: int = 256, cols: int = 256, precision: str = "int8",
                           adc_bits: int = 8, organization: str = "org_b") -> str:
    """Check whether an analog DRAM compute-in-memory array is usable in a PIM chiplet."""
    ecore = {"rows": rows, "cols": cols, "precision": precision, "adc_bits": adc_bits,
             "organization": organization}
    return f"""Goal: decide whether a {rows}x{cols} {precision} analog DRAM CIM array
({organization}, {adc_bits}-bit ADC) is worth putting in a PIM chiplet.

1. chip_quickstart() and check external_backends in chiplab://models/assumptions: the
   'ecore' backend must be usable (it needs Echo Lab installed).
2. simulate_ecore_array(rows={rows}, cols={cols}, precision="{precision}",
   adc_bits={adc_bits}, organization="{organization}")
   Read accuracy first (snr_db_vs_reference, rel_l2_error, method), then derived
   (array_tops, energy_pj_per_op, array_area_mm2), calibration and limitations. The
   per_mac_snr_db figure is not end-to-end accuracy.
3. If accuracy is usable, put it in a design: a chiplet with type "PIM", area_mm2 and
   compute: {{"architecture": "processing_in_memory", "ecore": {_j(ecore)}}}, then
   chip_evaluate on that design. Peak TOPS, power and energy/op of that chiplet now come
   from eCore and are listed in the run's assumptions with source "ecore:...".
4. Repeat step 2 for adc_bits 6/10 and the other organization; report the accuracy vs
   energy/op trade-off and every limitation that applies."""


PROMPTS = (moe_decode_chiplet_balance, riscv_soc_power_area_tradeoff,
           hbm_accelerator_cost_per_token, analog_pim_feasibility)

def create_custom_chip_from_idea(idea: str = "custom chip", target_node: str = "tsmc_n7") -> str:
    """A guided prompt that walks ChatGPT through: describing the idea -> chip_design_from_nl -> chip_validate_design -> chip_full_flow -> reviewing results -> iterating.

    Includes the exact tool call sequence for creating a custom chip from a natural language idea.
    """
    return """Goal: Create a custom chip from your idea: {idea}

I'll walk you through the complete chip design workflow using ChipLab MCP tools.

Step 1: Describe your idea in natural language
-------------------------------------------
Your idea: {idea}

Step 2: Parse natural language into a Design Specification
---------------------------------------------------------
Tool call: chip_design_from_nl(description="{idea}", constraints={{'target_node': '{target_node}'}})

This will parse your description and return a structured DesignSpec YAML with:
- Architecture type (heterogeneous_accelerator, systolic, tensor_engine, etc.)
- Chiplet types and counts (memory chiplets, compute chiplets)
- Memory technology (DRAM, HBM, SRAM)
- Package type (2.5D, 3D_stacking, monolithic)
- Interconnect topology

Step 3: Validate the design
---------------------------
Tool call: chip_validate_design(design_id_or_name="<design_name>")

Check for:
- Valid architecture configuration
- Proper chiplet counts and types
- Memory technology compatibility
- Package and interconnect validity
- Any missing parameters or constraints

Step 4: Run the full design flow
--------------------------------
Tool call: chip_full_flow(description="{idea}", workload={{}}, constraints={{'target_node': '{target_node}'}}, max_iterations=3)

This triggers the entire lifecycle:
1. Design creation from NL
2. Design validation
3. Initial simulation (analytical fidelity)
4. Bottleneck analysis
5. Automatic improvement suggestions
6. Parameter sweep optimization
7. Design comparison
8. Export of complete project bundle

Step 5: Review results
----------------------
Evaluate the output from chip_full_flow, which includes:
- Final design specification
- Simulation metrics (throughput, power, temperature)
- Bottleneck identification
- Improvement suggestions
- Comparison of design variants
- Exported project files (RTL, reports, constraint files)

Step 6: Iterate as needed
-------------------------
If the results don't meet your objectives, you can:
- Refine the natural language description
- Adjust constraints (power, area, cost)
- Request specific improvements
- Run another flow iteration

The chip_design_from_nl, chip_validate_design, and chip_full_flow tools form
the core workflow for turning any natural language chip idea into a working
design with simulation results and exported files.

Please provide your chip idea, and I'll guide you through the complete design process."""


def optimize_existing_chip(design_id_or_name: str = "default_design", objective: str = "ppu") -> str:
    """Prompt for optimizing an existing design: chip_get_design -> chip_analyze_bottleneck -> chip_suggest_improvements -> chip_sweep_parameters -> chip_compare_designs.

    Args:
        design_id_or_name: The design ID or name to optimize
        objective: Optimization objective: "ppu" (power per unit), "throughput", "area", etc.

    Returns:
        A structured prompt guiding the optimization workflow.
    """
    return """Goal: Optimize existing chip design: {design_id_or_name}

Objective: {objective} (power per unit, throughput, area, or other)

I'll guide you through optimizing an existing ChipLab design using the optimization workflow.

Step 1: Retrieve the existing design
------------------------------------
Tool call: chip_get_design(design_id_or_name="{design_id_or_name}")

This returns:
- Design specification (name, version, architecture, chiplets, package, etc.)
- Design history and provenance
- Current validation status

Step 2: Analyze the design bottleneck
--------------------------------------
Tool call: chip_analyze_bottleneck(design_id_or_name="{design_id_or_name}")

This returns:
- Primary bottleneck type (memory_bandwidth, compute_bound, thermal, power, balanced)
- Severity score
- Utilization breakdown across components
- Specific recommendations for improvement

Step 3: Get ranked improvement suggestions
-----------------------------------------
Tool call: chip_suggest_improvements(design_id_or_name="{design_id_or_name}", max_suggestions=5)

This returns a ranked list of concrete parameter changes with:
- Specific changes (increase memory bandwidth, add compute chiplets, change process node, adjust cache sizes)
- Rationale for each change
- Expected improvement quantification
- Priority ordering

Step 4: Run parameter sweep optimization
-----------------------------------------
Tool call: chip_sweep_parameters(design_id_or_name="{design_id_or_name}",
    parameters={{...sweep parameters from suggestions...}},
    objective="{objective}")

This runs a Cartesian product sweep of parameter values and returns:
- Best point for the given objective
- All swept points with metrics
- Objective values for comparison

Step 5: Compare design variants
--------------------------------
Tool call: chip_compare_designs(run_ids=[...])

If multiple sweep runs or design versions exist, compare them to:
- See metric deltas between variants
- Identify the winner by objective
- Review summary of trade-offs

Step 6: Implement and re-evaluate
----------------------------------
Based on the comparison:
- Select the best design variant
- Optionally run another optimization cycle
- Generate RTL for the optimized design
- Export the complete project bundle

The optimization workflow (chip_get_design -> chip_analyze_bottleneck ->
chip_suggest_improvements -> chip_sweep_parameters -> chip_compare_designs)
provides a systematic approach to improving existing chip designs for your
chosen objective (power per unit, throughput, area, etc.).

Please provide the design ID or name you want to optimize, and I'll guide you
through the complete optimization process."""
PROMPTS = (moe_decode_chiplet_balance, riscv_soc_power_area_tradeoff,
            hbm_accelerator_cost_per_token, analog_pim_feasibility,
            create_custom_chip_from_idea, optimize_existing_chip)