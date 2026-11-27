"""MCP industrial tools for ChipLab - clean JSON-serializable functions."""

from __future__ import annotations

from typing import Any

from chiplab.reliability.degradation import state_drift
from chiplab.reliability.aging import monte_carlo_lifetime
from chiplab.discovery.explorer import (
    BottleneckType,
    DesignSpace,
    evaluate_candidate,
    explore_design_space,
    ParetoFrontier,
    quick_exploration,
)
from chiplab.manufacturing.pipeline import ManufacturingPipeline
from chiplab.digital_twin.twin import DigitalTwin
from chiplab.multiscale.gpu_surrogate import GPUSurrogate, make_gpu_surrogate, pod_reduce, quick_pod_demo
import numpy as np

from chiplab.credibility import refuse_ferrofluid


def chiplab_reliability_analyze(
    technology: str,
    stress_temp_k: float,
    operating_temp_k: float,
    duration_hours: float,
    n_mc_samples: int = 1000,
) -> dict:
    """Analyze reliability degradation and lifetime using Arrhenius + Monte Carlo."""
    refuse_ferrofluid("the generic Arrhenius reliability model", technology)
    deg_result = state_drift(initial_value=1.0, drift_rate=0.01, time_elapsed=duration_hours)
    k_boltzmann = 8.617333262145e-5
    ea_typical = 0.7
    aa_factor = np.exp(ea_typical / k_boltzmann * (1.0 / operating_temp_k - 1.0 / stress_temp_k))
    age_tv = monte_carlo_lifetime(
        n_samples=n_mc_samples,
        temperature_k=operating_temp_k,
    )
    age_val = age_tv.value if hasattr(age_tv, "value") else age_tv
    if isinstance(age_val, dict):
        lifetimes = age_val.get("lifetime_hours", age_val.get("samples", []))
    elif isinstance(age_val, (list, np.ndarray)):
        lifetimes = list(age_val)
    else:
        lifetimes = [float(age_val)] if age_val is not None else []
    if len(lifetimes) > 0:
        lifetimes_arr = np.asarray(lifetimes, dtype=float)
        percentiles = {
            "p10": float(np.percentile(lifetimes_arr, 10)),
            "p50": float(np.percentile(lifetimes_arr, 50)),
            "p90": float(np.percentile(lifetimes_arr, 90)),
            "mean": float(np.mean(lifetimes_arr)),
            "median": float(np.median(lifetimes_arr)),
        }
    else:
        percentiles = {"p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "technology": technology,
        "stress_temp_k": stress_temp_k,
        "operating_temp_k": operating_temp_k,
        "duration_hours": duration_hours,
        "arrhenius_acceleration_factor": float(aa_factor),
        "degradation_metrics": {
            "threshold": deg_result.get("threshold", 1.0) if hasattr(deg_result, "get") else 1.0,
            "current_degradation": float(deg_result.get("current_degradation", 0.0)) if hasattr(deg_result, "get") else 0.0,
        },
        "monte_carlo_lifetime": {"samples": n_mc_samples, "percentiles_hours": percentiles},
        "provenance": {"degradation_model": "chiplab.reliability.degradation", "aging_model": "chiplab.reliability.aging"},
    }


def chiplab_discover_architectures(
    parameter_ranges: dict,
    n_candidates: int = 100,
    sampling_method: str = "latin_hypercube",
) -> dict:
    """Explore design space and return Pareto-optimal architectures with bottlenecks."""
    refuse_ferrofluid("architecture discovery", parameter_ranges)
    design_space = DesignSpace(parameters=parameter_ranges)
    frontier: ParetoFrontier = explore_design_space(
        design_space=design_space,
        n_candidates=n_candidates,
        sampling_method=sampling_method,
        verbose=False,
    )
    frontier_candidates = frontier.get_frontier_candidates()
    bottleneck_analysis = []
    for candidate in frontier_candidates:
        bottlenecks = []
        if candidate.power > 100:
            bottlenecks.append("thermal")
        if candidate.area > 50:
            bottlenecks.append("IR-drop")
        efficiency = candidate.performance / max(candidate.power, 1e-3)
        bottleneck_analysis.append(
            {
                "params": candidate.params,
                "ppa_tuple": candidate.ppa_tuple,
                "bottlenecks": bottlenecks,
                "efficiency_ops_per_watt": efficiency,
            }
        )
    return {
        "parameter_ranges": parameter_ranges,
        "n_candidates_generated": n_candidates,
        "sampling_method": sampling_method,
        "pareto_frontier_candidates": len(frontier_candidates),
        "frontier_candidates": bottleneck_analysis,
        "bottleneck_type_summary": {
            bt: sum(1 for ba in bottleneck_analysis if bt in ba["bottlenecks"])
            for bt in [bt.value for bt in BottleneckType]
        },
        "provenance": {"explorer_module": "chiplab.discovery.explorer", "core_func": "explore_design_space"},
    }


def chiplab_manufacturing_signoff(
    design_name: str,
    version: str = "1.0",
    material_spec: dict | None = None,
    device_params: dict | None = None,
    circuit_params: dict | None = None,
    rtl_params: dict | None = None,
) -> dict:
    """Run full manufacturing pipeline signoff and return signoff report."""
    from chiplab.manufacturing.pipeline import PipelineStage

    refuse_ferrofluid("manufacturing signoff", design_name, material_spec, device_params,
                      circuit_params, rtl_params)

    pipeline = ManufacturingPipeline(design_name=design_name, version=version)
    pipeline_data = {}
    if material_spec is not None:
        pipeline_data[PipelineStage.MATERIAL] = material_spec
    if device_params is not None:
        pipeline_data[PipelineStage.DEVICE] = device_params
    if circuit_params is not None:
        pipeline_data[PipelineStage.CIRCUIT] = circuit_params
    if rtl_params is not None:
        pipeline_data[PipelineStage.RTL] = rtl_params
    report = pipeline.run_full_pipeline(pipeline_data)
    return {
        "design_name": report.design_name,
        "version": report.version,
        "overall_pass": report.overall_pass,
        "stage_results": {stage.value: result.value for stage, result in report.stage_results.items()},
        "dfm_checks": {
            stage.value: {
                "pass": dfm.pass_flag,
                "checks": dfm.checks,
                "violations": dfm.violations,
                "area_mm2": dfm.area_mm2,
                "defect_density_cm2": dfm.defect_density_cm2,
                "yield_pct": dfm.yield_pct,
            }
            for stage, dfm in report.dfm_checks.items()
        },
        "pvt_summary": {
            "process_corners": report.pvt_summary.process_corners,
            "voltages": report.pvt_summary.voltages,
            "temperatures": report.pvt_summary.temperatures,
            "yield_at_corners": {
                f"{proc}_{vol}_{temp}": yield_pct
                for (proc, vol, temp), yield_pct in report.pvt_summary.yield_at_corners.items()
            },
            "worst_case_yield_pct": report.pvt_summary.worst_case_yield_pct,
            "best_case_yield_pct": report.pvt_summary.best_case_yield_pct,
        },
        "total_cost_usd": report.total_cost_usd,
        "yield_pct": report.yield_pct,
        "notes": report.notes,
        "provenance": {"pipeline_module": "chiplab.manufacturing.pipeline", "core_class": "ManufacturingPipeline"},
    }


def chiplab_digital_twin_manage(
    action: str,
    design_id: str | None = None,
    design_data: dict | None = None,
    calibration_data: dict | None = None,
    snapshot_name: str | None = None,
    save_path: str | None = None,
    load_path: str | None = None,
) -> dict:
    """Manage digital twin: create, evaluate, calibrate, snapshot, save, load."""
    refuse_ferrofluid("the digital twin", design_id, design_data, calibration_data)
    dt = DigitalTwin(persist_path=save_path or load_path)
    if action == "create":
        if design_data is None:
            raise ValueError("design_data is required for 'create' action")
        params = {"node": 7, "cores": 16, "vlen": 256, "dlen": 256, **design_data}
        eval_res = dt.evaluate(params=params)
        return {
            "action": "create",
            "design_id": design_id or "default",
            "state": dt.snapshot(),
            "evaluation": eval_res,
            "provenance": {"module": "chiplab.digital_twin.twin", "class": "DigitalTwin"},
        }
    elif action == "evaluate":
        params = {"node": 7, "cores": 16, "vlen": 256, "dlen": 256, **(design_data or {})}
        result = dt.evaluate(params=params)
        return {"action": "evaluate", "result": result, "provenance": {"module": "chiplab.digital_twin.twin", "class": "DigitalTwin"}}
    elif action == "calibrate":
        if calibration_data is None:
            raise ValueError("calibration_data is required for 'calibrate' action")
        dt.calibrate(factors=calibration_data)
        return {
            "action": "calibrate",
            "design_id": design_id or "default",
            "state": dt.snapshot(),
            "provenance": {"module": "chiplab.digital_twin.twin", "class": "DigitalTwin"},
        }
    elif action == "snapshot":
        snap = dt.snapshot()
        return {"action": "snapshot", "snapshot_name": snapshot_name or "latest", "state": snap, "provenance": {"module": "chiplab.digital_twin.twin", "class": "DigitalTwin"}}
    elif action == "save":
        if save_path is None:
            raise ValueError("save_path is required for 'save' action")
        dt.save(save_path)
        return {"action": "save", "save_path": save_path, "provenance": {"module": "chiplab.digital_twin.twin", "class": "DigitalTwin"}}
    elif action == "load":
        if load_path is None:
            raise ValueError("load_path is required for 'load' action")
        dt.load(load_path)
        return {"action": "load", "design_id": design_id or "default", "state": dt.snapshot(), "provenance": {"module": "chiplab.digital_twin.twin", "class": "DigitalTwin"}}
    else:
        raise ValueError(f"Unknown action: {action}. Must be one of: create, evaluate, calibrate, snapshot, save, load")


def chiplab_multiscale_surrogate(
    action: str,
    input_data: Any = None,
    n_components: int = 16,
    hidden_dim: int = 64,
    operator_type: str = "mlp",
    device: str = "auto",
    n_epochs: int = 10,
) -> dict | None:
    """Multi-scale surrogate modeling: POD/SVD reduction + neural operator prediction."""
    import numpy as np
    if action == "reduce":
        if input_data is None:
            raise ValueError("input_data is required for 'reduce' action")
        modes, singular_values = pod_reduce(X=input_data, n_components=n_components, device=device)
        return {
            "action": "reduce",
            "modes_shapes": {"modes": list(modes.shape), "singular_values": list(singular_values.shape)},
            "explained_variance": float(np.sum(singular_values ** 2) / np.sum((np.linalg.svd(input_data, compute_uv=False) ** 2))),
            "provenance": {"module": "chiplab.multiscale.gpu_surrogate", "func": "pod_reduce"},
        }
    elif action == "surrogate":
        if input_data is None:
            raise ValueError("input_data is required for 'surrogate' action")
        n_input = input_data.shape[-1] if hasattr(input_data, "shape") else 256
        n_output = input_data.shape[-1] if hasattr(input_data, "shape") else 256
        surrogate = make_gpu_surrogate(input_dim=n_input, output_dim=n_output, hidden_dim=hidden_dim, operator_type=operator_type, device=device)
        return {
            "action": "surrogate",
            "model_type": f"GPUSurrogate({operator_type}, {hidden_dim})",
            "input_dim": surrogate.input_dim,
            "output_dim": surrogate.output_dim,
            "provenance": {"module": "chiplab.multiscale.gpu_surrogate", "func": "make_gpu_surrogate"},
        }
    elif action == "predict":
        if input_data is None:
            raise ValueError("input_data is required for 'predict' action")
        n_input = input_data.shape[-1] if hasattr(input_data, "shape") else 256
        n_output = input_data.shape[-1] if hasattr(input_data, "shape") else 256
        surrogate = make_gpu_surrogate(input_dim=n_input, output_dim=n_output, hidden_dim=hidden_dim, operator_type=operator_type, device=device)
        surrogate.fit(X_train=input_data, y_train=input_data, n_epochs=n_epochs)
        predictions = surrogate.predict(X=input_data)
        mse = float(np.mean((predictions - input_data) ** 2)) if hasattr(predictions, "shape") else 0.0
        return {
            "action": "predict",
            "n_predictions": predictions.shape[0],
            "mse": mse,
            "provenance": {"module": "chiplab.multiscale.gpu_surrogate", "func": "GPUSurrogate.predict"},
        }
    elif action == "quick_demo":
        n_features = input_data.shape[-1] if input_data is not None else 256
        demo = quick_pod_demo(n_samples=50, n_features=n_features, n_modes=n_components, device=device)
        return {
            "action": "quick_demo",
            "modes": int(demo["modes"].shape[1]),
            "singular_values_count": int(len(demo["singular_values"])),
            "reconstruction_error": float(demo["reconstruction_error"]),
            "device_info": demo["device_info"],
            "provenance": {"module": "chiplab.multiscale.gpu_surrogate", "func": "quick_pod_demo"},
        }
    else:
        raise ValueError(f"Unknown action: {action}. Must be one of: reduce, surrogate, predict, quick_demo")
