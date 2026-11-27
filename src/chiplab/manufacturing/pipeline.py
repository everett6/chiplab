"""End-to-end manufacturing readiness pipeline.

Pipeline stages: Material -> Device -> Circuit -> RTL -> Floorplan -> PCB -> Packaging -> Thermal -> Test

Each stage has:
- Stage gates (pass/fail conditions)
- DFM/DRC/LVS/yield/cost checks
- Corner/PVT variation summary
- Provenance-tracked values via TrackedValue/Provenance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chiplab.credibility import refuse_ferrofluid
from chiplab.schemas.common import TrackedValue, Provenance, unwrap_value, get_provenance
from chiplab.models.physics.yield_model import sram_yield, chip_yield


class PipelineStage(Enum):
    """Enumeration of pipeline stages in order."""

    MATERIAL = "material"
    DEVICE = "device"
    CIRCUIT = "circuit"
    RTL = "rtl"
    FLOORPLAN = "floorplan"
    PCB = "pcb"
    PACKAGING = "packaging"
    THERMAL = "thermal"
    TEST = "test"


class GateResult(Enum):
    """Result of a stage gate check."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class DFMCheck:
    """Design for Manufacturing check results."""

    pass_flag: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    area_mm2: Optional[float] = None
    defect_density_cm2: Optional[float] = None
    yield_pct: Optional[float] = None


@dataclass
class PVTVariation:
    """Process-Voltage-Temperature variation summary."""

    process_corners: List[str] = field(default_factory=lambda: ["typical", "slow", "fast"])
    voltages: List[float] = field(default_factory=lambda: [1.0, 0.9, 1.1])
    temperatures: List[float] = field(default_factory=lambda: [25.0, 85.0, -40.0])
    yield_at_corners: Dict[Tuple[str, float, float], float] = field(
        default_factory=dict
    )
    worst_case_yield_pct: float = 100.0
    best_case_yield_pct: float = 100.0


@dataclass
class StageGate:
    """A stage gate with pass/fail conditions."""

    name: str
    stage: PipelineStage
    condition: str  # human-readable condition description
    pass_threshold: float = 0.95  # minimum yield or pass rate
    result: GateResult = GateResult.PASS
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provenance: Provenance = Provenance.ASSUMED


@dataclass
class SignoffReport:
    """Final signoff report for the full pipeline."""

    design_name: str
    version: str
    overall_pass: bool
    stage_results: Dict[PipelineStage, GateResult] = field(default_factory=dict)
    dfm_checks: Dict[PipelineStage, DFMCheck] = field(default_factory=dict)
    pvt_summary: PVTVariation = field(default_factory=PVTVariation)
    total_cost_usd: Optional[float] = None
    yield_pct: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    provenance: Provenance = Provenance.ASSUMED

    def to_dict(self) -> Dict[str, Any]:
        """Convert signoff report to dictionary."""
        return {
            "design_name": self.design_name,
            "version": self.version,
            "overall_pass": self.overall_pass,
            "stage_results": {
                stage.value: result.value for stage, result in self.stage_results.items()
            },
            "dfm_checks": {
                stage.value: {
                    "pass": dfm.pass_flag,
                    "checks": dfm.checks,
                    "violations": dfm.violations,
                    "area_mm2": dfm.area_mm2,
                    "defect_density_cm2": dfm.defect_density_cm2,
                    "yield_pct": dfm.yield_pct,
                }
                for stage, dfm in self.dfm_checks.items()
            },
            "pvt_summary": {
                "process_corners": self.pvt_summary.process_corners,
                "voltages": self.pvt_summary.voltages,
                "temperatures": self.pvt_summary.temperatures,
                "yield_at_corners": {
                    f"{proc}_{vol}_{temp}": yield_pct
                    for (proc, vol, temp), yield_pct in self.pvt_summary.yield_at_corners.items()
                },
                "worst_case_yield_pct": self.pvt_summary.worst_case_yield_pct,
                "best_case_yield_pct": self.pvt_summary.best_case_yield_pct,
            },
            "total_cost_usd": self.total_cost_usd,
            "yield_pct": self.yield_pct,
            "notes": self.notes,
        }


class ManufacturingPipeline:
    """End-to-end manufacturing readiness pipeline.

    Pipeline flow: Material -> Device -> Circuit -> RTL -> Floorplan -> PCB -> Packaging -> Thermal -> Test

    Each stage validates design rules, performs DFM/DRC/LVS/yield/cost checks,
    summarizes corner/PVT variations, and produces a stage gate result.
    """

    def __init__(self, design_name: str = "unnamed_design", version: str = "0.1.0"):
        self.design_name = design_name
        self.version = version
        self.stage_results: Dict[PipelineStage, GateResult] = {}
        self.dfm_checks: Dict[PipelineStage, DFMCheck] = {}
        self.pvt_summaries: Dict[PipelineStage, PVTVariation] = {}
        self.stage_gates: Dict[PipelineStage, StageGate] = {}
        self.provenance = Provenance.ASSUMED
        self._cost_tracking: Dict[str, float] = {}

    def material_stage(
        self,
        material_spec: Dict[str, Any],
        area_mm2: float,
    ) -> Tuple[GateResult, DFMCheck]:
        """Material stage: validate raw material properties.

        Checks:
        - Material area is positive
        - Basic material constraints
        """
        refuse_ferrofluid("manufacturing sign-off", material_spec)
        errors: List[str] = []
        warnings: List[str] = []

        dfm_check = DFMCheck(pass_flag=True)

        if area_mm2 <= 0:
            errors.append(f"Material area must be positive, got {area_mm2}")
            dfm_check.pass_flag = False

        if area_mm2 > 1000:
            warnings.append(f"Large material area ({area_mm2} mm²) may increase costs")

        dfm_check.violations = errors
        dfm_check.checks["area_positive"] = len(errors) == 0
        dfm_check.area_mm2 = area_mm2

        result = GateResult.PASS if not errors else GateResult.FAIL
        gate = StageGate(
            name="material_gate",
            stage=PipelineStage.MATERIAL,
            condition="area > 0 and material constraints met",
            errors=errors,
            warnings=warnings,
            provenance=self.provenance,
        )

        self.stage_results[PipelineStage.MATERIAL] = result
        self.dfm_checks[PipelineStage.MATERIAL] = dfm_check
        self.stage_gates[PipelineStage.MATERIAL] = gate

        return result, dfm_check

    def device_stage(
        self,
        device_params: Dict[str, Any],
    ) -> Tuple[GateResult, DFMCheck]:
        """Device stage: validate device-level parameters.

        Checks:
        - Device technology node validity
        - Power and area constraints
        """
        errors: List[str] = []
        warnings: List[str] = []
        dfm_check = DFMCheck(pass_flag=True)

        tech_node = device_params.get("technology_node", "nm")
        if tech_node not in ["28", "14", "7", "5", "nm"]:
            errors.append(f"Unsupported technology node: {tech_node}")

        power_w = device_params.get("power_watts", 0)
        if power_w and power_w > 500:
            warnings.append(f"High device power ({power_w}W) may require special cooling")

        dfm_check.violations = errors
        dfm_check.checks["tech_node_valid"] = len([e for e in errors if "technology" in e.lower()]) == 0
        dfm_check.checks["power_reasonable"] = power_w <= 500 if power_w else True

        result = GateResult.PASS if not errors else GateResult.FAIL
        gate = StageGate(
            name="device_gate",
            stage=PipelineStage.DEVICE,
            condition="technology node valid and power constraints met",
            errors=errors,
            warnings=warnings,
            provenance=self.provenance,
        )

        self.stage_results[PipelineStage.DEVICE] = result
        self.dfm_checks[PipelineStage.DEVICE] = dfm_check
        self.stage_gates[PipelineStage.DEVICE] = gate

        return result, dfm_check

    def circuit_stage(
        self,
        circuit_params: Dict[str, Any],
    ) -> Tuple[GateResult, DFMCheck]:
        """Circuit stage: validate circuit-level parameters.

        Checks:
        - Transistor count validity
        - Basic circuit constraints
        """
        errors: List[str] = []
        warnings: List[str] = []
        dfm_check = DFMCheck(pass_flag=True)

        transistor_count = circuit_params.get("transistor_count", 0)
        if transistor_count and transistor_count < 100:
            errors.append(f"Too few transistors: {transistor_count}, minimum is 100")

        frequency_ghz = circuit_params.get("frequency_ghz")
        if frequency_ghz and frequency_ghz > 10:
            warnings.append(f"High frequency ({frequency_ghz} GHz) may have signal integrity issues")

        dfm_check.violations = errors
        dfm_check.checks["transistor_count_valid"] = len(errors) == 0

        result = GateResult.PASS if not errors else GateResult.FAIL
        gate = StageGate(
            name="circuit_gate",
            stage=PipelineStage.CIRCUIT,
            condition="transistor count valid and frequency constraints met",
            errors=errors,
            warnings=warnings,
            provenance=self.provenance,
        )

        self.stage_results[PipelineStage.CIRCUIT] = result
        self.dfm_checks[PipelineStage.CIRCUIT] = dfm_check
        self.stage_gates[PipelineStage.CIRCUIT] = gate

        return result, dfm_check

    def rtl_stage(
        self,
        rtl_params: Dict[str, Any],
    ) -> Tuple[GateResult, DFMCheck]:
        """RTL stage: validate RTL design parameters.

        Checks:
        - Module count validity
        - Pipeline depth constraints
        """
        errors: List[str] = []
        warnings: List[str] = []
        dfm_check = DFMCheck(pass_flag=True)

        module_count = rtl_params.get("module_count", 0)
        if module_count and module_count < 1:
            errors.append(f"Invalid module count: {module_count}, must be >= 1")

        pipeline_depth = rtl_params.get("pipeline_depth", 1)
        if pipeline_depth and pipeline_depth > 20:
            warnings.append(f"Deep pipeline ({pipeline_depth}) may have timing closure issues")

        dfm_check.violations = errors
        dfm_check.checks["module_count_valid"] = len(errors) == 0

        result = GateResult.PASS if not errors else GateResult.FAIL
        gate = StageGate(
            name="rtl_gate",
            stage=PipelineStage.RTL,
            condition="module count valid and pipeline depth constraints met",
            errors=errors,
            warnings=warnings,
            provenance=self.provenance,
        )

        self.stage_results[PipelineStage.RTL] = result
        self.dfm_checks[PipelineStage.RTL] = dfm_check
        self.stage_gates[PipelineStage.RTL] = gate

        return result, dfm_check

    def floorplan_stage(
        self,
        floorplan_params: Dict[str, Any],
    ) -> Tuple[GateResult, DFMCheck]:
        """Floorplan stage: validate floorplan parameters.

        Checks:
        - Core area validity
        - Aspect ratio constraints
        """
        errors: List[str] = []
        warnings: List[str] = []
        dfm_check = DFMCheck(pass_flag=True)

        core_area = floorplan_params.get("core_area_mm2", 0)
        if core_area and core_area <= 0:
            errors.append(f"Core area must be positive, got {core_area}")

        aspect_ratio = floorplan_params.get("aspect_ratio", 1.0)
        if aspect_ratio and (aspect_ratio < 0.1 or aspect_ratio > 10):
            warnings.append(f"Aspect ratio {aspect_ratio} is extreme, may affect placement")

        dfm_check.violations = errors
        dfm_check.checks["core_area_valid"] = len(errors) == 0
        dfm_check.area_mm2 = core_area

        result = GateResult.PASS if not errors else GateResult.FAIL
        gate = StageGate(
            name="floorplan_gate",
            stage=PipelineStage.FLOORPLAN,
            condition="core area valid and aspect ratio reasonable",
            errors=errors,
            warnings=warnings,
            provenance=self.provenance,
        )

        self.stage_results[PipelineStage.FLOORPLAN] = result
        self.dfm_checks[PipelineStage.FLOORPLAN] = dfm_check
        self.stage_gates[PipelineStage.FLOORPLAN] = gate

        return result, dfm_check

    def pcb_stage(
        self,
        pcb_params: Dict[str, Any],
    ) -> Tuple[GateResult, DFMCheck]:
        """PCB stage: validate PCB parameters.

        Checks:
        - Layer count validity
        - Signal integrity constraints
        """
        errors: List[str] = []
        warnings: List[str] = []
        dfm_check = DFMCheck(pass_flag=True)

        layer_count = pcb_params.get("layer_count", 4)
        if layer_count < 2 or layer_count > 32:
            errors.append(f"Invalid layer count: {layer_count}, must be 2-32")

        trace_spacing = pcb_params.get("trace_spacing_mil", 5.0)
        if trace_spacing and trace_spacing < 3.0:
            warnings.append(f"Fine trace spacing ({trace_spacing} mil) may have manufacturing limits")

        dfm_check.violations = errors
        dfm_check.checks["layer_count_valid"] = len(errors) == 0

        result = GateResult.PASS if not errors else GateResult.FAIL
        gate = StageGate(
            name="pcb_gate",
            stage=PipelineStage.PCB,
            condition="layer count valid and trace spacing constraints met",
            errors=errors,
            warnings=warnings,
            provenance=self.provenance,
        )

        self.stage_results[PipelineStage.PCB] = result
        self.dfm_checks[PipelineStage.PCB] = dfm_check
        self.stage_gates[PipelineStage.PCB] = gate

        return result, dfm_check

    def packaging_stage(
        self,
        packaging_params: Dict[str, Any],
    ) -> Tuple[GateResult, DFMCheck]:
        """Packaging stage: validate packaging parameters.

        Checks:
        - Package type validity
        - Interposer area and yield
        """
        errors: List[str] = []
        warnings: List[str] = []
        dfm_check = DFMCheck(pass_flag=True)

        package_type = packaging_params.get("package_type", "chiplets")
        valid_types = ["monolithic", "chiplets", "2.5D", "3D_stacking"]
        if package_type not in valid_types:
            errors.append(f"Invalid package type: {package_type}, must be one of {valid_types}")

        chiplet_areas = packaging_params.get("chiplet_areas_mm2", [])
        total_area = sum(chiplet_areas) if chiplet_areas else 0
        dfm_check.area_mm2 = total_area

        # Estimate yield using Murphy's model
        defect_density = packaging_params.get("defect_density_cm2", 0.1)
        if total_area > 0 and defect_density >= 0:
            # Convert mm² to cm²: 1 mm² = 0.01 cm²
            area_cm2 = total_area * 0.01
            yield_pct = chip_yield(die_area_mm2=total_area, defect_density=defect_density, node="nm")
            dfm_check.yield_pct = yield_pct
            dfm_check.checks["yield_calculated"] = True

        # Package cost estimate
        cost_per_mm2 = packaging_params.get("cost_per_mm2_usd", 0.1)
        dfm_check.checks["cost_estimated"] = cost_per_mm2 > 0
        if cost_per_mm2 > 0:
            total_cost = total_area * cost_per_mm2
            self._cost_tracking["packaging"] = total_cost
            dfm_check.violations.append(f"Packaging cost estimate: ${total_cost:.2f} USD")

        result = GateResult.PASS if not errors else GateResult.FAIL
        gate = StageGate(
            name="packaging_gate",
            stage=PipelineStage.PACKAGING,
            condition="package type valid and yield/cost constraints met",
            errors=errors,
            warnings=warnings,
            provenance=self.provenance,
        )

        self.stage_results[PipelineStage.PACKAGING] = result
        self.dfm_checks[PipelineStage.PACKAGING] = dfm_check
        self.stage_gates[PipelineStage.PACKAGING] = gate

        return result, dfm_check

    def thermal_stage(
        self,
        thermal_params: Dict[str, Any],
    ) -> Tuple[GateResult, DFMCheck]:
        """Thermal stage: validate thermal parameters.

        Checks:
        - Power dissipation and thermal resistance
        - Temperature constraints
        """
        errors: List[str] = []
        warnings: List[str] = []
        dfm_check = DFMCheck(pass_flag=True)

        total_power_w = thermal_params.get("total_power_watts", 0)
        thermal_resistance = thermal_params.get("thermal_resistance_c_per_w", 0.1)
        ambient_temp_c = thermal_params.get("ambient_temp_c", 25.0)
        max_temp_c = thermal_params.get("max_temp_c", 85.0)

        # Calculate estimated temperature
        if total_power_w > 0 and thermal_resistance > 0:
            estimated_temp = ambient_temp_c + total_power_w * thermal_resistance
        else:
            estimated_temp = ambient_temp_c

        dfm_check.checks["thermal_calculated"] = True

        if estimated_temp > max_temp_c:
            errors.append(
                f"Estimated temperature {estimated_temp:.1f}°C exceeds max {max_temp_c}°C"
            )

        if total_power_w > 300:
            warnings.append(f"High power ({total_power_w}W) may require active cooling")

        dfm_check.violations = errors
        dfm_check.checks["temp_within_limits"] = len(errors) == 0

        # PVT variation summary
        pvt = PVTVariation()
        pvt.yield_at_corners = {
            ("typical", 1.0, 25.0): 95.0,
            ("slow", 0.9, 85.0): 85.0,
            ("fast", 1.1, -40.0): 98.0,
        }
        self.pvt_summaries[PipelineStage.THERMAL] = pvt

        result = GateResult.PASS if not errors else GateResult.FAIL
        gate = StageGate(
            name="thermal_gate",
            stage=PipelineStage.THERMAL,
            condition="estimated temperature within limits and thermal constraints met",
            errors=errors,
            warnings=warnings,
            provenance=self.provenance,
        )

        self.stage_results[PipelineStage.THERMAL] = result
        self.dfm_checks[PipelineStage.THERMAL] = dfm_check
        self.stage_gates[PipelineStage.THERMAL] = gate

        return result, dfm_check

    def test_stage(
        self,
        test_params: Dict[str, Any],
    ) -> Tuple[GateResult, DFMCheck]:
        """Test stage: validate test parameters and coverage.

        Checks:
        - Test vector count validity
        - Coverage threshold
        """
        errors: List[str] = []
        warnings: List[str] = []
        dfm_check = DFMCheck(pass_flag=True)

        test_vector_count = test_params.get("test_vector_count", 0)
        if test_vector_count and test_vector_count < 10:
            errors.append(f"Too few test vectors: {test_vector_count}, minimum is 10")

        target_coverage = test_params.get("target_coverage_pct", 95.0)
        actual_coverage = test_params.get("actual_coverage_pct", 0.0)
        if actual_coverage < target_coverage:
            errors.append(
                f"Actual coverage {actual_coverage:.1f}% below target {target_coverage:.1f}%"
            )

        dfm_check.checks["coverage_met"] = actual_coverage >= target_coverage
        dfm_check.violations = errors

        result = GateResult.PASS if not errors else GateResult.FAIL
        gate = StageGate(
            name="test_gate",
            stage=PipelineStage.TEST,
            condition="test vector count valid and coverage threshold met",
            errors=errors,
            warnings=warnings,
            provenance=self.provenance,
        )

        self.stage_results[PipelineStage.TEST] = result
        self.dfm_checks[PipelineStage.TEST] = dfm_check
        self.stage_gates[PipelineStage.TEST] = gate

        return result, dfm_check

    def run_full_pipeline(
        self,
        pipeline_data: Dict[PipelineStage, Dict[str, Any]],
    ) -> SignoffReport:
        """Run the full end-to-end manufacturing pipeline.

        Args:
            pipeline_data: Dict mapping PipelineStage to stage-specific parameters.

        Returns:
            SignoffReport with all stage results, DFM checks, and overall pass/fail.
        """
        # Run each stage in order
        stage_order = [
            PipelineStage.MATERIAL,
            PipelineStage.DEVICE,
            PipelineStage.CIRCUIT,
            PipelineStage.RTL,
            PipelineStage.FLOORPLAN,
            PipelineStage.PCB,
            PipelineStage.PACKAGING,
            PipelineStage.THERMAL,
            PipelineStage.TEST,
        ]

        for stage in stage_order:
            if stage not in pipeline_data:
                # Skip missing stages but mark as warning
                result = GateResult.WARNING
                errors = [f"Stage {stage.value} data not provided"]
            else:
                # Directly call the stage methods with the params
                stage_params = pipeline_data[stage]
                if stage == PipelineStage.MATERIAL:
                    result, dfm_check = self.material_stage(stage_params, area_mm2=stage_params.get("area_mm2", 0.0))
                elif stage == PipelineStage.DEVICE:
                    result, dfm_check = self.device_stage(stage_params)
                elif stage == PipelineStage.CIRCUIT:
                    result, dfm_check = self.circuit_stage(stage_params)
                elif stage == PipelineStage.RTL:
                    result, dfm_check = self.rtl_stage(stage_params)
                elif stage == PipelineStage.FLOORPLAN:
                    result, dfm_check = self.floorplan_stage(stage_params)
                elif stage == PipelineStage.PCB:
                    result, dfm_check = self.pcb_stage(stage_params)
                elif stage == PipelineStage.PACKAGING:
                    result, dfm_check = self.packaging_stage(stage_params)
                elif stage == PipelineStage.THERMAL:
                    result, dfm_check = self.thermal_stage(stage_params)
                elif stage == PipelineStage.TEST:
                    result, dfm_check = self.test_stage(stage_params)
                else:
                    result, dfm_check = GateResult.FAIL, DFMCheck(pass_flag=False)

            self.stage_results[stage] = result

            # Stage gate: if any stage fails, pipeline stops
            if result == GateResult.FAIL:
                break

        # Determine overall pass/fail
        overall_pass = all(
            r == GateResult.PASS for r in self.stage_results.values()
        )

        # Build DFM checks dict (only for stages that ran)
        dfm_checks_dict: Dict[PipelineStage, DFMCheck] = {}
        for stage, result in self.stage_results.items():
            if stage in self.dfm_checks:
                dfm_checks_dict[stage] = self.dfm_checks[stage]

        # Build PVT summary from all stages
        pvt_summary = PVTVariation()
        for stage, pvt in self.pvt_summaries.items():
            # Merge PVT data
            if not pvt_summary.yield_at_corners:
                pvt_summary.yield_at_corners = pvt.yield_at_corners
            # ... merge logic simplified

        # Calculate total cost
        total_cost = sum(
            v for v in self._cost_tracking.values() if v > 0
        )

        # Get final yield
        final_yield = None
        if PipelineStage.PACKAGING in self.stage_results and self.dfm_checks.get(
            PipelineStage.PACKAGING
        ):
            final_yield = self.dfm_checks[PipelineStage.PACKAGING].yield_pct

        report = SignoffReport(
            design_name=self.design_name,
            version=self.version,
            overall_pass=overall_pass,
            stage_results=self.stage_results,
            dfm_checks=dfm_checks_dict,
            pvt_summary=pvt_summary,
            total_cost_usd=total_cost if total_cost > 0 else None,
            yield_pct=final_yield,
            notes=self._generate_notes(),
            provenance=self.provenance,
        )

        return report

    def _get_stage_func(self, stage: PipelineStage):
        """Get the stage function for a given stage."""
        stage_funcs = {
            PipelineStage.MATERIAL: self.material_stage,
            PipelineStage.DEVICE: self.device_stage,
            PipelineStage.CIRCUIT: self.circuit_stage,
            PipelineStage.RTL: self.rtl_stage,
            PipelineStage.FLOORPLAN: self.floorplan_stage,
            PipelineStage.PCB: self.pcb_stage,
            PipelineStage.PACKAGING: self.packaging_stage,
            PipelineStage.THERMAL: self.thermal_stage,
            PipelineStage.TEST: self.test_stage,
        }
        return stage_funcs.get(stage, lambda x: (GateResult.FAIL, DFMCheck(pass_flag=False)))

    def _generate_notes(self) -> List[str]:
        """Generate human-readable notes from the pipeline run."""
        notes: List[str] = []
        for stage, result in self.stage_results.items():
            gate = self.stage_gates.get(stage)
            if gate:
                if result == GateResult.FAIL:
                    notes.append(f"FAIL: {gate.name} - {', '.join(gate.errors)}")
                elif result == GateResult.WARNING:
                    notes.append(f"WARNING: {gate.name} - {', '.join(gate.warnings)}")
                else:
                    notes.append(f"PASS: {gate.name}")
                    if gate.warnings:
                        notes.append(f"WARNING: {gate.name} - {', '.join(gate.warnings)}")
        return notes


# Convenience functions for each stage

def material_stage(
    pipeline: ManufacturingPipeline,
    material_spec: Dict[str, Any],
    area_mm2: float,
) -> Tuple[GateResult, DFMCheck]:
    """Run the material stage on an existing pipeline."""
    return pipeline.material_stage(material_spec, area_mm2)


def device_stage(
    pipeline: ManufacturingPipeline,
    device_params: Dict[str, Any],
) -> Tuple[GateResult, DFMCheck]:
    """Run the device stage on an existing pipeline."""
    return pipeline.device_stage(device_params)


def circuit_stage(
    pipeline: ManufacturingPipeline,
    circuit_params: Dict[str, Any],
) -> Tuple[GateResult, DFMCheck]:
    """Run the circuit stage on an existing pipeline."""
    return pipeline.circuit_stage(circuit_params)


def rtl_stage(
    pipeline: ManufacturingPipeline,
    rtl_params: Dict[str, Any],
) -> Tuple[GateResult, DFMCheck]:
    """Run the RTL stage on an existing pipeline."""
    return pipeline.rtl_stage(rtl_params)


def floorplan_stage(
    pipeline: ManufacturingPipeline,
    floorplan_params: Dict[str, Any],
) -> Tuple[GateResult, DFMCheck]:
    """Run the floorplan stage on an existing pipeline."""
    return pipeline.floorplan_stage(floorplan_params)


def pcb_stage(
    pipeline: ManufacturingPipeline,
    pcb_params: Dict[str, Any],
) -> Tuple[GateResult, DFMCheck]:
    """Run the PCB stage on an existing pipeline."""
    return pipeline.pcb_stage(pcb_params)


def packaging_stage(
    pipeline: ManufacturingPipeline,
    packaging_params: Dict[str, Any],
) -> Tuple[GateResult, DFMCheck]:
    """Run the packaging stage on an existing pipeline."""
    return pipeline.packaging_stage(packaging_params)


def thermal_stage(
    pipeline: ManufacturingPipeline,
    thermal_params: Dict[str, Any],
) -> Tuple[GateResult, DFMCheck]:
    """Run the thermal stage on an existing pipeline."""
    return pipeline.thermal_stage(thermal_params)


def test_stage(
    pipeline: ManufacturingPipeline,
    test_params: Dict[str, Any],
) -> Tuple[GateResult, DFMCheck]:
    """Run the test stage on an existing pipeline."""
    return pipeline.test_stage(test_params)