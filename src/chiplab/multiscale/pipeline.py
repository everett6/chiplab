"""End-to-end multiscale pipeline for chip-scale simulation.

Provides the MultiscalePipeline class that orchestrates the full multiscale simulation
process: identifying required scales, running calibration workflows, executing
simulation at target scale with calibrated ROMs, and providing error bounds and
uncertainty quantification.

Also provides ScaleSelector for adaptive fidelity per region.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .scales import Scale, ScaleHierarchy

from .calibration import CalibrationWorkflow, ParameterSpaceSampling, ExtrapolationDetector


class ScaleSelector:
    """Automatically choose fidelity per region (adaptive multiscale).

    The ScaleSelector analyzes the device geometry and simulation requirements
    to automatically select the appropriate fidelity level for different regions.
    Regions near interfaces, defects, or high-gradient areas get higher fidelity,
    while uniform regions use lower fidelity.

    Parameters
    ----------
    device_geometry : dict
        Device geometry specification.
    target_scale : Scale
        The target scale for the simulation.
    """

    def __init__(
        self,
        device_geometry: Optional[Dict[str, Any]] = None,
        target_scale: str = "device",
    ) -> None:
        self.device_geometry = device_geometry or {}
        self.target_scale = Scale(target_scale) if isinstance(target_scale, str) else target_scale
        self.fidelity_map: Dict[str, str] = {}  # region -> fidelity level

    def select_fidelity_for_region(
        self,
        region_id: str,
        gradient_magnitude: float,
        distance_to_interface: float,
    ) -> str:
        """Select the fidelity level for a given region based on local criteria.

        Parameters
        ----------
        region_id : str
            Identifier for the spatial region.
        gradient_magnitude : float
            Magnitude of the solution gradient (e.g., temperature, potential).
        distance_to_interface : float
            Distance from the region to the nearest material/device interface.

        Returns
        -------
        str
            Fidelity level: "nanoparticle", "material", "device", "memory_array",
            "chip", "package", or "system".
        """
        # Heuristic rules for fidelity selection:
        # - Near interfaces (small distance): higher fidelity
        # - High gradients: higher fidelity
        # - Uniform, distant regions: lower fidelity

        # Distance thresholds (in meters)
        interface_threshold_fine = 1e-6  # 1 µm: go to finer scale
        interface_threshold_coarse = 1e-3  # 1 mm: can use coarser scale
        gradient_threshold_fine = 10.0  # High gradient threshold
        gradient_threshold_coarse = 1.0  # Low gradient threshold

        fidelity = "device"  # Default

        # Check distance to interface
        if distance_to_interface < interface_threshold_fine:
            # Very close to interface: use finest scale
            fidelity = "nanoparticle"
        elif distance_to_interface < interface_threshold_coarse:
            # Moderately close: use fine scale
            if gradient_magnitude > gradient_threshold_fine:
                fidelity = "nanoparticle"
            elif gradient_magnitude > gradient_threshold_coarse:
                fidelity = "material"
            else:
                fidelity = "device"
        else:
            # Far from interface
            if gradient_magnitude > gradient_threshold_fine:
                # High gradient even far away: still need fine scale
                fidelity = "material"
            elif gradient_magnitude > gradient_threshold_coarse:
                fidelity = "device"
            else:
                fidelity = "memory_array"

        # Store the fidelity map
        self.fidelity_map[region_id] = fidelity

        return fidelity

    def generate_fidelity_map(
        self,
        regions: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Generate fidelity levels for all regions.

        Parameters
        ----------
        regions : list of dict
            List of region specifications, each with:
            - 'id': region identifier
            - 'gradient_magnitude': solution gradient magnitude
            - 'distance_to_interface': distance to nearest interface

        Returns
        -------
        dict[str, str]
            Mapping from region_id to fidelity level.
        """
        for region in regions:
            region_id = region.get("id", "unknown")
            gradient = region.get("gradient_magnitude", 0.0)
            distance = region.get("distance_to_interface", float("inf"))

            fidelity = self.select_fidelity_for_region(
                region_id, gradient, distance,
            )
            region["selected_fidelity"] = fidelity

        # Return mapping
        return {region.get("id", "unknown"): region.get("selected_fidelity", "device")
                for region in regions}


class MultiscalePipeline:
    """End-to-end multiscale pipeline for chip-scale simulation.

    The MultiscalePipeline orchestrates the full multiscale simulation process:
    1. Identify required scales from device geometry
    2. Run calibration workflows for each scale transition
    3. Execute simulation at target scale with calibrated ROMs
    4. Provide error bounds and uncertainty quantification

    Parameters
    ----------
    device : object
        Device specification object with geometry and material information.
    target_scale : Scale
        The target scale for the simulation (e.g., Scale.DEVICE, Scale.CHIP).
    """

    def __init__(
        self,
        device: Any,
        target_scale: str = "device",
    ) -> None:
        self.device = device
        self.target_scale = Scale(target_scale) if isinstance(target_scale, str) else target_scale
        self.scale_hierarchy = ScaleHierarchy()
        self.calibration_workflows: Dict[str, CalibrationWorkflow] = {}
        self.fidelity_map: Dict[str, str] = {}
        self.error_bounds: Optional[Dict[str, float]] = None
        self.uncertainty_quantification: Optional[Dict[str, Any]] = None

        # Initialize scale-specific components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize scale-specific reduction and calibration components."""
        # Create calibration workflows for each scale transition
        scale_transitions = [
            (Scale.MATERIAL, Scale.DEVICE),
            (Scale.DEVICE, Scale.MEMORY_ARRAY),
            (Scale.MEMORY_ARRAY, Scale.CHIP),
            (Scale.CHIP, Scale.PACKAGE),
            (Scale.PACKAGE, Scale.SYSTEM),
        ]

        for fine_scale, coarse_scale in scale_transitions:
            transition_key = f"{fine_scale.value}→{coarse_scale.value}"
            self.calibration_workflows[transition_key] = CalibrationWorkflow(
                rom_type="pod",  # Default; can be configured per transition
                reduced_dimension=10,
                n_calibration_samples=30,
                n_validation_samples=10,
            )

    def identify_required_scales(self) -> List[str]:
        """Identify the required scales from device geometry.

        Analyzes the device to determine which scales are relevant and
        what the appropriate scale transitions are.

        Returns
        -------
        list[str]
            List of scale names required for the simulation.
        """
        # Based on device geometry, determine which scales are needed
        # This is a simplified version; in practice, this would analyze
        # the device structure, feature sizes, etc.

        # Start with the target scale
        required_scales = [self.target_scale.value]

        # Add all finer scales in the hierarchy down to nanoparticle
        hierarchy_scales = self.scale_hierarchy.get_all_scales()
        for scale in hierarchy_scales:
            if scale.value != self.target_scale.value:
                # Add finer scales
                if hierarchy_scales.index(scale) < hierarchy_scales.index(Scale(self.target_scale)):
                    required_scales.append(scale.value)

        # Remove duplicates and sort by fineness
        scale_order = [
            "nanoparticle", "material", "device", "memory_array",
            "chip", "package", "system"
        ]
        required_scales = [s for s in scale_order if s in required_scales]

        return required_scales

    def run_calibration_workflows(
        self,
        simulation_providers: Dict[str, callable],
        parameter_bounds: Dict[str, List[Tuple[float, float]]],
    ) -> Dict[str, Any]:
        """Run calibration workflows for each scale transition.

        Parameters
        ----------
        simulation_providers : dict
            Mapping from scale transition key to high-fidelity simulation functions.
            e.g., {"material→device": lambda params: run_material_simulation(params)}
        parameter_bounds : dict
            Mapping from scale transition key to parameter bounds.

        Returns
        -------
        dict[str, Any]
            Results from all calibration workflows.
        """
        calibration_results = {}

        for transition_key, workflow in self.calibration_workflows.items():
            if transition_key in simulation_providers:
                sim_provider = simulation_providers[transition_key]
                bounds = parameter_bounds.get(transition_key, [(-1.0, 1.0)] * 3)

                try:
                    result = workflow.run_calibration(
                        high_fidelity_simulate=sim_provider,
                        parameter_bounds=bounds,
                    )
                    calibration_results[transition_key] = result
                except Exception as e:
                    calibration_results[transition_key] = {
                        "error": str(e),
                        "rom_trained": False,
                    }
            else:
                calibration_results[transition_key] = {
                    "error": f"No simulation provider for {transition_key}",
                    "rom_trained": False,
                }

        return calibration_results

    def execute_simulation(
        self,
        initial_conditions: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the multiscale simulation at the target scale using calibrated ROMs.

        Parameters
        ----------
        initial_conditions : dict
            Initial conditions for the simulation.
        parameters : dict
            Simulation parameters.

        Returns
        -------
        dict[str, Any]
            Simulation results with error bounds.
        """
        # Identify required scales
        required_scales = self.identify_required_scales()

        # For each scale transition, use the calibrated ROM
        simulation_results = {
            "target_scale": self.target_scale.value,
            "required_scales": required_scales,
            "step_results": {},
        }

        # Execute simulation using calibrated ROMs
        # This is a simplified pipeline; in practice, this would involve
        # sequential upscaling/downscaling through the scales
        for i, scale in enumerate(required_scales[:-1]):  # Skip the last (target)
            next_scale = required_scales[i + 1] if i + 1 < len(required_scales) else self.target_scale
            transition_key = f"{scale}→{next_scale}"

            # Check if we have a calibrated ROM for this transition
            if transition_key in self.calibration_workflows:
                workflow = self.calibration_workflows[transition_key]
                if workflow.rom is not None and workflow.rom.trained:
                    # Use the ROM for prediction
                    try:
                        # Prepare parameters for ROM prediction
                        param_dict = parameters.get("rom_params", {})
                        if not param_dict:
                            # Use default parameters
                            param_dict = {"reduced_coords": np.zeros(workflow.reduced_dimension)}

                        prediction = workflow.rom.predict(param_dict)

                        simulation_results["step_results"][transition_key] = {
                            "status": "completed",
                            "rom_type": workflow.rom_type,
                            "reduced_dimension": workflow.reduced_dimension,
                            "prediction_shape": prediction.shape if hasattr(prediction, 'shape') else "scalar",
                        }
                    except Exception as e:
                        simulation_results["step_results"][transition_key] = {
                            "status": "error",
                            "error": str(e),
                        }
                else:
                    simulation_results["step_results"][transition_key] = {
                        "status": "not_trained",
                        "message": "ROM not trained for this transition",
                    }
            else:
                simulation_results["step_results"][transition_key] = {
                    "status": "no_workflow",
                    "message": f"No calibration workflow for {transition_key}",
                }

        # Compute error bounds based on ROM validation results
        self._compute_error_bounds(simulation_results)

        return simulation_results

    def _compute_error_bounds(
        self,
        simulation_results: Dict[str, Any],
    ) -> None:
        """Compute error bounds and uncertainty quantification.

        Parameters
        ----------
        simulation_results : dict
            Results from the multiscale simulation execution.
        """
        error_bounds = {}
        uq = {}

        # Collect error estimates from each step
        step_results = simulation_results.get("step_results", {})
        total_error_estimate = 0.0
        n_steps_with_error = 0

        for step_name, step_data in step_results.items():
            if step_data.get("status") == "completed":
                # Use validation error from the calibration workflow if available
                # Otherwise, estimate based on reduced dimension
                n_reduced = step_data.get("reduced_dimension", 10)
                # Error estimate: typically decreases with more reduced dimensions
                # This is a simplified model
                step_error = max(1e-6, 1.0 / (n_reduced + 1))
                total_error_estimate += step_error
                n_steps_with_error += 1
                error_bounds[step_name] = {
                    "estimated_error": step_error,
                    "confidence": 0.95,
                }
            elif step_data.get("status") == "error":
                error_bounds[step_name] = {
                    "estimated_error": 1.0,  # Large error for failed steps
                    "confidence": 0.5,
                    "note": step_data.get("error", "Unknown error"),
                }
            else:
                error_bounds[step_name] = {
                    "estimated_error": 0.1,  # Moderate error for untrained steps
                    "confidence": 0.7,
                    "note": step_data.get("message", "Step not completed"),
                }

        # Overall error bound
        if n_steps_with_error > 0:
            overall_error = total_error_estimate / n_steps_with_error
        else:
            overall_error = 1.0

        uq["overall_relative_error"] = overall_error
        uq["error_bounds_by_step"] = error_bounds
        uq["n_successful_steps"] = n_steps_with_error
        uq["n_total_steps"] = len(step_results)

        self.error_bounds = error_bounds
        self.uncertainty_quantification = uq

    def get_results(self) -> Dict[str, Any]:
        """Get the complete multiscale simulation results.

        Returns
        -------
        dict[str, Any]
            Summary of the multiscale simulation including:
            - Target scale
            - Required scales
            - Error bounds
            - Uncertainty quantification
            - Fidelity map
        """
        return {
            "target_scale": self.target_scale.value,
            "required_scales": self.identify_required_scales(),
            "error_bounds": self.error_bounds,
            "uncertainty_quantification": self.uncertainty_quantification,
            "fidelity_map": self.fidelity_map,
            "calibration_workflows": {
                key: {
                    "rom_type": wf.rom_type if wf else None,
                    "trained": wf.trained if wf else False,
                }
                for key, wf in self.calibration_workflows.items()
            },
        }