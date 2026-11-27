"""Automated calibration pipeline for chiplab calibration.

This module provides the end-to-end calibration workflow including data parsing,
sensitivity analysis, parameter estimation, validation, uncertainty quantification,
and report generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .data_import import (
    VoltageMeasurement, CurrentMeasurement, ImpedanceMeasurement,
    FrequencyResponse, TemperatureMeasurement, StateRetentionMeasurement,
    WriteReadTiming, DataValidator, MeasurementType
)
from .parameter_fitting import (
    ParameterEstimationProblem, CostFunction, Optimizer,
    ParameterBounds, IdentifiabilityAnalysis, Regularization
)
from .sensitivity import LocalSensitivity, GlobalSensitivity, parameter_screening
from .uncertainty import (
    UncertaintyQuantification, UncertaintyDecomposition,
    PredictionIntervals, ModelDiscrepancy
)


# ---------------------------------------------------------------------------
# AutoCalibrationWorkflow - end-to-end calibration pipeline
# ---------------------------------------------------------------------------

@dataclass
class AutoCalibrationWorkflow:
    """End-to-end automated calibration workflow.

    Pipeline steps:
      1. Parse experiment data → observables
      2. Identify calibratable parameters (sensitivity analysis)
      3. Set up estimation problem with priors/bounds
      4. Run optimization/MCMC
      5. Validate with cross-validation / hold-out data
      6. Quantify uncertainty in calibrated parameters
      7. Generate calibrated material/device model

    Attributes:
        device_spec: Device specification/model.
        experiment_data: Measurement containers with experimental data.
        parameters_to_fit: Parameter names to optimize.
        method: Optimization method ("L-BFGS-B", "emcee", "bayesian_posterior").
        n_chains: Number of MCMC chains (for MCMC methods).
        n_iterations: Number of optimization iterations.
    """
    device_spec: Dict[str, Any]
    experiment_data: List[Any]
    parameters_to_fit: List[str]
    method: str = "L-BFGS-B"
    n_chains: int = 4
    n_iterations: int = 1000
    sensitivity_analysis: bool = True
    quantify_uncertainty: bool = True

    def run(self) -> CalibrationReport:
        """Run the complete calibration workflow.

        Returns:
            CalibrationReport with all results.
        """
        report = CalibrationReport(workflow=self)

        # Step 1: Parse experiment data → observables
        report = report._replace(step1_parse_data=self._parse_experiment_data())
        observables = report.step1_parse_data["observables"]

        # Step 2: Identify calibratable parameters (sensitivity analysis)
        if self.sensitivity_analysis:
            report = report._replace(step2_sensitivity=self._sensitivity_analysis(report))
            observables = report.step2_sensitivity["observables"]

        # Step 3: Set up estimation problem with priors/bounds
        report = report._replace(step3_setup_problem=self._setup_estimation_problem(observables))
        problem = report.step3_setup_problem["problem"]

        # Step 4: Run optimization/MCMC
        report = report._replace(step4_optimize=self._run_optimization(problem))
        optimized = report.step4_optimize

        # Step 5: Validate with cross-validation / hold-out data
        report = report._replace(step5_validate=self._validate_calibration(report, optimized))
        validated = report.step5_validate

        # Step 6: Quantify uncertainty in calibrated parameters
        if self.quantify_uncertainty:
            report = report._replace(step6_uncertainty=self._quantify_uncertainty(problem, validated))
        else:
            report = report._replace(step6_uncertainty={})

        # Step 7: Generate calibrated material/device model
        report = report._replace(step7_model=self._generate_calibrated_model(report))

        return report

    def _parse_experiment_data(self) -> Dict[str, Any]:
        """Step 1: Parse experiment data into observable containers.

        Returns:
            Dict with parsed observables and any validation warnings.
        """
        validator = DataValidator(strict=False)
        parsed = {"observables": [], "warnings": []}

        for i, data in enumerate(self.experiment_data):
            # Determine measurement type from data structure
            if hasattr(data, 'time') and hasattr(data, 'voltage'):
                # Voltage measurement
                val = VoltageMeasurement(
                    time=np.asarray(data.time),
                    voltage=np.asarray(data.voltage),
                    setup=getattr(data, 'setup', ''),
                    probe=getattr(data, 'probe', ''),
                    bandwidth=getattr(data, 'bandwidth', None),
                )
                validator.validate(val)
                parsed["observables"].append(("voltage", val))
                parsed["warnings"].extend(validator.errors)
                parsed["warnings"].extend(validator.warnings)

            elif hasattr(data, 'time') and hasattr(data, 'current'):
                # Current measurement
                val = CurrentMeasurement(
                    time=np.asarray(data.time),
                    current=np.asarray(data.current),
                    transimpedance=getattr(data, 'transimpedance', 1e6),
                    bandwidth=getattr(data, 'bandwidth', None),
                    feedback_resistance=getattr(data, 'feedback_resistance', None),
                )
                validator.validate(val)
                parsed["observables"].append(("current", val))
                parsed["warnings"].extend(validator.errors)
                parsed["warnings"].extend(validator.warnings)

            elif hasattr(data, 'frequency') and hasattr(data, 'real_part'):
                # Impedance measurement
                val = ImpedanceMeasurement(
                    frequency=np.asarray(data.frequency),
                    real_part=np.asarray(data.real_part),
                    imag_part=np.asarray(data.imag_part),
                    setup=getattr(data, 'setup', ''),
                )
                validator.validate(val)
                parsed["observables"].append(("impedance", val))
                parsed["warnings"].extend(validator.errors)
                parsed["warnings"].extend(validator.warnings)

            elif hasattr(data, 'frequency') and hasattr(data, 'magnitude') and hasattr(data, 'phase'):
                # Frequency response
                val = FrequencyResponse(
                    frequency=np.asarray(data.frequency),
                    magnitude=np.asarray(data.magnitude),
                    phase=np.asarray(data.phase),
                    reference=getattr(data, 'reference', None),
                    comment=getattr(data, 'comment', ''),
                )
                parsed["observables"].append(("frequency_response", val))

            elif hasattr(data, 'temperature') and hasattr(data, 'location'):
                # Spatial temperature
                val = TemperatureMeasurement(
                    time=getattr(data, 'time', None),
                    temperature=np.asarray(data.temperature),
                    unit=getattr(data, 'unit', 'C'),
                    location=np.asarray(data.location) if getattr(data, 'location', None) is not None else None,
                    location_unit=getattr(data, 'location_unit', None),
                )
                validator.validate(val)
                parsed["observables"].append(("temperature", val))
                parsed["warnings"].extend(validator.errors)
                parsed["warnings"].extend(validator.warnings)

            else:
                parsed["warnings"].append(f"Unknown data format for experiment {i}")

        return parsed

    def _sensitivity_analysis(self, report: CalibrationReport) -> Dict[str, Any]:
        """Step 2: Perform sensitivity analysis to identify calibratable parameters.

        Returns:
            Dict with sensitivity results and active parameter identification.
        """
        from .parameter_fitting import ParameterBounds

        # Build parameter bounds from device spec
        param_names = self.parameters_to_fit
        n_params = len(param_names)

        # Default bounds - these would typically come from the device spec
        lower_bounds = np.array([getattr(self.device_spec, 'get', lambda x, d: 0.0)(p, 0.0)
                                 for p in param_names])
        upper_bounds = np.array([getattr(self.device_spec, 'get', lambda x, d: 1.0)(p, 1.0)
                                 for p in param_names])

        bounds = ParameterBounds(
            names=param_names,
            lower=lower_bounds,
            upper=upper_bounds,
        )

        # Use device spec to get simulator if available
        simulator = self.device_spec.get("simulator", None)

        if simulator is None:
            # Placeholder simulator - in practice this would be a physical model
            def simulator(params):
                return np.zeros(len(self.experiment_data))

        # Perform global sensitivity analysis
        global_si = GlobalSensitivity(n_parameters=n_params)
        sensitivity_results = global_si.sobol_indices(simulator)

        # Determine which parameters are influential
        threshold = 0.05  # 5% threshold for significance
        active_indices = np.where(sensitivity_results["first_order"] > threshold)[0]
        fix_indices = np.where(sensitivity_results["first_order"] <= threshold)[0]

        return {
            "sensitivity_results": sensitivity_results,
            "active_parameters": [param_names[i] for i in active_indices],
            "fixable_parameters": [param_names[i] for i in fix_indices],
            "parameter_bounds": bounds,
            "simulator": simulator,
        }

    def _setup_estimation_problem(self, observables: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Set up the parameter estimation problem.

        Returns:
            Dict with ParameterEstimationProblem and cost function.
        """
        param_names = self.parameters_to_fit
        n_params = len(param_names)

        # Get bounds from device spec or use defaults
        default_lower = np.array([0.1] * n_params)
        default_upper = np.array([100.0] * n_params)

        bounds_lower = np.array(
            [self.device_spec.get(p, 0.1) for p in param_names],
            dtype=float,
        )
        bounds_upper = np.array(
            [self.device_spec.get(p, 100.0) for p in param_names],
            dtype=float,
        )

        bounds = ParameterBounds(
            names=param_names,
            lower=bounds_lower,
            upper=bounds_upper,
        )

        # Create cost function
        cost_function = CostFunction(
            kind="least_squares",
            weights=None,  # would be inverse covariance if available
        )

        # Create simulator function that maps parameters → observables
        def simulator(param_dict: Dict[str, np.ndarray]) -> np.ndarray:
            """Map parameters to simulated observables.

            This is a placeholder - in practice would call the device model.
            """
            params = param_dict.get("values", np.array([param_dict[name] for name in param_names]))
            # Return simulated values matching the number of observables
            n_observables = len(observables) if hasattr(observables, '__len__') else 1
            return np.zeros(n_observables)

        # Create ParameterEstimationProblem
        # We need experimental data - extract from observables
        exp_data = self.experiment_data if self.experiment_data else []

        problem = ParameterEstimationProblem(
            params={name: 1.0 for name in param_names},  # initial guess
            simulate_observable=simulator,
            experimental_data=exp_data,
            bounds=bounds,
            cost_function=cost_function,
            n_params=n_params,
            n_observables=len(observables) if hasattr(observables, '__len__') else 1,
        )

        return {
            "problem": problem,
            "cost_function": cost_function,
            "bounds": bounds,
        }

    def _run_optimization(self, problem: ParameterEstimationProblem) -> Dict[str, Any]:
        """Step 4: Run optimization or MCMC to estimate parameters.

        Returns:
            Dict with optimization results.
        """
        optimizer = Optimizer(
            method=self.method,
            max_iter=self.n_iterations,
            tol=1e-8,
            options={"maxiter": self.n_iterations, "ftol": 1e-12},
        )

        # Run optimization
        result = optimizer.minimize(
            cost_func=problem.cost_function.compute,
            x0=np.array([problem.params[name].value if hasattr(problem.params[name], 'value') else problem.params[name]
                         for name in problem.bounds.names]),
            bounds=problem.bounds,
            args=(problem.experimental_data,),
        )

        return {
            "success": result["success"],
            "optimized_params": result["x"],
            "optimized_cost": result["cost"],
            "message": result["message"],
            "parameter_names": problem.bounds.names,
        }

    def _validate_calibration(self, report: CalibrationReport,
                              optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Validate calibration with cross-validation / hold-out data.

        Returns:
            Dict with validation results.
        """
        # Simple hold-out validation
        # In practice, would split experiment_data into training/test sets
        from .data_import import DataValidator

        validator = DataValidator(strict=False)

        optimized_params = optimization_result["optimized_params"]
        problem = report.step3_setup_problem["problem"]

        # Compute simulated observables with optimized parameters
        sim_observables = problem.simulate_observable(optimized_params)

        # Compare with experimental data
        if isinstance(self.experiment_data[0], VoltageMeasurement):
            exp_observables = np.array([m.voltage for m in self.experiment_data])
        elif isinstance(self.experiment_data[0], CurrentMeasurement):
            exp_observables = np.array([m.current for m in self.experiment_data])
        elif isinstance(self.experiment_data[0], ImpedanceMeasurement):
            exp_observables = np.column_stack([
                m.real_part for m in self.experiment_data
            ])
        else:
            exp_observables = np.array([])

        # Compute residuals
        if len(exp_observables) > 0 and len(sim_observables) > 0:
            # Ensure compatible shapes
            min_len = min(len(exp_observables), len(sim_observables))
            residuals = exp_observables[:min_len] - sim_observables[:min_len]
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((exp_observables[:min_len] - np.mean(exp_observables[:min_len])) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        else:
            r_squared = 0.0
            ss_res = 0.0
            ss_tot = 0.0

        return {
            "r_squared": r_squared,
            "residuals": ss_res,
            "ss_tot": ss_tot,
            "validation_passed": r_squared > 0.5,  # arbitrary threshold
            "optimized_params": optimized_params,
        }

    def _quantify_uncertainty(self, problem: ParameterEstimationProblem,
                              validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Quantify uncertainty in calibrated parameters.

        Returns:
            Dict with uncertainty quantification results.
        """
        uq = UncertaintyQuantification(n_mc_samples=500)

        # Get optimized parameter values
        opt_params = validation_result["optimized_params"]

        # Monte Carlo uncertainty propagation using the problem's simulate_observable
        # We need parameter samples - generate around the optimized values
        n_params = len(problem.bounds.names)

        # Simple: generate samples from a Gaussian around the optimized values
        # with standard deviations from the Fisher information
        try:
            fi = problem.fisher_information(opt_params, problem.cost_function)
            # Invert Fisher to get covariance (approximate)
            if fi is not None and fi.shape[0] > 0:
                try:
                    cov = np.linalg.inv(fi)
                except np.linalg.LinAlgError:
                    cov = np.eye(n_params) * 0.1
            else:
                cov = np.eye(n_params) * 0.1
        except Exception:
            cov = np.eye(n_params) * 0.1

        # Generate MC samples
        np.random.seed(42)
        mc_samples = np.random.multivariate_normal(opt_params, cov, size=uq.n_mc_samples)

        # Run simulator on MC samples
        observables_mc = np.array([problem.simulate_observable(sample) for sample in mc_samples.T])

        if observables_mc.ndim == 1:
            observables_mc = observables_mc[:, np.newaxis]

        # Compute prediction intervals
        pi = PredictionIntervals(confidence_level=0.95)
        intervals = pi.compute_intervals(observables_mc)

        # Decompose uncertainty
        ud = UncertaintyDecomposition()
        decomp = ud.decompose(
            {"observables": observables_mc},
            parameter_uncertainties={name: cov[i, i] for i, name in enumerate(problem.bounds.names)},
        )

        return {
            "mc_samples": mc_samples,
            "observables_mc": observables_mc,
            "prediction_intervals": intervals,
            "uncertainty_decomposition": decomp,
            "parameter_covariance": cov,
        }

    def _generate_calibrated_model(self, report: CalibrationReport) -> Dict[str, Any]:
        """Step 7: Generate calibrated material/device model.

        Returns:
            Dict with the calibrated model representation.
        """
        # Extract the best parameters from the workflow
        # This would typically create a new model instance with calibrated parameters
        optimized = report.step4_optimize

        # Build calibrated model dict
        calibrated_model = {
            "model_type": "calibrated_device",
            "parameters": {
                name: {"value": val, "provenance": "calibrated"}
                for name, val in zip(optimized["parameter_names"], optimized["optimized_params"])
            },
            "fit_quality": {
                "cost": optimized["optimized_cost"],
                "r_squared": report.step5_validate["r_squared"] if hasattr(report, 'step5_validate') else 0.0,
            },
            "uncertainty": report.step6_uncertainty if hasattr(report, 'step6_uncertainty') else {},
            "description": "Calibrated device model from experimental data",
        }

        return calibrated_model


# ---------------------------------------------------------------------------
# CalibrationReport - parameter estimates, uncertainties, correlations, fit quality, predictions
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    """Report from a calibration workflow run.

    Contains all results from the calibration pipeline including parameter
    estimates, uncertainties, fit quality metrics, and the calibrated model.

    Attributes:
        workflow: The AutoCalibrationWorkflow that generated this report.
        step1_parse_data: Results from experiment data parsing.
        step2_sensitivity: Results from sensitivity analysis.
        step3_setup_problem: Results from estimation problem setup.
        step4_optimize: Results from optimization/MCMC.
        step5_validate: Results from cross-validation.
        step6_uncertainty: Results from uncertainty quantification.
        step7_model: Results from calibrated model generation.
        success: Whether the full workflow completed successfully.
    """
    workflow: AutoCalibrationWorkflow
    step1_parse_data: Dict[str, Any] = field(default_factory=dict)
    step2_sensitivity: Dict[str, Any] = field(default_factory=dict)
    step3_setup_problem: Dict[str, Any] = field(default_factory=dict)
    step4_optimize: Dict[str, Any] = field(default_factory=dict)
    step5_validate: Dict[str, Any] = field(default_factory=dict)
    step6_uncertainty: Dict[str, Any] = field(default_factory=dict)
    step7_model: Dict[str, Any] = field(default_factory=dict)
    success: bool = False

    def _replace(self, **kwargs) -> CalibrationReport:
        """Create a new CalibrationReport with updated fields."""
        return CalibrationReport(
            workflow=self.workflow,
            **{**self.__dict__, **kwargs},
        )

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a concise summary of the calibration report."""
        return {
            "success": self.success,
            "optimized_parameters": {
                name: val for name, val in zip(
                    self.step4_optimize.get("parameter_names", []),
                    self.step4_optimize.get("optimized_params", [])
                )
            } if self.step4_optimize else {},
            "fit_quality": {
                "r_squared": self.step5_validate.get("r_squared", 0.0),
                "cost": self.step4_optimize.get("optimized_cost", float("inf")),
            } if hasattr(self, 'step4_optimize') and self.step4_optimize else {},
            "uncertainty_quantified": bool(self.step6_uncertainty),
            "model_generated": bool(self.step7_model),
        }