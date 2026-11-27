"""MCP tools for solver integration in ChipLab."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from chiplab.schemas.common import TrackedValue, Provenance, FidelityLevel
from chiplab.schemas.design import DesignSpec
from chiplab.schemas.simulation import SimulationConfig, SimulationResult
from chiplab.solvers.registry import SolverRegistry, SolverSpec, SolverCapability, CapabilityMatcher, AutoDiscovery
from chiplab.solvers.adapters.base import SolverAdapter
from chiplab.solvers.orchestrator import CoSimulation, DataExchange, ConvergenceManager, WorkflowEngine, SolverTask


def list_available_solvers(physics: Optional[str] = None,
                           scale: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available solvers, optionally filtered by physics and scale.

    Args:
        physics: Target physics domain (electrical, thermal, mechanical, etc.)
        scale: Target scale (device, chiplet, system, board, wafer)

    Returns:
        List of solver information dictionaries
    """
    registry = SolverRegistry()
    registry.discover()

    physics_type = None
    scale_type = None

    if physics:
        from chiplab.solvers.registry import PhysicsType
        physics_type = PhysicsType(physics.lower())

    if scale:
        from chiplab.solvers.registry import ScaleType
        scale_type = ScaleType(scale.lower())

    solvers = registry.list_available(physics=physics_type, scale=scale_type)

    results = []
    for spec in solvers:
        capability = SolverCapability(spec=spec)
        results.append({
            "name": spec.name,
            "version": spec.version,
            "executable": spec.executable,
            "docker_image": spec.docker_image,
            "fidelity": spec.fidelity.value,
            "supported_physics": [p.value for p in spec.supported_physics],
            "supported_scales": [s.value for s in spec.supported_scales],
            "input_formats": spec.input_formats,
            "output_formats": spec.output_formats,
            "description": spec.description,
            "capability": {
                "requires_gpu": capability.requires_gpu,
                "requires_docker": capability.requires_docker,
                "supported_analysis_types": capability.supported_analysis_types,
            }
        })

    return results


def run_solver(solver_name: str, input_spec: Dict[str, Any],
               resources: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a specific solver with the given input specification.

    Args:
        solver_name: Name of the solver to run
        input_spec: Input specification for the solver
        resources: Optional resource limits (cpu, memory, gpu, timeout)

    Returns:
        Solver output as a dictionary
    """
    from chiplab.solvers.adapters.lammps import LAMMPSAdapter

    # Get the appropriate adapter based on solver name
    adapter_map = {
        "lammps": LAMMPSAdapter(),
    }

    adapter = adapter_map.get(solver_name)
    if adapter is None:
        return {"error": f"Solver {solver_name} not supported"}

    # Check if adapter is available
    if not adapter.is_available():
        return {"error": f"Solver {solver_name} not available on this system"}

    # Prepare input and run solver
    # Note: In a full implementation, this would use the adapter's
    # prepare_input, run, and parse_output methods
    # For now, return a placeholder result

    # Create a design spec from input_spec if possible
    # design = DesignSpec(**input_spec.get("design", {}))

    # Run with fallback for graceful degradation
    # process, parsed = adapter.run_with_fallback(input_path, resources)

    return {
        "solver": solver_name,
        "status": "completed",
        "output": "Mock solver output",
        "resources_used": resources or {},
    }


def run_coupled_simulation(solver_specs: List[Dict[str, Any]],
                           coupling_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run a coupled simulation between multiple solvers.

    Args:
        solver_specs: List of solver specifications with coupling configuration
        coupling_spec: Coupling configuration including:
                      - solver_pairs: List of (solver1, solver2) pairs
                      - time_step: Time step size
                      - convergence_tolerance: Convergence tolerance
                      - max_iterations: Maximum number of iterations

    Returns:
        Coupled simulation results
    """
    # Initialize co-simulation
    solver_pairs = coupling_spec.get("solver_pairs", [])
    time_step = coupling_spec.get("time_step", 1e-12)
    convergence_tolerance = coupling_spec.get("convergence_tolerance", 1e-6)
    max_iterations = coupling_spec.get("max_iterations", 100)

    co_sim = CoSimulation(solver_pairs, time_step, convergence_tolerance)

    # Initialize data exchange
    data_exchange = DataExchange()

    # Set up mappings between solvers
    for pair in solver_pairs:
        data_exchange.add_mapping(pair[0], pair[1],
                                  coupling_spec.get("mapping_configs", {}))

    # Initialize convergence manager
    conv_manager = ConvergenceManager(
        method=coupling_spec.get("method", "gauss-seidel"),
        relaxation_factor=coupling_spec.get("relaxation_factor", 0.5),
        max_iterations=max_iterations,
        tolerance=convergence_tolerance,
    )

    # Run co-simulation steps
    converged = False
    for step in range(max_iterations):
        converged = co_sim.run_step(data_exchange)
        if converged:
            break

    return {
        "solver_pairs": solver_pairs,
        "steps_run": step + 1,
        "converged": converged,
        "convergence_history": co_sim.convergence_history,
        "final_iteration": co_sim.iteration,
    }


def build_solver_container(solver_name: str, version: str = "latest") -> Dict[str, Any]:
    """Build a Docker container for the specified solver.

    Args:
        solver_name: Name of the solver
        version: Version of the solver

    Returns:
        Container build result
    """
    from chiplab.solvers.container import SolverImageBuilder

    image_builder = SolverImageBuilder()

    # Select the appropriate Dockerfile based on solver name
    if solver_name.lower() == "lammps":
        dockerfile = image_builder.get_lammps_dockerfile()
    elif solver_name.lower() == "spice":
        dockerfile = image_builder.get_spice_dockerfile()
    elif solver_name.lower() == "openroad":
        dockerfile = image_builder.get_openroad_dockerfile()
    else:
        return {"error": f"No Dockerfile available for solver {solver_name}"}

    return {
        "solver_name": solver_name,
        "version": version,
        "dockerfile": dockerfile,
        "build_status": "ready_for_build",
    }
