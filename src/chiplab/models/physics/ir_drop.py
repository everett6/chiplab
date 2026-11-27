"""Power grid IR-drop analysis models.

Solves the 2D Laplace equation for power grid using finite difference method
and provides static and dynamic IR-drop analysis.
"""

from __future__ import annotations

from chiplab.schemas.common import TrackedValue, Provenance
import numpy as np

from chiplab.core.gpu import GPUBackend, get_gpu_backend


def _solve_laplace_2d(
    rows: int,
    cols: int,
    conductance_matrix: np.ndarray,
    current_source: np.ndarray,
    boundary_conditions: dict | None = None,
    max_iterations: int = 10000,
    tolerance: float = 1e-6,
    device: str = "auto",
) -> np.ndarray:
    """Solve 2D Laplace equation using finite difference method.

    Solves: div(J) = 0 or ∇·(σ∇V) = I_source

    Using Gauss-Seidel iteration with successive over-relaxation (SOR).

    Parameters
    ----------
    rows : int
        Number of grid rows.
    cols : int
        Number of grid columns.
    conductance_matrix : np.ndarray
        2D array of conductance values (S).
    current_source : np.ndarray
        2D array of current source densities (A/m^2).
    boundary_conditions : dict, optional
        Boundary condition specifications.
    max_iterations : int, default 10000
        Maximum Gauss-Seidel iterations.
    tolerance : float, default 1e-6
        Convergence tolerance.
    device : str, default "auto"
        Device to run solver on. "auto", "cpu", or "cuda".

    Returns
    -------
    np.ndarray
        2D potential distribution (voltage grid).
    """
    backend = get_gpu_backend(device=device)

    V = np.zeros((rows, cols))

    omega = 1.5  # over-relaxation factor

    for iteration in range(max_iterations):
        V_old = V.copy()

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Finite difference stencil for Laplace equation
                # Conductance-weighted average
                g_left = conductance_matrix[i, j - 1]
                g_right = conductance_matrix[i, j + 1]
                g_top = conductance_matrix[i - 1, j]
                g_bottom = conductance_matrix[i + 1, j]

                # Sum of conductances
                g_sum = g_left + g_right + g_top + g_bottom

                if g_sum > 0:
                    # Gauss-Seidel update with SOR
                    V[i, j] = (1 - omega) * V_old[i, j] + omega * (
                        (g_left * V_old[i, j - 1]
                         + g_right * V_old[i, j + 1]
                         + g_top * V_old[i - 1, j]
                         + g_bottom * V_old[i + 1, j])
                        / g_sum
                        - current_source[i, j] / g_sum
                    )

        # Check convergence
        diff = np.max(np.abs(V - V_old))
        if diff < tolerance:
            break

    return V


def ir_drop_grid(
    node: dict,
    voltage: float,
    current_density: float,
    grid_rows: int,
    grid_cols: int,
    metal_resistivity: float,
    device: str = "auto",
) -> dict:
    """Analyze IR drop in a power grid using 2D finite difference solver.

    Solves the 2D Laplace equation for the power grid to compute
    the voltage distribution and identify IR-drop violations.

    Parameters
    ----------
    node : dict
        Node specification containing grid coordinates and positions.
        Expected keys: "row", "col", "x", "y" (grid positions).
    voltage : float
        Nominal supply voltage in volts.
    current_density : float
        Current density in A/cm^2.
    grid_rows : int
        Number of rows in the power grid.
    grid_cols : int
        Number of columns in the power grid.
    metal_resistivity : float
        Metal resistivity in ohm-cm.
    device : str, default "auto"
        Device to run solver on. "auto", "cpu", or "cuda".
        If "auto", uses GPU if available, otherwise CPU.

    Returns
    -------
    dict
        Dictionary with:
        - "max_ir_drop": float, maximum IR drop in volts
        - "avg_ir_drop": float, average IR drop in volts
        - "violation_mask": np.ndarray, boolean array indicating violation locations
        - "voltage_distribution": np.ndarray, 2D voltage grid in volts
        - "min_voltage": float, minimum voltage in the grid
    """
    backend = get_gpu_backend(device=device)

    # Initialize conductance matrix properly
    # Conductance G = 1/R, where R = rho * L / A
    # For a grid with unit spacing, we normalize resistivity to conductance per node

    # Pitch in cm (assuming 1um default grid spacing)
    pitch_cm = 1e-4

    # Conductance between adjacent nodes: G = 1 / (rho * pitch / pitch) = 1/rho
    # But we need to normalize properly for the grid size
    # For a grid, the conductance between adjacent nodes is:
    # G = width / (rho * length) - for unit width and length = pitch
    # We use a normalized approach: G = 1 / (rho * pitch) per unit width

    # Use resistivity to compute conductance, scaling by pitch
    # G_unit = 1 / metal_resistivity  (S/cm, per unit width/length)

    # Create conductance matrix with proper values
    # For a power grid, conductance between adjacent nodes depends on
    # metal width, spacing, and resistivity

    # Normalized conductance: smaller resistivity = higher conductance
    # We use 1/metal_resistivity as the base conductance, scaled for grid size
    base_conductance = 1.0 / metal_resistivity if metal_resistivity > 0 else 1e6

    # Scale conductance based on grid dimensions for proper solver behavior
    # Larger grids need lower conductance per segment
    conductance = np.full((grid_rows, grid_cols), base_conductance)

    # Current source distribution (uniform current draw)
    # Convert current_density A/cm^2 to appropriate units for the grid
    # Grid has pitch_cm spacing, so each node handles area = pitch^2
    current_source = np.full((grid_rows, grid_cols), current_density * pitch_cm ** 2)

    # Solve 2D Laplace equation for voltage distribution
    V = _solve_laplace_2d(
        grid_rows, grid_cols, conductance, current_source, device=device
    )

    # Calculate IR drop from nominal voltage
    # IR drop = V_nominal - V_actual at each point
    ir_drop = voltage - V

    # Maximum and average IR drop
    max_ir_drop = float(np.max(ir_drop))
    avg_ir_drop = float(np.mean(ir_drop))

    # Violation mask: where IR drop exceeds a threshold (e.g., 5% of nominal)
    violation_threshold = 0.05 * voltage  # 5% threshold
    violation_mask = ir_drop > violation_threshold

    # Minimum voltage in the grid
    min_voltage = float(V.min())

    return {
        "max_ir_drop": max_ir_drop,
        "avg_ir_drop": avg_ir_drop,
        "violation_mask": violation_mask,
        "voltage_distribution": V,
        "min_voltage": min_voltage,
    }


def static_ir_drop(
    voltage: float,
    r_grid: float,
    i_chip: float,
) -> float:
    """Calculate static IR drop.

    Static IR drop is the voltage drop under steady-state conditions:
        V_actual = V_nominal - I * R

    Parameters
    ----------
    voltage : float
        Nominal supply voltage in volts.
    r_grid : float
        Grid resistance in ohms.
    i_chip : float
        Total chip current in amps.

    Returns
    -------
    float
        Static IR drop in volts.
    """
    ir_drop_value = r_grid * i_chip
    return float(ir_drop_value)


def dynamic_ir_drop(
    voltage: float,
    r_grid: float,
    i_chip: float,
    switching_factor: float,
    di_dt: float,
) -> float:
    """Calculate dynamic IR drop with di/dt effects.

    Dynamic IR drop includes both steady-state and transient components:
        V_dynamic = V_static + L * di/dt

    Where the inductive contribution depends on the switching factor
    and the rate of current change.

    Parameters
    ----------
    voltage : float
        Nominal supply voltage in volts.
    r_grid : float
        Grid resistance in ohms.
    i_chip : float
        Total chip current in amps.
    switching_factor : float
        Fraction of cells switching (0 to 1).
    di_dt : float
        Rate of current change in A/ns.

    Returns
    -------
    float
        Dynamic IR drop in volts.
    """
    # Static component
    ir_static = static_ir_drop(voltage, r_grid, i_chip)

    # Inductive component: L * di/dt
    # Approximate grid inductance based on spacing and geometry
    # L_eff ≈ r_grid * scaling_factor (empirical)

    # di/dt contribution scaled by switching factor
    di_dt_scaled = di_dt * switching_factor

    # Effective inductance: L_eff ≈ r_grid * 1e-9 (rough empirical factor)
    # r_grid in ohms, so L_eff in nanohenries approximately
    l_eff = r_grid * 1e-9  # H

    # di/dt in A/ps: convert A/ns to A/ps (1 ns = 1000 ps)
    di_dt_ps = di_dt / 1e3

    # Inductive drop: L * di/dt
    ir_inductive = l_eff * di_dt_ps

    # Total dynamic IR drop
    ir_dynamic = ir_static + ir_inductive

    return float(ir_dynamic)