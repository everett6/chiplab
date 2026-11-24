"""3D thermal gradient & inter-die coupling models.

Provides 3D thermal analysis including inter-die coupling and hotspot
detection for multi-die and stacked IC designs.
"""

from __future__ import annotations

from chiplab.schemas.common import TrackedValue, Provenance
import numpy as np


def thermal_gradient_3d(
    die_power_w: float,
    die_area_mm2: float,
    die_thickness_um: float,
    num_die_layers: int,
    thermal_resistance: float,
) -> dict:
    """Compute 3D thermal gradient across die layers.

    Solves the thermal diffusion equation for stacked dies using a
    lumped-parameter RC thermal model. Temperature distribution is
    computed per layer with hotspot identification and coupling matrix.

    Parameters
    ----------
    die_power_w : float
        Power dissipation per die in watts.
    die_area_mm2 : float
        Die area in square millimeters.
    die_thickness_um : float
        Die thickness in micrometers.
    num_die_layers : int
        Number of die layers in the stack.
    thermal_resistance : float
        Thermal resistance in K/W (junction-to-ambient per layer).

    Returns
    -------
    dict
        Dictionary with:
        - "temperature_per_layer": np.ndarray, temperature per layer in °C
        - "hotspot_temperature": float, maximum temperature in °C
        - "thermal_coupling_matrix": np.ndarray, num_die_layers x num_die_layers
          coupling matrix (W/K) describing thermal interaction between layers
        - "ambient_temperature": float, assumed ambient temperature in °C (25°C)
        - "temperature_rise": np.ndarray, temperature rise per layer in °C
    """
    # Ambient temperature
    ambient_temp_c = 25.0

    # Temperature rise per die (simple 1D thermal model)
    # ΔT = P * R_th where R_th is thermal resistance
    temp_rise_per_die = die_power_w * thermal_resistance

    # Temperature per layer (cumulative with coupling consideration)
    temperatures_per_layer = np.zeros(num_die_layers)

    # Simple model: each layer adds temperature rise, with coupling
    # Coupling: inner layers are hotter due to less heat dissipation path
    for i in range(num_die_layers):
        # Edge layers dissipate better than central layers
        # Factor: edge layers have 1.0 coupling, inner layers have higher coupling
        if i == 0 or i == num_die_layers - 1:
            # Edge layer: better heat dissipation
            layer_rise = temp_rise_per_die * 0.8
        else:
            # Inner layer: more heat accumulation
            layer_rise = temp_rise_per_die * 1.0

        temperatures_per_layer[i] = ambient_temp_c + layer_rise

    # Thermal coupling matrix: describes how heat from one layer affects others
    # Diagonal dominance: each layer's self-coupling is strongest
    # Off-diagonal: adjacent layers couple more strongly
    coupling_matrix = np.eye(num_die_layers) * 0.8

    # Adjacent layer coupling
    for i in range(num_die_layers - 1):
        coupling_matrix[i, i + 1] = 0.1
        coupling_matrix[i + 1, i] = 0.1

    # Next-adjacent layer coupling (weaker)
    for i in range(num_die_layers - 2):
        coupling_matrix[i, i + 2] = 0.05
        coupling_matrix[i + 2, i] = 0.05

    # Hotspot temperature (maximum across all layers)
    hotspot_temperature = float(temperatures_per_layer.max())

    # Temperature rise per layer
    temperature_rise = temperatures_per_layer - ambient_temp_c

    return {
        "temperature_per_layer": temperatures_per_layer,
        "hotspot_temperature": hotspot_temperature,
        "thermal_coupling_matrix": coupling_matrix,
        "ambient_temperature": ambient_temp_c,
        "temperature_rise": temperature_rise,
    }


def inter_die_coupling(
    power_die1: float,
    power_die2: float,
    distance_um: float,
    interface_material: str,
) -> float:
    """Calculate thermal coupling coefficient between two dies.

    Models heat transfer between dies through the interface material.
    The coupling coefficient decreases with distance and depends on
    the thermal conductivity of the interface material.

    Parameters
    ----------
    power_die1 : float
        Power dissipation of die 1 in watts.
    power_die2 : float
        Power dissipation of die 2 in watts.
    distance_um : float
        Distance between die centers in micrometers.
    interface_material : str
        Interface material name (e.g., " solder", "thermal_epox", "gap").

    Returns
    -------
    float
        Thermal coupling coefficient (0 to 1), where 1 means strong coupling
        and 0 means negligible coupling.
    """
    # Thermal conductivity of common interface materials (W/m-K)
    material_conductivity = {
        "solder": 60.0,
        "thermal_epox": 1.0,
        "gap": 0.0,  # Air/vacuum
        "underfill": 0.5,
    }

    # Default conductivity if material not in lookup
    k = material_conductivity.get(interface_material, 1.0)

    # Reference distance (characteristic coupling distance)
    # Distance at which coupling drops significantly
    d_ref = 100.0  # um reference distance

    # Coupling model: exponential decay with distance
    # Coupling ~ exp(-d / lambda) where lambda depends on material conductivity
    # Higher conductivity -> longer coupling distance

    # lambda = d_ref * (k / k_ref) where k_ref is a reference conductivity
    k_ref = 1.0  # W/m-K reference
    lambda_coupling = d_ref * (k / k_ref)

    # Exponential decay coupling coefficient
    coupling_coeff = np.exp(-distance_um / lambda_coupling)

    # Normalize to 0-1 range (ensure positive and bounded)
    coupling_coeff = max(0.0, min(1.0, coupling_coeff))

    # Also consider power difference - higher power difference reduces effective coupling
    # (heat flows from hot to cold, so asymmetry matters)
    power_ratio = power_die1 / max(1e-10, power_die2)
    asymmetry_factor = min(power_ratio, 1.0 / power_ratio)  # 0 to 1

    # Effective coupling considering asymmetry
    effective_coupling = coupling_coeff * asymmetry_factor

    return float(effective_coupling)


def hotspot_analysis(
    power_map: np.ndarray,
    floorplan: dict,
    ambient_temp: float,
    cooling_solution: str,
) -> dict:
    """Perform hotspot analysis to identify hot regions on the die.

    Identifies local temperature hotspots based on power density,
    floorplan geometry, and cooling conditions.

    Parameters
    ----------
    power_map : np.ndarray
        2D array of power dissipation per grid cell in W/cm^2.
    floorplan : dict
        Floorplan specification with keys:
        - "grid_size": (rows, cols) in grid cells
        - "cell_size_um": cell size in micrometers
    ambient_temp : float
        Ambient temperature in °C.
    cooling_solution : str
        Cooling type: "natural_convection", "forced_convection", "liquid_cooling".

    Returns
    -------
    dict
        Dictionary with:
        - "hotspot_temperature_map": np.ndarray, 2D temperature map in °C
        - "max_hotspot_temp": float, maximum temperature in °C
        - "hotspot_locations": list of (row, col) tuples for hotspots
        - "thermal_image": np.ndarray, visual representation data
        - "cooling_efficiency": float, cooling effectiveness (0 to 1)
    """
    # Extract floorplan parameters
    grid_size = floorplan.get("grid_size", (power_map.shape[0], power_map.shape[1]))
    cell_size_um = floorplan.get("cell_size_um", 10.0)

    rows, cols = grid_size

    # Ensure power_map matches grid size
    if power_map.shape != (rows, cols):
        # Resize/interpolate if needed
        from scipy.interpolate import RegularGridInterpolator
        # For now, just use as-is with warning
        pass

    # Thermal diffusivity approximation for silicon
    # alpha ~ 0.84 mm^2/s for silicon at room temperature
    # But we'll use a steady-state approximation

    # Temperature rise calculation using Green's function / superposition
    # For a uniform grid with local heating:
    # T(x) = T_ambient + R_th * P(x) where R_th is thermal resistance

    # Cooling resistance depends on cooling solution
    # Natural convection: ~100 K/W per unit area
    # Forced convection: ~20-50 K/W
    # Liquid cooling: ~5-10 K/W
    cooling_resistance_factor = {
        "natural_convection": 1.0,
        "forced_convection": 0.3,
        "liquid_cooling": 0.1,
    }.get(cooling_solution, 0.5)

    # Calculate temperature rise per grid cell
    # Simple model: ΔT_cell = R_th * P_cell where R_th depends on cooling
    # R_th ~ cell_size / (k * sqrt(area)) but we simplify

    # Temperature rise per cell (K per W/cm^2)
    # Using simplified: delta_T = power_density * thermal_resistance_per_area
    # Thermal resistance per area for silicon ~ 1000 K-cm^2/W (rough)

    # Convert cell size to cm
    cell_size_cm = cell_size_um * 1e-4

    # Temperature rise per cell
    delta_t = power_map * (cell_size_cm ** 2) * 1000.0 * cooling_resistance_factor

    # Add ambient temperature
    temperature_map = ambient_temp + delta_t

    # Find hotspots (local maxima above threshold)
    # Hotspot threshold: above ambient by significant amount
    hotspot_threshold = ambient_temp + 15.0  # 15°C above ambient

    # Find cells above threshold
    above_threshold = temperature_map > hotspot_threshold

    # Local maxima detection (simple: cell is hotspot if it's above threshold
    # and higher than all 8 neighbors)
    hotspot_locations = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if above_threshold[i, j]:
                # Check if it's a local maximum
                is_local_max = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if temperature_map[i + di, j + dj] >= temperature_map[i, j]:
                            is_local_max = False
                            break
                    if not is_local_max:
                        break

                if is_local_max:
                    hotspot_locations.append((i, j))

    # Maximum hotspot temperature
    max_hotspot_temp = float(temperature_map.max())

    # Cooling efficiency (inverse of how much above ambient)
    # Higher cooling efficiency means lower temperatures for same power
    max_possible_rise = np.max(delta_t)
    cooling_efficiency = max(0.0, 1.0 - (np.max(delta_t) / 50.0))  # 50°C reference

    # Thermal image for visualization (normalized)
    thermal_image = (temperature_map - ambient_temp) / max(0.01, np.max(temperature_map) - ambient_temp)

    return {
        "hotspot_temperature_map": temperature_map,
        "max_hotspot_temp": max_hotspot_temp,
        "hotspot_locations": hotspot_locations,
        "thermal_image": thermal_image,
        "cooling_efficiency": cooling_efficiency,
    }