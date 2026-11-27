"""Upscaling from fine to coarse scales for multiscale simulation.

Provides upscaling methods for transitioning from fine-scale (nanoparticle, material)
representations to coarse-scale (device, chip, package, system) models.

Key methods:
- Homogenization: periodic/statistical microstructure → effective properties
- Volume Averaging: field averaging over RVE
- Asymptotic Expansion: ε→0 for scale separation
- Numerical Homogenization: solve cell problems on RVE mesh
- Effective Property Calculator: compute σ_eff, ε_eff, μ_eff, k_eff, η_eff
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Homogenization:
    """Homogenization for periodic/statistical microstructure → effective properties.

    Computes effective (homogenized) continuum properties from a detailed microstructure.
    Supports both periodic boundary conditions (for regular microstructures) and
    statistical averaging (for random microstructures).

    Parameters
    ----------
    rve_size : tuple[float, float, float]
        RVE dimensions (L, W, H) in meters.
    volume_fraction : float
        Volume fraction of the inclusions/phases.
    bc_type : str
        Boundary condition type: "periodic", "dirichlet", or "neumann".
    """

    def __init__(
        self,
        rve_size: Tuple[float, float, float] = (1e-6, 1e-6, 1e-6),
        volume_fraction: float = 0.5,
        bc_type: str = "periodic",
    ) -> None:
        self.rve_size = rve_size
        self.volume_fraction = volume_fraction
        self.bc_type = bc_type

    def compute_effective_conductivity(
        self, phase_conductivities: Tuple[float, float], geometry: Dict[str, Any]
    ) -> float:
        """Compute effective electrical/thermal conductivity using homogenization.

        Parameters
        ----------
        phase_conductivities : tuple of float
            (matrix_conductivity, inclusion_conductivity) in S/m or W/(m·K).
        geometry : dict
            Geometry parameters describing the microstructure.

        Returns
        -------
        float
            Effective conductivity in the same units.
        """
        k_m, k_i = phase_conductivities
        phi = self.volume_fraction  # inclusion volume fraction

        if self.bc_type == "periodic":
            # Hashin-Shtrikman bounds or self-consistent scheme
            # For spherical inclusions in 3D (periodic):
            if k_m == 0:
                return k_i
            if k_i == 0:
                return k_m

            # Hashin-Shtrikman upper bound (inclusions more conductive)
            if k_i > k_m:
                upper = k_i + phi * (k_m * (1 - phi)) / (k_m + 2 * k_i * (1 - phi) / 3)
                return upper
            else:
                # Hashin-Shtrikman lower bound
                lower = k_m + (1 - phi) * (k_i * (1 - k_m / k_i)) / (k_i + 2 * k_m * (1 - phi) / 3)
                return lower
        else:
            # Simple rule of mixtures for non-periodic
            # Voigt (upper) bound: k_eff = phi*k_i + (1-phi)*k_m
            # Reuss (lower) bound: 1/k_eff = phi/k_i + (1-phi)/k_m
            k_voigt = phi * k_i + (1 - phi) * k_m
            k_reuss = 1.0 / (phi / k_i + (1 - phi) / k_m) if k_m > 0 and k_i > 0 else max(k_m, k_i)
            return (k_voigt + k_reuss) / 2.0  # Average of bounds

    def compute_effective_permittivity(
        self, phase_permittivities: Tuple[float, float], geometry: Dict[str, Any]
    ) -> float:
        """Compute effective permittivity using homogenization.

        Parameters
        ----------
        phase_permittivities : tuple of float
            (matrix_permittivity, inclusion_permittivity) in F/m.
        geometry : dict
            Geometry parameters.

        Returns
        -------
        float
            Effective permittivity in F/m.
        """
        eps_m, eps_i = phase_permittivities
        phi = self.volume_fraction

        # Maxwell-Garnett effective medium approximation for spherical inclusions
        if eps_m == 0:
            return eps_i
        if eps_i == 0:
            return eps_m

        # MG formula: eps_eff = eps_m * (1 + 3*phi*(eps_i-eps_m)/(eps_i+2*eps_m-phi*(eps_i-eps_m)))
        numerator = 3 * phi * (eps_i - eps_m)
        denominator = eps_i + 2 * eps_m - phi * (eps_i - eps_m)
        if abs(denominator) < 1e-15:
            return eps_m
        eps_eff = eps_m * (1 + numerator / denominator)
        return eps_eff

    def compute_effective_viscosity(
        self, phase_viscosities: Tuple[float, float], geometry: Dict[str, Any]
    ) -> float:
        """Compute effective viscosity using homogenization.

        Parameters
        ----------
        phase_viscosities : tuple of float
            (matrix_viscosity, inclusion_viscosity) in Pa·s.
        geometry : dict
            Geometry parameters.

        Returns
        -------
        float
            Effective viscosity in Pa·s.
        """
        eta_m, eta_i = phase_viscosities
        phi = self.volume_fraction

        # Einstein-Rosinblat for dilute suspensions
        if phi >= 0.5:
            # Beyond Einstein regime, use Batchelor or numerical
            return eta_m * (1 + 2.5 * phi + 6.2 * phi ** 2)

        # Einstein formula: eta_eff = eta_m * (1 + 2.5*phi) for spherical particles
        return eta_m * (1 + 2.5 * phi)


class VolumeAveraging:
    """Volume averaging for field averaging over RVE.

    Computes volume-averaged fields across a representative volume element.
    Useful for computing effective properties from field solutions on the RVE mesh.

    Parameters
    ----------
    rve_mesh : np.ndarray, optional
        Mesh grid for the RVE (nx, ny, nz).
    """

    def __init__(self, rve_mesh: Optional[Tuple[int, int, int]] = None) -> None:
        self.rve_mesh = rve_mesh

    def average_field(self, field: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Compute the volume average of a field over the RVE.

        Parameters
        ----------
        field : np.ndarray
            Field values on the RVE mesh of shape (nx, ny, nz) or (n_points,).
        weights : np.ndarray, optional
            Volume weights for each point. If None, uniform weighting is used.

        Returns
        -------
        float
            The volume-averaged field value.
        """
        if weights is None:
            # Uniform averaging
            return float(np.mean(field))
        else:
            # Weighted averaging
            weights = np.asarray(weights)
            field = np.asarray(field)
            if weights.shape != field.shape:
                raise ValueError("Weights and field must have the same shape.")
            return float(np.sum(weights * field) / np.sum(weights))

    def effective_property_from_fields(
        self, field_sets: List[np.ndarray], property_getters: List[callable]
    ) -> Dict[str, float]:
        """Compute effective properties from multiple field solutions.

        Parameters
        ----------
        field_sets : list of np.ndarray
            List of field solutions from different RVE simulations.
        property_getters : list of callable
            Functions that extract a property from a field solution.

        Returns
        -------
        dict[str, float]
            Effective properties computed from the field averages.
        """
        results = {}
        for i, (field, getter) in enumerate(zip(field_sets, property_getters)):
            avg = self.average_field(field)
            prop = getter(avg)
            # Use descriptive keys based on the getter function name
            key = property_getters[i].__name__ if hasattr(property_getters[i], "__name__") else f"prop_{i}"
            results[key] = prop
        return results


class AsymptoticExpansion:
    """Asymptotic expansion for ε→0 (scale separation).

    Provides asymptotic expansion methods for problems with scale separation,
    where the small parameter ε = L_fine / L_coarse ≪ 1. The solution is
    expanded as u(x, ε) = u_0(x) + ε*u_1(x, x/ε) + ε^2*u_2(x, x/ε) + ...

    Parameters
    ----------
    epsilon : float
        The scale separation parameter ε = L_fine / L_coarse.
    order : int
        The order of the asymptotic expansion to compute (default: 1).
    """

    def __init__(self, epsilon: float = 1e-3, order: int = 1) -> None:
        self.epsilon = epsilon
        self.order = order

    def expand_solution(
        self, u0: np.ndarray, u1: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute the asymptotic expansion of the solution.

        Parameters
        ----------
        u0 : np.ndarray
            The leading-order (ε^0) solution.
        u1 : np.ndarray, optional
            The first-order correction (ε^1) solution. If None, zeros are used.

        Returns
        -------
        np.ndarray
            The expanded solution u_0 + ε*u_1 + O(ε^2).
        """
        if u1 is None:
            u1 = np.zeros_like(u0)

        expansion = u0 + self.epsilon * u1
        return expansion

    def compute_effective_coefficient(
        self, fine_scale_coefficient: float, microstructure_info: Dict[str, Any]
    ) -> float:
        """Compute the effective coefficient via asymptotic homogenization.

        Parameters
        ----------
        fine_scale_coefficient : float
            The coefficient at the fine scale.
        microstructure_info : dict
            Information about the microstructure (geometry, properties, etc.).

        Returns
        -------
        float
            The effective (coarse-scale) coefficient.
        """
    # Placeholder: in a full implementation, this would compute the
    # homogenized coefficient from the asymptotic expansion of the cell problems
        return fine_scale_coefficient


class NumericalHomogenization:
    """Numerical homogenization: solve cell problems on RVE mesh.

    Solves the periodic cell problems on a discretized RVE mesh to compute
    the effective properties of a composite material. The cell problems involve
    solving local PDEs with periodic boundary conditions on the RVE.

    Parameters
    ----------
    rve_size : tuple[float, float, float]
        RVE dimensions in meters.
    mesh_resolution : int
        Number of elements per RVE dimension (default: 20).
    """

    def __init__(
        self,
        rve_size: Tuple[float, float, float] = (1e-6, 1e-6, 1e-6),
        mesh_resolution: int = 20,
    ) -> None:
        self.rve_size = rve_size
        self.mesh_resolution = mesh_resolution
        self.mesh: Optional[np.ndarray] = None
        self.cell_problem_solutions: Dict[str, np.ndarray] = {}

    def generate_mesh(self) -> np.ndarray:
        """Generate a mesh for the RVE.

        Returns
        -------
        np.ndarray
            The mesh grid points.
        """
        Lx, Ly, Lz = self.rve_size
        nx, ny, nz = self.mesh_resolution

        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.linspace(0, Lz, nz)

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        self.mesh = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        return self.mesh

    def solve_cell_problem(
        self, problem_type: str, phase_properties: Dict[str, float], geometry: Dict[str, Any]
    ) -> np.ndarray:
        """Solve a cell problem on the RVE mesh.

        Parameters
        ----------
        problem_type : str
            Type of cell problem: "conductivity", "viscosity", "elasticity", etc.
        phase_properties : dict
            Phase properties (conductivities, viscosities, etc.).
        geometry : dict
            Geometry description of the microstructure.

        Returns
        -------
        np.ndarray
            Solution field on the RVE mesh.
        """
        # Generate mesh if not already
        if self.mesh is None:
            self.generate_mesh()

        n_points = self.mesh.shape[0]

        # Placeholder solution - in a full implementation, this would solve
        # the actual PDE (e.g., ∇·(k∇χ) = 0 with periodic BCs) on the RVE
        # using finite elements or finite volumes

        if problem_type == "conductivity":
            # Simple effective medium solution
            k_eff = phase_properties.get("k_eff", 1.0)
            solution = np.full(n_points, k_eff)
        elif problem_type == "viscosity":
            eta_eff = phase_properties.get("eta_eff", 1.0)
            solution = np.full(n_points, eta_eff)
        elif problem_type == "elasticity":
            # Placeholder for elasticity cell problem
            solution = np.full(n_points, 1.0)
        else:
            solution = np.ones(n_points)

        self.cell_problem_solutions[problem_type] = solution
        return solution

    def compute_effective_properties(
        self, phase_properties: Dict[str, float], geometry: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute effective properties by solving cell problems.

        Parameters
        ----------
        phase_properties : dict
            Phase-level properties (e.g., {"k_matrix": 1.0, "k_inclusion": 10.0}).
        geometry : dict
            Microstructure geometry info (e.g., {"volume_fraction": 0.3, "shape": "spherical"}).

        Returns
        -------
        dict[str, float]
            Effective properties computed from numerical homogenization.
        """
        # Solve multiple cell problems
        problem_types = ["conductivity", "viscosity"]

        effective = {}
        for ptype in problem_types:
            sol = self.solve_cell_problem(ptype, phase_properties, geometry)
            # The effective property is the volume average
            avg = np.mean(sol)
            effective[f"{ptype}_eff"] = float(avg)

        return effective


class EffectivePropertyCalculator:
    """Calculator for effective properties from microstructure.

    Computes effective material properties including:
    - σ_eff: electrical conductivity
    - ε_eff: permittivity/dielectric constant
    - μ_eff: magnetic permeability
    - k_eff: thermal conductivity
    - η_eff: dynamic viscosity

    Supports various microstructures: composites, porous materials, layered media,
    and random dispersions.

    Parameters
    ----------
    method : str
        Computation method: "homogenization", "rule_of_mixtures", "mgarnett", "mc".
    """

    def __init__(self, method: str = "homogenization") -> None:
        self.method = method

    def compute_sigma_eff(
        self, phase_conductivities: Tuple[float, float], volume_fraction: float, bc: str = "periodic"
    ) -> float:
        """Compute effective electrical conductivity.

        Parameters
        ----------
        phase_conductivities : tuple of float
            (matrix, inclusion) conductivities in S/m.
        volume_fraction : float
            Inclusion volume fraction.
        bc : str
            Boundary condition type.

        Returns
        -------
        float
            Effective conductivity σ_eff in S/m.
        """
        k_m, k_i = phase_conductivities
        phi = volume_fraction

        calculator = Homogenization(bc_type=bc, volume_fraction=phi)

        if self.method == "homogenization":
            return calculator.compute_effective_conductivity(phase_conductivities, {})
        elif self.method == "mgarnett":
            # Maxwell-Garnett
            eps_m, eps_i = k_m, k_i  # Using permittivity-like notation
            if eps_m == 0:
                return k_i
            num = 3 * phi * (eps_i - eps_m)
            den = eps_i + 2 * eps_m - phi * (eps_i - eps_m)
            if abs(den) < 1e-15:
                return eps_m
            return eps_m * (1 + num / den)
        elif self.method == "rule_of_mixtures":
            # Voigt upper bound
            return phi * k_i + (1 - phi) * k_m
        else:
            # Default: Hashin-Shtrikman
            if k_i > k_m:
                return k_i + phi * (k_m * (1 - phi)) / (k_m + 2 * k_i * (1 - phi) / 3)
            else:
                return k_m + (1 - phi) * (k_i * (1 - k_m / k_i)) / (k_i + 2 * k_m * (1 - phi) / 3)

    def compute_epsilon_eff(
        self, phase_permittivities: Tuple[float, float], volume_fraction: float, method: str = "mgarnett"
    ) -> float:
        """Compute effective permittivity.

        Parameters
        ----------
        phase_permittivities : tuple of float
            (matrix, inclusion) permittivities in F/m.
        volume_fraction : float
            Inclusion volume fraction.
        method : str
            Method: "mgarnett", "homogenization", "rule_of_mixtures".

        Returns
        -------
        float
            Effective permittivity ε_eff in F/m.
        """
        eps_m, eps_i = phase_permittivities
        phi = volume_fraction

        if method == "mgarnett":
            if eps_m == 0:
                return eps_i
            num = 3 * phi * (eps_i - eps_m)
            den = eps_i + 2 * eps_m - phi * (eps_i - eps_m)
            if abs(den) < 1e-15:
                return eps_m
            return eps_m * (1 + num / den)
        elif method == "homogenization":
            calculator = Homogenization(volume_fraction=phi)
            return calculator.compute_effective_permittivity(phase_permittivities, {})
        elif method == "rule_of_mixtures":
            # Voigt upper bound
            return phi * eps_i + (1 - phi) * eps_m
        else:
            return eps_m  # Default

    def compute_mu_eff(
        self, phase_permeabilities: Tuple[float, float], volume_fraction: float, method: str = "homogenization"
    ) -> float:
        """Compute effective magnetic permeability.

        Parameters
        ----------
        phase_permeabilities : tuple of float
            (matrix, inclusion) permeabilities (dimensionless or H/m).
        volume_fraction : float
            Inclusion volume fraction.
        method : str
            Computation method.

        Returns
        -------
        float
            Effective permeability μ_eff.
        """
        mu_m, mu_i = phase_permeabilities
        phi = volume_fraction

        if method == "homogenization":
            calculator = Homogenization(volume_fraction=phi)
            return calculator.compute_effective_conductivity(phase_permeabilities, {})
        elif method == "mgarnett":
            # Similar to permittivity MG
            if mu_m == 0:
                return mu_i
            num = 3 * phi * (mu_i - mu_m)
            den = mu_i + 2 * mu_m - phi * (mu_i - mu_m)
            if abs(den) < 1e-15:
                return mu_m
            return mu_m * (1 + num / den)
        elif method == "rule_of_mixtures":
            return phi * mu_i + (1 - phi) * mu_m
        else:
            return mu_m

    def compute_k_eff(
        self, phase_thermal_conductivities: Tuple[float, float], volume_fraction: float, method: str = "homogenization"
    ) -> float:
        """Compute effective thermal conductivity.

        Parameters
        ----------
        phase_thermal_conductivities : tuple of float
            (matrix, inclusion) thermal conductivities in W/(m·K).
        volume_fraction : float
            Inclusion volume fraction.
        method : str
            Computation method.

        Returns
        -------
        float
            Effective thermal conductivity k_eff in W/(m·K).
        """
        k_m, k_i = phase_thermal_conductivities
        phi = volume_fraction

        if method == "homogenization":
            calculator = Homogenization(volume_fraction=phi)
            return calculator.compute_effective_conductivity(phase_thermal_conductivities, {})
        elif method == "mgarnett":
            return self.compute_sigma_eff(phase_thermal_conductivities, volume_fraction, bc="periodic")
        elif method == "rule_of_mixtures":
            # Parallel (Voigt) and series (Reuss) bounds
            k_parallel = phi * k_i + (1 - phi) * k_m  # Parallel to layers
            k_series = 1.0 / (phi / k_i + (1 - phi) / k_m) if k_m > 0 and k_i > 0 else max(k_m, k_i)
            # Average
            return (k_parallel + k_series) / 2.0
        else:
            return k_m

    def compute_eta_eff(
        self, phase_viscosities: Tuple[float, float], volume_fraction: float, method: str = "einstein"
    ) -> float:
        """Compute effective dynamic viscosity.

        Parameters
        ----------
        phase_viscosities : tuple of float
            (matrix, inclusion) viscosities in Pa·s.
        volume_fraction : float
            Inclusion volume fraction.
        method : str
            Method: "einstein", "batchelor", "homogenization".

        Returns
        -------
        float
            Effective viscosity η_eff in Pa·s.
        """
        eta_m, eta_i = phase_viscosities
        phi = volume_fraction

        if method == "einstein":
            # Einstein formula for dilute suspensions
            if phi >= 0.5:
                return eta_m * (1 + 2.5 * phi + 6.2 * phi ** 2)
            return eta_m * (1 + 2.5 * phi)

        elif method == "batchelor":
            # Batchelor correction for higher concentrations
            if phi >= 0.5:
                return eta_m * (1 + 2.5 * phi + 6.2 * phi ** 2)
            return eta_m * (1 + 2.5 * phi + 6.2 * phi ** 2)

        elif method == "homogenization":
            calculator = Homogenization(volume_fraction=phi)
            return calculator.compute_effective_viscosity(phase_viscosities, {})

        else:
            return eta_m  # Default