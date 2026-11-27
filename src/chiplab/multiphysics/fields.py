"""Coupled field equations for multiphysics simulation.

This module defines the fundamental field equations that are coupled
through feedback loops in multiphysics simulations.
"""

from __future__ import annotations

from chiplab.schemas.common import Provenance, TrackedValue
from chiplab.core.gpu import get_gpu_backend
import numpy as np


class FieldSystem:
    """Container for coupled fields (E, H, T, u, c, σ).
    
    Holds all field variables that participate in the multiphysics coupling
    and provides methods for accessing and updating them with provenance tracking.
    """
    
    def __init__(
        self,
        E: np.ndarray | None = None,
        H: np.ndarray | None = None,
        T: float | TrackedValue | None = None,
        u: float | TrackedValue | None = None,
        c: float | TrackedValue | None = None,
        sigma: float | TrackedValue | None = None,
        device: str = "auto",
    ):
        """Initialize field system.

        Parameters
        ----------
        E : np.ndarray, optional
            Electric field vector/matrix (V/m)
        H : np.ndarray, optional
            Magnetic field vector/matrix (A/m)
        T : float or TrackedValue, optional
            Temperature (K)
        u : float or TrackedValue, optional
            Internal energy density (J/m³)
        c : float or TrackedValue, optional
            Particle concentration (1/m³)
        sigma : float or TrackedValue, optional
            Electrical conductivity (S/m)
        device : str
            Device to run on: "auto", "cuda", "gpu", or "cpu". Default "auto".
        """
        self.device = device
        self.backend = get_gpu_backend(device=device)
        self.E = E if E is not None else np.zeros(3)
        self.H = H if H is not None else np.zeros(3)
        self.T = (
            TrackedValue(value=T, unit="K", provenance="assumed")
            if isinstance(T, (float, int))
            else T
        )
        self.u = (
            TrackedValue(value=u, unit="J/m³", provenance="assumed")
            if isinstance(u, (float, int))
            else u
        )
        self.c = (
            TrackedValue(value=c, unit="1/m³", provenance="assumed")
            if isinstance(c, (float, int))
            else c
        )
        self.sigma = (
            TrackedValue(value=sigma, unit="S/m", provenance="assumed")
            if isinstance(sigma, (float, int))
            else sigma
        )
        self.magnetization = TrackedValue(value=0.0, unit="emu/cm³", provenance="assumed")
    
    def copy(self) -> FieldSystem:
        """Create a deep copy of the field system."""
        def _copy_tv(tv):
            if tv is None:
                return None
            return TrackedValue(
                value=tv.value, unit=tv.unit, provenance=tv.provenance,
                source=tv.source, uncertainty_pct=tv.uncertainty_pct,
                notes=tv.notes,
            )

        return FieldSystem(
            E=self.E.copy() if hasattr(self.E, 'copy') else self.E,
            H=self.H.copy() if hasattr(self.H, 'copy') else self.H,
            T=_copy_tv(self.T),
            u=_copy_tv(self.u),
            c=_copy_tv(self.c),
            sigma=_copy_tv(self.sigma),
            device=self.device,
        )
    
    def __repr__(self):
        T_val = float(self.T) if self.T and self.T.value is not None else 300.0
        u_val = float(self.u) if self.u and self.u.value is not None else 0.0
        c_val = float(self.c) if self.c and self.c.value is not None else 0.0
        sigma_val = float(self.sigma) if self.sigma and self.sigma.value is not None else 0.0
        return (f"FieldSystem(E={self.E.shape if hasattr(self.E, 'shape') else 'scalar'}, "
                f"H={self.H.shape if hasattr(self.H, 'shape') else 'scalar'}, "
                f"T={T_val:.1f}K, u={u_val:.2f}J/m³, "
                f"c={c_val:.2e}/m³, σ={sigma_val:.2e}S/m)")


class MaxwellEquations:
    """Maxwell's equations with material constitutive relations.
    
    Implements ∇×E = -∂B/∂t, ∇×H = J + ∂D/∂t with material constitutive relations
    that couple E, H, T, c, and σ fields.
    """
    
    def __init__(self, field_system: FieldSystem, grid: dict | None = None):
        """Initialize Maxwell equations solver.
        
        Parameters
        ----------
        field_system : FieldSystem
            Coupled field container
        grid : dict, optional
            Grid parameters {dx, dy, dz, dt}
        """
        self.field_system = field_system
        self.grid = grid or {"dx": 1e-6, "dy": 1e-6, "dz": 1e-6, "dt": 1e-12}
    
    def curl_E(self) -> np.ndarray:
        """Compute ∇×E - Faraday's law."""
        # Simplified: return curl of electric field
        Ex, Ey, Ez = self.field_system.E if self.field_system.E.ndim == 3 else [self.field_system.E]
        # In a full FEM implementation, this would use sparse matrix operators
        return np.zeros(3)  # placeholder
    
    def curl_H(self) -> np.ndarray:
        """Compute ∇×H - Ampere's law with displacement current.
        
        Handles both 1D (shape (3,)) and 3D (shape (3, 3)) H arrays.
        """
        H = self.field_system.H
        # Extract H components - handle both 1D and 3D cases
        if H.ndim == 3:
            Hx, Hy, Hz = H[0], H[1], H[2]
        else:
            # 1D case: shape (3,)
            Hx, Hy, Hz = H[0], H[1], H[2]
        # J = σE (Ohm's law), ∂D/∂t couples to temperature and concentration
        sigma_val = float(self.field_system.sigma)  # Unwrap TrackedValue
        E = self.field_system.E
        J = sigma_val * E
        return J  # placeholder
    
    def update_from_joule_heating(self):
        """Update temperature from Joule heating Q_joule = J·E."""
        J = self.curl_H()
        E = self.field_system.E
        Q_joule = float(np.sum(J * E))  # W/m³
        # Update temperature tracked value
        new_T = float(self.field_system.T) + Q_joule * self.grid["dt"] / 1000.0  # simplified
        self.field_system.T = TrackedValue(
            value=new_T, unit="K", provenance=self.field_system.T.provenance,
            source="MaxwellEquations"
        )
        return Q_joule
    
    def update_from_hysteresis(self) -> float:
        """Update temperature from hysteresis heating Q_hysteresis."""
        # Hysteresis heating depends on M·H loop area, coupled to particle dynamics
        H = self.field_system.H
        # Placeholder: return simplified hysteresis heating
        return 0.0
    
    def update_from_relaxation(self) -> float:
        """Update temperature from relaxation heating Q_relaxation."""
        # Relaxation heating from nanoparticle magnetic relaxation
        tau = float(self.field_system.c) if hasattr(self.field_system.c, '__float__') else 0.0
        return 0.0


class HeatEquation:
    """Heat equation: ρc_p ∂T/∂t = ∇·(k∇T) + Q_joule + Q_hysteresis + Q_relaxation.
    
    Couples temperature to electrical (Joule heating), magnetic (hysteresis),
    and particle dynamics (relaxation heating).
    """
    
    def __init__(self, field_system: FieldSystem, material_params: dict | None = None):
        """Initialize heat equation.
        
        Parameters
        ----------
        field_system : FieldSystem
            Coupled field container
        material_params : dict, optional
            Material parameters {rho, cp, k}
        """
        self.field_system = field_system
        self.material_params = material_params or {
            "rho": TrackedValue(value=2700.0, unit="kg/m³", provenance="assumed"),
            "cp": TrackedValue(value=900.0, unit="J/(kg·K)", provenance="assumed"),
            "k": TrackedValue(value=200.0, unit="W/(m·K)", provenance="assumed"),
        }
    
    def compute_laplacian_T(self) -> float:
        """Compute ∇·(k∇T) - thermal diffusion term."""
        k = float(self.material_params["k"])
        T = float(self.field_system.T)
        # Simplified: assume uniform temperature, no spatial diffusion
        return 0.0
    
    def compute_source_terms(self) -> dict[str, float]:
        """Compute all heat source terms.
        
        Returns
        -------
        dict with keys 'Q_joule', 'Q_hysteresis', 'Q_relaxation'
        """
        maxwell = MaxwellEquations(self.field_system)
        Q_joule = maxwell.update_from_joule_heating()
        Q_hysteresis = maxwell.update_from_hysteresis()
        Q_relaxation = maxwell.update_from_relaxation()
        
        return {
            "Q_joule": Q_joule,
            "Q_hysteresis": Q_hysteresis,
            "Q_relaxation": Q_relaxation,
        }
    
    def time_step(self, dt: float) -> dict:
        """Advance temperature by one time step.
        
        Parameters
        ----------
        dt : float
            Time step (s)
            
        Returns
        -------
        dict with updated field values
        """
        sources = self.compute_source_terms()
        
        # Simplified explicit time stepping
        rho_cp = float(self.material_params["rho"]) * float(self.material_params["cp"])
        k = float(self.material_params["k"])
        T_old = float(self.field_system.T)
        
        # Heat diffusion + sources
        dT_diff = self.compute_laplacian_T() * dt / rho_cp
        dT_source = (sources["Q_joule"] + sources["Q_hysteresis"] + sources["Q_relaxation"]) * dt / rho_cp
        
        new_T = T_old + dT_diff + dT_source
        
        self.field_system.T = TrackedValue(
            value=new_T, unit="K", provenance=self.field_system.T.provenance,
            source="HeatEquation"
        )
        
        return {"T": new_T, "sources": sources}


class ParticleDynamics:
    """Particle dynamics: ∂c/∂t = ∇·(D∇c) - ∇·(c v_drift) + R_aggregation.
    
    Models particle concentration evolution with diffusion, drift, and aggregation.
    Couples temperature (via D viscosity), electrical (via σ), and magnetic (via M) fields.
    """
    
    def __init__(self, field_system: FieldSystem, params: dict | None = None):
        """Initialize particle dynamics.
        
        Parameters
        ----------
        field_system : FieldSystem
            Coupled field container
        params : dict, optional
            Particle parameters {D0, eta0, alpha_agg}
        """
        self.field_system = field_system
        self.params = params or {
            "D0": TrackedValue(value=1e-12, unit="m²/s", provenance="assumed"),
            "eta0": TrackedValue(value=1e-3, unit="Pa·s", provenance="assumed"),
            "alpha_agg": TrackedValue(value=0.01, unit="1/s", provenance="assumed"),
        }
    
    def compute_diffusion_coefficient(self) -> float:
        """Compute D(T,c) = D0 * exp(-Ea/(kB*T)) * f(c)."""
        D0 = float(self.params["D0"])
        T = float(self.field_system.T)
        # Activation energy simplification
        kB = 1.38e-23
        Ea = 0.5  # eV, simplified
        Arrhenius = np.exp(-Ea / (kB * T)) if T > 0 else 0.0
        
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        # Concentration-dependent factor
        f_c = 1.0 / (1.0 + 10.0 * c) if c > 0 else 1.0
        
        return D0 * Arrhenius * f_c
    
    def compute_drift_velocity(self) -> np.ndarray:
        """Compute v_drift(E, H, T, c) - electrical/magnetic field-driven drift."""
        sigma = float(self.field_system.sigma) if self.field_system.sigma.value else 0.0
        H = self.field_system.H if hasattr(self.field_system, 'H') else np.zeros(3)
        T = float(self.field_system.T)
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        
        # Electrophoretic drift: v = μE, mobility depends on T and c
        E = self.field_system.E if hasattr(self.field_system, 'E') else np.zeros(3)
        
        # Simplified mobility: μ = μ0 / (1 + α*c) * f(T)
        mu0 = 1e-8  # m²/(V·s)
        mu = mu0 / (1.0 + 5.0 * c) * (300.0 / max(T, 1.0))
        
        drift = mu * E
        return drift
    
    def compute_aggregation_rate(self) -> float:
        """Compute R_aggregation(c, T, σ) - particle aggregation rate."""
        alpha_agg = float(self.params["alpha_agg"])
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        T = float(self.field_system.T)
        sigma = float(self.field_system.sigma) if self.field_system.sigma.value else 0.0
        
        # Aggregation enhanced by concentration, temperature, and conductivity
        # Reduced by thermal motion
        rate = alpha_agg * c * (1.0 + 0.1 * sigma) / (1.0 + 0.01 * T)
        return max(rate, 0.0)
    
    def time_step(self, dt: float, concentration_grid: np.ndarray | None = None) -> dict:
        """Advance particle concentration by one time step.
        
        Parameters
        ----------
        dt : float
            Time step (s)
        concentration_grid : np.ndarray, optional
            3D concentration grid for spatial diffusion
            
        Returns
        -------
        dict with updated field values
        """
        D = self.compute_diffusion_coefficient()
        v_drift = self.compute_drift_velocity()
        R_agg = self.compute_aggregation_rate()
        
        c_old = float(self.field_system.c)
        
        # Simplified update: c_new = c_old + dt * (diffusion + drift + aggregation)
        # In full FEM, this would solve ∂c/∂t = ∇·(D∇c) - ∇·(c v_drift) + R_aggregation
        
        # Concentration change from aggregation (dominant effect)
        dc_agg = R_agg * dt
        
        # Drift contribution (simplified uniform drift)
        dc_drift = 0.0  # would need spatial gradients
        
        # Diffusion contribution (simplified)
        dc_diff = 0.0  # would need Laplacian of c
        
        new_c = c_old + dc_agg + dc_diff + dc_drift
        new_c = max(0.0, new_c)  # concentration can't be negative
        
        self.field_system.c = TrackedValue(
            value=new_c, unit="1/m³", provenance=self.field_system.c.provenance,
            source="ParticleDynamics"
        )
        
        return {"c": new_c, "D": D, "v_drift": float(np.linalg.norm(v_drift)), "R_agg": R_agg}


class FluidDynamics:
    """Navier-Stokes for ferrofluid carrier fluid.
    
    Optional, for advanced multiphysics coupling where fluid flow affects
    particle dynamics and heat transfer.
    """
    
    def __init__(self, field_system: FieldSystem, params: dict | None = None):
        """Initialize fluid dynamics.
        
        Parameters
        ----------
        field_system : FieldSystem
            Coupled field container
        params : dict, optional
            Fluid parameters {rho, mu, g}
        """
        self.field_system = field_system
        self.params = params or {
            "rho": TrackedValue(value=1200.0, unit="kg/m³", provenance="assumed"),
            "mu": TrackedValue(value=0.1, unit="Pa·s", provenance="assumed"),
        }
    
    def compute_viscous_heating(self) -> float:
        """Compute viscous dissipation heating."""
        mu = float(self.params["mu"])
        # Simplified: Φ = μ(∇v)²
        return 0.0
    
    def time_step(self, dt: float) -> dict:
        """Advance fluid state by one time step."""
        # Placeholder for Navier-Stokes integration
        return {"viscous_heating": 0.0}


class MechanicalEquations:
    """Mechanical equilibrium: ∇·σ = 0 with thermal expansion, magnetostriction.
    
    Models stress-strain response coupled to temperature and magnetic field.
    """
    
    def __init__(self, field_system: FieldSystem, params: dict | None = None):
        """Initialize mechanical equations.
        
        Parameters
        ----------
        field_system : FieldSystem
            Coupled field container
        params : dict, optional
            Material parameters {E_mod, nu, alpha_th, lambda_ms}
        """
        self.field_system = field_system
        self.params = params or {
            "E_mod": TrackedValue(value=200e9, unit="Pa", provenance="assumed"),
            "nu": TrackedValue(value=0.3, unit="", provenance="assumed"),
            "alpha_th": TrackedValue(value=1e-5, unit="1/K", provenance="assumed"),
            "lambda_ms": TrackedValue(value=1e-6, unit="", provenance="assumed"),
        }
    
    def compute_thermal_stress(self) -> float:
        """Compute stress from thermal expansion: σ_th = E·α·ΔT."""
        E = float(self.params["E_mod"])
        alpha = float(self.params["alpha_th"])
        T = float(self.field_system.T)
        T_ref = 300.0  # reference temperature
        delta_T = T - T_ref
        sigma_th = E * alpha * delta_T
        return sigma_th
    
    def compute_magnetostrictive_stress(self) -> float:
        """Compute stress from magnetostriction: σ_ms = λ·H²."""
        lam = float(self.params["lambda_ms"])
        H = float(np.linalg.norm(self.field_system.H)) if hasattr(self.field_system.H, '__abs__') else 0.0
        sigma_ms = lam * H**2
        return sigma_ms
    
    def total_stress(self) -> float:
        """Compute total stress: σ_total = σ_th + σ_ms + σ_elastic."""
        sigma_th = self.compute_thermal_stress()
        sigma_ms = self.compute_magnetostrictive_stress()
        # Elastic stress from field system if present
        sigma_elastic = float(self.field_system.u) if self.field_system.u.value else 0.0
        return sigma_th + sigma_ms + sigma_elastic


class ConstitutiveRelations:
    """Constitutive relations: σ(E,H,T,c), ε(E,H,T,c), μ(H,T,c), k(T,c), η(T,c).
    
    Defines all material property dependencies on the coupled fields.
    """
    
    def __init__(self, field_system: FieldSystem):
        """Initialize constitutive relations.
        
        Parameters
        ----------
        field_system : FieldSystem
            Coupled field container
        """
        self.field_system = field_system
    
    def electrical_conductivity(self) -> float:
        """σ(E,H,T,c) - electrical conductivity dependence on all fields.
        
        Returns
        -------
        float
            Conductivity (S/m)
        """
        sigma0 = float(self.field_system.sigma)
        T = float(self.field_system.T)
        c = float(self.field_system.c)
        H = float(np.linalg.norm(self.field_system.H)) if hasattr(self.field_system.H, '__abs__') else 0.0
        
        # Conductivity increases with temperature (more carriers)
        # and concentration (more charge carriers), but decreases with strong magnetic field
        # (can align charges and reduce mobility)
        sigma_T = sigma0 * (1.0 + 0.01 * (T - 300.0))  # temp coefficient
        sigma_c = 1.0 + 10.0 * c  # concentration effect
        sigma_H = 1.0 / (1.0 + 0.001 * H**2)  # magnetic field effect
        
        return sigma_T * sigma_c * sigma_H
    
    def permittivity(self) -> float:
        """ε(E,H,T,c) - permittivity dependence on all fields."""
        epsilon0 = 8.854e-12
        T = float(self.field_system.T)
        c = float(self.field_system.c)
        
        # Permittivity decreases with temperature (Curie-Weiss)
        # and increases with concentration of polar particles
        epsilon = epsilon0 * (80.0 - 0.1 * (T - 300.0)) * (1.0 + 5.0 * c)
        return max(epsilon, epsilon0)
    
    def permeability(self) -> float:
        """μ(H,T,c) - magnetic permeability dependence."""
        mu0 = 4e-7 * np.pi
        H = float(np.linalg.norm(self.field_system.H)) if hasattr(self.field_system.H, '__abs__') else 0.0
        T = float(self.field_system.T)
        c = float(self.field_system.c)
        
        # Permeability decreases with temperature (Curie law)
        # and increases with particle concentration/organization
        mu_T = mu0 * (1.0 / (1.0 + (T / 500.0)))  # Curie-Weiss simplified
        mu_c = 1.0 + 2.0 * c  # concentration enhancement
        
        return mu_T * mu_c
    
    def thermal_conductivity(self) -> float:
        """k(T,c) - thermal conductivity dependence on temperature and concentration."""
        k0 = float(self.field_system.u) if hasattr(self.field_system.u, 'value') else 200.0
        T = float(self.field_system.T)
        c = float(self.field_system.c)
        
        # Thermal conductivity decreases with temperature for metals
        # but can increase with ceramic/concentration additions
        k_T = k0 / (1.0 + 0.001 * (T - 300.0))
        k_c = 1.0 + 50.0 * c  # particle enhancement
        
        return k_T * k_c
    
    def viscosity(self) -> float:
        """η(T,c) - viscosity dependence on temperature and concentration."""
        eta0 = float(self.field_system.c) if self.field_system.c.value else 1e-3
        T = float(self.field_system.T)
        c = float(self.field_system.c)
        
        # Viscosity decreases with temperature (Arrhenius)
        # and increases with particle concentration
        # Arrhenius: η = η0 * exp(Ea/(kB*T))
        kB = 1.38e-23
        Ea = 0.5  # eV
        eta_T = eta0 * np.exp(Ea / (kB * max(T, 1.0)))
        eta_c = 1.0 + 100.0 * c  # concentration thickening
        
        return eta_T * eta_c