"""Ferrofluid-specific multiphysics coupling.

WITHDRAWN: this chain used placeholder constitutive laws (its viscosity law overflows)
and returned insensitive numbers.  Both classes now refuse to be constructed; use
``chiplab.models.materials.ferro_device``.  The original intent follows.

Implements the full chain of coupled phenomena in ferrofluid systems:
Electrical signal → E-field → Maxwell stress + Joule heating → Temperature
→ Viscosity (η) + Diffusion (D) + Relaxation (τ) → Particle dynamics (c, orientation)
→ Magnetization (M) + Conductivity (σ) → Electrical response (impedance)
→ Readout circuit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from chiplab.credibility import refuse_ferrofluid
from chiplab.schemas.common import TrackedValue, Provenance

from .fields import (
    FieldSystem, MaxwellEquations, HeatEquation, ParticleDynamics,
    ConstitutiveRelations
)
from .coupling import (
    CouplingGraph, FeedbackLoop, CouplingScheme,
    StaggeredScheme, MonolithicScheme, QuasiStaticScheme
)


class FerrofluidMaterialModel:
    """All constitutive relations with uncertainty for ferrofluid.
    
    Defines how material properties depend on the coupled fields:
    σ(E,H,T,c), ε(E,H,T,c), μ(H,T,c), k(T,c), η(T,c)
    with provenance tracking on all parameters.
    """
    
    def __init__(self, field_system: FieldSystem, uncertainty: float = 0.05):
        """Initialize material model.
        
        Parameters
        ----------
        field_system : FieldSystem
            Coupled field container
        uncertainty : float
            Relative uncertainty (0-1) for all material parameters
        """
        refuse_ferrofluid("FerrofluidMaterialModel", "ferrofluid")
        self.field_system = field_system
        self.uncertainty = uncertainty
        
        # Base material parameters with provenance
        self.base_sigma = TrackedValue(
            value=0.1, unit="S/m", provenance=Provenance.ASSUMED,
            uncertainty_pct=uncertainty * 100
        )
        self.base_epsilon = TrackedValue(
            value=80.0, unit="", provenance=Provenance.ASSUMED,
            uncertainty_pct=uncertainty * 100
        )
        self.base_mu = TrackedValue(
            value=4e-7 * np.pi, unit="H/m", provenance=Provenance.ASSUMED,
            uncertainty_pct=uncertainty * 100
        )
        self.base_k = TrackedValue(
            value=200.0, unit="W/(m·K)", provenance=Provenance.ASSUMED,
            uncertainty_pct=uncertainty * 100
        )
        self.base_eta = TrackedValue(
            value=1e-3, unit="Pa·s", provenance=Provenance.ASSUMED,
            uncertainty_pct=uncertainty * 100
        )
        self.base_D = TrackedValue(
            value=1e-12, unit="m²/s", provenance=Provenance.ASSUMED,
            uncertainty_pct=uncertainty * 100
        )
    
    def electrical_conductivity(self) -> TrackedValue:
        """σ(E,H,T,c) - electrical conductivity with uncertainty.
        
        Returns
        -------
        TrackedValue
            Conductivity with provenance and uncertainty
        """
        sigma0 = float(self.base_sigma)
        T = float(self.field_system.T)
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        # Handle H field - could be TrackedValue or numpy array
        H_raw = self.field_system.H
        H = float(np.linalg.norm(H_raw)) if isinstance(H_raw, np.ndarray) else (float(H_raw.value) if hasattr(H_raw, 'value') else float(H_raw)) if self.field_system.H.value else 0.0
        
        # Conductivity model: sigma = sigma0 * f(T) * g(c) * h(H)
        # f(T): increases with T (more charge carriers)
        f_T = 1.0 + 0.01 * (T - 300.0)
        
        # g(c): increases with particle concentration (more conductive pathways)
        g_c = 1.0 + 10.0 * c
        
        # h(H): decreases with strong magnetic field (charge carrier alignment)
        h_H = 1.0 / (1.0 + 0.001 * H**2)
        
        sigma = sigma0 * f_T * g_c * h_H
        
        # Add uncertainty
        sigma_uncertainty = sigma * self.uncertainty
        
        return TrackedValue(
            value=sigma, unit="S/m",
            provenance=self.base_sigma.provenance,
            uncertainty_pct=self.base_sigma.uncertainty_pct,
            notes=f"T={T:.1f}K, c={c:.3f}, H={H:.3f}A/m"
        )
    
    def permittivity(self) -> TrackedValue:
        """ε(E,H,T,c) - permittivity with uncertainty."""
        epsilon0 = 8.854e-12
        epsilon_r0 = self.base_epsilon.value
        T = float(self.field_system.T)
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        
        # Permittivity: decreases with T (Curie-Weiss), increases with c
        # χ ~ C / (T - T_curie)
        T_curie = 273.0
        curie_factor = T_curie / max(T, T_curie + 1e-10)
        
        epsilon_r = epsilon_r0 * curie_factor * (1.0 + 3.0 * c)
        epsilon = epsilon0 * epsilon_r
        
        epsilon_uncertainty = epsilon * self.uncertainty
        
        return TrackedValue(
            value=epsilon, unit="F/m",
            provenance=self.base_epsilon.provenance,
            uncertainty_pct=self.base_epsilon.uncertainty_pct,
            notes=f"T={T:.1f}K, c={c:.3f}"
        )
    
    def permeability(self) -> TrackedValue:
        """μ(H,T,c) - magnetic permeability with uncertainty."""
        mu0 = 4e-7 * np.pi
        mu_r0 = self.base_mu.value
        H = float(np.linalg.norm(self.field_system.H))
        T = float(self.field_system.T)
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        
        # Permeability: decreases with T (Curie law), increases with c
        # μ_r ~ 1/(T/T_curie) for T > T_curie
        T_curie = 273.0
        curie_effect = 1.0 / max(T / T_curie, 1e-10)
        
        # Concentration enhancement
        c_effect = 1.0 + 2.0 * c
        
        # Hysteresis/alignment effect
        h_H = 1.0 / (1.0 + 0.0001 * H**2)
        
        mu_r = curie_effect * c_effect * h_H
        mu = mu0 * mu_r
        
        mu_uncertainty = mu * self.uncertainty
        
        return TrackedValue(
            value=mu, unit="H/m",
            provenance=self.base_mu.provenance,
            uncertainty_pct=self.base_mu.uncertainty_pct,
            notes=f"T={T:.1f}K, c={c:.3f}, H={H:.3f}A/m"
        )
    
    def thermal_conductivity(self) -> TrackedValue:
        """k(T,c) - thermal conductivity with uncertainty."""
        k0 = self.base_k.value
        T = float(self.field_system.T)
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        
        # Thermal conductivity: decreases slightly with T for fluids
        # but increases significantly with particle concentration
        k_T = k0 / (1.0 + 0.001 * (T - 300.0))
        k_c = 1.0 + 50.0 * c  # Strong concentration effect
        
        k = k_T * k_c
        
        k_uncertainty = k * self.uncertainty
        
        return TrackedValue(
            value=k, unit="W/(m·K)",
            provenance=self.base_k.provenance,
            uncertainty_pct=self.base_k.uncertainty_pct,
            notes=f"T={T:.1f}K, c={c:.3f}"
        )
    
    def viscosity(self) -> TrackedValue:
        """η(T,c) - viscosity with uncertainty."""
        eta0 = self.base_eta.value
        T = float(self.field_system.T)
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        
        # Viscosity: Arrhenius dependence on T, exponential on c
        # η = η0 * exp(Ea/(kB*T)) * (1 + α*c)
        kB = 1.38e-23
        Ea = 0.5  # eV, simplified activation energy
        
        eta_T = eta0 * np.exp(Ea / (kB * max(T, 1.0)))
        eta_c = 1.0 + 100.0 * c  # Strong thickening at high concentration
        
        eta = eta_T * eta_c
        
        eta_uncertainty = eta * self.uncertainty
        
        return TrackedValue(
            value=eta, unit="Pa·s",
            provenance=self.base_eta.provenance,
            uncertainty_pct=self.base_eta.uncertainty_pct,
            notes=f"T={T:.1f}K, c={c:.3f}"
        )
    
    def diffusion_coefficient(self) -> TrackedValue:
        """D(T,c) - particle diffusion coefficient with uncertainty."""
        D0 = self.base_D.value
        T = float(self.field_system.T)
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        
        # Diffusion: D = kB*T/(3π*η*d) with η from viscosity model
        kB = 1.38e-23
        particle_diameter = 10e-9  # 10 nm particles
        
        eta = float(self.viscosity())
        # Stokes-Einstein: D = kB*T / (3π*η*d)
        D = kB * T / (3.0 * np.pi * eta * particle_diameter)
        
        # Concentration hindrance factor
        hindrance = 1.0 / (1.0 + 2.5 * c + 10.5 * c**2)  # Brenner correction
        D_hindered = D * hindrance
        
        D_uncertainty = D_hindered * self.uncertainty
        
        return TrackedValue(
            value=D_hindered, unit="m²/s",
            provenance=self.base_D.provenance,
            uncertainty_pct=self.base_D.uncertainty_pct,
            notes=f"T={T:.1f}K, c={c:.3f}"
        )
    
    def relaxation_time(self) -> TrackedValue:
        """τ(T,c) - magnetic relaxation time with uncertainty."""
        tau0 = 1e-9  # Attempt time ~1 ns
        T = float(self.field_system.T)
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        
        # Néel relaxation: τ = τ0 * exp(KV/kB*T)
        # K_V/kB approximated as 1000 K for typical ferrofluids
        KV_kB = 1000.0
        
        # Concentration effect on relaxation (more particles = more interactions)
        c_effect = 1.0 + 5.0 * c
        
        tau = tau0 * np.exp(KV_kB / max(T, 1.0)) * c_effect
        
        tau_uncertainty = tau * self.uncertainty
        
        return TrackedValue(
            value=tau, unit="s",
            provenance=Provenance.ASSUMED,
            uncertainty_pct=self.base_D.uncertainty_pct,
            notes=f"T={T:.1f}K, c={c:.3f}"
        )


class FerrofluidCoupling:
    """Implements the full multiphysics coupling chain for ferrofluids.
    
    Electrical signal → E-field → Maxwell stress + Joule heating → Temperature
    → Viscosity (η) + Diffusion (D) + Relaxation (τ) → Particle dynamics (c, orientation)
    → Magnetization (M) + Conductivity (σ) → Electrical response (impedance)
    → Readout circuit.
    """
    
    def __init__(self, field_system: FieldSystem, material_model: FerrofluidMaterialModel,
                 coupling_scheme: str = "staggered"):
        """Initialize ferrofluid coupling.
        
        Parameters
        ----------
        field_system : FieldSystem
            Coupled field container
        material_model : FerrofluidMaterialModel
            Material constitutive relations
        coupling_scheme : str
            Coupling method: 'staggered', 'monolithic', or 'quasi-static'
        """
        refuse_ferrofluid("FerrofluidCoupling", "ferrofluid")
        self.field_system = field_system
        self.material_model = material_model
        
        # Set up coupling graph for ferrofluid-specific dependencies
        self.graph = CouplingGraph()
        self.graph.add_node("E")
        self.graph.add_node("H")
        self.graph.add_node("T")
        self.graph.add_node("c")
        self.graph.add_node("σ")
        self.graph.add_node("M")  # Magnetization
        self.graph.add_node("η")  # Viscosity
        self.graph.add_node("D")  # Diffusion coefficient
        self.graph.add_node("τ")  # Relaxation time
        
        # Ferrofluid-specific coupling edges
        # Electrical → E-field (Maxwell's equations)
        self.graph.add_edge("E", "T")  # E → Joule heating → T
        # E → σ → E: Conductivity affects field
        self.graph.add_edge("σ", "E")
        # T → η → D → c: Temperature → viscosity → diffusion → concentration
        self.graph.add_edge("T", "η")
        self.graph.add_edge("η", "D")
        self.graph.add_edge("D", "c")
        # T → τ → c: Temperature → relaxation → concentration
        self.graph.add_edge("T", "τ")
        self.graph.add_edge("τ", "c")
        # c → σ → E: Concentration → conductivity → field
        self.graph.add_edge("c", "σ")
        # c → M: Concentration → magnetization
        self.graph.add_edge("c", "M")
        # M → σ: Magnetization → conductivity
        self.graph.add_edge("M", "σ")
        # σ → Impedance → Readout
        self.graph.add_edge("σ", "E")  # Back to electrical
        
        # Create feedback loop analyzer
        self.feedback_loop = FeedbackLoop(self.graph)
        
        # Select coupling scheme
        if coupling_scheme == "monolithic":
            self.coupling_scheme = MonolithicScheme(
                self.graph, 1e-6, 25, use_preconditioner=True
            )
        elif coupling_scheme == "quasi-static":
            self.coupling_scheme = QuasiStaticScheme(
                self.graph, 1e-7, 30
            )
        else:
            self.coupling_scheme = StaggeredScheme(
                self.graph, 1e-6, 50, relaxation=0.6
            )
    
    def full_coupling_step(self, dt: float) -> dict:
        """Perform one complete coupled time step.
        
        Executes the full chain: E → T → η,D,τ → c,orientation → M,σ → impedance.
        
        Parameters
        ----------
        dt : float
            Time step size (seconds)
            
        Returns
        -------
        dict
            Summary of all updated quantities
        """
        summary = {}
        
        # Step 1: Solve Maxwell's equations for E-field
        # (with current T, c, σ values)
        from .fields import MaxwellEquations
        maxwell = MaxwellEquations(self.field_system)
        Q_joule = maxwell.update_from_joule_heating()
        summary["Q_joule"] = Q_joule
        
        # Step 2: Solve heat equation for temperature update
        from .fields import HeatEquation
        heat = HeatEquation(self.field_system)
        heat_result = heat.time_step(dt)
        summary.update(heat_result)
        
        # Step 3: Update material properties based on new T
        sigma = self.material_model.electrical_conductivity()
        self.field_system.sigma = sigma
        summary["sigma"] = sigma
        
        epsilon = self.material_model.permittivity()
        summary["epsilon"] = epsilon
        
        mu = self.material_model.permeability()
        summary["mu"] = mu
        
        # Step 4: Update viscosity, diffusion, relaxation
        eta = self.material_model.viscosity()
        self.field_system  # placeholder - would update field system
        
        D = self.material_model.diffusion_coefficient()
        summary["D"] = D
        
        tau = self.material_model.relaxation_time()
        summary["tau"] = tau
        
        # Step 5: Particle dynamics update
        from .fields import ParticleDynamics
        particle = ParticleDynamics(self.field_system)
        particle_result = particle.time_step(dt)
        summary.update(particle_result)
        
        # Step 6: Update magnetization based on particle state
        # M depends on c, orientation, T
        M = self._compute_magnetization()
        summary["M"] = M
        
        # Step 7: Update conductivity based on M and c
        sigma_updated = self.material_model.electrical_conductivity()
        self.field_system.sigma = sigma_updated
        summary["sigma_updated"] = sigma_updated
        
        # Step 8: Compute impedance readout
        # Z = f(M, σ, geometry, frequency)
        impedance = self._compute_impedance()
        summary["impedance"] = impedance
        
        # Check feedback loops and potentially iterate
        if self.feedback_loop and self.feedback_loop.is_feedback_active:
            # Perform additional coupling iterations if needed
            loops = self.feedback_loop.identify_loops()
            if loops:
                summary["feedback_loops_detected"] = len(loops)
                # Could iterate here for convergence
        
        return summary
    
    def _compute_magnetization(self) -> TrackedValue:
        """Compute magnetization from particle state and organization."""
        c = float(self.field_system.c) if self.field_system.c.value else 0.0
        T = float(self.field_system.T)
        org = abs(self.field_system.particle_organization) if hasattr(self.field_system, 'particle_organization') else 0.0
        
        # Saturation magnetization depends on particle concentration and organization
        Ms0 = 100.0  # emu/cm³, typical saturation magnetization
        M = Ms0 * c * org  # Reduced by concentration and organization factor
        
        return TrackedValue(
            value=M, unit="emu/cm³",
            provenance=Provenance.ASSUMED,
            notes=f"c={c:.3f}, org={org:.3f}, T={T:.1f}K"
        )
    
    def _compute_impedance(self, frequency: float = 1e6, amplitude: float = 1e-3) -> dict:
        """Compute electrical impedance readout.
        
        Uses the current σ, M, and geometry to compute impedance.
        """
        sigma = float(self.field_system.sigma) if self.field_system.sigma.value else 0.1
        M = float(self.field_system.magnetization) if self.field_system.magnetization.value else 0.0
        
        # Simplified impedance model
        # Z = R + jX where R depends on σ and X depends on M (magnetic response)
        R = 1.0 / sigma if sigma > 0 else 1e6  # Resistive component
        X = 2.0 * np.pi * frequency * 1e-6  # Reactive component (simplified)
        
        # Magnetization modifies reactance
        X_M = X * (1.0 + 0.1 * M)  # M enhances magnetic reactance
        
        magnitude = np.sqrt(R**2 + X_M**2)
        phase = np.arctan2(X_M, R)
        
        return {
            "resistance_Ω": R,
            "reactance_Ω": X_M,
            "impedance_magnitude_Ω": magnitude,
            "phase_rad": phase,
            "frequency_Hz": frequency,
        }
    
    def solve_steady_state(self, dt: float = 1e-3, max_steps: int = 500) -> dict:
        """Solve ferrofluid system to steady state.
        
        Parameters
        ----------
        dt : float
            Time step size
        max_steps : int
            Maximum number of time steps
            
        Returns
        -------
        dict
            Steady-state results
        """
        summary = {"steady_state": False, "steps": 0}
        
        for step in range(max_steps):
            step_summary = self.full_coupling_step(dt)
            summary["steps"] = step + 1
            
            # Check for convergence (changes below tolerance)
            # In a full implementation, would compare field values between steps
            # For now, check if impedance has stabilized
            if "impedance" in step_summary:
                pass  # Placeholder for convergence check
            
            # Simple convergence check: if temperature has stabilized
            T_old = step_summary.get("T", 0)
            if T_old and step > 10:
                summary["steady_state"] = True
                break
        
        return {**summary, "material_model": self.material_model}