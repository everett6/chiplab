"""Physics of a dilute magnetic nanoparticle suspension (ferrofluid).

Everything here is derived from a stated equation with a stated validity range
(see docs/ferrofluid_model.md). Nothing is calibrated unless the caller supplies
parameters fitted to their own measurements (see ``ferro_calibration``).

Units are SI throughout. Fields are H in A/m, magnetization M in A/m.

References
----------
[R]   R. E. Rosensweig, *Ferrohydrodynamics*, Cambridge Univ. Press, 1985.
[S74] M. I. Shliomis, Sov. Phys. Usp. 17, 153 (1974).
[N]   L. Neel, Ann. Geophys. 5, 99 (1949); W. F. Brown, Phys. Rev. 130, 1677 (1963).
[MRS] M. A. Martsenyuk, Yu. L. Raikher, M. I. Shliomis, Sov. Phys. JETP 38, 413 (1974).
[IK]  A. O. Ivanov, O. B. Kuznetsova, Phys. Rev. E 64, 041405 (2001).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

KB = 1.380649e-23  # J/K, exact (SI 2019)
MU0 = 1.25663706212e-6  # H/m, CODATA 2018

MODEL_VERSION = "ferro-particle-1.0"

# name: (default, unit, provenance of the default, note, low, high)
PARAMETER_TABLE: dict[str, tuple[float, str, str, str, float, float]] = {
    "volume_fraction": (0.01, "-", "assumed", "magnetic core volume fraction", 0.0, 0.3),
    "core_diameter_nm": (
        10.0, "nm", "assumed", "median magnetic core diameter (number distribution)", 2.0, 100.0),
    "size_sigma_ln": (0.2, "-", "assumed", "log-normal width of the core diameter", 0.0, 0.6),
    "coating_nm": (
        2.0, "nm", "assumed", "surfactant shell thickness added to the hydrodynamic radius",
        0.0, 100.0),
    "domain_magnetization_a_m": (
        4.46e5, "A/m", "published",
        "bulk magnetite saturation magnetization at room temperature [R]; "
        "nanoparticles are often lower", 1e4, 2e6),
    "anisotropy_j_m3": (
        2.0e4, "J/m^3", "assumed",
        "effective uniaxial anisotropy constant (reported nanoparticle values vary widely)",
        0.0, 1e7),
    "neel_attempt_time_s": (
        1.0e-9, "s", "assumed", "Neel attempt time, literature range about 1e-12 to 1e-9 s",
        1e-13, 1e-7),
    "viscosity_pa_s": (
        2.0e-3, "Pa*s", "assumed",
        "carrier viscosity at the operating temperature (T dependence not modelled)",
        1e-5, 1e3),
    "temperature_k": (300.0, "K", "assumed", "uniform fluid temperature", 1.0, 1000.0),
}

OPTION_DEFAULTS: dict[str, Any] = {
    "carrier_mobile": True,
    "interaction_model": "mmf1",
    "size_bins": 15,
}


@dataclass
class FerrofluidParams:
    """Material parameters of the suspension, each with a provenance label."""

    volume_fraction: float = PARAMETER_TABLE["volume_fraction"][0]
    core_diameter_nm: float = PARAMETER_TABLE["core_diameter_nm"][0]
    size_sigma_ln: float = PARAMETER_TABLE["size_sigma_ln"][0]
    coating_nm: float = PARAMETER_TABLE["coating_nm"][0]
    domain_magnetization_a_m: float = PARAMETER_TABLE["domain_magnetization_a_m"][0]
    anisotropy_j_m3: float = PARAMETER_TABLE["anisotropy_j_m3"][0]
    neel_attempt_time_s: float = PARAMETER_TABLE["neel_attempt_time_s"][0]
    viscosity_pa_s: float = PARAMETER_TABLE["viscosity_pa_s"][0]
    temperature_k: float = PARAMETER_TABLE["temperature_k"][0]
    carrier_mobile: bool = True  # False: particles immobilised (frozen or gelled carrier)
    interaction_model: str = "mmf1"  # "mmf1" [IK] or "none" (independent particles)
    size_bins: int = 15
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, (_, unit, _, _, low, high) in PARAMETER_TABLE.items():
            value = float(getattr(self, name))
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name}={value} is outside the accepted range "
                                 f"[{low}, {high}] {unit}")
            setattr(self, name, value)
        if not isinstance(self.carrier_mobile, bool):
            raise ValueError("carrier_mobile must be true or false")
        if self.interaction_model not in ("mmf1", "none"):
            raise ValueError("interaction_model must be 'mmf1' or 'none'")
        self.size_bins = int(self.size_bins)
        if not 1 <= self.size_bins <= 60:
            raise ValueError("size_bins must be between 1 and 60")
        diameter = self.core_diameter_nm
        hydrodynamic = self.volume_fraction * ((diameter + 2 * self.coating_nm) / diameter) ** 3
        if hydrodynamic > 0.5:
            raise ValueError(
                f"hydrodynamic volume fraction {hydrodynamic:.2f} exceeds 0.5: "
                "reduce volume_fraction or coating_nm")

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls) if f.name != "provenance"]

    @classmethod
    def from_dict(cls, overrides: dict[str, Any] | None = None,
                  label: str = "user_supplied") -> FerrofluidParams:
        overrides = dict(overrides or {})
        unknown = sorted(set(overrides) - set(cls.names()))
        if unknown:
            raise ValueError(f"unknown ferrofluid parameter(s) {unknown}; "
                             f"accepted: {cls.names()}")
        provenance = {name: label for name in overrides}
        return cls(**overrides, provenance=provenance)

    def describe(self) -> list[dict[str, Any]]:
        rows = []
        for name, (_, unit, default_prov, note, _, _) in PARAMETER_TABLE.items():
            rows.append({"name": name, "value": getattr(self, name), "unit": unit,
                         "provenance": self.provenance.get(name, default_prov), "note": note})
        for name in OPTION_DEFAULTS:
            rows.append({"name": name, "value": getattr(self, name), "unit": "-",
                         "provenance": self.provenance.get(name, "assumed"),
                         "note": "model option"})
        return rows


# ---------------------------------------------------------------------------
# Langevin function and field-dependent relaxation factors
# ---------------------------------------------------------------------------

_SMALL = 1e-2


def langevin(x: np.ndarray | float) -> np.ndarray:
    """L(x) = coth(x) - 1/x, with the series x/3 - x^3/45 near zero."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < _SMALL
    safe = np.where(small, 1.0, x)
    large = 1.0 / np.tanh(safe) - 1.0 / safe
    series = x / 3.0 - x**3 / 45.0 + 2.0 * x**5 / 945.0
    return np.where(small, series, large)


def langevin_derivative(x: np.ndarray | float) -> np.ndarray:
    """L'(x) = 1/x^2 - 1/sinh^2(x), with a series near zero."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < _SMALL
    safe = np.where(small, 1.0, x)
    csch2 = np.where(np.abs(safe) > 350.0, 0.0,
                     1.0 / np.sinh(np.clip(safe, -350.0, 350.0)) ** 2)
    large = 1.0 / safe**2 - csch2
    series = 1.0 / 3.0 - x**2 / 15.0 + 2.0 * x**4 / 189.0
    return np.where(small, series, large)


def log_sinhc(x: np.ndarray | float) -> np.ndarray:
    """ln(sinh(x)/x): the magnetic free energy (in kT) of one particle in a field."""
    x = np.abs(np.asarray(x, dtype=float))
    small = x < _SMALL
    safe = np.where(small, 1.0, x)
    large = safe + np.log1p(-np.exp(-2.0 * safe)) - np.log(2.0 * safe)
    return np.where(small, x**2 / 6.0 - x**4 / 180.0, large)


def brownian_field_factors(xi: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """tau_parallel/tau_B and tau_perp/tau_B in a bias field [MRS].

    parallel: d ln L / d ln xi;  perpendicular: 2 L / (xi - L).  Both tend to 1 at xi=0.
    """
    xi = np.abs(np.asarray(xi, dtype=float))
    small = xi < _SMALL
    safe = np.where(small, 1.0, xi)
    value = langevin(safe)
    parallel = safe * langevin_derivative(safe) / value
    perp = 2.0 * value / (safe - value)
    return (np.where(small, 1.0 - 2.0 * xi**2 / 15.0, parallel),
            np.where(small, 1.0 - xi**2 / 10.0, perp))


# ---------------------------------------------------------------------------
# Particle ensemble
# ---------------------------------------------------------------------------


class ParticleEnsemble:
    """A log-normal particle size distribution split into quadrature bins.

    Bins carry the *volume* fraction of each size.  A log-normal number distribution
    with median d0 and width s has a volume-weighted distribution that is log-normal
    with median d0*exp(3 s^2), so Gauss-Hermite nodes are placed on the latter.
    """

    def __init__(self, params: FerrofluidParams):
        self.params = params
        p = params
        sigma = p.size_sigma_ln
        if sigma == 0.0 or p.size_bins == 1:
            nodes, weights = np.array([0.0]), np.array([1.0])
        else:
            nodes, weights = np.polynomial.hermite.hermgauss(p.size_bins)
            weights = weights / np.sqrt(np.pi)
        shift = 3.0 * sigma**2 if len(nodes) > 1 else 0.0
        self.core_diameter_m = p.core_diameter_nm * 1e-9 * np.exp(
            shift + np.sqrt(2.0) * sigma * nodes)
        self.volume_weight = weights / weights.sum()
        self.core_volume = np.pi * self.core_diameter_m**3 / 6.0
        hydro_d = self.core_diameter_m + 2.0 * p.coating_nm * 1e-9
        self.hydrodynamic_diameter_m = hydro_d
        self.hydrodynamic_volume = np.pi * hydro_d**3 / 6.0
        self.moment = p.domain_magnetization_a_m * self.core_volume  # A m^2
        kt = KB * p.temperature_k
        # Rotational (Brownian) relaxation, tau_B = 3 eta V_h / kT [S74]
        if p.carrier_mobile:
            self.tau_brown = 3.0 * p.viscosity_pa_s * self.hydrodynamic_volume / kt
            # Stokes-Einstein translational diffusion
            self.diffusivity = kt / (3.0 * np.pi * p.viscosity_pa_s * hydro_d)
        else:
            self.tau_brown = np.full_like(self.core_volume, np.inf)
            self.diffusivity = np.zeros_like(self.core_volume)
        # Neel relaxation, tau_N = tau_0 exp(K V / kT) [N]
        self.barrier_ratio = p.anisotropy_j_m3 * self.core_volume / kt
        self.tau_neel = p.neel_attempt_time_s * np.exp(np.minimum(self.barrier_ratio, 650.0))
        self.xi_per_field = MU0 * self.moment / kt  # Langevin argument per A/m

    @property
    def typical_xi_per_field(self) -> float:
        """Volume-weighted mean Langevin argument per A/m (not the extreme size tail)."""
        return float(self.volume_weight @ self.xi_per_field)

    @property
    def bins(self) -> int:
        return len(self.core_volume)

    @property
    def tau_zero_field(self) -> np.ndarray:
        """Effective relaxation time per bin, 1/tau = 1/tau_B + 1/tau_N [S74]."""
        return 1.0 / (1.0 / self.tau_brown + 1.0 / self.tau_neel)

    def _phi(self, concentration: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
        base = self.params.volume_fraction * self.volume_weight
        if concentration is None:
            return np.broadcast_to(base, shape + (self.bins,))
        return base * (1.0 + concentration)

    def _independent(self, field_mag: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Non-interacting magnetization M_L(H) and its derivative dM_L/dH."""
        md = self.params.domain_magnetization_a_m
        xi = field_mag[..., None] * self.xi_per_field
        m_l = md * np.sum(phi * langevin(xi), axis=-1)
        dm_l = md * np.sum(phi * langevin_derivative(xi) * self.xi_per_field, axis=-1)
        return m_l, dm_l

    def effective_field(self, field_mag: np.ndarray, concentration: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Effective field H_e = H + M_L(H)/3 and dH_e/dH (first-order mean field [IK])."""
        field_mag = np.asarray(field_mag, dtype=float)
        if self.params.interaction_model == "none":
            return field_mag, np.ones_like(field_mag)
        phi = self._phi(concentration, field_mag.shape)
        m_l, dm_l = self._independent(field_mag, phi)
        return field_mag + m_l / 3.0, 1.0 + dm_l / 3.0

    def equilibrium(self, field_mag: np.ndarray, concentration: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Equilibrium magnetization per size bin (shape ...,bins) and xi at H_e."""
        field_mag = np.asarray(field_mag, dtype=float)
        h_e, _ = self.effective_field(field_mag, concentration)
        xi = h_e[..., None] * self.xi_per_field
        phi = self._phi(concentration, field_mag.shape)
        return self.params.domain_magnetization_a_m * phi * langevin(xi), xi

    def magnetization(self, field_mag: np.ndarray | float) -> np.ndarray:
        """Total equilibrium magnetization M(H) [A/m]."""
        return self.equilibrium(np.asarray(field_mag, dtype=float))[0].sum(axis=-1)

    def relaxation_times(self, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Parallel and perpendicular effective relaxation times at Langevin argument xi.

        The field dependence applies to the Brownian channel only [MRS].  The field
        dependence of Neel relaxation for randomly oriented easy axes is not modelled.
        """
        f_par, f_perp = brownian_field_factors(xi)
        neel_rate = 1.0 / self.tau_neel
        return (1.0 / (1.0 / (self.tau_brown * f_par) + neel_rate),
                1.0 / (1.0 / (self.tau_brown * f_perp) + neel_rate))

    def static_susceptibility_bins(self, field_mag: np.ndarray,
                                   concentration: np.ndarray | None = None
                                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Differential susceptibility per bin, parallel and perpendicular to a bias H.

        chi_par = dM_k/dH, chi_perp = M_k/H.  Returns (chi_par, chi_perp, xi).
        """
        field_mag = np.asarray(field_mag, dtype=float)
        h_e, dhe = self.effective_field(field_mag, concentration)
        xi = h_e[..., None] * self.xi_per_field
        phi = self._phi(concentration, field_mag.shape)
        md = self.params.domain_magnetization_a_m
        chi_par = md * phi * langevin_derivative(xi) * self.xi_per_field * dhe[..., None]
        zero = field_mag < 1e-9
        safe = np.where(zero, 1.0, field_mag)
        chi_perp = md * phi * langevin(xi) / safe[..., None]
        chi_perp = np.where(zero[..., None], chi_par, chi_perp)
        return chi_par, chi_perp, xi

    def complex_susceptibility(self, frequency_hz: np.ndarray, field_mag: float = 0.0
                               ) -> tuple[np.ndarray, np.ndarray]:
        """Small-signal chi*(f) = chi' - j chi'' parallel and perpendicular to a bias field.

        Sum of Debye terms chi_k / (1 + j w tau_k), one per size bin.
        """
        omega = 2.0 * np.pi * np.atleast_1d(np.asarray(frequency_hz, dtype=float))
        par, perp, xi = self.static_susceptibility_bins(np.asarray(field_mag, dtype=float))
        tau_par, tau_perp = self.relaxation_times(xi)
        w = omega[:, None]
        return (np.sum(par / (1.0 + 1j * w * tau_par), axis=1),
                np.sum(perp / (1.0 + 1j * w * tau_perp), axis=1))

    def remanence(self, time_s: np.ndarray, initial_field: float) -> np.ndarray:
        """M(t)/M(0) after an equilibrium magnetization at `initial_field` is released."""
        t = np.atleast_1d(np.asarray(time_s, dtype=float))
        m0, _ = self.equilibrium(np.asarray(initial_field, dtype=float))
        total = m0.sum()
        if total <= 0:
            raise ValueError("initial field and volume fraction must give a non-zero M(0)")
        return np.exp(-t[:, None] / self.tau_zero_field) @ m0 / total

    def concentration_equilibrium(self, field_mag: np.ndarray, cell_volume: np.ndarray
                                  ) -> np.ndarray:
        """Magnetophoretic Boltzmann equilibrium, relative deviation c/c0 - 1 per cell, bin.

        c(r) is proportional to sinh(xi)/xi [R], normalised so that each size bin keeps
        its particle number.  Dilute (ideal) particles; steric limits are not modelled.
        """
        field_mag = np.asarray(field_mag, dtype=float)
        energy = log_sinhc(field_mag[:, None] * self.xi_per_field)  # cells x bins
        energy = energy - energy.max(axis=0, keepdims=True)
        weight = np.exp(energy)
        mean = (cell_volume[:, None] * weight).sum(axis=0) / cell_volume.sum()
        return weight / mean - 1.0

    def diagnostics(self) -> dict[str, float]:
        """Volume-weighted dimensionless groups used to judge model validity."""
        p = self.params
        w = self.volume_weight
        kt = KB * p.temperature_k
        coupling = MU0 * self.moment**2 / (4.0 * np.pi * kt * self.hydrodynamic_diameter_m**3)
        chi_l = (p.volume_fraction * p.domain_magnetization_a_m
                 * np.sum(w * self.xi_per_field) / 3.0)
        chi0 = float(chi_l * (1.0 + chi_l / 3.0) if p.interaction_model == "mmf1" else chi_l)
        anisotropy_field = (2.0 * p.anisotropy_j_m3 / (MU0 * p.domain_magnetization_a_m))
        return {
            "initial_susceptibility": chi0,
            "saturation_magnetization_a_m": p.volume_fraction * p.domain_magnetization_a_m,
            "dipolar_coupling_lambda": float(np.sum(w * coupling)),
            "mean_barrier_kv_over_kt": float(np.sum(w * self.barrier_ratio)),
            "anisotropy_field_a_m": float(anisotropy_field),
            "tau_effective_min_s": float(self.tau_zero_field.min()),
            "tau_effective_max_s": float(self.tau_zero_field.max()),
        }

    def validity_warnings(self, max_field: float = 0.0, max_frequency: float = 0.0
                          ) -> list[str]:
        d = self.diagnostics()
        p = self.params
        out = []
        if d["dipolar_coupling_lambda"] > 1.0:
            out.append(f"dipolar coupling lambda={d['dipolar_coupling_lambda']:.2f} > 1: "
                       "chaining/aggregation likely and not modelled")
        if p.interaction_model == "mmf1" and p.volume_fraction > 0.15:
            out.append("volume_fraction > 0.15 is beyond the usual range of the first-order "
                       "mean-field interaction model")
        if p.anisotropy_j_m3 > 0 and max_field > 0.1 * d["anisotropy_field_a_m"]:
            out.append("field exceeds 10% of the anisotropy field: field-assisted Neel "
                       "switching and blocked-particle hysteresis are not modelled")
        if d["mean_barrier_kv_over_kt"] < 2.0:
            out.append("KV/kT < 2: the Neel-Arrhenius time is only an order-of-magnitude "
                       "estimate at low barriers")
        if not p.carrier_mobile:
            out.append("immobilised particles: the Langevin equilibrium is exact only for "
                       "weak fields when easy axes are frozen at random")
        if max_frequency > 1e8:
            out.append("frequency above 100 MHz: ferromagnetic resonance and intra-well "
                       "relaxation are not modelled")
        return out


# ---------------------------------------------------------------------------
# Particle-resolved reference simulation
# ---------------------------------------------------------------------------


def brownian_rotation_monte_carlo(
    xi: float,
    duration_over_tau_b: float,
    n_particles: int = 4000,
    n_steps: int = 400,
    start: str = "random",
    seed: int = 0,
) -> dict[str, Any]:
    """Stochastic rotational dynamics of individual non-interacting particle moments.

    Integrates the Ito/Euler-Maruyama form of rotational Brownian motion with unit
    vectors e and a field along z, with rotational diffusivity D_r = 1/(2 tau_B):

        de = D_r xi (z - (z.e) e) dt + sqrt(2 D_r dt) (I - e e^T) g,   then renormalise.

    Independent check of the reduced-order model: the mean of e.z tends to L(xi), and
    with xi=0 an aligned ensemble decays as exp(-t/tau_B).  Brownian rotation only.
    """
    if n_particles < 1 or n_steps < 1 or duration_over_tau_b <= 0:
        raise ValueError("n_particles, n_steps and duration must be positive")
    if start not in ("random", "aligned"):
        raise ValueError("start must be 'random' or 'aligned'")
    dt = duration_over_tau_b / n_steps
    if dt * max(1.0, abs(xi)) > 0.1:
        raise ValueError("time step too large for a stable result: increase n_steps")
    rng = np.random.default_rng(seed)
    if start == "aligned":
        e = np.tile(np.array([0.0, 0.0, 1.0]), (n_particles, 1))
    else:
        e = rng.normal(size=(n_particles, 3))
        e /= np.linalg.norm(e, axis=1, keepdims=True)
    z = np.array([0.0, 0.0, 1.0])
    diffusion = 0.5
    mean = np.empty(n_steps + 1)
    mean[0] = e[:, 2].mean()
    for step in range(n_steps):
        drift = diffusion * xi * (z - e[:, 2:3] * e) * dt
        noise = np.sqrt(2.0 * diffusion * dt) * rng.normal(size=e.shape)
        noise -= np.sum(noise * e, axis=1, keepdims=True) * e
        e = e + drift + noise
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        mean[step + 1] = e[:, 2].mean()
    return {
        "time_over_tau_b": np.linspace(0.0, duration_over_tau_b, n_steps + 1),
        "mean_alignment": mean,
        "standard_error": float(e[:, 2].std(ddof=1) / np.sqrt(n_particles))
        if n_particles > 1 else float("nan"),
        "moments": e,
    }
