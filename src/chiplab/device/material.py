"""chiplab.device.material - Material definition system.

Provides a comprehensive material definition system with composition, property models,
uncertainty tracking, and a material library with versioning and calibration history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np

from chiplab.schemas.common import TrackedValue, Provenance


class PropertyModelType(str, Enum):
    """Types of property models."""
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
    MAGNETIC = "magnetic"
    MECHANICAL = "mechanical"
    OPTICAL = "optical"


class PropertyModel(ABC):
    """Base class for all property models."""

    def __init__(self, provenance: Provenance = Provenance.ASSUMED):
        self.provenance = provenance

    @abstractmethod
    def evaluate(self, omega: Optional[float] = None, T: Optional[float] = None) -> Any:
        """Evaluate the property at given frequency and temperature."""
        pass


class ElectricalModel(PropertyModel):
    """Electrical property model: conductivity, permittivity, breakdown field."""

    def __init__(
        self,
        conductivity: Union[TrackedValue, float],
        conductivity_freq_dep: bool = False,
        permittivity: Union[TrackedValue, float] = 1.0,
        permittivity_freq_dep: bool = False,
        breakdown_field: Union[TrackedValue, float] = 1e6,
        provenance: Provenance = Provenance.ASSUMED,
    ):
        super().__init__(provenance=provenance)
        self.conductivity = conductivity if isinstance(conductivity, TrackedValue) \
            else TrackedValue(value=conductivity, provenance=provenance)
        self.conductivity_freq_dep = conductivity_freq_dep
        self.permittivity = permittivity if isinstance(permittivity, TrackedValue) \
            else TrackedValue(value=permittivity, provenance=provenance)
        self.permittivity_freq_dep = permittivity_freq_dep
        self.breakdown_field = breakdown_field if isinstance(breakdown_field, TrackedValue) \
            else TrackedValue(value=breakdown_field, provenance=provenance)

    def evaluate(self, omega: Optional[float] = None, T: Optional[float] = None) -> Dict[str, any]:
        """Evaluate electrical properties at given frequency (omega) and temperature (T)."""
        cond_val = float(self.conductivity)
        perm_val = float(self.permittivity)
        bf_val = float(self.breakdown_field)

        # Frequency-dependent effects (simplified Debye model)
        if omega is not None and self.conductivity_freq_dep:
            # Conductivity decreases with frequency in some materials
            omega_c = 1e9  # characteristic frequency
            cond_val = self.conductivity.value / (1 + (omega / omega_c) ** 2)

        if omega is not None and self.permittivity_freq_dep:
            # Permittivity follows Debye relaxation
            omega_c = 1e9
            perm_val = self.permittivity.value / (1 + 1j * omega / omega_c)

        return {
            "conductivity_S_m": cond_val,
            "permittivity_F_m": perm_val,
            "breakdown_field_V_m": bf_val,
        }


class ThermalModel(PropertyModel):
    """Thermal property model: conductivity, specific heat, thermal expansion."""

    def __init__(
        self,
        conductivity: Union[TrackedValue, float],
        specific_heat: Union[TrackedValue, float],
        thermal_expansion: Union[TrackedValue, float] = 0.0,
        provenance: Provenance = Provenance.ASSUMED,
    ):
        super().__init__(provenance=provenance)
        self.conductivity = conductivity if isinstance(conductivity, TrackedValue) \
            else TrackedValue(value=conductivity, provenance=provenance)
        self.specific_heat = specific_heat if isinstance(specific_heat, TrackedValue) \
            else TrackedValue(value=specific_heat, provenance=provenance)
        self.thermal_expansion = thermal_expansion if isinstance(thermal_expansion, TrackedValue) \
            else TrackedValue(value=thermal_expansion, provenance=provenance)

    def evaluate(self, T: Optional[float] = None) -> Dict[str, any]:
        """Evaluate thermal properties at given temperature."""
        k_val = float(self.conductivity)
        cp_val = float(self.specific_heat)
        alpha_val = float(self.thermal_expansion)

        # Simple temperature dependence
        if T is not None:
            # Linear temperature scaling as placeholder
            ref_T = 300.0
            k_val = k_val * (T / ref_T)
            cp_val = cp_val * (T / ref_T)

        return {
            "thermal_conductivity_W_m_K": k_val,
            "specific_J_kg_K": cp_val,
            "thermal_expansion_1_K": alpha_val,
        }


class MagneticModel(PropertyModel):
    """Magnetic property model: permeability, hysteresis, saturation, relaxation."""

    def __init__(
        self,
        permeability: Union[TrackedValue, float] = 1.0,
        saturation_magnetization: Union[TrackedValue, float] = 1.0e5,
        coercivity: Union[TrackedValue, float] = 1.0e3,
        hysteresis_loss: Union[TrackedValue, float] = 0.0,
        relaxation_time: Union[TrackedValue, float] = 1e-3,
        provenance: Provenance = Provenance.ASSUMED,
    ):
        super().__init__(provenance=provenance)
        self.permeability = permeability if isinstance(permeability, TrackedValue) \
            else TrackedValue(value=permeability, provenance=provenance)
        self.saturation_magnetization = saturation_magnetization
        self.coercivity = coercivity if isinstance(coercivity, TrackedValue) \
            else TrackedValue(value=coercivity, provenance=provenance)
        self.hysteresis_loss = hysteresis_loss
        self.relaxation_time = relaxation_time if isinstance(relaxation_time, TrackedValue) \
            else TrackedValue(value=relaxation_time, provenance=provenance)

    def evaluate(self, H: Optional[float] = None, T: Optional[float] = None) -> Dict[str, any]:
        """Evaluate magnetic properties at given field and temperature."""
        mu_val = float(self.permeability)
        Ms_val = float(self.saturation_magnetization)
        Hc_val = float(self.coercivity)

        # Simplified hysteresis loop
        if H is not None:
            # Saturation behavior
            if H > Hc_val:
                Ms_val = float(self.saturation_magnetization)
            elif H < -Hc_val:
                Ms_val = -float(self.saturation_magnetization)
            else:
                # Partial hysteresis
                Ms_val = float(self.saturation_magnetization) * (H / Hc_val)

        return {
            "permeability_mu_0": mu_val,
            "saturation_Ms_A_m": Ms_val,
            "coercivity_Hc_A_m": Hc_val,
        }


class MechanicalModel(PropertyModel):
    """Mechanical property model: Young's modulus, Poisson's ratio, yield strength, viscosity."""

    def __init__(
        self,
        young_modulus: Union[TrackedValue, float],
        poisson_ratio: Union[TrackedValue, float] = 0.3,
        yield_strength: Union[TrackedValue, float] = 1e6,
        viscosity: Union[TrackedValue, float] = 0.0,
        provenance: Provenance = Provenance.ASSUMED,
    ):
        super().__init__(provenance=provenance)
        self.young_modulus = young_modulus if isinstance(young_modulus, TrackedValue) \
            else TrackedValue(value=young_modulus, provenance=provenance)
        self.poisson_ratio = poisson_ratio if isinstance(poisson_ratio, TrackedValue) \
            else TrackedValue(value=poisson_ratio, provenance=provenance)
        self.yield_strength = yield_strength if isinstance(yield_strength, TrackedValue) \
            else TrackedValue(value=yield_strength, provenance=provenance)
        self.viscosity = viscosity if isinstance(viscosity, TrackedValue) \
            else TrackedValue(value=viscosity, provenance=provenance)

    def evaluate(self, stress: Optional[float] = None, strain: Optional[float] = None,
                 T: Optional[float] = None) -> Dict[str, any]:
        """Evaluate mechanical properties."""
        E_val = float(self.young_modulus)
        nu_val = float(self.poisson_ratio)
        sigma_y_val = float(self.yield_strength)

        return {
            "young_modulus_Pa": E_val,
            "poisson_ratio": nu_val,
            "yield_strength_Pa": sigma_y_val,
        }


class OpticalModel(PropertyModel):
    """Optical property model: refractive index, absorption."""

    def __init__(
        self,
        refractive_index: Union[TrackedValue, float] = 1.0,
        absorption_coeff: Union[TrackedValue, float] = 0.0,
        provenance: Provenance = Provenance.ASSUMED,
    ):
        super().__init__(provenance=provenance)
        self.refractive_index = refractive_index if isinstance(refractive_index, TrackedValue) \
            else TrackedValue(value=refractive_index, provenance=provenance)
        self.absorption_coeff = absorption_coeff if isinstance(absorption_coeff, TrackedValue) \
            else TrackedValue(value=absorption_coeff, provenance=provenance)

    def evaluate(self, omega: Optional[float] = None) -> Dict[str, any]:
        """Evaluate optical properties at given frequency."""
        n_val = float(self.refractive_index)
        k_val = float(self.absorption_coeff)

        # Simple Sellmeier-like dispersion
        if omega is not None:
            # ω0 is a characteristic frequency
            omega0 = 1e15
            n2 = 1.0  # simplified
            k = k_val * omega ** 2 / (omega0 ** 2 + omega ** 2)

        return {
            "refractive_index": n_val,
            "absorption_coefficient_m_1": k_val,
        }


class UncertainProperty:
    """Uncertain property: value + uncertainty + provenance."""

    def __init__(
        self,
        value: float,
        uncertainty_pct: float,
        provenance: Provenance = Provenance.ASSUMED,
        source: str = "estimated",
        notes: str = "",
    ):
        self.value = value
        self.uncertainty_pct = uncertainty_pct
        self.provenance = provenance
        self.source = source
        self.notes = notes

    @property
    def uncertainty_value(self) -> float:
        """Return the absolute uncertainty."""
        return self.value * self.uncertainty_pct / 100.0

    def __str__(self) -> str:
        return f"UncertainProperty(value={self.value}±{self.uncertainty_value:.2f}, " \
               f"provenance={self.provenance.value}, source={self.source})"


class MaterialDefinition:
    """Material definition: composition (constituents + fractions), property models."""

    def __init__(
        self,
        name: str,
        constituents: Dict[str, float],  # name -> volume fraction
        electrical: Optional[ElectricalModel] = None,
        thermal: Optional[ThermalModel] = None,
        magnetic: Optional[MagneticModel] = None,
        mechanical: Optional[MechanicalModel] = None,
        optical: Optional[OpticalModel] = None,
        provenance: Provenance = Provenance.ASSUMED,
    ):
        self.name = name
        self.constituents = constituents  # e.g., {"Fe3O4": 0.8, "gamma-Fe2O3": 0.2}
        self.electrical = electrical
        self.thermal = thermal
        self.magnetic = magnetic
        self.mechanical = mechanical
        self.optical = optical
        self.provenance = provenance
        self.calibration_history: List[Dict] = []

    def add_calibration(self, measurement: Dict, conditions: Dict):
        """Add a calibration entry to the history."""
        entry = {
            "measurement": measurement,
            "conditions": conditions,
            "provenance": self.provenance,
            "timestamp": len(self.calibration_history),  # placeholder
        }
        self.calibration_history.append(entry)

    def get_property(self, model_type: PropertyModelType,
                     omega: Optional[float] = None, T: Optional[float] = None) -> Dict:
        """Get a property model evaluation."""
        if model_type == PropertyModelType.ELECTRICAL:
            if self.electrical:
                return self.electrical.evaluate(omega, T)
            return {}
        elif model_type == PropertyModelType.THERMAL:
            if self.thermal:
                return self.thermal.evaluate(T)
            return {}
        elif model_type == PropertyModelType.MAGNETIC:
            if self.magnetic:
                return self.magnetic.evaluate(omega, T)
            return {}
        elif model_type == PropertyModelType.MECHANICAL:
            if self.mechanical:
                return self.mechanical.evaluate()
            return {}
        elif model_type == PropertyModelType.OPTICAL:
            if self.optical:
                return self.optical.evaluate(omega)
            return {}
        return {}


class MaterialLibrary:
    """Registry of materials with versioning, calibration history."""

    def __init__(self):
        self._library: Dict[str, MaterialDefinition] = {}
        self._version: Dict[str, int] = {}
        self._calibration_history: Dict[str, List] = {}

    def register(self, material: MaterialDefinition):
        """Register a material in the library."""
        name = material.name
        self._library[name] = material
        self._version[name] = material.provenance.value  # simplified versioning
        self._calibration_history[name] = material.calibration_history

    def unregister(self, name: str):
        """Unregister a material."""
        if name in self._library:
            del self._library[name]
            del self._version[name]
            del self._calibration_history[name]

    def get(self, name: str) -> Optional[MaterialDefinition]:
        """Get a material by name."""
        return self._library.get(name)

    def list_materials(self) -> List[str]:
        """List all material names."""
        return list(self._library.keys())

    def get_property(self, name: str, model_type: PropertyModelType,
                     omega: Optional[float] = None, T: Optional[float] = None) -> Dict:
        """Get a property from a registered material."""
        material = self.get(name)
        if material:
            return material.get_property(model_type, omega, T)
        return {}

    def get_all_properties(self, name: str,
                           omega: Optional[float] = None, T: Optional[float] = None) -> Dict[str, Dict]:
        """Get all property types from a registered material."""
        material = self.get(name)
        if material:
            return {
                pt.value: material.get_property(pt, omega, T)
                for pt in PropertyModelType
            }
        return {}
