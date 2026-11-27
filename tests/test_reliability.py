"""Tests for ChipLab reliability module.

Tests all degradation, aging, GPU/CPU fallback, and provenance tracking.
"""

import os
import numpy as np
import pytest

from chiplab.schemas.common import TrackedValue, Provenance
from chiplab.reliability.degradation import (
    state_drift,
    pcms_endurance,
    memristor_cycling_endurance,
    ferrofluid_retention,
    sram_retention,
    blacks_equation,
    nbti_aging,
    pbti_aging,
    hci_aging,
)

from chiplab.reliability.aging import (
    arrhenius_accelerated_aging,
    monte_carlo_lifetime,
    monte_carlo_ttf_distribution,
)

from chiplab.core.gpu import GPUBackend


@pytest.fixture
def reliability_backend():
    """Provide a GPUBackend instance for testing."""
    return GPUBackend(prefer="auto", device="auto")


@pytest.fixture
def cpu_backend():
    """Provide a CPU-only GPUBackend instance."""
    return GPUBackend(prefer="cpu", device="cpu")


# ============================================================
# Degradation Model Tests
# ============================================================


class TestStateDrift:
    """Tests for state_drift function."""

    def test_state_drift_basic(self):
        """Test basic state drift."""
        result = state_drift(initial_value=1.0, drift_rate=1e-9, time_elapsed=1e9)
        assert isinstance(result, TrackedValue)
        assert float(result) == 1.0 + 1e-9 * 1e9  # = 2.0
        assert result.provenance == "estimated"

    def test_state_drift_with_provenance(self):
        """Test state_drift with custom provenance."""
        result = state_drift(
            initial_value=5.0, drift_rate=0.1, time_elapsed=10, provenance="measured"
        )
        assert isinstance(result, TrackedValue)
        assert float(result) == 5.0 + 0.1 * 10  # = 6.0
        assert result.provenance == "measured"


class TestPCMSEndurance:
    """Tests for pcms_endurance function."""

    def test_pcms_endurance_basic(self):
        """Test basic PCM endurance."""
        result = pcms_endurance(cycle_count=1000, endurance_limit=10**6)
        assert isinstance(result, TrackedValue)
        assert result.unit == "cycles"
        assert result.value < 10**6  # Some degradation
        assert result.provenance == "estimated"

    def test_pcms_endurance_at_limit(self):
        """Test PCM endurance at cycle limit."""
        result = pcms_endurance(cycle_count=10**6, endurance_limit=10**6)
        assert isinstance(result, TrackedValue)
        assert result.value == 999000  # Small degradation at 1e6 cycles with 1e-9 per cycle
        assert result.unit == "cycles"


class TestMemristorCyclingEndurance:
    """Tests for memristor_cycling_endurance function."""

    def test_memristor_endurance_basic(self):
        """Test basic memristor endurance."""
        result = memristor_cycling_endurance(cycle_count=10**5, endurance_limit=10**9)
        assert isinstance(result, TrackedValue)
        assert result.unit == "cycles"
        assert result.value < 10**9  # Some degradation
        assert result.provenance == "estimated"


class TestFerrofluidRetention:
    """Tests for ferrofluid_retention function."""

    def test_ferrofluid_retention_basic(self):
        """Test basic ferrofluid retention."""
        result = ferrofluid_retention(time_elapsed=3600)
        assert isinstance(result, TrackedValue)
        assert 0 <= float(result) <= 1  # Fraction between 0 and 1
        assert result.unit == "fraction"
        assert result.provenance == "estimated"

    def test_ferrofluid_retention_short_time(self):
        """Test ferrofluid retention with short time."""
        result = ferrofluid_retention(time_elapsed=0.1)
        assert isinstance(result, TrackedValue)
        assert float(result) > 0.5  # Should retain most


class TestSRAMRetention:
    """Tests for sram_retention function."""

    def test_sram_retention_basic(self):
        """Test basic SRAM retention."""
        result = sram_retention(temperature_k=300, time_elapsed=1e3)
        assert isinstance(result, TrackedValue)
        assert 0 <= float(result) <= 1  # Fraction between 0 and 1
        assert result.unit == "fraction"
        assert result.provenance == "estimated"

    def test_sram_retention_temperature_dependence(self):
        """Test SRAM retention temperature dependence."""
        result_cold = sram_retention(temperature_k=260, time_elapsed=1e3)
        result_hot = sram_retention(temperature_k=340, time_elapsed=1e3)
        # Higher temperature should give lower retention
        assert float(result_cold) >= float(result_hot)


class TestBlacksEquation:
    """Tests for blacks_equation function."""

    def test_blacks_equation_basic(self):
        """Test basic Black's equation."""
        result = blacks_equation(current=1e-3, time_elapsed=1e3)
        assert isinstance(result, TrackedValue)
        assert 0 <= float(result) <= 1  # Degradation fraction
        assert result.unit == "fraction"
        assert result.provenance == "estimated"

    def test_blacks_equation_with_provenance(self):
        """Test Black's equation with custom provenance."""
        result = blacks_equation(
            current=1e-3, time_elapsed=1e3, provenance="measured"
        )
        assert result.provenance == "measured"


class TestNBTIAgng:
    """Tests for nbti_aging function."""

    def test_nbti_aging_basic(self):
        """Test basic NBTI aging."""
        result = nbti_aging(stress_time=1e4, temperature_k=300, voltage=1.0)
        assert isinstance(result, TrackedValue)
        assert result.unit == "V"
        assert float(result) >= 0  # Vth shift should be non-negative
        assert result.provenance == "estimated"

    def test_nbti_aging_temperature_dependence(self):
        """Test NBTI aging temperature dependence."""
        result_cold = nbti_aging(stress_time=1e4, temperature_k=260, voltage=1.0)
        result_hot = nbti_aging(stress_time=1e4, temperature_k=340, voltage=1.0)
        # Lower temperature should give larger Vth shift (arrhenius factor)
        assert float(result_cold) >= float(result_hot)


class TestPBTIAgng:
    """Tests for pbti_aging function."""

    def test_pbti_aging_basic(self):
        """Test basic PBTI aging."""
        result = pbti_aging(stress_time=1e4, temperature_k=300, voltage=1.0)
        assert isinstance(result, TrackedValue)
        assert result.unit == "V"
        assert float(result) >= 0  # Vth shift should be non-negative
        assert result.provenance == "estimated"


class TestHCIAgng:
    """Tests for hci_aging function."""

    def test_hci_aging_basic(self):
        """Test basic HCI aging."""
        result = hci_aging(drain_voltage=1.0, gate_voltage=0.5, time_elapsed=1e3)
        assert isinstance(result, TrackedValue)
        assert result.unit == "arbitrary"
        assert float(result) >= 0  # Degradation should be non-negative
        assert result.provenance == "estimated"


# ============================================================
# Aging Model Tests
# ============================================================


class TestArrheniusAcceleratedAging:
    """Tests for arrhenius_accelerated_aging function."""

    def test_arrhenius_basic(self):
        """Test basic Arrhenius aging."""
        result = arrhenius_accelerated_aging(
            temp_operating_k=350, temp_reference_k=300, time_operating=1e6
        )
        assert isinstance(result, TrackedValue)
        assert result.unit == "seconds"
        assert float(result) > 1e6  # Should be accelerated
        assert result.provenance == "estimated"

    def test_arrhenius_same_temp(self):
        """Test Arrhenius aging at same operating and reference temperature."""
        result = arrhenius_accelerated_aging(
            temp_operating_k=300, temp_reference_k=300, time_operating=1e6
        )
        assert isinstance(result, TrackedValue)
        # At same temp, equivalent time should equal operating time
        assert abs(float(result) - 1e6) < 1.0  # Should be very close to 1e6


class TestMonteCarloLifeTime:
    """Tests for monte_carlo_lifetime function."""

    def test_monte_carlo_lifetime_basic(self):
        """Test basic Monte Carlo lifetime."""
        result = monte_carlo_lifetime(
            n_samples=100,
            activation_energy_mean_ev=0.7,
            temperature_k=300,
            current_density_a_per_m2=1e6,
        )
        assert isinstance(result, TrackedValue)
        assert result.value > 0  # MTTF should be positive
        assert result.unit == "seconds"
        assert result.provenance == "estimated"

    def test_monte_carlo_lifetime_with_gpu(self, reliability_backend):
        """Test Monte Carlo lifetime with GPU backend."""
        result = monte_carlo_lifetime(
            n_samples=100,
            activation_energy_mean_ev=0.7,
            temperature_k=300,
            current_density_a_per_m2=1e6,
            device="auto",
        )
        assert isinstance(result, TrackedValue)

    def test_monte_carlo_lifetime_different_temps(self):
        """Test Monte Carlo lifetime temperature dependence."""
        result_cold = monte_carlo_lifetime(
            n_samples=50,
            activation_energy_mean_ev=0.7,
            temperature_k=260,
            current_density_a_per_m2=1e6,
        )
        result_hot = monte_carlo_lifetime(
            n_samples=50,
            activation_energy_mean_ev=0.7,
            temperature_k=340,
            current_density_a_per_m2=1e6,
        )
        # Higher temperature should give shorter MTTF
        assert float(result_cold.value) >= float(result_hot.value)


class TestMonteCarloTTFDistribution:
    """Tests for monte_carlo_ttf_distribution function."""

    def test_monte_carlo_ttf_basic(self):
        """Test basic Monte Carlo TTF distribution."""
        result = monte_carlo_ttf_distribution(
            n_samples=100,
            activation_energy_mean_ev=0.7,
            temperature_k=300,
            current_density_a_per_m2=1e6,
        )
        assert isinstance(result, TrackedValue)
        assert result.value > 0  # TTF should be positive
        assert result.unit == "seconds"
        assert result.provenance == "estimated"

    def test_monte_carlo_ttf_with_gpu(self, reliability_backend):
        """Test Monte Carlo TTF with GPU backend."""
        result = monte_carlo_ttf_distribution(
            n_samples=100,
            activation_energy_mean_ev=0.7,
            temperature_k=300,
            current_density_a_per_m2=1e6,
            device="auto",
        )
        assert isinstance(result, TrackedValue)

    def test_monte_carlo_ttf_percentiles(self):
        """Test TTF distribution percentiles."""
        result = monte_carlo_ttf_distribution(
            n_samples=1000,
            activation_energy_mean_ev=0.7,
            temperature_k=300,
            current_density_a_per_m2=1e6,
        )
        # Check that the value (median TTF) is positive
        assert result.value > 0


# ============================================================
# GPU/CPU Fallback Tests
# ============================================================


class TestGPUCPUFallback:
    """Tests for GPU/CPU fallback behavior."""

    def test_gpu_backend_initialization(self, cpu_backend):
        """Test GPUBackend initialization falls back to CPU."""
        assert cpu_backend is not None
        assert cpu_backend.is_gpu == False
        assert cpu_backend.device == "cpu"

    def test_gpu_backend_gpu_if_available(self, reliability_backend):
        """Test GPUBackend initialization with GPU if available."""
        assert reliability_backend is not None
        # Backend may be GPU or CPU depending on availability

    def test_monte_carlo_cpu_backend(self, cpu_backend):
        """Test Monte Carlo functions work with CPU backend."""
        result = monte_carlo_lifetime(
            n_samples=50,
            activation_energy_mean_ev=0.7,
            temperature_k=300,
            current_density_a_per_m2=1e6,
            device="cpu",
        )
        assert isinstance(result, TrackedValue)

    def test_monte_carlo_ttf_cpu_backend(self, cpu_backend):
        """Test Monte Carlo TTF functions work with CPU backend."""
        result = monte_carlo_ttf_distribution(
            n_samples=50,
            activation_energy_mean_ev=0.7,
            temperature_k=300,
            current_density_a_per_m2=1e6,
            device="cpu",
        )
        assert isinstance(result, TrackedValue)

    def test_all_degradation_provenance_tracking(self):
        """Test that all degradation models track provenance correctly."""
        # Test each degradation model returns TrackedValue with provenance
        models = [
            ("state_drift", state_drift(initial_value=1.0)),
            ("pcms_endurance", pcms_endurance(cycle_count=100)),
            ("memristor_endurance", memristor_cycling_endurance(cycle_count=100)),
            ("ferrofluid", ferrofluid_retention(time_elapsed=100)),
            ("sram_retention", sram_retention(temperature_k=300, time_elapsed=100)),
            ("blacks_eq", blacks_equation(current=1e-3, time_elapsed=100)),
            ("nbti", nbti_aging(stress_time=100)),
            ("pbti", pbti_aging(stress_time=100)),
            ("hci", hci_aging(drain_voltage=1.0, gate_voltage=0.5)),
        ]

        for name, result in models:
            assert isinstance(result, TrackedValue), f"{name} should return TrackedValue"
            assert hasattr(result, "provenance"), f"{name} should have provenance"
            assert result.provenance in [
                "measured",
                "published",
                "estimated",
                "assumed",
            ], f"{name} should have valid provenance"

    def test_all_aging_provenance_tracking(self):
        """Test that all aging models track provenance correctly."""
        # Test each aging model returns TrackedValue with provenance
        models = [
            ("arrhenius", arrhenius_accelerated_aging(temp_operating_k=350)),
            ("monte_carlo_lt", monte_carlo_lifetime(n_samples=50)),
            ("monte_carlo_ttf", monte_carlo_ttf_distribution(n_samples=50)),
        ]

        for name, result in models:
            assert isinstance(result, TrackedValue), f"{name} should return TrackedValue"
            assert hasattr(result, "provenance"), f"{name} should have provenance"
            assert result.provenance in [
                "measured",
                "published",
                "estimated",
                "assumed",
            ], f"{name} should have valid provenance"
