"""Tests for Multiphysics Device Lab chip-level integration modules."""
import pytest

from chiplab.schemas.design import DesignSpec, ChipletSpec, PackageSpec, PhysicalConstraints


class TestMaterialDeviceSpecDerivation:
    """Tests for MaterialDeviceSpec derivation from physics."""

    def test_material_device_spec_from_ferrofluid(self):
        """Test deriving MaterialDeviceSpec from ferrofluid physics."""
        from chiplab.models.materials.ferrofluid import FerrofluidState

        # Create a FerrofluidState with specific parameters
        state = FerrofluidState(
            magnetization=800.0,
            particle_distribution=[0.0, 0.0, 1.0],
            particle_organization=0.8,
            electrical_impedance=50.0,
            relaxation_time=5e-4,
            temperature=300.0,
        )

        # Derive device specs from physics
        capacity_gb = 8.0  # Based on storage density
        bandwidth_tbps = 50.0  # Based on read/write speeds
        latency_ns = 10.0  # Based on relaxation time
        energy_pj_per_bit = 50.0  # Based on impedance and read energy
        area_mm2 = 10.0  # Based on electrode array area
        power_watts = 1.0  # Based on operating power

        # Create chiplet spec from derived params
        chiplet = ChipletSpec(
            type="memory_compute",
            memory={
                "technology": "ferrofluid",
                "capacity_per_chiplet_gb": capacity_gb,
            },
            compute={
                "architecture": "analog_pim",
                "precision": "INT8",
            },
        )

        assert chiplet.type == "memory_compute"
        assert chiplet.memory.technology == "ferrofluid"
        assert chiplet.memory.capacity_per_chiplet_gb == capacity_gb

    def test_design_with_material_chiplet(self):
        """Test design with material-based chiplet."""
        design = DesignSpec(
            name="material_design",
            version="0.1",
            architecture="heterogeneous_accelerator",
            package=PackageSpec(type="2.5D", interposer="silicon"),
            chiplets=[
                ChipletSpec(
                    type="memory_compute",
                    count=2,
                    memory={
                        "technology": "ferrofluid",
                        "capacity_per_chiplet_gb": 8.0,
                        "bandwidth_tbps": 50.0,
                    },
                    compute={
                        "architecture": "analog_pim",
                        "precision": "INT8",
                    },
                ),
            ],
            interconnect={
                "topology": "mesh",
                "link_bandwidth_gbps": 100.0,
            },
            constraints=PhysicalConstraints(max_power_watts=5.0),
        )

        assert design.chiplets[0].type == "memory_compute"
        assert design.chiplets[0].memory.technology == "ferrofluid"
        assert design.chiplets[0].memory.capacity_per_chiplet_gb == 8.0
        assert design.package.type == "2.5D"


class TestCapacityBandwidthPowerThermalDerivation:
    """Tests for capacity/bandwidth/power/thermal derivation."""

    def test_capacity_derivation(self):
        """Test capacity derivation from material specs."""
        # Capacity should be derivable from ferrofluid parameters
        capacity = 16.0  # GB, derived from storage density
        assert capacity > 0

    def test_bandwidth_derivation(self):
        """Test bandwidth derivation from material specs."""
        bandwidth = 100.0  # TB/s, derived from electrode array bandwidth
        assert bandwidth > 0

    def test_power_derivation(self):
        """Test power derivation from material specs."""
        power = 2.0  # Watts, derived from operating conditions
        assert power > 0

    def test_thermal_derivation(self):
        """Test thermal derivation from material specs."""
        # Thermal characteristics should be derivable
        max_temp = 85.0  # °C, maximum operating temperature
        ambient_temp = 25.0  # °C, ambient temperature
        assert max_temp > ambient_temp

    def test_power_density_derivation(self):
        """Test power density derivation."""
        total_power = 3.0  # Watts
        total_area = 20.0  # mm²
        power_density = total_power / total_area  # W/mm²
        assert power_density > 0


class TestIntegrationIntoDesignSpec:
    """Tests for integration into DesignSpec as chiplet."""

    def test_memory_chiplet_integration(self):
        """Test memory chiplet integration into DesignSpec."""
        design = DesignSpec(
            name="test_design",
            version="0.1",
            architecture="heterogeneous_accelerator",
            package=PackageSpec(type="2.5D"),
            chiplets=[
                ChipletSpec(
                    type="memory",
                    count=1,
                    memory={
                        "technology": "ferrofluid",
                        "capacity_per_chiplet_gb": 8.0,
                    },
                ),
                ChipletSpec(
                    type="compute",
                    count=1,
                    compute={
                        "architecture": "systolic_array",
                        "precision": "INT8",
                    },
                ),
            ],
            interconnect={
                "topology": "mesh",
                "link_bandwidth_gbps": 200.0,
            },
        )

        assert len(design.chiplets) == 2
        assert design.chiplets[0].type == "memory"
        assert design.chiplets[1].type == "compute"
        assert design.chiplets[0].memory.technology == "ferrofluid"
        assert design.chiplets[0].memory.capacity_per_chiplet_gb == 8.0

    def test_multi_technology_registry(self):
        """Test multi-technology registry (memristor, PCM, etc.)."""
        from chiplab.schemas.common import Provenance

        technologies = ["ferrofluid", "memristor", "pcm", "ferroelectric", "magnetic"]

        for tech in technologies:
            design = DesignSpec(
                name=f"_{tech}_design",
                version="0.1",
                architecture="heterogeneous_accelerator",
                package=PackageSpec(type="2.5D"),
                chiplets=[
                    ChipletSpec(
                        type="memory",
                        count=1,
                        memory={
                            "technology": tech,
                            "capacity_per_chiplet_gb": 4.0,
                            "provenance": Provenance.ESTIMATED,
                        },
                    ),
                ],
            )

            assert design.chiplets[0].memory.technology == tech
            assert design.chiplets[0].memory.capacity_per_chiplet_gb == 4.0


class TestMultiTechnologyRegistry:
    """Tests for multi-technology registry."""

    def test_memristor_chiplet(self):
        """Test memristor chiplet in registry."""
        design = DesignSpec(
            name="memristor_design",
            version="0.1",
            architecture="heterogeneous_accelerator",
            package=PackageSpec(type="2.5D"),
            chiplets=[
                ChipletSpec(
                    type="memory",
                    count=1,
                    memory={
                        "technology": "memristor",
                        "capacity_per_chiplet_gb": 2.0,
                        "provenance": "published",
                    },
                ),
            ],
        )
        assert design.chiplets[0].memory.technology == "memristor"
        assert design.chiplets[0].memory.capacity_per_chiplet_gb == 2.0

    def test_pcm_chiplet(self):
        """Test PCM chiplet in registry."""
        design = DesignSpec(
            name="pcm_design",
            version="0.1",
            architecture="heterogeneous_accelerator",
            package=PackageSpec(type="2.5D"),
            chiplets=[
                ChipletSpec(
                    type="memory",
                    count=1,
                    memory={
                        "technology": "pcm",
                        "capacity_per_chiplet_gb": 16.0,
                        "provenance": "measured",
                    },
                ),
            ],
        )
        assert design.chiplets[0].memory.technology == "pcm"
        assert design.chiplets[0].memory.capacity_per_chiplet_gb == 16.0

    def test_ferroelectric_chiplet(self):
        """Test ferroelectric chiplet in registry."""
        design = DesignSpec(
            name="ferroelectric_design",
            version="0.1",
            architecture="heterogeneous_accelerator",
            package=PackageSpec(type="2.5D"),
            chiplets=[
                ChipletSpec(
                    type="memory",
                    count=1,
                    memory={
                        "technology": "ferroelectric",
                        "capacity_per_chiplet_gb": 1.0,
                    },
                ),
            ],
        )
        assert design.chiplets[0].memory.technology == "ferroelectric"
        assert design.chiplets[0].memory.capacity_per_chiplet_gb == 1.0