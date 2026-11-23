"""Tests for DesignSpec YAML loading, TrackedValue provenance, and validate_design."""

import pytest
import yaml
from chiplab.schemas.design import DesignSpec, ChipletSpec, MemorySpec, ComputeSpec, PackageSpec, InterconnectSpec, PhysicalConstraints
from chiplab.schemas.common import FidelityLevel, Provenance, unwrap_value, get_provenance
from chiplab.engine.validator import validate_design


class TestDesignSpecYAMLLoading:
    """Test DesignSpec YAML loading from various project files."""

    def test_echo_v0_1_yaml_loading(self):
        """Load echo_v0_1.yaml and verify DesignSpec."""
        with open("/home/everett/Chip Lab/projects/echo/echo_v0_1.yaml") as f:
            data = yaml.safe_load(f)
        design = DesignSpec.from_yaml(yaml.dump(data))
        assert design.name == "Echo Chiplet v0.1"
        assert design.version == "0.1"
        assert len(design.chiplets) == 3
        assert design.chiplets[0].type == "DRAM"
        assert design.chiplets[1].type == "PIM"
        assert design.chiplets[2].type == "SIMD"

    def test_echo_v0_2_yaml_loading(self):
        """Load echo_v0_2.yaml and verify DesignSpec."""
        with open("/home/everett/Chip Lab/projects/echo/echo_v0_2.yaml") as f:
            data = yaml.safe_load(f)
        design = DesignSpec.from_yaml(yaml.dump(data))
        assert design.name == "Echo Chiplet v0.2"
        assert design.version == "0.2"
        assert design.parent_version == "0.1"
        assert len(design.chiplets) == 3
        # v0.2 has HBM3 chiplet added
        types = [c.type for c in design.chiplets]
        assert "DRAM" in types
        assert "PIM" in types
        assert "HBM3" in types

    def test_transformer_accelerator_yaml_loading(self):
        """Load transformer_accelerator.yaml and verify DesignSpec."""
        with open("/home/everett/Chip Lab/examples/transformer_accelerator.yaml") as f:
            data = yaml.safe_load(f)
        design = DesignSpec.from_yaml(yaml.dump(data))
        assert design.name == "Transformer Accelerator"
        assert design.version == "1.0"
        assert len(design.chiplets) == 3
        types = [c.type for c in design.chiplets]
        assert "HBM3" in types
        assert "systolic_array" in types
        assert "vector_processor" in types


class TestTrackedValueProvenance:
    """Test TrackedValue provenance tracking."""

    def test_provenance_types(self):
        """Test that Provenance enum has correct values."""
        from chiplab.schemas.common import Provenance
        assert Provenance.MEASURED.value == "measured"
        assert Provenance.PUBLISHED.value == "published"
        assert Provenance.ESTIMATED.value == "estimated"
        assert Provenance.ASSUMED.value == "assumed"

    def test_unwrap_value(self):
        """Test unwrap_value function."""
        from chiplab.schemas.common import unwrap_value
        # Raw value
        assert unwrap_value(32.0) == 32.0
        # TrackedValue
        from chiplab.schemas.common import TrackedValue
        tv = TrackedValue(value=32.0, unit="GB", provenance=Provenance.PUBLISHED)
        assert unwrap_value(tv) == 32.0
        assert get_provenance(tv) == Provenance.PUBLISHED

    def test_get_provenance(self):
        """Test get_provenance function."""
        from chiplab.schemas.common import get_provenance, Provenance
        assert get_provenance(32.0) == Provenance.ASSUMED
        from chiplab.schemas.common import TrackedValue
        tv = TrackedValue(value=32.0, provenance=Provenance.MEASURED)
        assert get_provenance(tv) == Provenance.MEASURED


class TestValidateDesign:
    """Test validate_design with various designs."""

    def test_valid_echo_v0_1(self):
        """echo_v0_1.yaml is valid: its PIM chiplet is a memoryless compute chiplet."""
        with open("/home/everett/Chip Lab/projects/echo/echo_v0_1.yaml") as f:
            design = DesignSpec.from_yaml(f.read())
        result = validate_design(design, allow_estimated_defaults=True)
        assert result.valid, result.errors
        pim = next(c for c in design.chiplets if c.id == "pim_0")
        assert pim.memory is None and pim.compute is not None

    def test_valid_echo_v0_1_with_defaults_false(self):
        with open("/home/everett/Chip Lab/projects/echo/echo_v0_1.yaml") as f:
            design = DesignSpec.from_yaml(f.read())
        result = validate_design(design, allow_estimated_defaults=False)
        assert result.valid, result.errors

    def test_explicit_zero_capacity_is_rejected(self):
        """A memory block that says 0 GB is an error; omit `memory:` for memoryless chiplets."""
        design = DesignSpec(name="zero_cap", package=PackageSpec(type="chiplets"), chiplets=[
            ChipletSpec(id="pim", type="PIM", memory=MemorySpec(technology="pim",
                                                                capacity_per_chiplet_gb=0.0),
                        compute=ComputeSpec(architecture="pim", peak_tops=10.0))])
        result = validate_design(design, allow_estimated_defaults=True)
        assert not result.valid
        assert any("capacity_per_chiplet_gb must be positive" in e for e in result.errors)

    def test_monolithic_with_multiple_chiplets(self):
        """Monolithic package with >1 chiplet should fail."""
        design = DesignSpec(
            name="bad_design",
            package=PackageSpec(type="monolithic"),
            chiplets=[
                ChipletSpec(type="compute"),
                ChipletSpec(type="memory"),
            ],
        )
        result = validate_design(design, allow_estimated_defaults=True)
        assert result.valid == False
        assert "Monolithic package type cannot have more than 1 chiplet" in str(result.errors)

    def test_memory_chiplet_without_MemorySpec_allow_defaults_false(self):
        """Memory chiplet without MemorySpec should fail with allow_estimated_defaults=False."""
        design = DesignSpec(
            name="bad_design",
            package=PackageSpec(type="chiplets"),
            chiplets=[
                ChipletSpec(type="memory"),  # No MemorySpec provided
            ],
        )
        result = validate_design(design, allow_estimated_defaults=False)
        assert result.valid == False
        assert any("memory type requires MemorySpec" in e for e in result.errors)

    def test_memory_chiplet_without_MemorySpec_allow_defaults_true(self):
        """Memory chiplet without MemorySpec should pass with warning with allow_estimated_defaults=True."""
        design = DesignSpec(
            name="bad_design",
            package=PackageSpec(type="chiplets"),
            chiplets=[
                ChipletSpec(type="memory"),  # No MemorySpec provided
            ],
        )
        result = validate_design(design, allow_estimated_defaults=True)
        assert result.valid == True
        # Should produce a warning, not an error
        assert len(result.warnings) > 0

    def test_design_with_parent_version(self):
        """Design with parent_version should work."""
        with open("/home/everett/Chip Lab/projects/echo/echo_v0_2.yaml") as f:
            data = yaml.safe_load(f)
        design = DesignSpec.from_yaml(yaml.dump(data))
        assert design.parent_version == "0.1"
        result = validate_design(design, allow_estimated_defaults=True)
        assert result.valid, result.errors

    def test_design_no_chiplets(self):
        """Design with no chiplets should fail."""
        design = DesignSpec(name="no_chiplets", chiplets=[])
        result = validate_design(design, allow_estimated_defaults=True)
        assert result.valid == False
        assert "Design must have at least one chiplet" in str(result.errors)
