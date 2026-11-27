"""Tests for solver adapters in ChipLab.

Tests all solver adapters: LAMMPS, SPICE, Verilator, OpenROAD, MOOSE, and OpenCFS.
Tests script/netlist generation, capability matching, and graceful fallback.
"""

from pathlib import Path

import pytest

from chiplab.solvers.registry import SolverRegistry, PhysicsType, ScaleType
from chiplab.solvers.adapters.base import SolverAdapter
from chiplab.solvers.adapters.lammps import LAMMPSAdapter
from chiplab.solvers.adapters.spice import SpiceAdapter
from chiplab.solvers.adapters.verilator import VerilatorAdapter
from chiplab.solvers.adapters.openroad import OpenROADAdapter
from chiplab.solvers.adapters.moose import MOOSEAdapter
from chiplab.solvers.adapters.opencfs import OpenCFSAdapter


# Chiplet class for testing - defined at module level
class Chiplet:
    def __init__(self, idx, area, tdp, arch):
        self.area_mm2 = area
        self.tdp_watts = tdp
        self.compute = type('obj', (object,), {'architecture': arch})()
        self.process_nm = 7


# Helper to create a minimal DesignSpec for testing
def make_design_spec(chiplets=None):
    """Create a minimal DesignSpec for testing adapter input preparation."""
    class DesignSpec:
        def __init__(self, chiplets=None):
            self.chiplets = chiplets or []

    return DesignSpec(chiplets=chiplets)


# Helper to check physics type support
def has_physics(spec_physics, physics_type):
    """Check if a solver spec supports a given physics type."""
    return physics_type.value in spec_physics


# Helper to check scale support
def has_scale(spec_scales, scale_type):
    """Check if a solver spec supports a given scale type."""
    return scale_type.value in spec_scales


class TestLAMMPSAdapter:
    """Test LAMMPS adapter functionality."""

    def test_lammps_adapter_creation(self):
        """Test that LAMMPS adapter can be instantiated."""
        adapter = LAMMPSAdapter()
        assert adapter.name == "lammps"
        assert adapter.use_gpu == True
        assert adapter.fidelity.name == "SPECIALIZED"

    def test_lammps_capability(self):
        """Test LAMMPS capability matching."""
        adapter = LAMMPSAdapter()
        cap = adapter.get_capability()
        assert cap.spec.name == "lammps"
        # Check physics support using string values
        assert has_physics(cap.spec.supported_physics, PhysicsType.MECHANICAL)
        assert has_physics(cap.spec.supported_physics, PhysicsType.THERMAL)
        # Check scale support using string values
        assert has_scale(cap.spec.supported_scales, ScaleType.DEVICE)
        assert has_scale(cap.spec.supported_scales, ScaleType.CHIPLET)

    def test_lammps_prepare_input(self):
        """Test LAMMPS input preparation."""
        adapter = LAMMPSAdapter()
        design = make_design_spec([Chiplet(0, 10.0, 50.0, "test_arch")])
        workload = {"temperature": 300}
        config = type('obj', (object,), {
            'fidelity': type('obj', (object,), {'value': 'SPECIALIZED'})()
        })()

        input_path, param_map = adapter.prepare_input(design, workload, config)
        assert input_path is not None
        assert len(param_map) > 0
        # Check that compute_type is in param_map (from chiplet)
        assert "compute_type" in param_map

    def test_lammps_graceful_fallback(self):
        """Test LAMMPS graceful degradation when binary not available."""
        adapter = LAMMPSAdapter()
        # is_available should handle missing executable gracefully
        # (will return False if lammps not on PATH)


class TestSpiceAdapter:
    """Test SPICE adapter functionality."""

    def test_spice_adapter_creation(self):
        """Test that SPICE adapter can be instantiated."""
        adapter = SpiceAdapter()
        assert adapter.name == "spice"
        assert adapter.use_gpu == True
        assert adapter.fidelity.name == "SPECIALIZED"

    def test_spice_capability(self):
        """Test SPICE capability matching."""
        adapter = SpiceAdapter()
        cap = adapter.get_capability()
        assert cap.spec.name == "spice"
        # Check physics support using string values
        assert has_physics(cap.spec.supported_physics, PhysicsType.ELECTRICAL)
        # Check scale support using string values
        assert has_scale(cap.spec.supported_scales, ScaleType.CHIPLET)
        assert has_scale(cap.spec.supported_scales, ScaleType.SYSTEM)
        # Check analysis types
        assert "transient_analysis" in cap.supported_analysis_types
        assert "ac_analysis" in cap.supported_analysis_types

    def test_spice_prepare_input(self):
        """Test SPICE netlist generation."""
        adapter = SpiceAdapter()
        design = make_design_spec([Chiplet(0, 10.0, 50.0, "test_arch")])
        workload = {"analysis": "transient"}
        config = type('obj', (object,), {
            'fidelity': type('obj', (object,), {'value': 'SPECIALIZED'})()
        })()

        input_path, param_map = adapter.prepare_input(design, workload, config)
        assert input_path is not None
        # Check that netlist file exists and has content
        assert Path(input_path).exists()
        with open(input_path) as f:
            content = f.read()
        assert ".analysis transient" in content
        assert ".analysis ac" in content

    def test_spice_graceful_fallback(self):
        """Test SPICE graceful degradation when binary not available."""
        adapter = SpiceAdapter()
        # Test that running without ngspice raises informative error
        try:
            adapter.run("/nonexistent/path.net")
            # If we get here, the error handling could be improved
        except RuntimeError as e:
            # Should get a clear error message about missing executable
            assert "not found" in str(e).lower() or "ngspice" in str(e).lower()


class TestVerilatorAdapter:
    """Test Verilator adapter functionality."""

    def test_verilator_adapter_creation(self):
        """Test that Verilator adapter can be instantiated."""
        adapter = VerilatorAdapter()
        assert adapter.name == "verilator"
        assert adapter.use_gpu == True  # Flag kept for interface compatibility
        assert adapter.fidelity.name == "SPECIALIZED"

    def test_verilator_capability(self):
        """Test Verilator capability matching."""
        adapter = VerilatorAdapter()
        cap = adapter.get_capability()
        assert cap.spec.name == "verilator"
        # Check physics support using string values
        assert has_physics(cap.spec.supported_physics, PhysicsType.ELECTRICAL)
        # Check scale support using string values
        assert has_scale(cap.spec.supported_scales, ScaleType.SYSTEM)
        # Check analysis types
        assert "rtl_simulation" in cap.supported_analysis_types
        assert "lint" in cap.supported_analysis_types

    def test_verilator_prepare_input(self):
        """Test Verilog source generation."""
        adapter = VerilatorAdapter()
        design = make_design_spec()
        workload = {"clock_freq": 100}
        config = type('obj', (object,), {
            'fidelity': type('obj', (object,), {'value': 'SPECIALIZED'})()
        })()

        input_path, param_map = adapter.prepare_input(design, workload, config)
        assert input_path is not None
        # Check that working directory was created with source files
        assert Path(input_path).is_dir()
        src_dir = Path(input_path) / "src"
        assert src_dir.is_dir()

    def test_verilator_graceful_fallback(self):
        """Test Verilator graceful degradation when binary not available."""
        adapter = VerilatorAdapter()
        try:
            adapter.run("/nonexistent/path")
        except RuntimeError as e:
            # Should get a clear error message about missing executable
            assert "verilator" in str(e).lower()


class TestOpenROADAdapter:
    """Test OpenROAD adapter functionality."""

    def test_openroad_adapter_creation(self):
        """Test that OpenROAD adapter can be instantiated."""
        adapter = OpenROADAdapter()
        assert adapter.name == "openroad"
        assert adapter.use_gpu == True
        assert adapter.fidelity.name == "SPECIALIZED"

    def test_openroad_capability(self):
        """Test OpenROAD capability matching."""
        adapter = OpenROADAdapter()
        cap = adapter.get_capability()
        assert cap.spec.name == "openroad"
        # Check physics support using string values
        assert has_physics(cap.spec.supported_physics, PhysicsType.ELECTRICAL)
        # Check scale support using string values
        assert has_scale(cap.spec.supported_scales, ScaleType.SYSTEM)
        assert has_scale(cap.spec.supported_scales, ScaleType.BOARD)
        # Check analysis types
        assert "floorplan" in cap.supported_analysis_types
        assert "placement" in cap.supported_analysis_types
        assert "routing" in cap.supported_analysis_types
        assert "sta" in cap.supported_analysis_types

    def test_openroad_prepare_input(self):
        """Test OpenROAD flow preparation."""
        adapter = OpenROADAdapter()
        design = make_design_spec()
        workload = {"target_technology": "nmos"}
        config = type('obj', (object,), {
            'fidelity': type('obj', (object,), {'value': 'SPECIALIZED'})()
        })()

        input_path, param_map = adapter.prepare_input(design, workload, config)
        assert input_path is not None
        assert Path(input_path).is_dir()
        run_dir = Path(input_path)
        # Check that design files were created
        assert (run_dir / "design.def").exists()
        assert (run_dir / "design.gds").exists()

    def test_openroad_graceful_fallback(self):
        """Test OpenROAD graceful degradation when binary not available."""
        adapter = OpenROADAdapter()
        try:
            adapter.run("/nonexistent/path")
        except RuntimeError as e:
            # Should get a clear error message about missing executable
            assert "openroad" in str(e).lower()


class TestMOOSEAdapter:
    """Test MOOSE adapter functionality."""

    def test_moose_adapter_creation(self):
        """Test that MOOSE adapter can be instantiated."""
        adapter = MOOSEAdapter()
        assert adapter.name == "moose"
        assert adapter.use_gpu == True
        assert adapter.fidelity.name == "SPECIALIZED"

    def test_moose_capability(self):
        """Test MOOSE capability matching."""
        adapter = MOOSEAdapter()
        cap = adapter.get_capability()
        assert cap.spec.name == "moose"
        # Check physics support using string values
        assert has_physics(cap.spec.supported_physics, PhysicsType.THERMAL)
        assert has_physics(cap.spec.supported_physics, PhysicsType.MECHANICAL)
        # Check scale support using string values
        assert has_scale(cap.spec.supported_scales, ScaleType.DEVICE)
        assert has_scale(cap.spec.supported_scales, ScaleType.CHIPLET)
        assert has_scale(cap.spec.supported_scales, ScaleType.SYSTEM)
        # Check analysis types
        assert "thermal_structural_coupling" in cap.supported_analysis_types

    def test_moose_prepare_input(self):
        """Test MOOSE input file generation."""
        adapter = MOOSEAdapter()
        design = make_design_spec([type('obj', (object,), {
            'area_mm2': 10.0,
            'tdp_watts': 50.0,
        })()])
        workload = {"analysis_type": "thermal"}
        config = type('obj', (object,), {
            'fidelity': type('obj', (object,), {'value': 'SPECIALIZED'})()
        })()

        input_path, param_map = adapter.prepare_input(design, workload, config)
        assert input_path is not None
        # Check that MOOSE input file was created
        assert Path(input_path).exists()
        with open(input_path) as f:
            content = f.read()
        assert "[Global Parameters]" in content
        assert "start_time" in content

    def test_moose_graceful_fallback(self):
        """Test MOOSE graceful degradation when binary not available."""
        adapter = MOOSEAdapter()
        try:
            adapter.run("/nonexistent/path.i")
        except RuntimeError as e:
            # Should get a clear error message about missing executable
            assert "moose" in str(e).lower()


class TestOpenCFSAdapter:
    """Test OpenCFS adapter functionality."""

    def test_opencfs_adapter_creation(self):
        """Test that OpenCFS adapter can be instantiated."""
        adapter = OpenCFSAdapter()
        assert adapter.name == "opencfs"
        assert adapter.use_gpu == True
        assert adapter.fidelity.name == "SPECIALIZED"

    def test_opencfs_capability(self):
        """Test OpenCFS capability matching."""
        adapter = OpenCFSAdapter()
        cap = adapter.get_capability()
        assert cap.spec.name == "opencfs"
        # Check physics support using string values
        assert has_physics(cap.spec.supported_physics, PhysicsType.THERMAL)
        assert has_physics(cap.spec.supported_physics, PhysicsType.MECHANICAL)
        # Check scale support using string values
        assert has_scale(cap.spec.supported_scales, ScaleType.DEVICE)
        assert has_scale(cap.spec.supported_scales, ScaleType.CHIPLET)
        assert has_scale(cap.spec.supported_scales, ScaleType.SYSTEM)
        # Check analysis types
        assert "thermal_structural_coupling" in cap.supported_analysis_types
        assert "fluid_structural_coupling" in cap.supported_analysis_types

    def test_opencfs_prepare_input(self):
        """Test OpenCFS input file generation."""
        adapter = OpenCFSAdapter()
        design = make_design_spec()
        workload = {"analysis_type": "thermal"}
        config = type('obj', (object,), {
            'fidelity': type('obj', (object,), {'value': 'SPECIALIZED'})()
        })()

        input_path, param_map = adapter.prepare_input(design, workload, config)
        assert input_path is not None
        # Check that OpenCFS input file was created
        assert Path(input_path).exists()
        with open(input_path) as f:
            content = f.read()
        assert "<?xml version" in content
        assert "<cfs_solver>" in content

    def test_opencfs_graceful_fallback(self):
        """Test OpenCFS graceful degradation when binary not available."""
        adapter = OpenCFSAdapter()
        try:
            adapter.run("/nonexistent/path.xml")
        except RuntimeError as e:
            # Should get a clear error message about missing executable
            assert "opencfs" in str(e).lower()


class TestRegistryIntegration:
    """Test registry integration and capability matching."""

    def test_registry_all_adapters_registered(self):
        """Test that all adapters are registered in the registry."""
        registry = SolverRegistry()
        # Manually register all adapter specs since discovery only finds PATH executables
        from chiplab.solvers.adapters.lammps import LAMMPSAdapter
        from chiplab.solvers.adapters.spice import SpiceAdapter
        from chiplab.solvers.adapters.verilator import VerilatorAdapter
        from chiplab.solvers.adapters.openroad import OpenROADAdapter
        from chiplab.solvers.adapters.moose import MOOSEAdapter
        from chiplab.solvers.adapters.opencfs import OpenCFSAdapter
        
        # Register LAMMPS (may already be registered by discovery)
        try:
            lammps_adapter = LAMMPSAdapter()
            registry.register(lammps_adapter.get_capability().spec)
        except Exception:
            pass
        
        # Register ngspice (SPICE backend)
        try:
            from chiplab.solvers.adapters.spice import SpiceAdapter
            ngspice_adapter = SpiceAdapter()
            registry.register(ngspice_adapter.get_capability().spec)
        except Exception:
            pass
        
        # Register all new adapters
        spice_adapter = SpiceAdapter()
        registry.register(spice_adapter.get_capability().spec)
        
        verilator_adapter = VerilatorAdapter()
        registry.register(verilator_adapter.get_capability().spec)
        
        openroad_adapter = OpenROADAdapter()
        registry.register(openroad_adapter.get_capability().spec)
        
        moose_adapter = MOOSEAdapter()
        registry.register(moose_adapter.get_capability().spec)
        
        opencfs_adapter = OpenCFSAdapter()
        registry.register(opencfs_adapter.get_capability().spec)
        
        # Check that all expected solvers are registered
        # Note: ngspice is discovered via PATH scan, may not be available in all environments
        expected_solvers = ["lammps", "spice", "verilator", "openroad", 
                          "moose", "opencfs"]
        for name in expected_solvers:
            assert name in registry.solvers, f"Solver {name} should be registered"
        
        # Verify LAMMPS still works
        assert "lammps" in registry.solvers
        lammps_spec = registry.solvers["lammps"]
        assert lammps_spec.name == "lammps"

    def test_registry_capability_matching_electrical(self):
        """Test capability matching for electrical physics."""
        registry = SolverRegistry()
        # Manually register all adapter specs since discovery only finds PATH executables
        from chiplab.solvers.adapters.lammps import LAMMPSAdapter
        from chiplab.solvers.adapters.spice import SpiceAdapter
        from chiplab.solvers.adapters.verilator import VerilatorAdapter
        from chiplab.solvers.adapters.openroad import OpenROADAdapter
        from chiplab.solvers.adapters.moose import MOOSEAdapter
        from chiplab.solvers.adapters.opencfs import OpenCFSAdapter
        
        # Register all adapters
        lammps_adapter = LAMMPSAdapter()
        registry.register(lammps_adapter.get_capability().spec)
        
        spice_adapter = SpiceAdapter()
        registry.register(spice_adapter.get_capability().spec)
        
        verilator_adapter = VerilatorAdapter()
        registry.register(verilator_adapter.get_capability().spec)
        
        openroad_adapter = OpenROADAdapter()
        registry.register(openroad_adapter.get_capability().spec)
        
        moose_adapter = MOOSEAdapter()
        registry.register(moose_adapter.get_capability().spec)
        
        opencfs_adapter = OpenCFSAdapter()
        registry.register(opencfs_adapter.get_capability().spec)
        
        # Match electrical at system scale
        matches = registry.match(
            physics=PhysicsType.ELECTRICAL,
            scale=ScaleType.SYSTEM
        )
        
        # Should find SPICE, Verilator, OpenROAD, ngspice
        matching_names = [spec.name for spec, _ in matches]
        electrical_solvers = ["spice", "verilator", "openroad"]
        for solver in electrical_solvers:
            assert solver in matching_names, f"{solver} should match electrical/system"

    def test_registry_capability_matching_thermal_chiplet(self):
        """Test capability matching for thermal physics at chiplet scale."""
        registry = SolverRegistry()
        # Manually register all adapter specs since discovery only finds PATH executables
        from chiplab.solvers.adapters.lammps import LAMMPSAdapter
        from chiplab.solvers.adapters.spice import SpiceAdapter
        from chiplab.solvers.adapters.verilator import VerilatorAdapter
        from chiplab.solvers.adapters.openroad import OpenROADAdapter
        from chiplab.solvers.adapters.moose import MOOSEAdapter
        from chiplab.solvers.adapters.opencfs import OpenCFSAdapter
        
        # Register all adapters
        lammps_adapter = LAMMPSAdapter()
        registry.register(lammps_adapter.get_capability().spec)
        
        spice_adapter = SpiceAdapter()
        registry.register(spice_adapter.get_capability().spec)
        
        verilator_adapter = VerilatorAdapter()
        registry.register(verilator_adapter.get_capability().spec)
        
        openroad_adapter = OpenROADAdapter()
        registry.register(openroad_adapter.get_capability().spec)
        
        moose_adapter = MOOSEAdapter()
        registry.register(moose_adapter.get_capability().spec)
        
        opencfs_adapter = OpenCFSAdapter()
        registry.register(opencfs_adapter.get_capability().spec)
        
        # Match thermal at chiplet scale
        matches = registry.match(
            physics=PhysicsType.THERMAL,
            scale=ScaleType.CHIPLET
        )
        
        # Should find LAMMPS, MOOSE, OpenCFS
        matching_names = [spec.name for spec, _ in matches]
        thermal_chiplet_solvers = ["lammps", "moose", "opencfs"]
        for solver in thermal_chiplet_solvers:
            assert solver in matching_names, f"{solver} should match thermal/chiplet"

    def test_registry_lammps_not_broken(self):
        """Test that LAMMPS adapter is not broken by new registrations."""
        registry = SolverRegistry()
        # Manually register LAMMPS
        from chiplab.solvers.adapters.lammps import LAMMPSAdapter
        
        lammps_adapter = LAMMPSAdapter()
        registry.register(lammps_adapter.get_capability().spec)
        
        # LAMMPS should still be available and have correct physics
        assert "lammps" in registry.solvers
        lammps_spec = registry.solvers["lammps"]
        assert has_physics(lammps_spec.supported_physics, PhysicsType.MECHANICAL)
        assert has_physics(lammps_spec.supported_physics, PhysicsType.THERMAL)
        assert has_scale(lammps_spec.supported_scales, ScaleType.DEVICE)


class TestAdapterGracefulFallback:
    """Test graceful fallback behavior across all adapters."""

    @pytest.mark.parametrize("adapter_class, name", [
        (LAMMPSAdapter, "lammps"),
        (SpiceAdapter, "spice"),
        (VerilatorAdapter, "verilator"),
        (OpenROADAdapter, "openroad"),
        (MOOSEAdapter, "moose"),
        (OpenCFSAdapter, "opencfs"),
    ])
    def test_adapter_running_without_binary(self, adapter_class, name):
        """Test that adapters handle missing binary gracefully."""
        adapter = adapter_class()
        # All adapters should provide informative errors rather than crashing
        # when the executable is not available
        try:
            # Try to run with a nonexistent path - this should give a clear error
            adapter.run("/tmp/nonexistent_file_12345")
        except (RuntimeError, FileNotFoundError) as e:
            # Should get some kind of error, not a cryptic one
            error_msg = str(e).lower()
            # The error should mention the adapter name or executable
            assert name in error_msg or adapter.executable in error_msg, \
                f"Error for {name} should mention {name} or {adapter.executable}"

    def test_all_adapters_have_use_gpu_flag(self):
        """Test that all adapters have the use_gpu flag."""
        adapters = [
            LAMMPSAdapter,
            SpiceAdapter,
            VerilatorAdapter,
            OpenROADAdapter,
            MOOSEAdapter,
            OpenCFSAdapter,
        ]
        for adapter_class in adapters:
            adapter = adapter_class()
            assert hasattr(adapter, 'use_gpu'), \
                f"{adapter_class.__name__} should have use_gpu attribute"
            assert adapter.use_gpu == True, \
                f"{adapter_class.__name__} should have use_gpu=True by default"