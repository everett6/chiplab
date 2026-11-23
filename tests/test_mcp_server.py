"""Tests for chiplab MCP server - all 9 tools, 4 resources, and 1 prompt."""

from __future__ import annotations

import json
import os
import tempfile

import chiplab.mcp_server
from chiplab.experiments.store import ExperimentStore


class MCPTestDB:
    """Context manager for temporary MCP test database."""

    def __init__(self):
        self.db_path = None
        self.store = None
        self._orig_store = None

    def __enter__(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.store = ExperimentStore(db_path=self.db_path)
        self._orig_store = chiplab.mcp_server.store
        chiplab.mcp_server.store = self.store
        return self.store

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._orig_store is not None:
            chiplab.mcp_server.store = self._orig_store
        if self.db_path and os.path.exists(self.db_path):
            os.unlink(self.db_path)


def _sample_spec(name: str = "test_design", version: str = "0.1") -> dict:
    return {
        "name": name,
        "version": version,
        "architecture": "heterogeneous_accelerator",
        "package": {"type": "2.5D", "interposer": "silicon"},
        "chiplets": [
            {
                "type": "digital_compute",
                "count": 2,
                "compute": {"architecture": "tensor_engine", "precision": "INT8", "peak_tops": 200.0},
            },
            {
                "type": "memory",
                "count": 4,
                "memory": {"technology": "DRAM", "capacity_per_chiplet_gb": 16.0, "channels": 8},
            },
        ],
        "interconnect": {"topology": "mesh", "link_bandwidth_gbps": 100.0},
    }


class TestMCPTools:
    """Test all 9 MCP tools."""

    def test_chip_create_design_dict_and_yaml(self):
        with MCPTestDB() as store:
            res = chiplab.mcp_server.chip_create_design(_sample_spec("alpha", "0.1"))
            assert res["success"] is True
            assert res["design_id"] == "alpha_0.1"
            _, loaded = store.get_design(res["design_id"])
            assert loaded.name == "alpha"

            yaml_spec = """
design:
  name: beta
  version: "0.1"
  chiplets:
    - type: memory
      count: 2
      memory:
        technology: HBM
        capacity_per_chiplet_gb: 24
"""
            res2 = chiplab.mcp_server.chip_create_design(yaml_spec)
            assert res2["success"] is True
            assert res2["name"] == "beta"

    def test_chip_get_design(self):
        with MCPTestDB():
            created = chiplab.mcp_server.chip_create_design(_sample_spec("get_test", "0.1"))
            res = chiplab.mcp_server.chip_get_design(created["design_id"], "0.1")
            assert res["design_id"] == created["design_id"]
            assert res["name"] == "get_test"
            assert len(res["history"]) == 1

    def test_chip_validate_design_valid_and_invalid(self):
        with MCPTestDB():
            created = chiplab.mcp_server.chip_create_design(_sample_spec("val_test", "0.1"))
            val = chiplab.mcp_server.chip_validate_design(created["design_id"], "0.1", True)
            assert val["valid"] is True
            assert val["errors"] == []

            invalid_spec = {
                "name": "bad_monolithic",
                "version": "0.1",
                "package": {"type": "monolithic"},
                "chiplets": [
                    {"type": "compute", "count": 1},
                    {"type": "memory", "count": 1},
                ],
            }
            bad_created = chiplab.mcp_server.chip_create_design(invalid_spec)
            bad_val = chiplab.mcp_server.chip_validate_design(bad_created["design_id"])
            assert bad_val["valid"] is False
            assert len(bad_val["errors"]) > 0

    def test_chip_run_simulation_analytical_and_discrete_event(self):
        with MCPTestDB():
            created = chiplab.mcp_server.chip_create_design(_sample_spec("sim_test", "0.1"))
            workload = {"name": "moe_w", "type": "moe_decode", "batch_size": 1}

            res_a = chiplab.mcp_server.chip_run_simulation(
                created["design_id"], workload=workload, fidelity="analytical"
            )
            assert res_a["fidelity"] == "analytical"
            assert "not cycle-accurate" in res_a["fidelity_disclaimer"].lower()
            assert res_a["metrics"]["total_memory_capacity_gb"] > 0

            res_d = chiplab.mcp_server.chip_run_simulation(
                created["design_id"], workload=workload, fidelity="discrete_event", seed=42, num_requests=16
            )
            assert res_d["fidelity"] == "discrete_event"
            assert "discrete-event" in res_d["fidelity_disclaimer"].lower()

    def test_chip_sweep_parameters(self):
        with MCPTestDB():
            created = chiplab.mcp_server.chip_create_design(_sample_spec("sweep_test", "0.1"))
            sweep = chiplab.mcp_server.chip_sweep_parameters(
                created["design_id"],
                parameters={"chiplets[0].count": [1, 2, 4]},
                fidelity="analytical",
                objective="throughput_tokens_per_sec",
            )
            assert len(sweep["points"]) == 3
            assert sweep["best_point"] is not None

    def test_chip_compare_designs(self):
        with MCPTestDB():
            c1 = chiplab.mcp_server.chip_create_design(_sample_spec("cmp_a", "0.1"))
            c2 = chiplab.mcp_server.chip_create_design(_sample_spec("cmp_b", "0.1"))
            r1 = chiplab.mcp_server.chip_run_simulation(c1["design_id"], fidelity="analytical")
            r2 = chiplab.mcp_server.chip_run_simulation(c2["design_id"], fidelity="analytical")
            cmp_res = chiplab.mcp_server.chip_compare_designs([r1["run_id"], r2["run_id"]])
            assert "designs" in cmp_res
            assert len(cmp_res["designs"]) == 2

    def test_chip_analyze_bottleneck_and_get_results(self):
        with MCPTestDB():
            created = chiplab.mcp_server.chip_create_design(_sample_spec("bn_test", "0.1"))
            sim = chiplab.mcp_server.chip_run_simulation(created["design_id"], fidelity="analytical")
            bn = chiplab.mcp_server.chip_analyze_bottleneck(sim["run_id"])
            assert "primary_bottleneck" in bn
            assert "recommendations" in bn

            got_one = chiplab.mcp_server.chip_get_results(run_id=sim["run_id"])
            assert got_one["run_id"] == sim["run_id"]

            got_list = chiplab.mcp_server.chip_get_results(design_id=created["design_id"])
            assert len(got_list["results"]) == 1

    def test_chip_export_report(self):
        with MCPTestDB():
            created = chiplab.mcp_server.chip_create_design(_sample_spec("rep_test", "0.1"))
            sim = chiplab.mcp_server.chip_run_simulation(created["design_id"], fidelity="analytical")
            md = chiplab.mcp_server.chip_export_report(sim["run_id"], fmt="markdown")
            assert "Fidelity" in md["report"]
            js = chiplab.mcp_server.chip_export_report(sim["run_id"], fmt="json")
            parsed = json.loads(js["report"])
            assert parsed["run_id"] == sim["run_id"]


class TestMCPResourcesAndPrompts:
    """Test all 4 MCP resources and the explore_architecture prompt."""

    def test_resources(self):
        with MCPTestDB():
            lib = chiplab.mcp_server.get_library_components()
            if isinstance(lib, str):
                lib = json.loads(lib)
            assert "components" in lib or "SRAM" in str(lib)

            assumptions = chiplab.mcp_server.get_models_assumptions()
            assert "fidelity_levels" in assumptions
            assert "provenance_categories" in assumptions

            created = chiplab.mcp_server.chip_create_design(_sample_spec("res_test", "0.1"))
            design_res = chiplab.mcp_server.get_design_resource(created["design_id"])
            if isinstance(design_res, str):
                design_res = json.loads(design_res)
            assert design_res["name"] == "res_test"

            sim = chiplab.mcp_server.chip_run_simulation(created["design_id"], fidelity="analytical")
            sim_res = chiplab.mcp_server.get_result_resource(sim["run_id"])
            if isinstance(sim_res, str):
                sim_res = json.loads(sim_res)
            assert sim_res["run_id"] == sim["run_id"]

    def test_prompt_and_server_import(self):
        prompt = chiplab.mcp_server.explore_architecture("Design a low-power MoE accelerator")
        assert "workflow" in prompt
        assert hasattr(chiplab.mcp_server, "main")
