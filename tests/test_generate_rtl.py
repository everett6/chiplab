"""chip_generate_rtl top-level text for 0, 1 and 2 memory and compute chiplets."""
from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from chiplab import mcp_server
from chiplab.experiments.store import ExperimentStore
from chiplab.schemas.design import DesignSpec

STUBS = """
module memory_subsystem #(parameter tech = "", parameter count = 1) (
    input clk, input rst_n);
endmodule
module compute_subsystem #(parameter arch = "", parameter precision = "") (
    input clk, input rst_n, output [31:0] result);
    assign result = 32'd0;
endmodule
"""


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    store = ExperimentStore(db_path=str(tmp_path / "rtl.db"))
    monkeypatch.setattr(mcp_server, "store", store)
    return store


def memory(i: int) -> dict:
    return {"id": f"mem{i}", "type": "memory", "count": i + 2,
            "memory": {"technology": ["HBM", "SRAM"][i], "capacity_per_chiplet_gb": 1,
                       "channels": 4 + i}}


def compute(i: int, kind: str) -> dict:
    return {"id": f"comp{i}", "type": kind,
            "compute": {"architecture": ["systolic", "tensor engine"][i], "precision": "INT8"}}


def generate(store, chiplets: list[dict]) -> dict:
    design = DesignSpec.model_validate({"name": "synthetic", "chiplets": chiplets})
    return mcp_server.chip_generate_rtl(store.save_design(design), target="system_verilog")


def check_top(text: str, n_mem: int, n_comp: int) -> None:
    assert text.count("module top_level_chip") == 1
    assert text.count("endmodule") == 1
    assert "{" not in text and "}" not in text  # no unfilled placeholders
    assert text.count("(") == text.count(")")
    assert not re.search(r",\s*\)", text)  # no trailing comma before a closing paren
    for module, prefix, n in (("memory_subsystem", "mem", n_mem),
                              ("compute_subsystem", "comp", n_comp)):
        blocks = re.findall(rf"{module} #\((.*?)\) (\w+) \((.*?)\);", text, re.DOTALL)
        assert len(re.findall(rf"{module} #\(", text)) == n
        assert [name for _, name, _ in blocks] == [f"{prefix}_{i}" for i in range(n)]
        for params, _, ports in blocks:
            assert "module" not in params + ports and "_subsystem" not in params + ports
    # each compute instance drives its own output
    results = re.findall(r"\.result\((\w+)\)", text)
    assert len(results) == len(set(results)) == n_comp
    for signal in results:
        assert re.search(rf"output \[31:0\] {signal}\b", text)


@pytest.mark.parametrize("kind", ["compute", "digital_compute"])
@pytest.mark.parametrize("n_comp", [0, 1, 2])
@pytest.mark.parametrize("n_mem", [0, 1, 2])
def test_top_module_is_well_formed(isolated_store, tmp_path, n_mem, n_comp, kind):
    chiplets = ([memory(i) for i in range(n_mem)]
                + [compute(i, kind) for i in range(n_comp)])
    out = generate(isolated_store, chiplets)
    text = out["rtl_code"]
    check_top(text, n_mem, n_comp)
    for i in range(n_mem):
        assert f"inout [{3 + i}:0] hbm_dq_{i}" in text
        assert f".count({i + 2})" in text
        assert f'.tech("{["HBM", "SRAM"][i]}")' in text
    for i in range(n_comp):
        assert f'.arch("{["systolic", "tensor engine"][i]}")' in text
    assert [f["chiplet"] for f in out["rtl_files"]] == [c["id"] for c in chiplets]

    if shutil.which("verilator"):
        path = tmp_path / "top_level_chip.sv"
        path.write_text(text + STUBS)
        lint = subprocess.run(
            ["verilator", "--lint-only", "-Wno-fatal", "--top-module", "top_level_chip",
             str(path)], capture_output=True, text=True, timeout=60)
        assert lint.returncode == 0, lint.stderr


def test_memory_chiplet_without_memory_spec(isolated_store):
    text = generate(isolated_store, [{"id": "m", "type": "memory"}])["rtl_code"]
    check_top(text, 1, 0)
    assert '.tech("DRAM")' in text


def test_tracked_values_are_written_as_numbers(isolated_store):
    chiplet = memory(0)
    chiplet["count"] = {"value": 2}
    chiplet["memory"]["channels"] = {"value": 4}
    out = generate(isolated_store, [chiplet])
    assert "inout [3:0] hbm_dq_0" in out["rtl_code"]
    assert ".count(2)" in out["rtl_code"]
    assert "for 2 chiplets" in out["rtl_files"][0]["rtl_code"]


def test_export_full_project_with_compute_only_design(isolated_store):
    design = DesignSpec.model_validate({"name": "c", "chiplets": [compute(0, "compute")]})
    design_id = isolated_store.save_design(design)
    out = mcp_server.chip_export_full_project(design_id)
    assert f"rtl_{design_id}.sv" in out["file_paths"]
