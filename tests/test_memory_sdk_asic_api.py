"""Memory compiler, component SDK, shared ASIC API limits and the /asic web routes.

Synthetic specs only. Verilator-dependent tests skip when it is not installed."""

from __future__ import annotations

import pytest

from chiplab import asic_api
from chiplab.memory_compiler import compiler, rtl, views
from chiplab.sdk import expr
from chiplab.sdk.components import ComponentError, Registry, elaborate, simulate, to_verilog

HAVE_VERILATOR = rtl.verilator() is not None


def _spec(**kw):
    d = {"name": "m", "kind": "sram", "words": 1024, "width": 16, "pdk": "sky130"}
    d.update(kw)
    return compiler.spec_from_dict(d)


# ------------------------------------------------------------------ memory compiler
def test_compile_sram_defaults_to_6t_and_reports_honest_fields():
    r = compiler.compile_memory(_spec())
    b = r["best"]
    assert b["cell"] == "sram6t"
    assert b["banks"] * b["rows"] * b["cols"] == 1024 * 16
    assert b["area_um2"] > 0 and b["access_ns"] > 0 and b["read_energy_pj"] > 0
    assert r["fidelity"] == "analytical"
    assert any(c["pareto"] for c in r["candidates"])
    drc = [c for c in r["checks"] if c["check"] in ("DRC", "LVS")]
    assert drc and all(c["ok"] is None for c in drc)          # never claimed as run


def test_rom_smaller_than_sram_and_dram_infeasible_on_logic_pdk():
    rows = compiler.compare([_spec(kind="mask_rom", read_ports=1, rw_ports=0, contents=[i & 0xFFFF for i in range(1024)]),
                             _spec(), _spec(kind="dram")])
    rom, sram, dram = rows
    assert rom["area_um2"] < sram["area_um2"]
    assert not dram["feasible"]


def test_spec_validation_rejects_nonsense():
    assert _spec(words=0).validate()
    assert _spec(width=0).validate()
    with pytest.raises((ValueError, TypeError)):
        compiler.spec_from_dict({"kind": "sram", "words": 16, "width": 8, "bogus": 1})


def test_rom_site_map_matches_contents():
    s = _spec(kind="mask_rom", words=64, width=8, read_ports=1, rw_ports=0,
              contents=[(i * 37) & 0xFF for i in range(64)])
    best = compiler.compile_memory(s)["best"]
    sites = views.site_map(s, best)
    mux = best["column_mux"]
    for a in (0, 5, 63):
        for b in range(8):
            assert sites[a // mux][b * mux + a % mux] == (s.contents[a] >> b & 1)


def test_all_views_are_svg():
    s = _spec(kind="mask_rom", words=64, width=8, read_ports=1, rw_ports=0, contents=list(range(64)))
    v = views.all_views(s, compiler.compile_memory(s))
    assert set(v) == {"logical", "circuit", "physical", "block"}
    assert all(x.lstrip().startswith("<svg") for x in v.values())


def test_golden_model_rom_and_sram():
    s = _spec(words=16, width=8)
    ops = rtl._ops(s, 50, 3)
    exp = rtl.golden(s, ops)
    assert len(exp) == len(ops)
    rom = _spec(kind="mask_rom", words=16, width=8, read_ports=1, rw_ports=0, contents=list(range(16)))
    assert "case" in rtl.gen_rtl(rom)


@pytest.mark.skipif(not HAVE_VERILATOR, reason="verilator not installed")
@pytest.mark.parametrize("kw", [dict(kind="mask_rom", read_ports=1, rw_ports=0, contents=[(i * 7) & 0xFF for i in range(64)]),
                                dict(kind="sram"), dict(kind="fifo")])
def test_generated_rtl_passes_its_testbench(kw):
    r = rtl.run_tests(_spec(words=64, width=8, **kw), 200)
    assert r["ran"] and r["passed"] and r["errors"] == 0


@pytest.mark.skipif(not HAVE_VERILATOR, reason="verilator not installed")
def test_edram_retention_flags_error_without_refresh():
    s = _spec(kind="edram", words=32, width=8)
    ok = rtl.run_tests(s, 300, retention_cycles=200, refresh_interval=2, refresh=True)
    bad = rtl.run_tests(s, 300, retention_cycles=200, refresh_interval=2, refresh=False)
    assert ok["passed"] and ok["retention_error"] == 0
    assert bad["retention_error"] == 1


# ------------------------------------------------------------------ SDK sandbox
@pytest.mark.parametrize("src", ["__import__('os').system('true')", "open('/etc/passwd')", "(1).__class__",
                                 "[x for x in range(3)]", "lambda: 1", "a.b", "9**9**9", "1 << 100000", "x" * 700])
def test_expression_sandbox_rejects(src):
    with pytest.raises(expr.ExprError):
        expr.evaluate(src, {"a": 1, "x": 1})


def test_expression_sandbox_allows_math():
    assert expr.evaluate("clog2(N) + max(a, 3)", {"N": 1000, "a": 1}) == 13
    assert expr.evaluate("select(s, a, b)", {"s": 0, "a": 4, "b": 5}) in (4, 5)


DEF = {"name": "acc_demo", "version": "1.0", "category": "arithmetic", "description": "test",
       "params": {"W": {"default": 8, "provenance": "assumed"}},
       "ports": [{"name": "a", "dir": "in", "width": "W"}, {"name": "y", "dir": "out", "width": "W"}],
       "behavior": {"type": "comb", "outputs": {"y": "(a + 1) & mask(W)"}},
       "models": {"area_um2": "W * gate_um2 * 6"}, "backend": "functional"}


def test_registry_versions_are_immutable(tmp_path):
    reg = Registry(tmp_path)
    reg.register(DEF)
    with pytest.raises(ComponentError):
        reg.register(DEF)
    reg.register({**DEF, "version": "1.1"})
    assert Registry(tmp_path).get("acc_demo@1.1").version == "1.1"      # persisted


def test_registry_rejects_unsafe_model(tmp_path):
    with pytest.raises((ComponentError, expr.ExprError)):
        Registry(tmp_path).register({**DEF, "name": "evil", "models": {"area_um2": "__import__('os')"}})


def test_mac_template_simulates_and_exports(tmp_path):
    reg = Registry(tmp_path)
    t = elaborate(reg, "mac_unit_pipelined@1.0", {})
    assert not t["errors"] and t["rollup"]["area_um2"] > 0
    stim = [{"a": 3, "b": 5, "clr": 0}, {"a": 2, "b": 2, "clr": 0}, {"a": 1, "b": 1, "clr": 0},
            {"a": 0, "b": 0, "clr": 0}, {"a": 0, "b": 0, "clr": 0}]
    assert [c["acc"] for c in simulate(t, stim)] == [0, 0, 15, 19, 20]
    v = to_verilog(reg, t)
    assert "module mac_unit_pipelined" in v and "endmodule" in v


# ------------------------------------------------------------------ shared API limits
def test_api_limits():
    with pytest.raises(ValueError):
        asic_api.asic_simulate("nop", cycles=10**7)
    with pytest.raises(ValueError):
        asic_api.asic_simulate("nop", script="X rm -rf /")
    with pytest.raises(ValueError):
        asic_api.asic_export("../escape")
    with pytest.raises(ValueError):
        asic_api.asic_run("shell")
    with pytest.raises(ValueError):
        asic_api.memory_compare([])
    with pytest.raises(ValueError):
        asic_api.component_simulate("mac_unit_pipelined@1.0", [])


def test_protocol_lab_examples_run_on_model():
    ex = asic_api._examples()
    r = asic_api.asic_simulate(ex["uart_rx"]["source"], ex["uart_rx"]["regs"], 500, ex["uart_rx"]["bench"], 0, 2)
    assert r["ok"] and [v >> 8 for a, v in r["reads"] if a == 0x40] == [0x5A]
    spi = ex["spi_mode0"]
    r = asic_api.asic_simulate(spi["source"], spi["regs"], 500, spi["bench"], 0, 2)
    assert [v & 0xFF for a, v in r["reads"] if a == 0x40] == [0x96]


def test_simulation_levels_cover_ten():
    lv = asic_api.simulation_levels()
    assert [x["level"] for x in lv] == list(range(1, 11))


# ------------------------------------------------------------------ web routes
def test_asic_routes(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from chiplab.app.server import create_app
    monkeypatch.setattr(asic_api, "_registry", lambda: Registry(tmp_path / "comps"))
    c = TestClient(create_app(db_path=tmp_path / "x.db"))
    assert c.get("/asic").status_code == 200 and "ASIC Studio" in c.get("/asic").text
    assert c.get("/asic.js").status_code == 200
    ov = c.get("/api/asic/overview").json()
    assert {"architecture", "report", "checklist", "levels"} <= set(ov)
    origin = {"Origin": "http://testserver"}
    body = {"spec": {"name": "m", "kind": "sram", "words": 256, "width": 8, "pdk": "sky130"}, "views": False}
    assert c.post("/api/memory/compile", json=body, headers={"Origin": "http://evil.example"}).status_code == 403
    r = c.post("/api/memory/compile", json=body, headers=origin)
    assert r.status_code == 200 and r.json()["best"]["cell"] == "sram6t"
    assert c.post("/api/asic/assemble", json={"source": "nop"}, headers=origin).json()["ok"]
    r = c.post("/api/components/create", json={"definition": DEF}, headers=origin)
    assert r.status_code == 200 and r.json()["created"] == "acc_demo@1.0"
    assert c.post("/api/components/create", json={"definition": DEF}, headers=origin).status_code == 400
