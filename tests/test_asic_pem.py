"""PEM competition ASIC: assembler, reference model, RTL lockstep, protocol tests, top level,
report honesty. RTL tests need Verilator and are skipped without it."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chiplab.asic import assembler, flow, isa, iss, protocols, report, verify

HAVE_VERILATOR = shutil.which("verilator") or (Path.home() / ".local/bin/verilator").exists()
rtl = pytest.mark.skipif(not HAVE_VERILATOR, reason="verilator not installed")


def test_assembler_roundtrip_and_errors():
    src, _ = protocols.usb_ls_tx()
    p = assembler.assemble(src)
    assert len(p.words) <= isa.IMEM_WORDS and p.start < p.wrap
    for w in p.words.values():
        again = assembler.assemble(assembler.disassemble(w))
        assert list(again.words.values()) == [w]
    for bad in ["jmp nowhere", "out pins, 17", "set pins, 3, 5", "bogus", "wait 2, gpio, 1"]:
        with pytest.raises(assembler.AsmError):
            assembler.assemble(bad)


def test_model_basic_semantics():
    p = assembler.assemble("setx 5\nloop:\njmp x--, loop\nmov isr, x\npush block\nhalt")
    sc = verify.load_script([(0, p, {})], 1) + "C 30\nR 40\nR 54\n"
    out = iss.run_script(sc)
    assert "R 40 ffff" in out          # x-- leaves x = 0xFFFF after the final test of 0
    assert "R 54 ffff" in out


def test_crc_engine_matches_independent_crc16():
    # feed bytes through `out pins, 1` with CRC snoop; compare with the decoder's CRC-16/USB
    src = """
        crc start
    l:  pull block
        setx 7
    b:  out pins, 1
        jmp x--, b
        jmp l
    """
    p = assembler.assemble(src)
    regs = {"crc_poly_lo": 0xA001, "crc_init_lo": 0xFFFF, "crc_cfg": isa.crc_cfg(16, False)}
    data = b"\x12\x34\xFF"
    sc = verify.load_script([(0, p, regs)], 1) + "".join(f"W 40 {b:x}\nC 40\n" for b in data)
    m = iss.PEM()
    iss.run_script(sc, m)
    assert (m.lanes[0].crc ^ 0xFFFF) == protocols.crc16_usb(data)


@pytest.mark.parametrize("name", list(protocols.TESTS))
def test_protocols_on_reference_model(name):
    r = protocols.TESTS[name](backend="model")
    assert r["passed"], r


@rtl
@pytest.mark.parametrize("name", list(protocols.TESTS))
def test_protocols_on_rtl(name):
    r = protocols.TESTS[name](backend="rtl")
    assert r["passed"] and r["fidelity"].startswith("RTL"), r


@rtl
def test_random_lockstep_rtl_vs_model():
    r = verify.regression(seeds=8, cycles=2500, coverage=False)
    assert r["passed"] == 8, r["failed"]


@rtl
def test_mutants_are_detected():
    r = verify.mutation_check(seeds=8, cycles=2500)
    assert r["killed"] == r["mutants"], r["results"]


@rtl
def test_tiny_tapeout_top_through_spi():
    r = verify.tt_smoke()
    assert r["passed"], r


@rtl
def test_rtl_lints_clean():
    assert flow.lint()["ok"]


def test_report_never_invents_physical_results(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    r = report.competition_report()
    for k in ("tile_utilization", "mapped_cell_area_um2", "fmax_mhz", "power_mw", "timing_margin_ns"):
        assert r[k]["value"] is None and r[k]["status"] == "not run"
    assert r["protocol_tests"] == {"status": "not run"}
    ck = {c["item"]: c["status"] for c in report.checklist()}
    assert ck["Passing tests"] == "missing"
    if not flow.flow_status()["tools"]["tools"]["openroad"]["installed"]:
        assert ck["Successful Tiny Tapeout flow"] == "missing"


def test_export_tiny_tapeout_project(tmp_path):
    out = flow.export_tt_project(tmp_path / "tt")
    root = Path(out["dest"])
    assert (root / "info.yaml").read_text().count("tt_um_chiplab_pem") >= 1
    for f in flow.SOURCES:
        assert (root / "src" / f).exists()
    assert (root / "programs" / "usb_ls_tx.pem").exists() and (root / "docs" / "isa.md").exists()
    assert '"TODO"' in (root / "info.yaml").read_text()      # no personal data filled in
