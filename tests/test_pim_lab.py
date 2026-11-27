"""PIM lab: SRAM near-memory bank (RTL vs golden model) and DRAM charge sharing."""

from __future__ import annotations

import pytest

from chiplab.pim import _eda, dram, sram
from chiplab.imc_lab.digital import LIBERTY
from chiplab.tools import registry

HAVE_IV = _eda.have("iverilog", "vvp")
HAVE_YS = _eda.have("yosys") and LIBERTY.is_file()
HAVE_SPICE = registry.get("ngspice").available


@pytest.mark.skipif(not HAVE_IV, reason="iverilog missing")
@pytest.mark.parametrize("spec", [{"rows": 2, "lanes": 1, "lane_bits": 2},
                                  {"rows": 8, "lanes": 4, "lane_bits": 8},
                                  {"rows": 16, "lanes": 3, "lane_bits": 7}])
def test_sram_nmc_rtl_bit_exact(spec):
    r = sram.simulate(spec, n_ops=50, seed=5)
    assert r["bit_exact"], r["first_mismatch"]
    assert set(r["ops_covered"]) == set(sram.OPS)


def test_sram_golden_model_semantics():
    p = sram.validate({"rows": 4, "lanes": 2, "lane_bits": 4})
    m = sram.NmcModel(p)
    m.mem = [0x3F, 0x21, 0x3F, 0x00]
    m.op("ADD", 0, 1, 3)
    assert m.mem[3] == 0x50                     # lanes F+1 -> 0 (wrap), 3+2 -> 5
    m.op("SEARCH", 0, 0, 3)
    assert m.mem[3] == 0b0101
    m.op("MAC", 0, 1, 0)
    assert m.acc == 15 * 1 + 3 * 2


@pytest.mark.skipif(not HAVE_YS, reason="yosys/liberty missing")
def test_sram_flops_match_structure():
    spec = {"rows": 4, "lanes": 2, "lane_bits": 4}
    assert sram.synthesize(spec)["flip_flops"] == sram.predict(spec)["flip_flops"]["value"]


def test_sram_cpu_comparison_is_consistent():
    c = sram.compare_with_cpu({"rows": 8, "lanes": 4, "lane_bits": 8})
    per = {r["op"]: r for r in c["per_op"]}
    assert all(r["cpu_bus_bits"] >= r["nmc_bus_bits"] or r["op"] == "ACCCLR"
               for r in c["per_op"])
    assert per["SEARCH"]["energy_ratio"] > per["AND"]["energy_ratio"]
    # with no wire or instruction cost the only difference is gone
    flat = sram.compare_with_cpu({"rows": 8}, {"AND": 1}, {"wire_pj_per_bit_mm": 0,
                                                           "instr_overhead_pj": 0})
    assert flat["total"]["energy_ratio_cpu_over_nmc"] == pytest.approx(1.0)


def test_sram_validation():
    with pytest.raises(sram.SramNmcError):
        sram.validate({"rows": 32, "lanes": 1, "lane_bits": 4})   # SEARCH needs rows <= width


def test_dram_margins_closed_form():
    m = dram.margins()
    p = dram.params()
    cc, cb, v = p["c_cell_f"], p["c_bitline_f"], p["vdd"]
    assert m["single_row_mv"] == pytest.approx(1e3 * cc * v / 2 / (cb + cc))
    assert m["tra_mv"] == pytest.approx(1e3 * cc * v / 2 / (cb + 3 * cc))
    assert m["tra_over_single"] < 1


@pytest.mark.skipif(not HAVE_SPICE, reason="ngspice missing")
@pytest.mark.parametrize("cells", [[1.2], [1.2, 1.2, 0.0], [0.0, 0.0, 1.2]])
def test_dram_charge_sharing_matches_ngspice(cells):
    assert dram.spice_check(cells)["agree"]


def test_dram_majority_logic_ideal():
    p = dram.params()
    for bits in [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]:
        dv = dram.deviation([b * p["vdd"] for b in bits], p)
        assert (dv > 0) == (sum(bits) >= 2)


def test_dram_errors_grow_with_offset_and_short_sharing():
    base = dram.monte_carlo("and", 20000, seed=1)["bit_error_rate"]
    off = dram.monte_carlo("and", 20000, seed=1,
                           overrides={"sa_offset_sigma_v": 0.05})["bit_error_rate"]
    short = dram.monte_carlo("and", 20000, seed=1,
                             overrides={"t_share_s": 0.05e-9})["bit_error_rate"]
    assert base < off and base < short


def test_dram_tra_less_robust_than_copy():
    big = {"sa_offset_sigma_v": 0.035}
    assert dram.monte_carlo("and", 40000, 2, big)["bit_error_rate"] > \
        dram.monte_carlo("copy", 40000, 2, big)["bit_error_rate"]


def test_dram_timing_and_read_disturb():
    te = dram.timing_energy("and")
    assert te["aaps"] == 4 and te["latency_ns"] == pytest.approx(4 * 48.75)
    rd = dram.read_disturb(1e6)
    assert rd["at_risk"] and rd["max_safe_ops_per_s"] < 1e6


def test_dram_energy_model_review_runs():
    r = dram.benchmark_review("and")
    assert r["verdict"] in ("validated", "rejected", "plausible")


# ---- builder and designer -------------------------------------------------------------------
from chiplab.pim import builder, dram_designer  # noqa: E402


@pytest.mark.skipif(not HAVE_IV, reason="iverilog missing")
@pytest.mark.parametrize("spec", [{"rows": 4, "lanes": 1, "lane_bits": 4},
                                  {"rows": 8, "lanes": 4, "lane_bits": 8}])
def test_tt_wrapper_pin_level(spec):
    assert builder.verify_wrapper(spec)["pin_level_bit_exact"]


@pytest.mark.skipif(not (HAVE_IV and HAVE_YS), reason="iverilog/yosys missing")
def test_builder_sram_end_to_end(tmp_path):
    r = builder.build("sram", ["AND", "MAC"], {"rows": 4, "lanes": 2, "lane_bits": 4},
                      export_dir=tmp_path)
    assert r["ok"] and r["verdict"] == "validated"
    exp = [s for s in r["stages"] if s["stage"] == "tt_export"][0]
    assert exp["submitted"] is False and (tmp_path / "info.yaml").is_file()


def test_builder_refuses_unbuildable():
    assert not builder.build("ferrofluid", ["mvm"])["ok"]
    assert not builder.build("sram", ["FLY"])["ok"]
    d = builder.build("dram", ["and"])
    assert d["buildable_on_digital_shuttle"] is False
    assert any(s["status"] == "blocked" for s in d["stages"])


def test_designer_retention_tra_stricter():
    r = dram_designer.retention()
    assert r["tra"]["fail_fraction"] > r["single_read"]["fail_fraction"]
    assert 0 < r["refresh_time_fraction"] < 0.1


def test_designer_schedule_respects_faw():
    s = dram_designer.schedule(512, "and")
    g = dram_designer.geometry()
    acts = 512 * 8
    assert s["makespan_s"] >= (acts / 4 - 1) * g["t_faw_s"]
    assert 0 < s["efficiency"] <= 1


def test_designer_comparison_sanity():
    r = dram_designer.compare("bitwise_and", 2**20)
    a = r["architectures"]
    assert a["cpu"]["bytes_over_channel"] == 3 * 2**20
    assert a["in_array"]["energy_j"] < a["cpu"]["energy_j"]
    assert not dram_designer.compare("dot_int8", 2**20)["architectures"]["in_array"][
        "supported"]
    area = dram_designer.area_overhead()
    assert 0 < area["in_array_fraction"] < area["near_bank_fraction"] < 1


def test_dram_idd_activation_energy_and_implied_diagnostic():
    from chiplab.pim import dram
    base = dram.timing_energy("and")
    assert base["activation_energy_method"] == "bitline_lower_bound"
    idd = dram.timing_energy("and", {"idd0_a": 0.05, "idd3n_a": 0.04, "idd2n_a": 0.03})
    assert idd["activation_energy_method"] == "datasheet_idd"
    # E = VDD [IDD0 tRC - (IDD3N tRAS + IDD2N tRP)] = 1.2 (2.4375 - 1.4 - 0.4125) nJ
    assert idd["activation_energy_nj"] == pytest.approx(0.75, rel=1e-6)
    with pytest.raises(dram.DramError):
        dram.timing_energy("and", {"idd0_a": 0.01, "idd3n_a": 0.04, "idd2n_a": 0.03})
    a, c = dram.implied_activation_energy("and"), dram.implied_activation_energy("copy")
    assert 1 < a["factor"] < 10 and 1 < c["factor"] < 10
    # the two independent publications should imply similar activation energies
    assert 0.5 < a["implied_activation_nj"] / c["implied_activation_nj"] < 2
