"""rc_delay against SkyWater SKY130 RCX table values (hand-calculated references)."""

import pytest

from chiplab.models.physics.rc_extraction import SKY130_LAYERS, rc_delay, sky130_wire_rc


def test_met1_1mm_matches_hand_calculation():
    # R = 0.125 ohm/sq * 1000 sq = 125 ohm; C = (114.197*1 + 2*59.496) aF/um * 1000 um
    r, c = 125.0, (114.197 + 2 * 59.496) * 1e-15
    assert rc_delay(1.0, 1.0, "met1") == pytest.approx(0.5 * r * c * 1e12, rel=1e-6)
    assert rc_delay(1.0, 1.0, "M1") == rc_delay(1.0, 1.0, "met1")


def test_capacitance_in_textbook_range_and_ordering():
    for layer in ("met1", "met2", "met3", "met4", "met5"):
        c_ff_um = sky130_wire_rc(1.0, layer)["c_f_per_um"] * 1e15
        assert 0.1 < c_ff_um < 0.4               # classic ~0.2 fF/um for on-chip wires
    assert SKY130_LAYERS["met5"]["r_sq_ohm"] < SKY130_LAYERS["met1"]["r_sq_ohm"]
    assert rc_delay(1, 1, "met5") < rc_delay(1, 1, "met1") < rc_delay(1, 1, "li1")


def test_quadratic_in_length_and_overrides():
    assert rc_delay(2, 1, "met2") == pytest.approx(4 * rc_delay(1, 1, "met2"))
    # copper 1.7 uohm cm, 1 x 0.5 um -> 34 ohm/mm; 0.2 fF/um -> 0.5*34*200e-15 s = 3.4 ps
    d = rc_delay(1, 1, "M9", resistivity_uc_ohm_cm=1.7, thickness_um=0.5, c_ff_per_um=0.2)
    assert d == pytest.approx(3.4, rel=1e-6)
    with pytest.raises(KeyError):
        rc_delay(1, 1, "M8")
    with pytest.raises(ValueError):
        rc_delay(1, 1, "met1", resistivity_uc_ohm_cm=1.7)
