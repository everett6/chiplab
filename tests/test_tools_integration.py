"""External-tool views: ngspice, Xschem, KiCad, KLayout, OpenFOAM.

Everything that does not need a tool runs everywhere (templates, validation, raw parsing,
netlist/schematic/.kicad_pcb/case generation, API refusals). Tests that drive a real tool skip
when it is not installed.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from chiplab.tools import cfd, kicad, layout, registry, schematic, spice

HAVE_NGSPICE = registry.get("ngspice").available
HAVE_XSCHEM = registry.get("xschem").available
HAVE_KICAD = registry.get("kicad").available
HAVE_KLAYOUT = layout.available()
HAVE_FOAM = registry.get("openfoam").available


def _gpu_scene():
    from test_studio import _gpu

    from chiplab.studio.scene import build_scene
    return build_scene(_gpu(), "studio_gpu_0.1")


# ---------------------------------------------------------------- registry / safety
def test_status_lists_every_tool():
    st = registry.status()
    assert set(st) == {"ngspice", "xschem", "kicad", "openfoam", "klayout"}
    for v in st.values():
        assert isinstance(v["available"], bool) and v["purpose"]


def test_run_refuses_unknown_or_missing_tools():
    with pytest.raises(registry.ToolError):
        registry.run(registry.Tool("bash", "/bin/bash"), ["-c", "true"])
    with pytest.raises(registry.ToolError):
        registry.run(registry.Tool("ngspice", None), ["-v"])
    if HAVE_NGSPICE:
        with pytest.raises(registry.ToolError):
            registry.run(registry.get("ngspice"), ["-b", 3])  # type: ignore[list-item]


# ---------------------------------------------------------------- SPICE templates
def test_templates_and_validation():
    keys = {t["key"] for t in spice.templates()}
    assert keys == {"dram_sense_amp", "dram_1t1c", "gain_cell_2t", "ecore_mac"}
    with pytest.raises(spice.CircuitError):
        spice.validate("rm -rf", {})
    with pytest.raises(spice.CircuitError):
        spice.validate("dram_sense_amp", {"vdd": 9})
    with pytest.raises(spice.CircuitError):
        spice.validate("dram_sense_amp", {"bit": 0.5})
    with pytest.raises(spice.CircuitError):
        spice.validate("dram_sense_amp", {".control": 1})
    with pytest.raises(spice.CircuitError):
        spice.validate("dram_sense_amp", {"vdd": "1; .shell ls"})
    with pytest.raises(spice.CircuitError):
        spice.validate("dram_sense_amp", {"vdd": float("nan")})


@pytest.mark.parametrize("tpl", sorted(spice.SPECS))
def test_netlist_is_only_template_text(tpl):
    p, circ = spice.build(tpl)
    text = spice.netlist(tpl, p, circ)
    assert ".tran" in text and text.rstrip().endswith(".end")
    low = text.lower()
    for bad in (".control", "shell", ".include", ".lib ", "system"):
        assert bad not in low
    nets = spice.nets(circ)
    assert "0" not in nets and len(nets) >= 3


def _raw_binary(names, data):
    head = ["Title: t", "Plotname: Transient Analysis", "Flags: real",
            f"No. Variables: {len(names)}", f"No. Points: {len(data)}", "Variables:"]
    head += [f"\t{i}\t{n}\tvoltage" for i, n in enumerate(names)]
    head.append("Binary:")
    body = b"".join(struct.pack("<" + "d" * len(names), *row) for row in data)
    return ("\n".join(head) + "\n").encode() + body


def test_parse_raw_binary_and_ascii():
    rows = [(0.0, 0.0, 1.0), (1e-9, 0.5, 0.9), (2e-9, 1.0, 0.8)]
    sig = spice.parse_raw(_raw_binary(["time", "v(a)", "v(b)"], rows))
    assert np.allclose(sig["v(a)"], [0, 0.5, 1.0]) and np.allclose(sig["time"][-1], 2e-9)
    ascii_raw = ("Title: t\nFlags: real\nNo. Variables: 2\nNo. Points: 2\nVariables:\n"
                 "\t0\ttime\ttime\n\t1\tv(x)\tvoltage\nValues:\n0\t0.0\n\t0.25\n1\t1e-9\n\t0.75\n")
    sig = spice.parse_raw(ascii_raw.encode())
    assert np.allclose(sig["v(x)"], [0.25, 0.75])
    with pytest.raises(spice.CircuitError):
        spice.parse_raw(b"Title: nothing\n")


@pytest.mark.skipif(not HAVE_NGSPICE, reason="ngspice not installed")
@pytest.mark.parametrize("tpl", ["dram_sense_amp", "dram_1t1c", "gain_cell_2t"])
@pytest.mark.parametrize("bit", [0, 1])
def test_ngspice_reads_back_the_stored_bit(tpl, bit):
    r = spice.simulate(tpl, {"bit": bit})
    assert r["simulator"]["name"] == "ngspice" and r["simulator"]["points"] > 100
    assert r["metrics"]["read_bit"] == bit and r["metrics"]["correct"]
    assert len(r["time_ns"]) == len(next(iter(r["voltages"].values())))
    assert r["currents_ua"], "device currents (savecurrents) missing"


@pytest.mark.skipif(not HAVE_NGSPICE, reason="ngspice not installed")
def test_ngspice_sense_amp_matches_charge_sharing():
    m = spice.simulate("dram_sense_amp")["metrics"]
    # bit-line signal before sensing ~ ideal Cs/(Cs+Cbl) charge share
    assert abs(m["signal_mv"] - m["ideal_charge_share_mv"]) < 15
    assert m["sense_time_ps"] and m["restored_cell_v"] > 1.0


@pytest.mark.skipif(not HAVE_NGSPICE, reason="ngspice not installed")
def test_ngspice_ecore_column_decodes_dot_product():
    for w, x in (((1, 0, 1, 1), (1, 1, 1, 0)), ((1, 1, 1, 1), (1, 1, 1, 1)),
                 ((0, 0, 0, 0), (1, 0, 1, 0))):
        params = {f"w{i}": w[i] for i in range(4)} | {f"x{i}": x[i] for i in range(4)}
        m = spice.simulate("ecore_mac", params)["metrics"]
        assert m["dot_product"] == sum(a * b for a, b in zip(w, x))
        assert m["correct"], m


# ---------------------------------------------------------------- schematics
def test_generated_schematic_tags_every_net():
    _, circ = spice.build("dram_sense_amp")
    svg = schematic.generated_svg(circ["devices"])
    for n in spice.nets(circ):
        assert f'data-net="{n}"' in svg
    sch = schematic.xschem_sch(circ["devices"])
    assert sch.startswith("v {xschem") and "{nmos4.sym}" in sch and "{lab_pin.sym}" in sch


@pytest.mark.skipif(not HAVE_XSCHEM, reason="xschem not installed")
def test_xschem_exports_svg_with_net_labels():
    _, circ = spice.build("gain_cell_2t")
    out = schematic.schematic(circ["devices"], spice.nets(circ))
    assert out["source"] == "xschem", out.get("note")
    assert out["svg"].lstrip().startswith("<svg") and 'data-net="sn"' in out["svg"]


# ---------------------------------------------------------------- KiCad
def test_kicad_pcb_export_is_balanced_and_complete():
    sc = _gpu_scene()
    text = kicad.kicad_pcb(sc)
    assert text.startswith("(kicad_pcb (version 20221018)")
    depth = 0
    in_str = False
    prev = ""
    for ch in text:
        if ch == '"' and prev != "\\":
            in_str = not in_str
        elif not in_str:
            depth += ch == "("
            depth -= ch == ")"
            assert depth >= 0
        prev = ch
    assert depth == 0
    n_parts = len(sc["pcb"]["parts"])
    assert text.count("(footprint ") == n_parts + 1      # + GPU package
    assert '"Edge.Cuts"' in text and '(net_name "GND")' in text
    s = kicad.summary(sc)
    assert s["vrm"]["phases"] == sc["pcb"]["vrm"]["phases"]


@pytest.mark.skipif(not HAVE_KICAD, reason="kicad-cli not installed")
def test_kicad_cli_renders_and_checks_the_board():
    r = kicad.render(_gpu_scene())
    assert not r["errors"], r["errors"]
    assert r["svg"].lstrip().startswith("<?xml") or "<svg" in r["svg"][:500]
    assert len(r["png_b64"]) > 1000
    assert "violations" in r["drc"] and r["drc"]["unconnected"] >= 0


# ---------------------------------------------------------------- KLayout
def test_layout_upload_validation(tmp_path):
    with pytest.raises(layout.LayoutError):
        layout.store_upload(tmp_path, "x.gds", b"")
    with pytest.raises(layout.LayoutError):
        layout.store_upload(tmp_path, "x.gds", b"#!/bin/sh\necho hi\n")
    with pytest.raises(layout.LayoutError):
        layout.store_upload(tmp_path, "evil.sh", b"\x00\x06\x00\x02" + b"\x00" * 10)
    with pytest.raises(layout.LayoutError):
        layout.info(tmp_path, "../../etc/passwd")
    lid = layout.store_upload(tmp_path, "../../a.gds", b"\x00\x06\x00\x02" + b"\x00" * 10)
    assert "/" not in lid and (tmp_path / lid).is_file()


@pytest.mark.skipif(not HAVE_KLAYOUT, reason="klayout not installed")
def test_klayout_info_and_render(tmp_path):
    import klayout.db as kdb
    ly = kdb.Layout()
    top = ly.create_cell("TOP")
    for lay_, dt in ((1, 0), (2, 0)):
        top.shapes(ly.layer(lay_, dt)).insert(kdb.DBox(0, 0, 10 * lay_, 5))
    f = tmp_path / "t.oas"
    ly.write(str(f))
    lid = layout.store_upload(tmp_path / "store", "t.oas", f.read_bytes())
    info = layout.info(tmp_path / "store", lid)
    assert info["format"] == "OASIS" and info["top_cells"] == ["TOP"]
    assert {(x["layer"], x["datatype"]) for x in info["layers"]} == {(1, 0), (2, 0)}
    r = layout.render(tmp_path / "store", lid, ["1/0"], 400, 300)
    import base64
    assert base64.b64decode(r["png_b64"])[:4] == b"\x89PNG"
    with pytest.raises(layout.LayoutError):
        layout.render(tmp_path / "store", lid, ["1/0; rm"])


@pytest.mark.skipif(not HAVE_KLAYOUT, reason="klayout not installed")
def test_floorplan_gds_round_trip(tmp_path):
    sc = _gpu_scene()
    lid = layout.floorplan_gds(tmp_path, sc)
    info = layout.info(tmp_path, lid)
    by = {(x["layer"], x["datatype"]): x["shapes"] for x in info["layers"]}
    assert by[(10, 0)] == sum(c["kind"] == "compute_die" for c in sc["components"])
    assert by[(20, 0)] == sum(c["kind"] == "hbm_stack" for c in sc["components"])


# ---------------------------------------------------------------- OpenFOAM
def test_cfd_case_and_h_from_outlet():
    sc = _gpu_scene()
    pv = cfd.preview(sc)
    files = pv["files"]
    assert {"system/blockMeshDict", "system/controlDict", "0/U", "0/T"} <= set(files)
    ch = pv["channel"]
    assert ch["wall_c"] > ch["ambient_c"] and ch["gap_mm"] > 0
    # outlet at ambient -> no heat; outlet near wall -> large h; monotonic in between
    lo = cfd.h_from_outlet(ch, ch["ambient_c"] + 0.1 * (ch["wall_c"] - ch["ambient_c"]))
    hi = cfd.h_from_outlet(ch, ch["ambient_c"] + 0.5 * (ch["wall_c"] - ch["ambient_c"]))
    assert 0 < lo < hi


@pytest.mark.skipif(not HAVE_FOAM, reason="OpenFOAM not installed")
def test_openfoam_channel_runs():
    r = cfd.run(_gpu_scene())
    assert r["h_cfd_w_m2k"] > 0 and r["outlet_c"] > r["channel"]["ambient_c"]


# ---------------------------------------------------------------- HTTP API
def test_tools_api(tmp_path):
    from starlette.testclient import TestClient
    from test_studio import _gpu

    from chiplab.app.server import create_app
    from chiplab.experiments.store import ExperimentStore

    db = str(tmp_path / "s.db")
    ExperimentStore(db_path=db).save_design(_gpu())
    c = TestClient(create_app(db_path=db))
    assert c.get("/toolviews.js").status_code == 200
    assert "klayout" in c.get("/api/tools/status").json()
    assert len(c.get("/api/tools/circuit/templates").json()) == 4
    evil = {"Origin": "http://evil.example"}
    body = {"template": "dram_sense_amp", "params": {"bit": 0}}
    assert c.post("/api/tools/circuit/run", json=body, headers=evil).status_code == 403
    assert c.post("/api/tools/circuit/run", json={"template": "x"}).status_code == 400
    assert c.post("/api/tools/circuit/run",
                  json={"template": "dram_sense_amp", "params": {"vdd": "1 .shell"}}
                  ).status_code == 400
    r = c.post("/api/tools/circuit/run", json=body)
    assert r.status_code == 200
    j = r.json()
    assert "svg" in j["schematic"] and ".tran" in j["netlist"]
    if HAVE_NGSPICE:
        assert j["metrics"]["read_bit"] == 0
    else:
        assert j["missing"] == "ngspice"
    f = c.get("/api/tools/board/kicad_pcb", params={"design": "studio_gpu_0.1"})
    assert f.status_code == 200 and f.text.startswith("(kicad_pcb")
    assert "attachment" in f.headers["content-disposition"]
    up = c.post("/api/tools/layout/upload", content=b"not a layout",
                headers={"Content-Type": "application/octet-stream", "X-Filename": "a.gds"})
    assert up.status_code in (400, 409)
    assert c.post("/api/tools/layout/upload", content=b"x",
                  headers={"Content-Type": "application/octet-stream", **evil}).status_code == 403
    cf = c.post("/api/tools/cfd", json={"design": "studio_gpu_0.1"})
    assert cf.status_code == 200 and "files" in cf.json()
