"""SCALE-Sim runner. Executed by ChipLab as
    <scalesim-venv>/bin/python -I _scalesim_runner.py <request.json> <result.json>
with an argument list (no shell). It does not import ChipLab. The request holds a systolic
array configuration and a list of GEMM shapes (bounded ints); the runner writes SCALE-Sim's
config/topology files into a temporary directory, runs it without saving traces, and writes
the per-layer compute report as JSON.
"""

import contextlib
import csv
import glob
import io
import json
import os
import sys
import tempfile

DATAFLOWS = ("os", "ws", "is")
MAX_DIM = 1024
MAX_GEMM = 1 << 16
MAX_LAYERS = 32

CFG = """[general]
run_name = chiplab

[architecture_presets]
ArrayHeight: {rows}
ArrayWidth: {cols}
IfmapSramSzkB: {ifmap_kb}
FilterSramSzkB: {filter_kb}
OfmapSramSzkB: {ofmap_kb}
IfmapOffset: 0
FilterOffset: 10000000
OfmapOffset: 20000000
Bandwidth: 10
Dataflow: {dataflow}
MemoryBanks: 1
ReadRequestBuffer: 32
WriteRequestBuffer: 32

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: 10
IfmapSRAMBankNum: 10
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: 10
FilterSRAMBankNum: 10
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport: false
SparseRep: ellpack_block
OptimizedMapping: false
BlockSize: 8
RandomNumberGeneratorSeed: 40

[run_presets]
InterfaceBandwidth: CALC
UseRamulatorTrace: False
"""


def _int(v, lo, hi, name):
    v = int(v)
    if not lo <= v <= hi:
        raise SystemExit(f"{name} out of range")
    return v


def main():
    req = json.load(open(sys.argv[1]))
    rows = _int(req["array_rows"], 4, MAX_DIM, "array_rows")
    cols = _int(req["array_cols"], 4, MAX_DIM, "array_cols")
    df = req["dataflow"]
    if df not in DATAFLOWS:
        raise SystemExit("invalid dataflow")
    kb = {k: _int(req[k], 16, 1 << 20, k) for k in ("ifmap_kb", "filter_kb", "ofmap_kb")}
    gemms = req["gemms"]
    if not 1 <= len(gemms) <= MAX_LAYERS:
        raise SystemExit("1-32 GEMMs")
    lines = ["Layer, M, N, K,"]
    for i, g in enumerate(gemms):
        m, n, k = (_int(g[x], 1, MAX_GEMM, x) for x in ("M", "N", "K"))
        lines.append(f"layer{i}, {m}, {n}, {k},")

    from scalesim.scale_sim import scalesim  # only available in the SCALE-Sim venv

    with tempfile.TemporaryDirectory() as tmp:
        cfg, topo, layout = (os.path.join(tmp, f) for f in ("chiplab.cfg", "gemm.csv",
                                                             "layout.csv"))
        with open(cfg, "w") as f:
            f.write(CFG.format(rows=rows, cols=cols, dataflow=df, **kb))
        with open(topo, "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(layout, "w") as f:
            f.write("Layer,\n")
        out_dir = os.path.join(tmp, "out")
        with contextlib.redirect_stdout(io.StringIO()):
            s = scalesim(save_disk_space=True, verbose=False, config=cfg, topology=topo,
                         layout=layout, input_type_gemm=True)
            s.run_scale(top_path=out_dir)
        report = glob.glob(os.path.join(out_dir, "*", "COMPUTE_REPORT.csv"))
        bw = glob.glob(os.path.join(out_dir, "*", "BANDWIDTH_REPORT.csv"))
        if not report:
            raise SystemExit("SCALE-Sim produced no COMPUTE_REPORT.csv")
        layers = []
        with open(report[0]) as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                layers.append({
                    "total_cycles_incl_prefetch": int(float(row["Total Cycles (incl. prefetch)"])),
                    "total_cycles": int(float(row["Total Cycles"])),
                    "stall_cycles": int(float(row["Stall Cycles"])),
                    "overall_util_pct": float(row["Overall Util %"]),
                    "mapping_efficiency_pct": float(row["Mapping Efficiency %"]),
                    "compute_util_pct": float(row["Compute Util %"]),
                })
        if bw:
            with open(bw[0]) as f:
                for layer, row in zip(layers, csv.DictReader(f, skipinitialspace=True)):
                    layer["dram_words_per_cycle"] = sum(
                        float(row[k]) for k in ("Avg IFMAP DRAM BW", "Avg FILTER DRAM BW",
                                                "Avg OFMAP DRAM BW"))
    for layer, g in zip(layers, gemms):
        layer.update({"M": int(g["M"]), "N": int(g["N"]), "K": int(g["K"])})
    with open(sys.argv[2], "w") as f:
        json.dump({"array_rows": rows, "array_cols": cols, "dataflow": df, **kb,
                   "layers": layers, "scalesim_version": "3.0.0"}, f)


if __name__ == "__main__":
    main()
