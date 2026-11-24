"""Ramulator 2.x runner. Executed by ChipLab as
    <ramulator-venv>/bin/python -I _ramulator_runner.py <request.json> <result.json>
with an argument list (no shell). It does not import ChipLab. The request names a DRAM
standard from the fixed table below plus bounded trace parameters; the runner generates the
trace itself, runs one memory channel and writes the statistics as JSON.
"""

import json
import os
import random
import sys
import tempfile

# standard -> (dram class, org preset, timing preset, controller class)
STANDARDS = {
    "HBM2": ("HBM2", "HBM2_8Gb", "HBM2_2400Mbps", "HBM12"),
    "HBM3": ("HBM3", "HBM3_16Gb_8hi", "HBM3_6400Mbps", "HBM34"),
    "HBM4": ("HBM4", "HBM4_32Gb_8Hi", "HBM4_8000Mbps", "HBM34"),
    "LPDDR5": ("LPDDR5", "LPDDR5_16Gb_x16", "LPDDR5_6400", "LPDDR5"),
    "DDR4": ("DDR4", "DDR4_8Gb_x8", "DDR4_3200AA", "GenericDDR"),
    "DDR5": ("DDR5", "DDR5_16Gb_x8", "DDR5_6400AN", "GenericDDR"),
    "GDDR7": ("GDDR7", "GDDR7_16Gb_x8", "GDDR7_28000_PAM3", "GDDR7"),
}
PATTERNS = ("sequential", "random")
LINE = 64


def _write_trace(path, pattern, n, write_fraction, seed):
    rng = random.Random(seed)
    # Reads and writes go to separate 512 MB regions so writes don't alias read rows.
    w_base = 1 << 29
    every = int(round(1 / write_fraction)) if write_fraction > 0 else 0
    with open(path, "w") as f:
        r_i = w_i = 0
        for i in range(n):
            is_w = every > 0 and i % every == every - 1
            if pattern == "sequential":
                addr = (w_base + w_i * LINE) if is_w else r_i * LINE
            else:
                addr = rng.randrange(0, 1 << 29) // LINE * LINE + (w_base if is_w else 0)
            if is_w:
                w_i += 1
            else:
                r_i += 1
            f.write(("ST " if is_w else "LD ") + str(addr) + "\n")


def main():
    req = json.load(open(sys.argv[1]))
    std = req["standard"]
    pattern = req["pattern"]
    n = int(req["n_requests"])
    wf = float(req["write_fraction"])
    seed = int(req["seed"])
    if std not in STANDARDS or pattern not in PATTERNS:
        raise SystemExit("invalid standard or pattern")
    if not 1000 <= n <= 2_000_000 or not 0.0 <= wf <= 0.5:
        raise SystemExit("n_requests or write_fraction out of range")

    import ramulator  # noqa: E402  (only available in the Ramulator venv)

    dram_cls, org, timing, ctrl_cls = STANDARDS[std]
    kwargs = {"rank": 1} if ctrl_cls == "GenericDDR" and std.startswith("DDR") else {}
    dram = getattr(ramulator.dram, dram_cls)(org_preset=org, timing_preset=timing, **kwargs)
    ctrl = getattr(ramulator.controller, ctrl_cls)(
        dram=dram, scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.MOP4CLXOR())
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=1, controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    with tempfile.TemporaryDirectory() as tmp:
        trace = os.path.join(tmp, "chiplab.trace")
        _write_trace(trace, pattern, n, wf, seed)
        fe = ramulator.frontend.LoadStoreTrace(clock_ratio=1, path=trace)
        sim = ramulator.Simulation(fe, mem)
        sim.run()
        stats = sim.stats
    c = stats["memory_system"]["controller"]
    o = getattr(ramulator.dram, dram_cls).org_presets[org]
    t = getattr(ramulator.dram, dram_cls).timing_presets[timing]
    out = {
        "standard": std, "dram": dram_cls, "org_preset": org, "timing_preset": timing,
        "controller": ctrl_cls, "addr_mapper": "MOP4CLXOR", "scheduler": "FRFCFS",
        "row_policy": "Open", "refresh": "AllBank",
        "rate_mtps": t["rate"], "channel_width_bits": o["channel_width"], "tck_ps": t["tCK_ps"],
        "cycles": c["cycles"],
        "read_throughput_mbps": c["read_throughput_MBps"],
        "write_throughput_mbps": c["write_throughput_MBps"],
        "total_throughput_mbps": c["total_throughput_MBps"],
        "avg_read_latency_cycles": c["avg_read_latency"],
        "num_reads": c["num_read_reqs"], "num_writes": c["num_write_reqs"],
        "row_hits": c["row_hits"], "row_misses": c["row_misses"],
        "row_conflicts": c["row_conflicts"],
        "ramulator_version": getattr(ramulator, "__version__", "2.1"),
    }
    with open(sys.argv[2], "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
