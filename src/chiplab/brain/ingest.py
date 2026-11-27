"""Fill the Project Brain from what ChipLab already knows. Every ingest is idempotent (keyed by
source_key), so it can run at every start without duplicating memories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from chiplab.brain.store import Brain

_ROOT = Path(__file__).resolve().parents[3]

# Curated scientific knowledge. Each entry names authors, venue and year (DOIs were not
# recorded here; look them up before citing). Numbers are quoted only where the
# source states them. (title, body, tags, source, status)
SCIENCE: tuple[tuple[str, str, tuple[str, ...], str, str], ...] = (
    ("SKY130 interconnect R and C from the SkyWater PDK RCX tables",
     "Sheet resistance li1 12.8, met1/met2 0.125, met3/met4 0.047, met5 0.029 ohm/sq; "
     "interlayer plate and fringe capacitance tables give ~0.2-0.3 fF/um for a 1 um wire over "
     "the layer below. Used by models.physics.rc_extraction.rc_delay (1 mm met1: ~15 ps).",
     ("sky130", "wire", "rc", "interconnect", "resistance", "capacitance"),
     "skywater-pdk docs/rules/rcx (resistance-values.tsv, capacitance-*.tsv)", "active"),
    ("Ambit: bulk bitwise AND/OR by triple-row activation in DRAM",
     "Activating three DRAM rows at once makes the bitline settle to the majority of the three "
     "cells (charge sharing). With a control row of all 0s the result is AND, all 1s gives OR. "
     "NOT uses a dual-contact cell that connects a row to the sense amplifier's negated "
     "bitline. Operands are first copied into reserved rows with RowClone because triple-row "
     "activation destroys the source rows.",
     ("dram", "ambit", "charge sharing", "tra", "bitwise", "pim"),
     "Seshadri et al., MICRO 2017", "active"),
    ("RowClone: in-DRAM bulk copy and initialisation",
     "Two back-to-back row activations in the same subarray copy one row into another through "
     "the sense amplifiers (Fast Parallel Mode) without moving data over the memory channel. "
     "Copies across banks use a slower pipelined serial mode.",
     ("dram", "rowclone", "copy", "pim", "data movement"),
     "Seshadri et al., MICRO 2013", "active"),
    ("ComputeDRAM: multi-row activation on unmodified commodity DDR3",
     "Issuing ACT-PRE-ACT with timing parameters violated opens several rows in some "
     "off-the-shelf DDR3 modules, giving in-memory copy and majority/AND/OR. Behaviour and "
     "error rates vary between modules and vendors; many modules do not support it.",
     ("dram", "computedram", "multi-row", "timing", "error rate", "pim"),
     "Gao, Addisie, Zhuo, MICRO 2019", "active"),
    ("RowHammer: repeated activation disturbs neighbouring DRAM rows",
     "Repeatedly activating a row can flip bits in adjacent rows. Any in-DRAM compute scheme "
     "that activates rows heavily must budget for read disturb and refresh.",
     ("dram", "rowhammer", "read disturb", "reliability"),
     "Kim et al., ISCA 2014", "active"),
    ("Compute Caches: bitwise operations on SRAM bitlines",
     "Activating two SRAM wordlines at once and sensing the bitline pair gives AND (BL) and NOR "
     "(BLB) of the two rows; XOR and comparison follow with small peripheral logic. Operands "
     "must share bitlines (same subarray).",
     ("sram", "compute cache", "bitline", "bitwise", "nmc", "in-memory"),
     "Aga et al., HPCA 2017", "active"),
    ("Neural Cache: bit-serial arithmetic inside SRAM arrays",
     "Transposed data layout plus bitline logic performs bit-serial add, multiply and "
     "reduction across many columns in parallel. Throughput comes from column parallelism; "
     "latency per operation grows with bit width.",
     ("sram", "neural cache", "bit-serial", "mac", "add", "nmc"),
     "Eckert et al., ISCA 2018", "active"),
    ("ISAAC: analog crossbar dot products with ADC cost dominating",
     "Memristor crossbars compute matrix-vector products in the analog domain; in the ISAAC "
     "design the ADCs take a large share of power and area, which sets the practical "
     "precision per column.",
     ("memristor", "crossbar", "analog", "adc", "mvm", "isaac"),
     "Shafiee et al., ISCA 2016", "active"),
    ("Samsung HBM-PIM: SIMD FP16 units at DRAM banks",
     "A commercial HBM2-based PIM places programmable FP16 SIMD units near the banks and runs "
     "memory-bound kernels (e.g. GEMV) without changing the DRAM cell array.",
     ("dram", "hbm", "pim", "bank", "near memory", "commercial"),
     "Lee et al., ISCA 2021", "active"),
    ("UPMEM: general-purpose cores inside DRAM chips",
     "Real UPMEM DIMMs place small in-order cores (DPUs) next to DRAM banks. Benchmarks show "
     "gains on memory-bound workloads with little inter-DPU communication, and weak results "
     "for workloads that need multiplication-heavy or cross-DPU traffic.",
     ("dram", "upmem", "pim", "near memory", "benchmark"),
     "Gomez-Luna et al., IEEE Access 2022", "active"),
    ("Energy cost of compute versus data movement (45 nm)",
     "Approximate energies at 45 nm: 8-bit integer add 0.03 pJ, 32-bit add 0.1 pJ, 8-bit "
     "multiply 0.2 pJ, 32-bit multiply 3.1 pJ, 32-bit float multiply 3.7 pJ; 64-bit cache "
     "read 10 pJ (8 KB), 20 pJ (32 KB), 100 pJ (1 MB); DRAM access 1.3-2.6 nJ. Moving data "
     "costs far more than computing on it, which motivates processing in or near memory.",
     ("energy", "data movement", "dram", "sram", "pj", "compute"),
     "Horowitz, ISSCC 2014", "active"),
    ("Landauer limit: minimum energy to erase a bit",
     "Erasing one bit dissipates at least kT ln 2 (about 2.9e-21 J at 300 K). ChipLab's review "
     "gate rejects any energy-per-bit claim below this.",
     ("energy", "landauer", "physics", "limit"),
     "Landauer, IBM J. Res. Dev. 5(3), 1961", "active"),
)

LIMITATIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("DRAM PIM energy model underestimates activation energy by ~3.5-4.4x",
     "chiplab.pim.dram counts only bitline swing per activation (~0.51 nJ per 8 Kb row). The "
     "Ambit (35x) and RowClone (74.4x) energy ratios each imply ~1.8-2.2 nJ "
     "(implied_activation_energy), so PIM gains are over-predicted (122x / 325x). Supply the "
     "part's datasheet IDD0/IDD2N/IDD3N (idd0_a, idd2n_a, idd3n_a) to use the TN-41-01 method.",
     ("dram", "energy", "pim", "ambit", "rowclone")),
    ("Digital shuttles cannot fabricate DRAM, memristor or ferrofluid devices",
     "Tiny Tapeout and other SKY130/IHP digital shuttles offer standard cells only. DRAM "
     "storage capacitors, memristors, PCM and ferrofluid cells cannot be built there; on a "
     "digital shuttle in-memory computing must use flip-flop/latch arrays or an SRAM macro.",
     ("tapeout", "tiny tapeout", "dram", "memristor", "ferrofluid", "fabrication")),
    ("dram_cim_bitcell has no compute operation",
     "The SDK component dram_cim_bitcell is an access transistor + capacitor + sense amp. It "
     "stores and senses one bit; it does not implement any logic. Use the DRAM charge-sharing "
     "simulator (chiplab.pim.dram) for Ambit-style operations.",
     ("dram", "component", "cim", "bitcell")),
    ("Ferrofluid memory model is not validated",
     "The ferrofluid electrical state-space model has no measurement data behind it; the "
     "published two-port experiment needs the authors' raw data. Treat all ferrofluid "
     "numbers as hypotheses.",
     ("ferrofluid", "validation", "model")),
    ("Learning loop cannot detect fabricated measurements",
     "Measurements are gated on source kind, method, date, units and uncertainty, and every "
     "point records who submitted it, but invented data that passes these checks would be "
     "learned from.",
     ("learning", "measurement", "validation")),
    ("Analog crossbar lab uses ideal op-amps",
     "imc_lab analog_crossbar models TIAs as finite-gain VCVS with a single-pole settling "
     "estimate; op-amp offset, noise, slew and input bias are not modelled.",
     ("analog", "crossbar", "opamp", "imc")),
    ("TT SKY26 shuttle: do not submit unverified experimental designs",
     "The next Tiny Tapeout SKY26 shuttle closes 2026-11-30. Only designs that pass "
     "verification, review and gate-level test should go; experimental PIM designs are not "
     "ready for that run.",
     ("tapeout", "tiny tapeout", "schedule", "pim")),
)


def seed_science(brain: Brain) -> int:
    for title, body, tags, src, status in SCIENCE:
        brain.remember("science", title, body, tags=tags, source=src, status=status,
                       source_key=f"science:{title}")
    for title, body, tags in LIMITATIONS:
        brain.remember("project", title, body, tags=tags, status="limitation",
                       source="chiplab", source_key=f"limitation:{title}")
    return len(SCIENCE) + len(LIMITATIONS)


def ingest_ledger(brain: Brain, ledger: Any = None, limit: int = 500) -> int:
    from chiplab.experiment_db.ledger import ExperimentLedger, default_db_path
    ledger = ledger or ExperimentLedger(default_db_path())
    runs = ledger.query_runs(limit=limit)
    for r in runs:
        tech = r["params"].get("technology", "") if isinstance(r["params"], dict) else ""
        metrics = ", ".join(f"{k}={v}" for k, v in list(r["metrics"].items())[:8])
        excl = f" Excluded from learning: {r['exclusion_reason']}." if r[
            "excluded_from_learning"] else ""
        brain.remember(
            "experiment", f"Run {r['run_id']} ({r['design_name']}, {r['fidelity']})",
            f"Parameters: {r['params']}. Metrics: {metrics}. Repeats: {r['repeats']}.{excl}",
            tags=(r["fidelity"] or "unknown", str(tech), r["design_name"]),
            source="experiment_ledger", source_key=f"ledger:{r['run_id']}",
            status="superseded" if r["excluded_from_learning"] else "active",
            data={"run_id": r["run_id"], "fidelity": r["fidelity"]})
    return len(runs)


def ingest_learning(brain: Brain, loop: Any = None) -> int:
    from chiplab.learning import LearningLoop
    loop = loop or LearningLoop()
    st = loop.status()
    for m in st["corrections"]:
        brain.remember(
            "experiment", f"Model correction {m['model_id']}.{m['metric']} v{m['version']}",
            f"Status {m['status']} from {m['n']} validated measurements. {m.get('why') or ''}",
            tags=("learning", "correction", m["model_id"].split(".")[0]),
            source="learning_loop", source_key=f"learning:{m['model_id']}:{m['metric']}")
    return len(st["corrections"])


def ingest_models(brain: Brain) -> int:
    from chiplab import model_registry
    models = model_registry.describe()["models"]
    for m in models:
        brain.remember(
            "tool", f"Model {m['id']}: {m['title']}",
            f"Technology {m['technology']}, fidelity {m['fidelity']}, validation: "
            f"{m['validation_status']}. Effects: {', '.join(m['supported_effects'])}.",
            tags=("model", m["technology"], m["fidelity"]), source="model_registry",
            source_key=f"model:{m['id']}")
        for lim in m["known_limitations"]:
            brain.remember("tool", f"{m['id']} limitation", lim,
                           tags=("model", m["technology"], "limitation"), status="limitation",
                           source="model_registry", source_key=f"model:{m['id']}:lim:{lim[:80]}")
    return len(models)


def ingest_tools(brain: Brain) -> int:
    from chiplab.tools import registry
    st = registry.status()
    for key, info in st.items():
        avail = info.get("available") if isinstance(info, dict) else None
        brain.remember("tool", f"EDA tool {key}",
                       f"{'Installed' if avail else 'Not installed'}. {info}",
                       tags=("eda", key, "installed" if avail else "missing"),
                       source="tools.registry", source_key=f"eda:{key}")
    return len(st)


def ingest_components(brain: Brain) -> int:
    from chiplab.sdk.components import LIBRARY, TEMPLATES
    n = 0
    for d in list(LIBRARY) + list(TEMPLATES):
        brain.remember("tool", f"Component {d['name']} ({d.get('category', '')})",
                       d.get("description", ""),
                       tags=("component", d.get("category", ""), d["name"]),
                       source="sdk.components", source_key=f"component:{d['name']}")
        n += 1
    return n


def ingest_imc(brain: Brain) -> int:
    from chiplab import imc_lab
    cat = imc_lab.catalog()
    items = cat.items() if isinstance(cat, dict) else ((c.get("family"), c) for c in cat)
    n = 0
    for fam, info in items:
        brain.remember("architecture", f"IMC lab family {fam}", str(info)[:2000],
                       tags=("imc", "pim", "buildable", str(fam)), source="imc_lab",
                       source_key=f"imc:{fam}")
        n += 1
    return n


def ingest_docs(brain: Brain, docs_dir: Path | None = None) -> int:
    docs_dir = docs_dir or _ROOT / "docs"
    n = 0
    for p in sorted(docs_dir.glob("*.md")):
        sections = re.split(r"(?m)^#{1,3} ", p.read_text(errors="replace"))
        for i, sec in enumerate(sections):
            sec = sec.strip()
            if len(sec) < 40:
                continue
            title, _, body = sec.partition("\n")
            brain.remember("project", f"{p.stem}: {title.strip()[:120]}", body.strip()[:4000],
                           tags=("doc", p.stem), source=f"docs/{p.name}",
                           source_key=f"doc:{p.name}:{i}")
            n += 1
    return n


def ingest_all(brain: Brain | None = None) -> dict[str, Any]:
    brain = brain or Brain()
    counts: dict[str, Any] = {}
    for name, fn in (("science", seed_science), ("ledger", ingest_ledger),
                     ("learning", ingest_learning), ("models", ingest_models),
                     ("tools", ingest_tools), ("components", ingest_components),
                     ("imc", ingest_imc), ("docs", ingest_docs)):
        try:
            counts[name] = fn(brain)
        except Exception as e:  # noqa: BLE001 - one bad source must not stop the others
            counts[name] = f"error: {type(e).__name__}: {e}"
    return counts | {"stats": brain.stats()}
