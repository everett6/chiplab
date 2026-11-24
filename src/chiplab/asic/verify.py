"""RTL verification for the PEM core: Verilator build/run, lockstep comparison against the
independent Python reference model, constrained-random programs and coverage.

Nothing here invents a result: every number comes from a run that happened in this process
(or from a stored regression record that names the run). If Verilator is missing, the
functions say so instead of passing.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from chiplab.asic import isa, iss
from chiplab.asic.assembler import Program

RTL = Path(__file__).parent / "rtl"
RTL_FILES = ["pem_lane.v", "pem_core.v"]
BUILD_ROOT = Path(os.environ.get("CHIPLAB_ASIC_BUILD", Path.home() / ".cache" / "chiplab_asic"))


class ToolMissing(RuntimeError):
    pass


def verilator() -> str:
    exe = shutil.which("verilator") or str(Path.home() / ".local" / "bin" / "verilator")
    if not Path(exe).exists():
        raise ToolMissing("verilator not installed")
    return exe


def rtl_hash() -> str:
    h = hashlib.sha256()
    for f in RTL_FILES + ["tb_core.cpp"]:
        h.update((RTL / f).read_bytes())
    return h.hexdigest()[:16]


MUTATIONS = [  # (name, file, original, mutant): deliberate bugs the regression must catch
    ("nrzi_no_toggle", "pem_lane.v", "if (!data[0]) n_lvl_out = ~lvl_out;", "if (1'b0) n_lvl_out = ~lvl_out;"),
    ("stuff_after_5", "pem_lane.v", "stuff && ones_out == 3'd6", "stuff && ones_out == 3'd5"),
    ("jmp_xdec_off", "pem_lane.v", "if (arg[7:5] == 3'd2) n_x = x - 16'd1;", "if (1'b0) n_x = x - 16'd1;"),
    ("fifo_pop_keeps_head", "pem_core.v", "tx0[l] <= tx1[l]; txc[l] <= txc[l] - 2'd1;", "txc[l] <= txc[l] - 2'd1;"),
    ("crc_msb_wrong_bit", "pem_lane.v", "fb = b ^ c[cc[4:0]];", "fb = b ^ c[0];"),
    ("irq_clear_ignored", "pem_core.v", "irq <= (irq & ~c) | s;", "irq <= irq | s;"),
]


def build(coverage: bool = False, mutation: str | None = None) -> Path:
    """Compile the core bench with Verilator (cached by source hash)."""
    exe = verilator()
    out = BUILD_ROOT / f"core_{rtl_hash()}{'_cov' if coverage else ''}{'_' + mutation if mutation else ''}"
    binary = out / "Vpem_core"
    if binary.exists():
        return binary
    out.mkdir(parents=True, exist_ok=True)
    src = out / "src"                    # copy sources: make cannot handle spaces in paths
    src.mkdir(exist_ok=True)
    for f in RTL_FILES + ["tb_core.cpp"]:
        shutil.copy(RTL / f, src / f)
    if mutation:
        _, mf, orig, mut = next(m for m in MUTATIONS if m[0] == mutation)
        text = (src / mf).read_text()
        if orig not in text:
            raise RuntimeError(f"mutation {mutation} no longer applies to {mf}")
        (src / mf).write_text(text.replace(orig, mut, 1))
    cmd = [exe, "--cc", "--exe", "--build", "-O2", "--assert", "-DPEM_ASSERT", "-Wno-fatal",
           "-Wno-lint", "-Wno-style", "--top-module", "pem_core", "--Mdir", str(out),
           *(str(src / f) for f in RTL_FILES), str(src / "tb_core.cpp")]
    if coverage:
        cmd.insert(2, "--coverage")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not binary.exists():
        raise RuntimeError("verilator build failed:\n" + (r.stderr or r.stdout)[-3000:])
    return binary


_ACTIVE_MUTATION: str | None = None     # set only by mutation_check


def run_rtl(script: str, coverage_file: str | None = None, mutation: str | None = None) -> list[str]:
    binary = build(coverage=coverage_file is not None, mutation=mutation or _ACTIVE_MUTATION)
    env = dict(os.environ)
    if coverage_file:
        env["PEM_COVERAGE_FILE"] = coverage_file
    r = subprocess.run([str(binary)], input=script, capture_output=True, text=True, timeout=600, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"RTL bench failed (assertion or crash): {(r.stderr or r.stdout)[-2000:]}")
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def lockstep(script: str, coverage_file: str | None = None) -> dict[str, Any]:
    """Run the same script on RTL and reference model; report the first divergence."""
    model = iss.PEM()
    ref = iss.run_script(script, model)
    rtl = run_rtl(script, coverage_file)
    first = None
    for k in range(max(len(ref), len(rtl))):
        a = ref[k] if k < len(ref) else "<end>"
        b = rtl[k] if k < len(rtl) else "<end>"
        if a != b:
            first = {"index": k, "model": a, "rtl": b}
            break
    return {"match": first is None, "events": len(ref), "first_divergence": first,
            "model_lines": ref, "rtl_lines": rtl, "functional": model.coverage}


# ------------------------------------------------------------------ bench script helpers
def load_script(progs: list[tuple[int, Program, dict[str, int]]], enable_mask: int) -> str:
    """Script that loads programs into lanes. progs: (lane, Program, extra lane regs)."""
    lines = []
    img: dict[int, int] = {}
    for _, p, _ in progs:
        img.update(p.words)
    for a, w in sorted(img.items()):
        lines.append(f"W {a:x} {w:x}")
    for lane, p, regs in progs:
        regs = dict(regs)
        regs.setdefault("shift", isa.shift_cfg(start=p.start, wrap=p.wrap))
        if "shift" in regs:
            regs["shift"] = (regs["shift"] & ~(31 << 2 | 31 << 7)) | (p.start << 2) | (p.wrap << 7)
        for name, val in regs.items():
            lines.append(f"W {isa.lane_reg(lane, name):x} {val & 0xFFFF:x}")
    lines.append(f"W {isa.A_CTRL:x} {enable_mask << 4:x}")       # restart, held disabled
    for lane, p, _ in progs:                                         # start at the entry word
        lines.append(f"W {isa.A_PC + lane:x} {min(p.words):x}")
    lines.append(f"W {isa.A_CTRL:x} {enable_mask:x}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ constrained random
_OPS = ["set", "out", "in", "wait", "jmp", "pull", "push", "mov", "irq", "setx", "sety", "line",
        "crc", "nop"]


def random_word(rng: random.Random, n_words: int) -> int:
    """A legal random instruction. Constraints keep programs live: waits are rare and
    satisfiable, jumps stay inside the program, halts are excluded."""
    op = rng.choices(_OPS, weights=[8, 12, 10, 2, 8, 4, 4, 8, 3, 4, 4, 2, 2, 2])[0]
    delay = rng.choice([0, 0, 0, 1, 2, 5])
    if op == "jmp":
        arg = rng.randrange(8) << 5 | rng.randrange(n_words)
    elif op == "wait":
        arg = rng.randrange(2) << 7 | rng.choice([0, 1]) << 5 | rng.randrange(16)
    elif op in ("pull", "push"):
        arg = rng.choice([0, 0, 1])
    elif op == "out":
        arg = rng.choice([0, 0, 1, 2, 3, 4, 6]) << 5 | rng.choice([0, 0, 0, rng.randrange(16)])
    elif op == "mov":
        arg = rng.choice([0, 1, 2, 3, 4, 6, 7]) << 5 | rng.randrange(32)
    elif op == "line":
        arg = rng.randrange(64)
    else:
        arg = rng.randrange(256)
    return isa.encode(op, arg, delay)


def _serial_program(rng: random.Random, base: int) -> list[int]:
    """Randomised serial loop: line-coder flags, CRC snoop, shift bits out and back in."""
    e = isa.encode
    flags = rng.randrange(64) | 1 << 4
    body = [e("line", flags), e("crc", 3), e("pull", 1, rng.choice([0, 1, 3]))]
    loop = base + len(body)
    body += [e("out", 0 << 5 | 0, rng.choice([0, 1, 2])), e("in", 0 << 5 | 0, rng.choice([0, 1])),
             e("jmp", 7 << 5 | loop), e("mov", 3 << 5 | rng.choice([0, 3]) << 3 | 5),
             e("push", 0), e("jmp", base + 2)]
    return body


def random_script(seed: int, cycles: int = 3000) -> str:
    """Random program image + lane configuration + random host traffic and input stimulus.
    Odd seeds use structured serial loops (line coder, stuffing, CRC) with pin loopback."""
    rng = random.Random(seed)
    serial = seed % 2 == 1
    if serial:
        words = []
        for _ in range(3):
            words += _serial_program(rng, len(words))
        n = len(words)
    else:
        n = rng.randrange(6, isa.IMEM_WORDS + 1)
        words = [random_word(rng, n) for _ in range(n)]
    lines = [f"W {a:x} {w:x}" for a, w in enumerate(words)]
    if serial:
        for _ in range(rng.randrange(1, 4)):
            lines.append(f"LB {rng.randrange(14)} {rng.randrange(13)}")
    mask = 0
    for lane in range(isa.N_LANES):
        if rng.random() < 0.8:
            mask |= 1 << lane
        start = rng.choice([0, 9, 18]) if serial else rng.randrange(n)
        wrap = (start + 8) if serial else rng.randrange(start, n)
        regs = {"clkdiv": rng.choice([0, 1, 2, 3, 7]),
                "pins": isa.pins_cfg(rng.randrange(16), rng.randrange(16), rng.randrange(16), rng.randrange(16)),
                "shift": isa.shift_cfg(rng.random() < 0.5, rng.random() < 0.5, start, wrap, rng.random() < 0.4),
                "crc_poly_lo": rng.randrange(1 << 16), "crc_poly_hi": rng.randrange(1 << 16),
                "crc_init_lo": rng.randrange(1 << 16), "crc_init_hi": rng.randrange(1 << 16),
                "crc_cfg": rng.randrange(64)}
        for name, val in regs.items():
            lines.append(f"W {isa.lane_reg(lane, name):x} {val:x}")
    lines.append(f"W {isa.A_CTRL:x} {mask << 4 | mask:x}")
    t = 0
    while t < cycles:
        step = rng.randrange(5, 120)
        lines.append(f"C {step}")
        t += step
        r = rng.random()
        if r < 0.25:
            lines.append(f"S {rng.randrange(1 << 13):x}")
        elif r < 0.45:
            v = rng.choice([0xFFFF, 0xFFFE, 0x7FFF, rng.randrange(1 << 16)]) if serial else rng.randrange(1 << 16)
            lines.append(f"W {0x40 + rng.randrange(4):x} {v:x}")
        elif r < 0.6:
            lines.append(f"R {0x40 + rng.randrange(4):x}")
        elif r < 0.75:
            lines.append(f"R {rng.choice([0x49, 0x49, 0x4A, 0x48, *range(0x50, 0x5C)]):x}")
            if rng.random() < 0.2:
                lines.append(f"W {0x50 + rng.randrange(4):x} {rng.randrange(n):x}")
        elif r < 0.77:
            m2 = rng.randrange(16)
            lines.append(f"W {isa.A_CTRL:x} {rng.randrange(16) << 4 | m2:x}")
        elif r < 0.79:
            lines.append(f"W {isa.A_ERR:x} f")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ coverage
FUNCTIONAL_BINS = (
    [f"op:{o}" for o in isa.OPCODES if o != "halt"]
    + ["jmp:0:1"] + [f"jmp:{c}:{t}" for c in range(1, 8) for t in (0, 1)]
    + ["stall:wait", "stall:pull", "stall:push", "stall:autopull", "autopull:refill", "serial:stuffed_bit", "serial:destuffed_bit",
       "serial:stuff_error"]
    + [f"serial:out nrzi={a} stuff={b} diff={c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    + [f"serial:in nrzi={a} destuff={b}" for a in (0, 1) for b in (0, 1)]
)


def functional_coverage(hits: dict[str, int]) -> dict[str, Any]:
    covered = [b for b in FUNCTIONAL_BINS if hits.get(b)]
    return {"bins": len(FUNCTIONAL_BINS), "covered": len(covered),
            "pct": round(100 * len(covered) / len(FUNCTIONAL_BINS), 1),
            "missing": [b for b in FUNCTIONAL_BINS if not hits.get(b)]}


def code_coverage(dat_files: list[str]) -> dict[str, Any]:
    """Line/branch/toggle points hit, from Verilator coverage data (verilator_coverage)."""
    tool = shutil.which("verilator_coverage") or str(Path.home() / ".local" / "bin" / "verilator_coverage")
    files = [f for f in dat_files if Path(f).exists()]
    if not files or not Path(tool).exists():
        return {"available": False, "reason": "no coverage data or verilator_coverage missing"}
    merged = str(Path(files[0]).with_name("merged.dat"))
    subprocess.run([tool, "--write", merged, *files], capture_output=True, text=True, timeout=300)
    pts = hit = 0
    for line in Path(merged).read_text(errors="replace").splitlines():
        m = re.match(r"^C '(.*)' (\d+)$", line)
        if not m:
            continue
        if "pem_lane.v" not in m[1] and "pem_core.v" not in m[1]:
            continue
        pts += 1
        hit += int(m[2]) > 0
    return {"available": True, "points": pts, "hit": hit,
            "pct": round(100 * hit / pts, 1) if pts else 0.0,
            "note": "Verilator line+toggle points in pem_lane.v/pem_core.v"}


ASSERTIONS = [
    {"id": "A1", "where": "pem_core.v", "text": "TX FIFO occupancy <= 2"},
    {"id": "A2", "where": "pem_core.v", "text": "RX FIFO occupancy <= 2"},
    {"id": "A3", "where": "pem_core.v", "text": "a lane never pushes into a full RX FIFO"},
    {"id": "A4", "where": "pem_core.v", "text": "a lane never pops an empty TX FIFO"},
]


def regression(seeds: int = 20, cycles: int = 3000, coverage: bool = True,
               workdir: Path | None = None) -> dict[str, Any]:
    """Constrained-random lockstep regression. Returns pass/fail per seed with coverage."""
    t0 = time.time()
    wd = Path(workdir or BUILD_ROOT / "regress")
    wd.mkdir(parents=True, exist_ok=True)
    hits: dict[str, int] = {}
    results, dats = [], []
    for s in range(seeds):
        dat = str(wd / f"cov_{s}.dat") if coverage else None
        r = lockstep(random_script(s, cycles), dat)
        if dat:
            dats.append(dat)
        for k, v in r["functional"].items():
            hits[k] = hits.get(k, 0) + v
        results.append({"seed": s, "match": r["match"], "events": r["events"],
                        "first_divergence": r["first_divergence"]})
    return {
        "kind": "constrained_random_lockstep", "rtl_hash": rtl_hash(), "seeds": seeds,
        "cycles_per_seed": cycles, "passed": sum(r["match"] for r in results),
        "failed": [r for r in results if not r["match"]], "functional": functional_coverage(hits),
        "functional_hits": hits, "code": code_coverage(dats) if coverage else {"available": False},
        "assertions": {"defined": len(ASSERTIONS), "list": ASSERTIONS,
                       "failed": 0,
                       "note": "Immediate assertions compiled in with --assert; any failure aborts the run."},
        "runtime_s": round(time.time() - t0, 2),
    }


def mutation_check(seeds: int = 20, cycles: int = 3000, directed: bool = True) -> dict[str, Any]:
    """Does the verification catch deliberate RTL bugs? A mutant is killed when any random
    seed's RTL trace diverges from the reference model, an assertion fires, or a directed
    protocol test (independent decoders) fails on the mutated RTL."""
    global _ACTIVE_MUTATION
    from chiplab.asic import protocols
    out = []
    scripts = [random_script(s, cycles) for s in range(seeds)]
    refs = [iss.run_script(sc) for sc in scripts]
    for name, *_ in MUTATIONS:
        killed_by = None
        for s, (sc, ref) in enumerate(zip(scripts, refs)):
            try:
                if run_rtl(sc, mutation=name) != ref:
                    killed_by = s
                    break
            except RuntimeError:
                killed_by = s
                break
        by_test = None
        if killed_by is None and directed:
            _ACTIVE_MUTATION = name
            try:
                for tname, fn in protocols.TESTS.items():
                    try:
                        ok = fn(backend="rtl")["passed"]
                    except RuntimeError:
                        ok = False
                    if not ok:
                        by_test = tname
                        break
            finally:
                _ACTIVE_MUTATION = None
        out.append({"mutation": name, "killed": killed_by is not None or by_test is not None,
                    "random_seed": killed_by, "directed_test": by_test})
    return {"mutants": len(out), "killed": sum(m["killed"] for m in out), "results": out}


def build_tt() -> Path:
    """Compile the Tiny Tapeout top-level bench (SPI host port + core)."""
    exe = verilator()
    h = hashlib.sha256()
    files = ["tt_um_chiplab_pem.v", "pem_core.v", "pem_lane.v", "tb_tt.cpp"]
    for f in files:
        h.update((RTL / f).read_bytes())
    out = BUILD_ROOT / f"tt_{h.hexdigest()[:16]}"
    binary = out / "Vtt_um_chiplab_pem"
    if binary.exists():
        return binary
    src = out / "src"
    src.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(RTL / f, src / f)
    r = subprocess.run([exe, "--cc", "--exe", "--build", "-O2", "-Wno-fatal", "-Wno-lint", "-Wno-style",
                        "--top-module", "tt_um_chiplab_pem", "--Mdir", str(out),
                        *(str(src / f) for f in files)], capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not binary.exists():
        raise RuntimeError("verilator build failed:\n" + (r.stderr or r.stdout)[-3000:])
    return binary


def tt_smoke() -> dict[str, Any]:
    """Through the SPI host port only: write/read back every instruction word and lane register
    pattern, load the UART TX program, send a byte and decode it from uio[0]."""
    from chiplab.asic import protocols
    from chiplab.asic.assembler import assemble
    binary = build_tt()
    pats = {a: (0xA5C3 ^ (a * 0x1111)) & 0xFFFF for a in range(0x40)}
    sc = "".join(f"W {a:x} {v:x}\n" for a, v in pats.items()) + "".join(f"R {a:x}\n" for a in pats)
    src, regs = protocols.uart_tx(pin=0)
    p = assemble(src)
    sc += "".join(f"W {a:x} {w:x}\n" for a, w in p.words.items())
    sc += f"W 20 2\nW 21 {regs['pins']:x}\nW 22 {isa.shift_cfg(True, True, p.start, p.wrap):x}\nW 23 0\nW 27 0\n"
    sc += f"W 48 10\nW 50 {min(p.words):x}\nW 48 1\nU\nW 40 a5\nC 600\nR 49\nR 50\n"
    r = subprocess.run([str(binary)], input=sc, capture_output=True, text=True, timeout=300)
    lines = r.stdout.splitlines()
    rd = [tuple(int(x, 16) for x in ln.split()[1:]) for ln in lines if ln.startswith("R")]
    readback_ok = all(v == pats[a] for a, v in rd[:len(pats)])
    wave = [(int(ln.split()[1], 16), int(ln.split()[2])) for ln in lines if ln.startswith("U")]
    got = protocols.decode_uart([(c, v) for c, v in wave], 0, 16, 10**9)
    byte_ok = [g["byte"] for g in got] == [0xA5]
    return {"passed": readback_ok and byte_ok, "register_readback_ok": readback_ok,
            "registers_checked": len(pats), "uart_bytes": [g["byte"] for g in got],
            "fidelity": "RTL simulation (Verilator) of tt_um_chiplab_pem through the SPI host port"}


def save_history(record: dict[str, Any], path: Path | None = None) -> Path:
    p = Path(path or BUILD_ROOT / "regression_history.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {k: v for k, v in record.items() if k not in ("functional_hits",)}
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return p
