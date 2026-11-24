"""Two-pass assembler for PEM-16.

Syntax, one instruction per line, `;` or `#` comments:

    .origin 4            ; place the next words at address 4
    .wrap_target         ; lane start address  (default: first word)
    loop:
        pull block
        set pins, 0 [3]   ; [n] = delay ticks after the instruction
        out pins, 1 [2]
        jmp !osre, loop
    .wrap                ; lane wraps from here back to .wrap_target

Returns words plus `start`/`wrap` addresses, ready for the lane `shift` register.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from chiplab.asic import isa


class AsmError(ValueError):
    pass


@dataclass
class Program:
    words: dict[int, int]
    start: int
    wrap: int
    labels: dict[str, int] = field(default_factory=dict)
    source: str = ""

    def image(self) -> list[int]:
        img = [0] * isa.IMEM_WORDS
        for a, w in self.words.items():
            img[a] = w
        return img


def _num(tok: str, labels: dict[str, int] | None = None) -> int:
    tok = tok.strip()
    if labels is not None and tok in labels:
        return labels[tok]
    try:
        return int(tok, 0)
    except ValueError as e:
        raise AsmError(f"expected a number or label, got {tok!r}") from e


def _encode(op: str, args: list[str], labels: dict[str, int]) -> int:
    a = [x.strip().lower() for x in args]
    if op in ("nop", "halt"):
        return 0
    if op == "set":
        if a[0] not in ("pins", "pindirs"):
            raise AsmError("set target must be pins or pindirs")
        v, n = _num(a[1]), _num(a[2]) if len(a) > 2 else 1
        if not 1 <= n <= 4 or not 0 <= v <= 15:
            raise AsmError("set: n must be 1-4, value 0-15")
        return (a[0] == "pindirs") << 7 | (n - 1) << 5 | v
    if op == "out":
        n = _num(a[1])
        if a[0] not in isa.OUT_DEST or not 1 <= n <= 16:
            raise AsmError(f"out: bad destination {a[0]!r} or count {n}")
        return isa.OUT_DEST[a[0]] << 5 | (n - 1)
    if op == "in":
        n = _num(a[1])
        if a[0] not in isa.IN_SRC or not 1 <= n <= 16:
            raise AsmError(f"in: bad source {a[0]!r} or count {n}")
        return isa.IN_SRC[a[0]] << 5 | (n - 1)
    if op == "wait":
        pol, src, idx = _num(a[0]), a[1], _num(a[2]) if len(a) > 2 else 0
        if pol not in (0, 1) or src not in isa.WAIT_SRC or not 0 <= idx <= 15:
            raise AsmError("wait: wait <0|1>, <gpio|pin|irq|jmppin>, <index>")
        return pol << 7 | isa.WAIT_SRC[src] << 5 | idx
    if op == "jmp":
        cond, tgt = ("always", a[0]) if len(a) == 1 else (a[0], a[1])
        if cond not in isa.JMP_COND:
            raise AsmError(f"jmp: unknown condition {cond!r}")
        addr = _num(tgt, labels)
        if not 0 <= addr < isa.IMEM_WORDS:
            raise AsmError("jmp: address out of range")
        return isa.JMP_COND[cond] << 5 | addr
    if op in ("pull", "push"):
        mode = a[0] if a and a[0] else "block"
        if mode not in ("block", "noblock"):
            raise AsmError(f"{op}: block or noblock")
        return int(mode == "block")
    if op == "mov":
        dst, src = a[0], a[1].replace(" ", "")
        mop = ""
        for p in ("hi", "::", "~"):
            if src.startswith(p):
                mop, src = p, src[len(p):]
                break
        if dst not in isa.MOV_DST or src not in isa.MOV_SRC:
            raise AsmError(f"mov: bad operands {a!r}")
        return isa.MOV_DST[dst] << 5 | isa.MOV_OP[mop] << 3 | isa.MOV_SRC[src]
    if op == "irq":
        mode, n = a[0], _num(a[1])
        if mode not in ("set", "clear") or not 0 <= n < isa.N_IRQ:
            raise AsmError("irq: irq <set|clear>, <0-7>")
        return (mode == "clear") << 3 | n
    if op in ("setx", "sety"):
        v = _num(a[0])
        if not 0 <= v <= 255:
            raise AsmError(f"{op}: immediate must be 0-255")
        return v
    if op == "line":
        bits = 0
        for f in a:
            if f and f not in isa.LINE_BITS:
                raise AsmError(f"line: unknown flag {f!r}")
            if f:
                bits |= 1 << isa.LINE_BITS[f]
        return bits
    if op == "crc":
        if a[0] not in isa.CRC_OPS:
            raise AsmError("crc: init, on, off or start")
        return isa.CRC_OPS[a[0]]
    raise AsmError(f"unknown instruction {op!r}")


_LINE = re.compile(r"^(?P<op>[a-z]+)\s*(?P<args>[^\[]*?)\s*(\[(?P<delay>\d+)\])?\s*$")


def assemble(src: str) -> Program:
    """Assemble PEM-16 source. Raises AsmError with the line number on any problem."""
    lines = []
    for n, raw in enumerate(src.splitlines(), 1):
        s = re.split(r"[;#]", raw, maxsplit=1)[0].strip()
        if s:
            lines.append((n, s))
    labels: dict[str, int] = {}
    addr, start, wrap, items = 0, None, None, []
    for n, s in lines:                                       # pass 1: addresses
        while ":" in s and re.match(r"^[A-Za-z_]\w*:", s):
            lab, s = s.split(":", 1)
            if lab in labels:
                raise AsmError(f"line {n}: duplicate label {lab!r}")
            labels[lab] = addr
            s = s.strip()
        if not s:
            continue
        if s.startswith("."):
            d = s.split()
            if d[0] == ".origin":
                addr = _num(d[1])
            elif d[0] == ".wrap_target":
                start = addr
            elif d[0] == ".wrap":
                wrap = addr - 1
            else:
                raise AsmError(f"line {n}: unknown directive {d[0]}")
            continue
        items.append((n, addr, s))
        addr += 1
    if not items:
        raise AsmError("empty program")
    words: dict[int, int] = {}
    for n, a, s in items:                                    # pass 2: encode
        if a >= isa.IMEM_WORDS:
            raise AsmError(f"line {n}: program exceeds {isa.IMEM_WORDS} words")
        m = _LINE.match(s.lower())
        if not m:
            raise AsmError(f"line {n}: cannot parse {s!r}")
        op = m["op"]
        args = [x for x in re.split(r",", m["args"])] if m["args"] else [""]
        delay = int(m["delay"] or 0)
        try:
            arg = _encode(op, args, labels)
            words[a] = isa.encode(op, arg, delay)
        except (AsmError, ValueError, IndexError) as e:
            raise AsmError(f"line {n}: {e}") from e
    first = min(words)
    return Program(words=words, start=first if start is None else start,
                   wrap=max(words) if wrap is None else wrap, labels=labels, source=src)


def disassemble(word: int) -> str:
    op, delay, arg = isa.decode(word)
    name = {v: k for k, v in isa.OPCODES.items()}.get(op, "nop")
    inv = lambda d: {v: k for k, v in d.items()}  # noqa: E731
    if name == "set":
        body = f"set {'pindirs' if arg >> 7 else 'pins'}, {arg & 15}, {(arg >> 5 & 3) + 1}"
    elif name == "out":
        body = f"out {inv(isa.OUT_DEST).get(arg >> 5, 'null')}, {(arg & 15) + 1}"
    elif name == "in":
        body = f"in {inv(isa.IN_SRC)[arg >> 5]}, {(arg & 15) + 1}"
    elif name == "wait":
        body = f"wait {arg >> 7}, {inv(isa.WAIT_SRC)[arg >> 5 & 3]}, {arg & 15}"
    elif name == "jmp":
        body = f"jmp {inv(isa.JMP_COND)[arg >> 5]}, {arg & 31}"
    elif name in ("pull", "push"):
        body = f"{name} {'block' if arg & 1 else 'noblock'}"
    elif name == "mov":
        body = f"mov {inv(isa.MOV_DST)[arg >> 5]}, {inv(isa.MOV_OP)[arg >> 3 & 3]}{inv(isa.MOV_SRC)[arg & 7]}"
    elif name == "irq":
        body = f"irq {'clear' if arg & 8 else 'set'}, {arg & 7}"
    elif name in ("setx", "sety"):
        body = f"{name} {arg}"
    elif name == "line":
        body = "line " + ", ".join(k for k, b in isa.LINE_BITS.items() if arg >> b & 1)
    elif name == "crc":
        body = f"crc {inv(isa.CRC_OPS)[arg & 3]}"
    else:
        body = name
    return body + (f" [{delay}]" if delay else "")
