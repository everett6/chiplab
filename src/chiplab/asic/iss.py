"""Cycle-accurate Python reference model of the PEM core (independent of the RTL).

It runs the same bench script as the Verilator testbench (see `run_script`) and prints the
same event lines, so RTL and model can be compared line by line:

    W addr data     bus write during the next clock
    R addr          bus read during the next clock  -> prints "R addr value"
    C n             n clocks, bus idle
    S value         drive the external input bits (bits 0-7 uio_in, 8-12 ui_in[7:3])
    LB src dst      environment: input dst follows output src (pull-up when not driven)
    I2C scl sda a   environment: I2C target at 7-bit address a on the two pins (open drain)

Output lines: "P cycle uio_out uio_oe uo_hi" when the pins change, "R addr value",
and I2C target events "TS", "TP", "TA addr rw", "TW byte", "TR byte ack".
Numbers are printed in hex without prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chiplab.asic import isa

M16 = 0xFFFF


def _bitrev16(v: int) -> int:
    return int(f"{v & M16:016b}"[::-1], 2)


def _mask(n: int) -> int:
    return (1 << n) - 1


@dataclass
class Lane:
    pc: int = 0
    delay: int = 0
    div: int = 0
    x: int = 0
    y: int = 0
    osr: int = 0
    osr_cnt: int = 16
    isr: int = 0
    isr_cnt: int = 0
    crc: int = 0
    snoop: int = 0
    nrzi: int = 0
    stuff: int = 0
    nrzi_in: int = 0
    destuff: int = 0
    diff: int = 0
    lvl_out: int = 1
    ones_out: int = 0
    lvl_in: int = 1
    ones_in: int = 0
    txf: list[int] = field(default_factory=list)
    rxf: list[int] = field(default_factory=list)


class PEM:
    def __init__(self) -> None:
        self.imem = [0] * isa.IMEM_WORDS
        self.cfg = [[0] * 8 for _ in range(isa.N_LANES)]
        self.lanes = [Lane() for _ in range(isa.N_LANES)]
        self.out_val = 0          # 14 bits: 0-7 uio, 8-13 uo_out[7:2]
        self.oe = 0               # 8 bits
        self.irq = 0
        self.enable = 0
        self.err = 0
        self.sync1 = 0
        self.sync2 = 0
        self.cycle = 0
        self.coverage: dict[str, int] = {}

    # ---------------------------------------------------------------- helpers
    def _cfgv(self, lane: int, name: str) -> int:
        return self.cfg[lane][isa.LANE_REGS.index(name)]

    def _cov(self, key: str) -> None:
        self.coverage[key] = self.coverage.get(key, 0) + 1

    def _crc_bit(self, i: int, L: Lane, b: int) -> None:
        cc = self._cfgv(i, "crc_cfg")
        w = (cc & 31) + 1
        poly = (self._cfgv(i, "crc_poly_hi") << 16 | self._cfgv(i, "crc_poly_lo")) & _mask(w)
        if cc >> 5 & 1:
            fb = b ^ (L.crc >> (w - 1) & 1)
            L.crc = (L.crc << 1) & _mask(w)
        else:
            fb = b ^ (L.crc & 1)
            L.crc = L.crc >> 1
        if fb:
            L.crc ^= poly

    def _restart(self, i: int) -> None:
        sh = self._cfgv(i, "shift")
        self.lanes[i] = Lane(pc=sh >> 2 & 31)

    def status(self) -> int:
        rxne = sum((len(L.rxf) > 0) << i for i, L in enumerate(self.lanes))
        txnf = sum((len(L.txf) < isa.FIFO_DEPTH) << i for i, L in enumerate(self.lanes))
        return self.irq << 8 | txnf << 4 | rxne

    def bus_read(self, addr: int) -> int:
        if addr < 0x20:
            return self.imem[addr]
        if addr < 0x40:
            return self.cfg[(addr - 0x20) >> 3][addr & 7]
        if 0x40 <= addr < 0x44:
            f = self.lanes[addr - 0x40].rxf
            return f[0] if f else 0
        if addr == isa.A_CTRL:
            return self.enable
        if addr == isa.A_STATUS:
            return self.status()
        if addr == isa.A_ERR:
            return self.err
        if 0x50 <= addr < 0x54:
            return self.lanes[addr - 0x50].pc
        if 0x54 <= addr < 0x58:
            return self.lanes[addr - 0x54].x
        if 0x58 <= addr < 0x5C:
            return self.lanes[addr - 0x58].y
        return 0

    # ---------------------------------------------------------------- one lane, one tick
    def _lane(self, i: int, pins: int, act: dict) -> None:
        L = self.lanes[i]
        if not (self.enable >> i & 1):
            return
        cd = max(self._cfgv(i, "clkdiv"), 1)
        tick = L.div == 0
        L.div = 0 if L.div + 1 >= cd else L.div + 1
        if not tick:
            return
        if L.delay:
            L.delay -= 1
            return
        pcfg, sh = self._cfgv(i, "pins"), self._cfgv(i, "shift")
        out_base, set_base, in_base, jmp_pin = pcfg & 15, pcfg >> 4 & 15, pcfg >> 8 & 15, pcfg >> 12 & 15
        out_right, in_right, start, wrap = sh & 1, sh >> 1 & 1, sh >> 2 & 31, sh >> 7 & 31
        autopull = sh >> 12 & 1
        op, delay, arg = isa.decode(self.imem[L.pc])
        nxt = start if L.pc == wrap else (L.pc + 1) & 31
        outcome = "done"
        pin = lambda k: pins >> (k & 15) & 1  # noqa: E731

        def wpin(k: int, v: int) -> None:
            k &= 15
            if k < 14:
                act["pin_we"] |= 1 << k
                act["pin_val"] = (act["pin_val"] & ~(1 << k)) | (v & 1) << k

        def wdir(k: int, v: int) -> None:
            k &= 15
            if k < 8:
                act["dir_we"] |= 1 << k
                act["dir_val"] = (act["dir_val"] & ~(1 << k)) | (v & 1) << k

        def emit(v: int) -> None:                    # serial line coder, output side
            if L.nrzi:
                L.lvl_out = L.lvl_out if v else L.lvl_out ^ 1
                v = L.lvl_out
            wpin(out_base, v)
            if L.diff:
                wpin(out_base + 1, v ^ 1)

        def shift_out(n: int) -> int:
            if out_right:
                d = L.osr & _mask(n)
                L.osr = 0 if n == 16 else L.osr >> n
            else:
                d = L.osr >> (16 - n) & _mask(n)
                L.osr = 0 if n == 16 else (L.osr << n) & M16
            L.osr_cnt = min(16, L.osr_cnt + n)
            return d

        def shift_in(d: int, n: int) -> None:
            if n == 16:
                L.isr = d & M16
            elif in_right:
                L.isr = (L.isr >> n) | (d & _mask(n)) << (16 - n)
            else:
                L.isr = ((L.isr << n) | (d & _mask(n))) & M16
            L.isr_cnt = min(16, L.isr_cnt + n)

        rot = ((pins >> in_base) | (pins << (16 - in_base))) & M16
        name = {v: k for k, v in isa.OPCODES.items()}.get(op, "nop")
        self._cov(f"op:{name}")
        if name == "set":
            n, v = (arg >> 5 & 3) + 1, arg & 15
            for k in range(n):
                (wdir if arg >> 7 else wpin)(set_base + k, v >> k & 1)
        elif name == "out":
            n, dest = (arg & 15) + 1, arg >> 5
            serial = dest == isa.OUT_DEST["pins"] and n == 1
            if not (serial and L.stuff and L.ones_out == 6) and autopull and L.osr_cnt == 16:
                if L.txf:
                    act["tx_pop"] = 1
                    L.osr, L.osr_cnt = L.txf[0], 0
                    self._cov("autopull:refill")
                else:
                    outcome = "stall"
                    self._cov("stall:autopull")
            if outcome == "stall":
                pass
            elif serial:
                if L.stuff and L.ones_out == 6:
                    emit(0)
                    L.ones_out = 0
                    outcome = "repeat"
                    self._cov("serial:stuffed_bit")
                else:
                    b = shift_out(1)
                    emit(b)
                    L.ones_out = (min(7, L.ones_out + 1)) if b else 0
                    if L.snoop:
                        self._crc_bit(i, L, b)
                    self._cov(f"serial:out nrzi={L.nrzi} stuff={L.stuff} diff={L.diff}")
            else:
                d = shift_out(n)
                if dest == 0:
                    for k in range(n):
                        wpin(out_base + k, d >> k & 1)
                elif dest == 1:
                    for k in range(n):
                        wdir(out_base + k, d >> k & 1)
                elif dest == 2:
                    L.x = d
                elif dest == 3:
                    L.y = d
                elif dest == 5:
                    nxt = d & 31
                elif dest == 6:
                    L.isr, L.isr_cnt = d, n
        elif name == "in":
            n, src = (arg & 15) + 1, arg >> 5
            if src == 0 and n == 1:
                r = pin(in_base)
                b = (1 if r == L.lvl_in else 0) if L.nrzi_in else r
                if L.nrzi_in:
                    L.lvl_in = r
                if L.destuff and L.ones_in == 6:
                    if b:
                        act["err"] = 1
                        self._cov("serial:stuff_error")
                    else:
                        self._cov("serial:destuffed_bit")
                    L.ones_in = 0
                    outcome = "repeat"
                else:
                    shift_in(b, 1)
                    L.ones_in = min(7, L.ones_in + 1) if b else 0
                    if L.snoop:
                        self._crc_bit(i, L, b)
                    self._cov(f"serial:in nrzi={L.nrzi_in} destuff={L.destuff}")
            else:
                d = {0: rot, 1: L.x, 2: L.y, 3: 0, 4: M16, 5: L.crc, 6: L.osr, 7: 0}[src]
                shift_in(d, n)
        elif name == "wait":
            pol, src, idx = arg >> 7, arg >> 5 & 3, arg & 15
            v = [pin(idx), pin(in_base + idx), self.irq >> (idx & 7) & 1, pin(jmp_pin)][src]
            if v != pol:
                outcome = "stall"
                self._cov("stall:wait")
        elif name == "jmp":
            cond, addr = arg >> 5, arg & 31
            take = [True, L.x == 0, L.x != 0, L.y == 0, L.y != 0, L.x != L.y,
                    bool(pin(jmp_pin)), L.osr_cnt < 16][cond]
            if cond == 2:
                L.x = (L.x - 1) & M16
            if cond == 4:
                L.y = (L.y - 1) & M16
            self._cov(f"jmp:{cond}:{int(take)}")
            if take:
                nxt = addr
        elif name == "pull":
            if L.txf:
                act["tx_pop"] = 1
                L.osr, L.osr_cnt = L.txf[0], 0
            elif arg & 1:
                outcome = "stall"
                self._cov("stall:pull")
            else:
                L.osr, L.osr_cnt = L.x, 0
        elif name == "push":
            if len(L.rxf) < isa.FIFO_DEPTH:
                act["rx_push"] = L.isr
                L.isr, L.isr_cnt = 0, 0
            elif arg & 1:
                outcome = "stall"
                self._cov("stall:push")
            else:
                L.isr, L.isr_cnt = 0, 0
        elif name == "mov":
            dst, mop, src = arg >> 5, arg >> 3 & 3, arg & 7
            st = len(L.rxf) | len(L.txf) << 2 | (self.err >> i & 1) << 4
            v = {0: rot, 1: L.x, 2: L.y, 3: L.isr, 4: L.osr, 5: L.crc, 6: 0, 7: st}[src]
            v = [v & M16, ~v & M16, _bitrev16(v), v >> 16 & M16][mop]
            if dst == 0:
                for k in range(8):
                    wpin(out_base + k, v >> k & 1)
            elif dst == 1:
                L.x = v
            elif dst == 2:
                L.y = v
            elif dst == 3:
                L.isr, L.isr_cnt = v, 0
            elif dst == 4:
                L.osr, L.osr_cnt = v, 0
            elif dst == 5:
                nxt = v & 31
            elif dst == 6:
                for k in range(8):
                    wdir(out_base + k, v >> k & 1)
        elif name == "irq":
            act["irq_clr" if arg >> 3 & 1 else "irq_set"] |= 1 << (arg & 7)
        elif name == "setx":
            L.x = arg
        elif name == "sety":
            L.y = arg
        elif name == "line":
            L.nrzi, L.stuff, L.nrzi_in, L.destuff, L.diff = (arg & 1, arg >> 1 & 1, arg >> 2 & 1,
                                                              arg >> 3 & 1, arg >> 5 & 1)
            if arg >> 4 & 1:
                L.ones_out = L.ones_in = 0
                L.lvl_out = L.lvl_in = 1
        elif name == "crc":
            if arg & 3 in (0, 3):
                w = (self._cfgv(i, "crc_cfg") & 31) + 1
                L.crc = (self._cfgv(i, "crc_init_hi") << 16 | self._cfgv(i, "crc_init_lo")) & _mask(w)
            if arg & 3 in (1, 3):
                L.snoop = 1
            if arg & 3 == 2:
                L.snoop = 0
        elif name == "halt":
            outcome = "stall"
        if outcome == "done":
            L.pc, L.delay = nxt, delay
        elif outcome == "repeat":
            L.delay = self._cfgv(i, "crc_cfg") >> 6 & 15

    # ---------------------------------------------------------------- one clock
    def clock(self, we: bool = False, re: bool = False, addr: int = 0, wdata: int = 0,
              raw_in: int = 0) -> int | None:
        pins = self.sync2
        rdata = self.bus_read(addr) if re else None
        acts = []
        for i in range(isa.N_LANES):
            act = {"pin_we": 0, "pin_val": 0, "dir_we": 0, "dir_val": 0, "irq_set": 0,
                   "irq_clr": 0, "tx_pop": 0, "rx_push": None, "err": 0}
            self._lane(i, pins, act)
            acts.append(act)
        # FIFOs: decisions use the pre-edge occupancy
        host_push = we and 0x40 <= addr < 0x44
        host_pop = re and 0x40 <= addr < 0x44
        for i, (L, a) in enumerate(zip(self.lanes, acts)):
            tx_full = len(L.txf) >= isa.FIFO_DEPTH
            rx_has = len(L.rxf) > 0
            if a["tx_pop"]:
                L.txf.pop(0)
            if host_push and addr - 0x40 == i and not tx_full:
                L.txf.append(wdata & M16)
            if host_pop and addr - 0x40 == i and rx_has:
                L.rxf.pop(0)
            if a["rx_push"] is not None:
                L.rxf.append(a["rx_push"])
        for a in acts:                                   # highest lane wins
            self.out_val = (self.out_val & ~a["pin_we"]) | (a["pin_val"] & a["pin_we"])
            self.oe = (self.oe & ~a["dir_we"]) | (a["dir_val"] & a["dir_we"])
        s = c = 0
        for i, a in enumerate(acts):
            s |= a["irq_set"]
            c |= a["irq_clr"]
            if a["err"]:
                self.err |= 1 << i
        self.irq = (self.irq & ~c | s) & 0xFF
        if we:
            if addr < 0x20:
                self.imem[addr] = wdata & M16
            elif addr < 0x40:
                self.cfg[(addr - 0x20) >> 3][addr & 7] = wdata & M16
            elif addr == isa.A_CTRL:
                self.enable = wdata & 15
                for i in range(isa.N_LANES):
                    if wdata >> (4 + i) & 1:
                        self._restart(i)
            elif addr == isa.A_ERR:
                self.err &= ~wdata & 15
            elif 0x50 <= addr < 0x54:
                self.lanes[addr - 0x50].pc, self.lanes[addr - 0x50].delay = wdata & 31, 0
        self.sync2, self.sync1 = self.sync1, raw_in & 0x1FFF
        self.cycle += 1
        return rdata

    def outputs(self) -> tuple[int, int, int]:
        return self.out_val & 0xFF, self.oe, (self.out_val >> 8 & 0x3F) << 2 | (self.irq >> 7 & 1) << 1


class I2CTarget:
    """Minimal I2C target used as a bench environment (mirrors tb_core.cpp exactly)."""

    def __init__(self, addr: int, log: list[str]) -> None:
        self.addr, self.log = addr, log
        self.state, self.bit, self.sh, self.rw, self.byte = "idle", 0, 0, 0, 0xA5
        self.pull, self.pscl, self.psda = 0, 1, 1

    def step(self, scl: int, sda: int) -> None:
        ps, pd = self.pscl, self.psda
        if scl and ps and pd and not sda:
            self.state, self.bit, self.sh, self.pull = "addr", 0, 0, 0
            self.log.append("TS")
        elif scl and ps and not pd and sda:
            self.state, self.pull = "idle", 0
            self.log.append("TP")
        elif not ps and scl:
            if self.state in ("addr", "wdata"):
                self.sh = (self.sh << 1 | sda) & 0xFF
                self.bit += 1
            elif self.state == "rack":
                self.log.append(f"TR {self.byte:x} {int(not sda)}")
                self.ack = not sda
        elif ps and not scl:
            st = self.state
            if st in ("addr", "wdata") and self.bit == 8:
                if st == "addr":
                    if self.sh >> 1 == self.addr:
                        self.rw = self.sh & 1
                        self.log.append(f"TA {self.addr:x} {self.rw}")
                        self.pull, self.state = 1, "acka"
                    else:
                        self.state = "idle"
                else:
                    self.log.append(f"TW {self.sh:x}")
                    self.pull, self.state = 1, "ackw"
            elif st == "acka":
                if self.rw:
                    self.state, self.bit = "rdata", 0
                    self.pull = int(not (self.byte >> 7 & 1))
                else:
                    self.pull, self.state, self.bit, self.sh = 0, "wdata", 0, 0
            elif st == "ackw":
                self.pull, self.state, self.bit, self.sh = 0, "wdata", 0, 0
            elif st == "rdata":
                self.bit += 1
                if self.bit < 8:
                    self.pull = int(not (self.byte >> (7 - self.bit) & 1))
                else:
                    self.pull, self.state = 0, "rack"
            elif st == "rack":
                if getattr(self, "ack", False):
                    self.byte = (self.byte + 1) & 0xFF
                    self.state, self.bit = "rdata", 0
                    self.pull = int(not (self.byte >> 7 & 1))
                else:
                    self.state, self.pull = "idle", 0
        self.pscl, self.psda = scl, sda


def run_script(script: str, model: PEM | None = None) -> list[str]:
    """Run a bench script on the reference model; returns the event lines."""
    m = model or PEM()
    log: list[str] = []
    ext, loops, i2c = 0, [], None
    last = None

    def env() -> int:
        uo, oe, _ = m.outputs()
        raw = ext
        for src, dst in loops:
            drv = (m.out_val >> src & 1) if (src >= 8 or oe >> src & 1) else 1
            raw = (raw & ~(1 << dst)) | drv << dst
        if i2c:
            t, scl_p, sda_p = i2c
            scl = 0 if (oe >> scl_p & 1 and not uo >> scl_p & 1) else 1
            drv_sda = oe >> sda_p & 1 and not uo >> sda_p & 1
            sda = 0 if (drv_sda or t.pull) else 1
            t.step(scl, sda)
            sda = 0 if (drv_sda or t.pull) else 1
            raw = (raw & ~(1 << scl_p | 1 << sda_p)) | scl << scl_p | sda << sda_p
        return raw

    raw = env()

    def tick(**kw) -> int | None:
        nonlocal raw, last
        r = m.clock(raw_in=raw, **kw)
        o = m.outputs()
        if o != last:
            log.append(f"P {m.cycle:x} {o[0]:x} {o[1]:x} {o[2]:x}")
            last = o
        raw = env()
        return r

    last = m.outputs()
    for line in script.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "W":
            tick(we=True, addr=int(t[1], 16), wdata=int(t[2], 16))
        elif t[0] == "R":
            a = int(t[1], 16)
            log.append(f"R {a:x} {tick(re=True, addr=a):x}")
        elif t[0] == "C":
            for _ in range(int(t[1])):
                tick()
        elif t[0] == "S":
            ext = int(t[1], 16)
            raw = env()
        elif t[0] == "LB":
            loops.append((int(t[1]), int(t[2])))
            raw = env()
        elif t[0] == "I2C":
            i2c = (I2CTarget(int(t[3], 16), log), int(t[1]), int(t[2]))
            raw = env()
        elif t[0] == "Q":
            break
    run_script.last_model = m  # type: ignore[attr-defined]
    return log
