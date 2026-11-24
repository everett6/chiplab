"""Protocol Laboratory for the PEM ASIC: protocol programs, bench scripts and independent
decoders.

Each test loads a PEM-16 program, drives it with host traffic and bench environments
(pin loopback, an I2C target model) and checks the pins with a decoder written separately
from both the RTL and the reference model (UART/SPI/USB decoders below sample the pin
waveform; the I2C target lives in the bench). Tests run on the RTL (Verilator) or on the
reference model; results say which.
"""

from __future__ import annotations

from typing import Any, Callable

from chiplab.asic import isa, iss
from chiplab.asic.assembler import Program, assemble

# ------------------------------------------------------------------ programs
def uart_tx(origin: int = 0, pin: int = 0) -> tuple[str, dict[str, int]]:
    """8N1 UART transmitter, LSB first, 8 lane ticks per bit."""
    src = f"""
    .origin {origin}
        set pindirs, 1
        set pins, 1
    .wrap_target
        pull block
        set pins, 0 [6]       ; start bit (7 + setx = 8 ticks)
        setx 7
    bit:
        out pins, 1 [6]
        jmp x--, bit
        set pins, 1 [6]       ; stop bit (+ pull)
    .wrap
    """
    return src, {"pins": isa.pins_cfg(out_base=pin, set_base=pin),
                 "shift": isa.shift_cfg(out_right=True)}


def uart_rx(origin: int = 0, pin: int = 0) -> tuple[str, dict[str, int]]:
    """8N1 UART receiver: waits for the start edge, samples mid-bit. Byte lands in ISR[15:8]."""
    src = f"""
    .origin {origin}
    .wrap_target
        wait 0, pin, 0
        setx 7 [10]           ; to the middle of data bit 0 (1.5 bits after the edge)
    bit:
        in pins, 1 [6]
        jmp x--, bit
        push block
        wait 1, pin, 0        ; stop bit
    .wrap
    """
    return src, {"pins": isa.pins_cfg(in_base=pin, jmp_pin=pin),
                 "shift": isa.shift_cfg(in_right=True)}


def spi_controller(mode: int, origin: int = 0, base: int = 0) -> tuple[str, dict[str, int]]:
    """SPI controller, modes 0-3, 8-bit MSB first. Pins: MOSI=base, SCK=base+1, CS_n=base+2,
    MISO=base+3. Host pushes byte<<8; the received byte comes back in the RX FIFO low byte."""
    cpol, cpha = mode >> 1 & 1, mode & 1
    lead, trail = cpol ^ 1, cpol
    if cpha == 0:
        body = f"""
        out pins, 1 [1]
        set pins, {lead}, 2 [1]
        in pins, 1
        set pins, {trail}, 2
        jmp x--, bit"""
    else:
        body = f"""
        set pins, {lead}, 2
        out pins, 1 [1]
        set pins, {trail}, 2
        in pins, 1
        jmp x--, bit"""
    src = f"""
    .origin {origin}
        set pindirs, 3, 2        ; SCK, CS_n outputs (only this lane's pins)
        mov osr, ~null
        out pindirs, 1           ; MOSI output
        set pins, {cpol | 2}, 2
    .wrap_target
        pull block
        set pins, {cpol}, 2       ; CS_n low
        setx 7
    bit:{body}
        push block
        set pins, {cpol | 2}, 2 [1] ; CS_n high
    .wrap
    """
    return src, {"pins": isa.pins_cfg(out_base=base, set_base=base + 1, in_base=base + 3),
                 "shift": isa.shift_cfg(out_right=False, in_right=False)}


def i2c_controller(origin: int = 0, sda: int = 4) -> tuple[str, dict[str, int]]:
    """I2C controller (open drain through pin directions; SCL = sda+1). One host word per
    byte: [15] START, [14:7] byte, [6] ack slot (1 = release), [5] STOP. The RX FIFO gets
    9 bits per byte: [8:1] byte seen on SDA, [0] ack (0 = acknowledged)."""
    src = f"""
    .origin {origin}
    .wrap_target
    top:
        pull block
        mov osr, ~osr
        out x, 1              ; x = not START
        jmp x--, bits
        set pindirs, 0 [1]    ; release SCL
        sety 1
        mov pindirs, y [1]    ; SDA low while SCL high: START
        set pindirs, 1        ; SCL low
    bits:
        setx 8
    bit:
        out pindirs, 1        ; SDA = data (driven low for 0)
        set pindirs, 0 [1]    ; SCL high
        in pins, 1
        set pindirs, 1 [1]    ; SCL low
        jmp x--, bit
        push noblock
        out x, 1              ; x = not STOP
        jmp x--, top
        sety 3
        mov pindirs, y [1]    ; SDA low, SCL low
        set pindirs, 0 [1]    ; SCL high
        sety 0
        mov pindirs, y [1]    ; release SDA: STOP
    .wrap
    """
    return src, {"pins": isa.pins_cfg(out_base=sda, set_base=sda + 1, in_base=sda),
                 "shift": isa.shift_cfg(out_right=False, in_right=False)}


USB_D = 3                     # out delay: bit time = USB_D + 2 ticks


def usb_ls_tx(origin: int = 0) -> tuple[str, dict[str, int]]:
    """Low-speed USB packet transmitter using the hardware line coder: NRZI, bit stuffing,
    differential drive (D- = pin 0, D+ = pin 1), CRC16 snoop and autopull, then EOP
    (SE0 for two bits, J). Host pushes: (data bits - 2), SYNC+PID word, data words (LSB
    first). Loop overhead is folded into the delays of the peeled last bit of each section,
    so every bit lasts exactly USB_D + 2 ticks; stuffed bits last 1 + stuff_pad ticks."""
    d = USB_D
    src = f"""
    .origin {origin}
        set pindirs, 3, 2
        set pins, 1, 2         ; J (low speed: D- high)
    .wrap_target
        out y, 16              ; header (autopull)
        line nrzi, stuff, diff, reset
        crc init
        setx 14
    p:  out pins, 1 [{d}]       ; SYNC + PID, bits 0-14
        jmp x--, p
        out pins, 1 [{d - 1}]   ; PID bit 15
        crc start
        mov x, y
    q:  out pins, 1 [{d}]       ; payload (autopull refills the OSR)
        jmp x--, q
        out pins, 1 [{d - 2}]   ; last payload bit
        mov osr, ~crc
        crc off
        setx 14
    c:  out pins, 1 [{d}]       ; CRC16, inverted, LSB first
        jmp x--, c
        out pins, 1 [{d}]
        line reset
        set pins, 0, 2 [{2 * d + 3}]   ; SE0 for two bit times
        set pins, 1, 2 [{d + 1}]       ; J
    .wrap
    """
    return src, {"pins": isa.pins_cfg(out_base=0, set_base=0),
                 "shift": isa.shift_cfg(out_right=True, autopull=True),
                 "crc_poly_lo": 0xA001, "crc_init_lo": 0xFFFF,
                 "crc_cfg": isa.crc_cfg(16, False, stuff_pad=d + 1)}


# ------------------------------------------------------------------ running
def run(script: str, backend: str = "rtl") -> list[str]:
    if backend == "rtl":
        from chiplab.asic import verify
        return verify.run_rtl(script)
    if backend == "gate":                              # Yosys netlist under Verilator
        from chiplab.asic import eda
        return eda.run_gate(script)
    return iss.run_script(script)


def setup(progs: list[tuple[int, str, dict[str, int]]], clkdiv: int = 1) -> tuple[str, list[Program]]:
    from chiplab.asic.verify import load_script
    items, assembled = [], []
    for lane, src, regs in progs:
        p = assemble(src)
        assembled.append(p)
        items.append((lane, p, {"clkdiv": clkdiv, **regs}))
    return load_script(items, sum(1 << lane for lane, _, _ in progs)), assembled


def waveform(lines: list[str], field: int = 0) -> list[tuple[int, int]]:
    """(cycle, value) change list from P events. field 0 = uio_out, 1 = uio_oe, 2 = uo_out."""
    out = [(0, 0)]
    for ln in lines:
        t = ln.split()
        if t[0] == "P":
            out.append((int(t[1], 16), int(t[2 + field], 16)))
    return out


def sample(wave: list[tuple[int, int]], cycle: float, bit: int) -> int:
    v = 0
    for c, val in wave:
        if c > cycle:
            break
        v = val
    return v >> bit & 1


def reads(lines: list[str]) -> list[tuple[int, int]]:
    return [(int(t[1], 16), int(t[2], 16)) for t in (ln.split() for ln in lines) if t[0] == "R"]


# ------------------------------------------------------------------ independent decoders
def decode_uart(wave, pin: int, period: float, end: int) -> list[dict[str, int]]:
    """Software UART decoder (8N1): finds start edges, samples mid-bit."""
    out, c = [], 0
    edges = [cy for (cy, v), (_, pv) in zip(wave[1:], wave[:-1]) if not (v >> pin & 1) and (pv >> pin & 1)]
    last_end = -1
    for e in edges:
        if e <= last_end:
            continue
        if sample(wave, e + period / 2, pin) != 0:
            continue
        byte = sum(sample(wave, e + period * (k + 1.5), pin) << k for k in range(8))
        stop = sample(wave, e + period * 9.5, pin)
        if e + period * 10 > end:
            break
        out.append({"byte": byte, "stop_ok": stop, "start": e})
        last_end = int(e + period * 9.5)
        c += 1
    return out


def decode_spi(lines: list[str], mode: int, mosi: int, sck: int, cs: int) -> list[int]:
    """Samples MOSI on the mode's sampling edge while CS_n is low (MSB first)."""
    cpol, cpha = mode >> 1 & 1, mode & 1
    sample_on_rise = (cpol == 0) == (cpha == 0)
    wave = waveform(lines)
    out, bits, val, prev = [], 0, 0, None
    for _, v in wave:
        s, c, d = v >> sck & 1, v >> cs & 1, v >> mosi & 1
        if prev is not None:
            ps, pc = prev
            if c:
                bits, val = 0, 0
            elif ps != s and (s == 1) == sample_on_rise:
                val = (val << 1) | d
                bits += 1
                if bits == 8:
                    out.append(val)
                    bits, val = 0, 0
        prev = (s, c)
    return out


def crc16_usb(data: bytes) -> int:
    """CRC-16/USB, table-free MSB-first formulation with explicit reflection
    (a different algorithm from the lane's reflected bit-serial hardware)."""
    def refl(v: int, w: int) -> int:
        return int(f"{v:0{w}b}"[::-1], 2)
    crc = 0xFFFF
    for byte in data:
        crc ^= refl(byte, 8) << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return refl(crc, 16) ^ 0xFFFF


def decode_usb_ls(lines: list[str], period: float, dminus: int = 0, dplus: int = 1) -> dict[str, Any]:
    """Low-speed USB packet decoder: line states -> NRZI -> destuff -> bytes, then checks
    SYNC, PID and CRC16."""
    wave = waveform(lines)
    start = next((c for (c, v), (_, pv) in zip(wave[1:], wave[:-1])
                  if (pv >> dminus & 1, pv >> dplus & 1) == (1, 0) and (v >> dminus & 1, v >> dplus & 1) == (0, 1)), None)
    if start is None:
        return {"ok": False, "error": "no packet"}
    prev, ones, bits, k, stuff_err = "J", 0, [], 0, False
    while k < 2000:
        t = start + period * (k + 0.5)
        dm, dp = sample(wave, t, dminus), sample(wave, t, dplus)
        st = "SE0" if (dm, dp) == (0, 0) else "J" if (dm, dp) == (1, 0) else "K" if (dm, dp) == (0, 1) else "SE1"
        if st in ("SE0", "SE1"):
            break
        b = 1 if st == prev else 0
        prev = st
        k += 1
        if ones == 6:
            if b:
                stuff_err = True
            ones = 0
            continue
        ones = ones + 1 if b else 0
        bits.append(b)
    nbytes = len(bits) // 8
    data = bytes(sum(bits[i * 8 + j] << j for j in range(8)) for i in range(nbytes))
    res: dict[str, Any] = {"bytes": data.hex(), "bit_errors": len(bits) % 8, "stuff_error": stuff_err}
    if nbytes < 4:
        return {**res, "ok": False, "error": "packet too short"}
    sync, pid, payload, crc = data[0], data[1], data[2:-2], data[-2] | data[-1] << 8
    res.update(sync_ok=sync == 0x80, pid=pid, pid_ok=(pid & 15) == (~pid >> 4 & 15),
               payload=payload.hex(), crc_rx=crc, crc_ref=crc16_usb(payload))
    res["ok"] = res["sync_ok"] and res["pid_ok"] and crc == res["crc_ref"] and not stuff_err and not res["bit_errors"]
    return res


# ------------------------------------------------------------------ tests
def _result(proto: str, case: str, ok: bool, backend: str, **details: Any) -> dict[str, Any]:
    return {"protocol": proto, "case": case, "passed": bool(ok), "backend": backend,
            "fidelity": "RTL simulation (Verilator)" if backend == "rtl" else "reference model",
            **details}


def test_uart_tx(backend: str = "rtl", data: bytes = b"\x55\xA3\x00\xFF", clkdiv: int = 2):
    src, regs = uart_tx(pin=0)
    sc, _ = setup([(0, src, regs)], clkdiv)
    for b in data:
        sc += f"W 40 {b:x}\nC {12 * 8 * clkdiv}\n"
    sc += "C 400\n"
    lines = run(sc, backend)
    end = int(waveform(lines)[-1][0]) + 400
    got = decode_uart(waveform(lines), 0, 8 * clkdiv, end + 10**6)
    ok = [g["byte"] for g in got] == list(data) and all(g["stop_ok"] for g in got)
    return _result("UART", "tx 8N1", ok, backend, sent=data.hex(), decoded=[g["byte"] for g in got])


def _uart_frame(byte: int, pin: int, period: int) -> str:
    """Bench stimulus for one 8N1 frame on an input pin (driven by the bench, not the DUT)."""
    s, bits = "", [0] + [byte >> k & 1 for k in range(8)] + [1]
    for b in bits:
        s += f"S {(b << pin):x}\nC {period}\n"
    return s


def test_uart_rx(backend: str = "rtl", data: bytes = b"\x5A\x01\xC7", clkdiv: int = 2):
    src, regs = uart_rx(pin=2)
    sc, _ = setup([(0, src, regs)], clkdiv)
    sc = f"S {1 << 2:x}\n" + sc + "C 50\n"   # line idles high before the lane starts
    for b in data:
        sc += _uart_frame(b, 2, 8 * clkdiv) + "C 20\nR 40\n"
    got = [v >> 8 for a, v in reads(run(sc, backend)) if a == 0x40]
    return _result("UART", "rx 8N1", got == list(data), backend, sent=data.hex(), received=got)


def test_spi(mode: int, backend: str = "rtl", data: bytes = b"\x96\x3C\xF0", clkdiv: int = 1):
    src, regs = spi_controller(mode, base=0)
    sc, _ = setup([(0, src, regs)], clkdiv)
    sc += "LB 0 3\n"
    for b in data:
        sc += f"W 40 {b << 8:x}\nC {80 * clkdiv}\nR 40\n"
    lines = run(sc, backend)
    wire = decode_spi(lines, mode, 0, 1, 2)
    back = [v & 0xFF for a, v in reads(lines) if a == 0x40]
    ok = wire == list(data) and back == list(data)
    return _result("SPI", f"mode {mode} controller, MOSI->MISO loopback", ok, backend,
                   sent=data.hex(), on_wire=wire, received=back)


def _i2c_word(byte: int, start: bool = False, stop: bool = False, ack_slot: int = 1) -> int:
    return int(start) << 15 | (byte & 0xFF) << 7 | (ack_slot & 1) << 6 | int(stop) << 5


def test_i2c(backend: str = "rtl", addr: int = 0x42, payload: bytes = b"\x10\xEE", clkdiv: int = 2):
    src, regs = i2c_controller(sda=4)
    sc, _ = setup([(0, src, regs)], clkdiv)
    sc += f"I2C 5 4 {addr:x}\n"
    words = [_i2c_word(addr << 1, start=True)] + [_i2c_word(b, stop=(i == len(payload) - 1))
                                                  for i, b in enumerate(payload)]
    words += [_i2c_word(addr << 1 | 1, start=True), _i2c_word(0xFF, ack_slot=0),
              _i2c_word(0xFF, ack_slot=1, stop=True)]
    for w in words:
        sc += f"W 40 {w:x}\nC {110 * clkdiv}\nR 40\n"
    sc += f"C {40 * clkdiv}\n"
    lines = run(sc, backend)
    target = [ln for ln in lines if ln[0] == "T"]
    rx = [v for a, v in reads(lines) if a == 0x40]
    acks = [v & 1 for v in rx]
    expect_t = (["TS", f"TA {addr:x} 0"] + [f"TW {b:x}" for b in payload] + ["TP", "TS", f"TA {addr:x} 1",
                "TR a5 1", "TR a6 0", "TP"])
    read_bytes = [v >> 1 & 0xFF for v in rx[-2:]]
    ok = target == expect_t and acks[: 1 + len(payload) + 1] == [0] * (len(payload) + 2) and read_bytes == [0xA5, 0xA6]
    return _result("I2C", "controller write + repeated-start read", ok, backend,
                   target_log=target, controller_rx=[hex(v) for v in rx], read=read_bytes)


def test_i2c_nack(backend: str = "rtl", clkdiv: int = 2):
    """Error condition: address with no target must read back NACK (ack bit 1)."""
    src, regs = i2c_controller(sda=4)
    sc, _ = setup([(0, src, regs)], clkdiv)
    sc += "I2C 5 4 42\n" + f"W 40 {_i2c_word(0x21 << 1, start=True, stop=True):x}\nC {110 * clkdiv}\nR 40\n"
    lines = run(sc, backend)
    rx = [v for a, v in reads(lines) if a == 0x40]
    ok = rx and rx[0] & 1 == 1 and not any(ln.startswith("TA") for ln in lines)
    return _result("I2C", "NACK on absent address (error condition)", ok, backend, controller_rx=[hex(v) for v in rx])


def test_usb_ls_tx(backend: str = "rtl", pid: int = 0xC3, payload: bytes = b"\xFF\xFF\x01\x80", clkdiv: int = 2):
    """USB low-speed DATA0 packet with a payload that forces bit stuffing."""
    src, regs = usb_ls_tx()
    sc, _ = setup([(0, src, regs)], clkdiv)
    words = [payload[i] | payload[i + 1] << 8 for i in range(0, len(payload), 2)]
    sc += f"C 20\nW 40 {8 * len(payload) - 2:x}\nW 40 {0x80 | pid << 8:x}\n"
    for w in words:
        sc += f"C {14 * (USB_D + 2) * clkdiv}\nW 40 {w:x}\n"
    sc += f"C {80 * (USB_D + 2) * clkdiv}\n"
    lines = run(sc, backend)
    dec = decode_usb_ls(lines, (USB_D + 2) * clkdiv)
    ok = dec.get("ok") and dec.get("payload") == payload.hex() and dec.get("pid") == pid
    return _result("USB-LS", "DATA0 packet: NRZI + bit stuffing + CRC16 + EOP", bool(ok), backend, decoded=dec)


def test_concurrent(backend: str = "rtl", clkdiv: int = 2):
    """Three lanes at once: UART TX (pin 5) looped to UART RX (pin 6) and an SPI mode-0
    controller (pins 0-3), sharing the instruction memory."""
    tx_src, tx_regs = uart_tx(origin=0, pin=5)
    rx_src, rx_regs = uart_rx(origin=9, pin=6)
    spi_src, spi_regs = spi_controller(0, origin=15, base=0)   # 14 words: 15-28
    sc, progs = setup([(0, tx_src, tx_regs), (1, rx_src, rx_regs), (2, spi_src, spi_regs)], clkdiv)
    sc = "LB 5 6\nLB 0 3\n" + sc       # environment wired before the lanes start
    data = b"\x3E\xC1"
    for b in data:
        sc += f"W 40 {b:x}\nW 42 {b << 8:x}\nC {12 * 8 * clkdiv}\nR 41\nR 42\n"
    lines = run(sc, backend)
    r = reads(lines)
    uart = [v >> 8 for a, v in r if a == 0x41]
    spi = [v & 0xFF for a, v in r if a == 0x42]
    words = sum(len(p.words) for p in progs)
    ok = uart == list(data) and spi == list(data)
    return _result("Concurrent", "UART TX->RX + SPI on three lanes", ok, backend,
                   uart_rx=uart, spi_rx=spi, imem_words_used=words)


def test_irq_sync(backend: str = "rtl", clkdiv: int = 1):
    """Synchronization: lane 1 raises IRQ flag 0 every 32 ticks; lane 0 waits on it, clears
    it and emits one pulse. Pulse count and spacing must follow lane 1 exactly."""
    a = """
    .origin 0
        set pindirs, 1
    .wrap_target
        wait 1, irq, 0
        irq clear, 0
        set pins, 1 [3]
        set pins, 0
    .wrap
    """
    b = """
    .origin 6
    .wrap_target
        irq set, 0 [15]
        nop [15]
    .wrap
    """
    sc, _ = setup([(0, a, {"pins": isa.pins_cfg(set_base=7)}), (1, b, {})], clkdiv)
    sc += f"C {32 * 10 * clkdiv + 20}\n"
    lines = run(sc, backend)
    rises = [c for (c, v), (_, pv) in zip(waveform(lines)[1:], waveform(lines)[:-1]) if v >> 7 & 1 and not pv >> 7 & 1]
    gaps = sorted({b - a for a, b in zip(rises, rises[1:])})
    ok = len(rises) in (10, 11) and gaps == [32 * clkdiv]
    return _result("Sync", "IRQ handshake between lanes", ok, backend, pulses=len(rises), spacing=gaps)


PLANNED = [
    {"protocol": "JTAG", "status": "not yet written", "note": "TAP driver fits the SPI-style shifter (TMS via set pins)"},
    {"protocol": "SWD", "status": "not yet written", "note": "needs turnaround: out pindirs, 1 between phases"},
    {"protocol": "PS/2", "status": "not yet written", "note": "device-clocked: wait on clock pin, in pins, 1, odd parity in software"},
    {"protocol": "CAN", "status": "not yet written", "note": "CRC-15 MSB-first and stuffing exist in hardware; arbitration needs wait-and-compare"},
    {"protocol": "10BASE-T Ethernet", "status": "not supported in hardware",
     "note": "Manchester coding is not in the line coder; software Manchester cannot reach 10 Mb/s at plausible clocks. CRC-32 is supported."},
    {"protocol": "USB-LS receive", "status": "datapath exists, no directed test yet", "note": "nrzi_in + destuff + stuff-error flag are covered by random lockstep only"},
]

TESTS: dict[str, Callable[..., dict[str, Any]]] = {
    "uart_tx": test_uart_tx, "uart_rx": test_uart_rx,
    "spi_mode0": lambda backend="rtl": test_spi(0, backend), "spi_mode1": lambda backend="rtl": test_spi(1, backend),
    "spi_mode2": lambda backend="rtl": test_spi(2, backend), "spi_mode3": lambda backend="rtl": test_spi(3, backend),
    "i2c": test_i2c, "i2c_nack": test_i2c_nack, "irq_sync": test_irq_sync, "usb_ls_tx": test_usb_ls_tx, "concurrent": test_concurrent,
}


def run_all(backend: str = "rtl") -> dict[str, Any]:
    results = []
    for name, fn in TESTS.items():
        try:
            results.append({"name": name, **fn(backend=backend)})
        except Exception as e:  # report, never hide
            results.append({"name": name, "passed": False, "backend": backend, "error": f"{type(e).__name__}: {e}"})
    return {"backend": backend, "passed": sum(r["passed"] for r in results), "total": len(results),
            "results": results, "planned": PLANNED}
