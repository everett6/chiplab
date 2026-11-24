"""PEM-16: the instruction set of ChipLab's Protocol Emulation Machine (competition ASIC).

PEM is a small reprogrammable pin processor for the Tiny Tapeout IHP 130 nm shuttle: four
lanes (state machines) share one 32-word instruction memory, eight bidirectional pins plus
five extra inputs and six extra outputs, eight IRQ/sync flags and a host port. What sets it
apart from a plain GPIO sequencer is the per-lane serial datapath in hardware: NRZI line coding
(optionally differential), USB-style bit stuffing / destuffing with error flags, and a CRC
engine with configurable width (1-32), polynomial, seed and bit order that can "snoop" every
serial data bit. Those are the parts of USB, CAN and Ethernet framing that are hard to do
in software at line rate.

Encoding (16 bits):  [15:12] opcode  [11:8] delay (extra lane ticks after completion)  [7:0] arg

This module is the single source of truth for the encoding: the assembler, the Python
reference model (iss.py), the RTL constants and the generated ISA document all use it.
"""

from __future__ import annotations

N_LANES = 4
IMEM_WORDS = 32
FIFO_DEPTH = 2
N_IRQ = 8
WORD = 16

OPCODES = {
    "nop": 0, "set": 1, "out": 2, "in": 3, "wait": 4, "jmp": 5, "pull": 6, "push": 7,
    "mov": 8, "irq": 9, "setx": 10, "sety": 11, "line": 12, "crc": 13, "halt": 14,
}
OUT_DEST = {"pins": 0, "pindirs": 1, "x": 2, "y": 3, "null": 4, "pc": 5, "isr": 6}
IN_SRC = {"pins": 0, "x": 1, "y": 2, "zeros": 3, "ones": 4, "crc": 5, "osr": 6, "null": 7}
WAIT_SRC = {"gpio": 0, "pin": 1, "irq": 2, "jmppin": 3}
JMP_COND = {"always": 0, "!x": 1, "x--": 2, "!y": 3, "y--": 4, "x!=y": 5, "pin": 6, "!osre": 7}
MOV_DST = {"pins": 0, "x": 1, "y": 2, "isr": 3, "osr": 4, "pc": 5, "pindirs": 6, "null": 7}
MOV_SRC = {"pins": 0, "x": 1, "y": 2, "isr": 3, "osr": 4, "crc": 5, "null": 6, "status": 7}
MOV_OP = {"": 0, "~": 1, "::": 2, "hi": 3}          # none, invert, bit-reverse, upper 16 bits
LINE_BITS = {"nrzi": 0, "stuff": 1, "nrzi_in": 2, "destuff": 3, "reset": 4, "diff": 5}
CRC_OPS = {"init": 0, "on": 1, "off": 2, "start": 3}  # start = init + snoop on

# Host / bus address map (shared by the RTL core bus, the SPI host port and the model)
A_IMEM = 0x00            # 0x00-0x1F instruction words
A_LANE = 0x20            # 0x20 + lane*8 + reg
LANE_REGS = ("clkdiv", "pins", "shift", "crc_poly_lo", "crc_poly_hi", "crc_init_lo",
             "crc_init_hi", "crc_cfg")
A_FIFO = 0x40            # 0x40 + lane: write = push TX FIFO, read = pop RX FIFO
A_CTRL = 0x48            # write {restart[7:4], enable[3:0]}; read returns enable
A_STATUS = 0x49          # read {irq[15:8], tx_not_full[7:4], rx_not_empty[3:0]}
A_ERR = 0x4A             # read sticky {stuff_err[3:0]}; write 1s to clear
A_PC = 0x50              # 0x50 + lane: read PC; write = jump (entry point / debug)
A_DBG_X = 0x54           # 0x54 + lane: read X (debug)
A_DBG_Y = 0x58           # 0x58 + lane: read Y (debug)


def lane_reg(lane: int, name: str) -> int:
    return A_LANE + lane * 8 + LANE_REGS.index(name)


def pins_cfg(out_base: int = 0, set_base: int = 0, in_base: int = 0, jmp_pin: int = 0) -> int:
    return (out_base & 15) | (set_base & 15) << 4 | (in_base & 15) << 8 | (jmp_pin & 15) << 12


def shift_cfg(out_right: bool = True, in_right: bool = True, start: int = 0, wrap: int = 31,
              autopull: bool = False) -> int:
    """shift register: out_right [0], in_right [1], start [6:2], wrap [11:7], autopull [12]
    (an OUT with an empty OSR refills it from the TX FIFO in the same tick, or stalls)."""
    return (int(out_right) | int(in_right) << 1 | (start & 31) << 2 | (wrap & 31) << 7
            | int(autopull) << 12)


def crc_cfg(width: int = 16, msb_first: bool = False, stuff_pad: int = 0) -> int:
    """crc_cfg register: width-1 [4:0], MSB-first [5], stuff_pad [9:6]: a stuffed/destuffed bit
    lasts 1 + stuff_pad ticks (set it to bit time - 1, independent of the loop's instructions)."""
    return ((width - 1) & 31) | int(msb_first) << 5 | (stuff_pad & 15) << 6


def encode(op: str, arg: int = 0, delay: int = 0) -> int:
    if op not in OPCODES:
        raise ValueError(f"unknown opcode {op!r}")
    if not 0 <= delay <= 15:
        raise ValueError("delay must be 0-15")
    if not 0 <= arg <= 255:
        raise ValueError("argument must fit 8 bits")
    return OPCODES[op] << 12 | delay << 8 | arg


def decode(word: int) -> tuple[int, int, int]:
    return word >> 12 & 15, word >> 8 & 15, word & 255


ISA_DOC = """# PEM-16 instruction set

Each instruction is 16 bits: `[15:12]` opcode, `[11:8]` delay, `[7:0]` argument. An
instruction runs on a lane tick (the lane clock divider), then the lane waits `delay` more
ticks. A *stall* repeats the instruction on the next tick with no delay. A *repeat* (used by
bit stuffing) applies the delay but does not advance the PC, so an inserted bit lasts exactly
one bit time: the repeat waits `stuff_pad` ticks (lane register `crc_cfg[9:6]`), so a
stuffed bit lasts `1 + stuff_pad` ticks whatever instruction emitted it. With *autopull*
(`shift[12]`) an OUT that finds the OSR empty refills it from the TX FIFO in the same tick
(or stalls), so a bit stream crosses 16-bit word boundaries with no gap.

| Op | Mnemonic | Argument | Effect |
|---|---|---|---|
| 0 | `nop` | - | nothing |
| 1 | `set pins|pindirs, v [, n]` | `[7]` dirs, `[6:5]` n-1, `[3:0]` v | drive n (1-4) pins from `set_base` |
| 2 | `out dest, n` | `[7:5]` dest, `[3:0]` n-1 | shift n bits out of OSR to pins, pindirs, x, y, null, pc, isr |
| 3 | `in src, n` | `[7:5]` src, `[3:0]` n-1 | shift n bits into ISR from pins, x, y, zeros, ones, crc, osr, null |
| 4 | `wait pol, gpio|pin|irq|jmppin, idx` | `[7]` pol, `[6:5]` src, `[3:0]` idx | stall until the source equals pol |
| 5 | `jmp [cond,] addr` | `[7:5]` cond, `[4:0]` addr | always, !x, x--, !y, y--, x!=y, pin, !osre |
| 6 | `pull [block|noblock]` | `[0]` block | OSR <- TX FIFO (empty: stall, or OSR <- x) |
| 7 | `push [block|noblock]` | `[0]` block | RX FIFO <- ISR, ISR cleared (full: stall, or drop) |
| 8 | `mov dst, [op]src` | `[7:5]` dst, `[4:3]` op, `[2:0]` src | op: none, `~` invert, `::` reverse, `hi` upper half |
| 9 | `irq set|clear, n` | `[3]` clear, `[2:0]` flag | shared flags for lane sync; flag 7 drives the host IRQ pin |
| 10 | `setx imm` | imm8 | x <- imm |
| 11 | `sety imm` | imm8 | y <- imm |
| 12 | `line flags...` | bits: nrzi, stuff, nrzi_in, destuff, reset, diff | configure the serial line coder |
| 13 | `crc init|on|off|start` | `[1:0]` | CRC seed / snoop control |
| 14 | `halt` | - | stall forever (until restart) |

**Serial datapath.** For `out pins, 1` and `in pins, 1` the lane line coder applies:
bit stuffing (a 0 after six 1s, inserted as a repeat), NRZI (level toggles on 0), and
differential drive (`diff`: pin `out_base+1` carries the complement). On input it undoes NRZI
and removes stuffed bits, setting a sticky stuff-error flag when the bit after six 1s is not 0.
With CRC snoop on, every data bit (pre-stuffing on output, post-destuffing on input) updates
the lane CRC: width 1-32, polynomial and seed from the lane registers, MSB-first or
LSB-first (reflected) order.

**Pins.** Pin space 0-15: 0-7 are the bidirectional `uio` pins, 8-13 output-only `uo_out[7:2]`
(inputs 8-12 read `ui_in[7:3]`). Inputs pass a two-flop synchroniser. When several lanes
write the same pin in one cycle, the highest lane wins.

**Host port.** SPI mode 0 on `ui_in[0]` SCK, `ui_in[1]` CS_n, `ui_in[2]` MOSI, `uo_out[0]` MISO.
First byte `{rw, addr[6:0]}`; writes send 16 data bits, reads return 16 bits. `uo_out[1]` is
IRQ flag 7. Address map: 0x00-0x1F instructions, 0x20+8*lane+reg lane registers
(clkdiv, pins, shift, crc_poly lo/hi, crc_init lo/hi, crc_cfg), 0x40+lane FIFO,
0x48 control, 0x49 status, 0x4A stuff errors, 0x50+lane PC (write to jump, e.g. to a program's entry before its wrap target), 0x54+lane X, 0x58+lane Y (debug).
"""
