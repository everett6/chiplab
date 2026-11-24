# PEM-16 instruction set

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
