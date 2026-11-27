"""Chipro-inspired conversational EDA workflow tools for ChipLab."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from chiplab.models.materials.process_nodes import get_process_node
from chiplab.schemas.common import Provenance


def _sanitize_identifier(name: str, default: str = "top") -> str:
    """Return a safe Verilog identifier for filenames and CLI arguments."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").strip())
    if not cleaned or cleaned[0].isdigit():
        return default
    return cleaned


def _strip_verilog_comments_preserve_lines(code: str) -> str:
    """Replace // and /* ... */ comments with spaces while preserving newlines."""
    out: list[str] = []
    i = 0
    n = len(code)
    in_string = False
    while i < n:
        ch = code[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(code[i + 1])
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and code[i + 1] == "/":
            while i < n and code[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch == "/" and i + 1 < n and code[i + 1] == "*":
            out.append("  ")
            i += 2
            while i < n:
                if code[i] == "*" and i + 1 < n and code[i + 1] == "/":
                    out.append("  ")
                    i += 2
                    break
                out.append("\n" if code[i] == "\n" else " ")
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _line_number_at(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _extract_always_blocks(clean_code: str) -> list[dict[str, Any]]:
    """Extract `always ...` blocks with their sensitivity header, body, and start line."""
    blocks: list[dict[str, Any]] = []
    pattern = re.compile(r"\b(always_comb|always_ff|always)\b")
    n = len(clean_code)

    for match in pattern.finditer(clean_code):
        kw = match.group(1)
        pos = match.end()
        # Parse optional @( ... ) sensitivity list
        header = ""
        while pos < n and clean_code[pos].isspace():
            pos += 1
        if pos < n and clean_code[pos] == "@":
            pos += 1
            while pos < n and clean_code[pos].isspace():
                pos += 1
            if pos < n and clean_code[pos] == "(":
                depth = 1
                h_start = pos + 1
                pos += 1
                while pos < n and depth > 0:
                    if clean_code[pos] == "(":
                        depth += 1
                    elif clean_code[pos] == ")":
                        depth -= 1
                    pos += 1
                header = clean_code[h_start : pos - 1]
            else:
                # e.g. always @*
                h_start = pos
                while pos < n and not clean_code[pos].isspace():
                    pos += 1
                header = clean_code[h_start:pos]

        while pos < n and clean_code[pos].isspace():
            pos += 1

        # Extract begin ... end block or single statement
        body_start = pos
        if clean_code[pos : pos + 5] == "begin" and (
            pos + 5 == n or not (clean_code[pos + 5].isalnum() or clean_code[pos + 5] == "_")
        ):
            depth = 1
            pos += 5
            # Skip optional `: block_name`
            tok_iter = re.finditer(r"\b(begin|end)\b", clean_code[pos:])
            body_end = n
            for tok in tok_iter:
                if tok.group(1) == "begin":
                    depth += 1
                else:
                    depth -= 1
                    if depth == 0:
                        body_end = pos + tok.end()
                        break
            body = clean_code[body_start:body_end]
        else:
            semi = clean_code.find(";", pos)
            body_end = semi + 1 if semi != -1 else n
            body = clean_code[body_start:body_end]

        blocks.append(
            {
                "keyword": kw,
                "header": header,
                "body": body,
                "body_offset": body_start,
                "line": _line_number_at(clean_code, match.start()),
            }
        )
    return blocks


def _lint_verilog_structural(rtl_code: str) -> list[dict[str, Any]]:
    """Fast Python structural/AST-like Verilog linter."""
    issues: list[dict[str, Any]] = []
    if not rtl_code or not rtl_code.strip():
        issues.append(
            {
                "severity": "error",
                "rule": "empty_rtl",
                "line": 1,
                "message": "RTL code is empty.",
            }
        )
        return issues

    clean = _strip_verilog_comments_preserve_lines(rtl_code)

    # 1. Check module / endmodule balance
    mod_tokens = list(re.finditer(r"\b(module|endmodule)\b", clean))
    depth = 0
    module_count = 0
    endmodule_count = 0
    for tok in mod_tokens:
        word = tok.group(1)
        line = _line_number_at(clean, tok.start())
        if word == "module":
            module_count += 1
            depth += 1
            if depth > 1:
                issues.append(
                    {
                        "severity": "error",
                        "rule": "nested_module",
                        "line": line,
                        "message": "Nested 'module' declaration before previous 'endmodule'.",
                    }
                )
        else:
            endmodule_count += 1
            depth -= 1
            if depth < 0:
                issues.append(
                    {
                        "severity": "error",
                        "rule": "unmatched_endmodule",
                        "line": line,
                        "message": "Unmatched 'endmodule' without opening 'module'.",
                    }
                )
                depth = 0

    if module_count == 0:
        issues.append(
            {
                "severity": "error",
                "rule": "missing_module",
                "line": 1,
                "message": "No 'module' declaration found.",
            }
        )
    elif module_count != endmodule_count:
        issues.append(
            {
                "severity": "error",
                "rule": "module_balance",
                "line": _line_number_at(clean, len(clean)),
                "message": f"Unbalanced module/endmodule count ({module_count} module vs {endmodule_count} endmodule).",
            }
        )

    # 2. Check always blocks for sequential blocking '=' and combinational latches
    blocks = _extract_always_blocks(clean)
    # Match blocking '=' that is NOT '<=', '==', '!=', '>=', '===', '!=='
    blocking_assign_re = re.compile(r"(?<![!<>=])=(?![=])")

    for blk in blocks:
        kw = blk["keyword"]
        hdr = blk["header"]
        body = blk["body"]
        body_offset = blk["body_offset"]
        is_seq = kw == "always_ff" or bool(re.search(r"\b(posedge|negedge)\b", hdr))
        is_comb = kw == "always_comb" or hdr.strip() in ("*", "* ")

        if is_seq:
            for m in blocking_assign_re.finditer(body):
                # Ensure it's inside a statement assignment (not a for-loop header or parameter)
                line_start = body.rfind("\n", 0, m.start()) + 1
                prefix = body[line_start : m.start()].strip()
                if prefix.startswith("for") or not prefix:
                    continue
                issues.append(
                    {
                        "severity": "warning",
                        "rule": "blocking_in_sequential",
                        "line": _line_number_at(clean, body_offset + m.start()),
                        "message": (
                            f"Blocking assignment '=' used inside sequential {kw} block; "
                            "prefer non-blocking '<='."
                        ),
                    }
                )

        if is_comb:
            if_count = len(re.findall(r"\bif\s*\(", body))
            else_count = len(re.findall(r"\belse\b", body))
            if if_count > else_count:
                # Check if there is a default assignment before the first 'if'
                first_if = re.search(r"\bif\s*\(", body)
                preamble = body[: first_if.start()] if first_if else ""
                has_default_assign = bool(blocking_assign_re.search(preamble))
                if not has_default_assign:
                    issues.append(
                        {
                            "severity": "warning",
                            "rule": "incomplete_if_latch",
                            "line": blk["line"],
                            "message": (
                                "Incomplete 'if' without 'else' (and no default assignment) in "
                                "combinational block may infer an unintended latch."
                            ),
                        }
                    )

    # 3. Check undriven/unconnected signals (wire, reg, logic, output)
    # Collect declared internal/output signals and check for drivers.
    declared: dict[str, int] = {}
    inputs: set[str] = set()
    verilog_keywords = {
        "module", "endmodule", "input", "output", "inout", "wire", "reg", "logic",
        "signed", "unsigned", "parameter", "localparam", "integer", "genvar",
        "begin", "end", "if", "else", "case", "endcase", "default", "for",
        "always", "always_comb", "always_ff", "assign", "posedge", "negedge",
    }

    # Port declarations & internal net/var declarations
    decl_re = re.compile(
        r"\b(input|output|inout|wire|reg|logic)\b(?:\s+(?:wire|reg|logic|signed|unsigned))*"
        r"(?:\s*\[[^\]]+\])*\s+([^;)\n]+)"
    )
    for m in decl_re.finditer(clean):
        kind = m.group(1)
        names_part = m.group(2)
        # Strip inline assignment like `wire a = b;`
        for raw_item in names_part.split(","):
            lhs = raw_item.split("=")[0].strip()
            ident_m = re.match(r"^([A-Za-z_][A-Za-z0-9_$]*)$", lhs)
            if not ident_m:
                continue
            sig = ident_m.group(1)
            if sig in verilog_keywords:
                continue
            if kind in ("input", "inout"):
                inputs.add(sig)
            else:
                declared.setdefault(sig, _line_number_at(clean, m.start()))

    # Collect driven signals: `assign sig =`, `wire sig =`, `sig <=`, `sig =`
    driven: set[str] = set(inputs)
    for m in re.finditer(r"\bassign\s+([A-Za-z_][A-Za-z0-9_$]*)\b", clean):
        driven.add(m.group(1))
    for m in re.finditer(r"\b(?:wire|logic|reg)\b[^;]*\b([A-Za-z_][A-Za-z0-9_$]*)\s*=", clean):
        driven.add(m.group(1))
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_$]*)\s*(?:\[[^\]]+\])?\s*(?:<=|(?<![!<>=])=(?![=]))", clean):
        driven.add(m.group(1))
    # Named port connections `.port(sig)` could drive `sig`
    for m in re.finditer(r"\.\s*[A-Za-z_][A-Za-z0-9_$]*\s*\(\s*([A-Za-z_][A-Za-z0-9_$]*)\s*\)", clean):
        driven.add(m.group(1))

    for sig, line in declared.items():
        if sig not in driven:
            issues.append(
                {
                    "severity": "warning",
                    "rule": "undriven_signal",
                    "line": line,
                    "message": f"Signal '{sig}' appears undriven or unconnected.",
                }
            )

    return issues


def chipro_rtl_lint(rtl_code: str, top_module: str = "top") -> dict[str, Any]:
    """Lint Verilog/SystemVerilog RTL using Verilator (if installed) and a fast Python structural linter."""
    safe_top = _sanitize_identifier(top_module, default="top")
    issues = _lint_verilog_structural(rtl_code)

    verilator_bin = shutil.which("verilator")
    verilator_available = verilator_bin is not None
    verilator_exit_code: int | None = None
    verilator_output = ""

    if verilator_available and rtl_code.strip():
        with tempfile.TemporaryDirectory(prefix="chiplab_lint_") as tmpdir:
            v_file = Path(tmpdir) / f"{safe_top}.v"
            v_file.write_text(rtl_code, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [
                        verilator_bin,  # type: ignore[list-item]
                        "--lint-only",
                        "-Wall",
                        "--top-module",
                        safe_top,
                        str(v_file),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                verilator_exit_code = proc.returncode
                verilator_output = (proc.stdout + "\n" + proc.stderr).strip()
                if proc.returncode != 0 and verilator_output:
                    for line in verilator_output.splitlines():
                        if line.startswith("%Error") or line.startswith("%Warning"):
                            sev = "error" if line.startswith("%Error") else "warning"
                            issues.append(
                                {
                                    "severity": sev,
                                    "rule": "verilator",
                                    "line": None,
                                    "message": line,
                                }
                            )
            except (subprocess.SubprocessError, OSError) as exc:
                verilator_output = f"Verilator execution failed: {exc}"

    error_count = sum(1 for i in issues if i["severity"] == "error")
    warning_count = sum(1 for i in issues if i["severity"] == "warning")
    ok = error_count == 0 and (verilator_exit_code in (None, 0))

    return {
        "ok": ok,
        "top_module": safe_top,
        "verilator_available": verilator_available,
        "verilator_exit_code": verilator_exit_code,
        "verilator_output": verilator_output,
        "issues": issues,
        "error_count": error_count,
        "warning_count": warning_count,
        "summary": (
            f"Lint {'passed' if ok else 'failed'}: {error_count} error(s), {warning_count} warning(s)."
        ),
    }


def _estimate_rtl_complexity(rtl_code: str) -> tuple[int, int, int]:
    """Return (estimated_gates, estimated_ffs, logic_depth) from RTL text."""
    clean = _strip_verilog_comments_preserve_lines(rtl_code)
    seq_blocks = sum(
        1
        for b in _extract_always_blocks(clean)
        if b["keyword"] == "always_ff" or re.search(r"\b(posedge|negedge)\b", b["header"])
    )
    nonblocking_assigns = len(re.findall(r"<=", clean))
    # Estimate bit-widths from vector declarations `[hi:lo]`
    bit_widths: list[int] = []
    for m in re.finditer(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", clean):
        hi, lo = int(m.group(1)), int(m.group(2))
        bit_widths.append(abs(hi - lo) + 1)
    avg_width = max(1.0, sum(bit_widths) / len(bit_widths)) if bit_widths else 1.0

    estimated_ffs = max(seq_blocks, int(round(nonblocking_assigns * avg_width)))
    assign_count = len(re.findall(r"\bassign\b", clean))
    arith_ops = len(re.findall(r"(\+|-|\*|/|<<|>>)", clean))
    logic_ops = len(re.findall(r"(&&|\|\||&|\||\^|~|\?:)", clean))
    branch_count = len(re.findall(r"\b(if|case)\b", clean))

    comb_gates = int(
        round(
            (assign_count * 4 + arith_ops * 18 + logic_ops * 2 + branch_count * 6) * avg_width
        )
    )
    estimated_gates = max(10, comb_gates + estimated_ffs * 6)
    logic_depth = max(4, min(64, 4 + arith_ops * 2 + branch_count * 2 + logic_ops // 2))
    return estimated_gates, estimated_ffs, logic_depth


def chipro_run_pnr(
    rtl_code: str,
    top_module: str = "top",
    pdk: str = "sky130",
    node_nm: float = 130.0,
) -> dict[str, Any]:
    """Run or analytically estimate Place-and-Route PPA metrics scaled by process node."""
    safe_top = _sanitize_identifier(top_module, default="top")
    librelane_available = shutil.which("librelane") is not None
    openroad_available = shutil.which("openroad") is not None

    node = get_process_node(node_nm=float(node_nm))
    est_gates, est_ffs, logic_depth = _estimate_rtl_complexity(rtl_code)

    fo4_ps = float(node.fo4_delay_ps.value)
    gate_density_kgates_mm2 = max(1e-3, float(node.gate_density_kgates_mm2.value))
    dyn_energy_fj = float(node.dynamic_energy_fj_per_gate.value)
    leakage_nw = float(node.leakage_nw_per_gate.value)
    wire_r = float(node.wire_r_ohm_per_um.value)
    wire_c = float(node.wire_c_ff_per_um.value)
    vdd_v = float(node.vdd.value)

    # Analytical core area with 65% standard-cell utilization
    utilization = 0.65
    stdcell_area_um2 = (est_gates / (gate_density_kgates_mm2 * 1e3)) * 1e6
    core_area_um2 = stdcell_area_um2 / utilization

    # Analytical critical path: logic FO4 stages + Elmore wire RC delay
    avg_wire_len_um = max(5.0, (core_area_um2 ** 0.5) * 0.25)
    wire_delay_ps = 0.5 * wire_r * (wire_c * 1e-3) * (avg_wire_len_um ** 2) * logic_depth
    critical_path_delay_ns = (logic_depth * fo4_ps + wire_delay_ps) * 1e-3
    max_freq_mhz = 1000.0 / max(1e-6, critical_path_delay_ns)

    target_period_ns = 10.0  # 100 MHz reference clock
    wns_ns = target_period_ns - critical_path_delay_ns
    tns_ns = min(0.0, wns_ns) * max(1, est_ffs // 4)

    # Dynamic power at min(100 MHz, max_freq_mhz) and 15% activity factor
    eval_freq_hz = min(100e6, max_freq_mhz * 1e6)
    activity_factor = 0.15
    dynamic_power_mw = est_gates * dyn_energy_fj * 1e-15 * activity_factor * eval_freq_hz * 1e3
    leakage_power_mw = est_gates * leakage_nw * 1e-6
    total_power_mw = dynamic_power_mw + leakage_power_mw

    return {
        "ok": True,
        "top_module": safe_top,
        "pdk": pdk,
        "node_nm": float(node.node_nm.value),
        "architecture": node.architecture.value,
        "vdd_v": round(vdd_v, 4),
        "mode": "analytical",
        "librelane_available": librelane_available,
        "openroad_available": openroad_available,
        "estimated_gate_count": est_gates,
        "estimated_ff_count": est_ffs,
        "logic_depth_levels": logic_depth,
        "critical_path_delay_ns": round(critical_path_delay_ns, 4),
        "max_freq_mhz": round(max_freq_mhz, 2),
        "target_period_ns": target_period_ns,
        "wns_ns": round(wns_ns, 4),
        "tns_ns": round(tns_ns, 4),
        "core_area_um2": round(core_area_um2, 4),
        "dynamic_power_mw": round(dynamic_power_mw, 6),
        "leakage_power_mw": round(leakage_power_mw, 6),
        "total_power_mw": round(total_power_mw, 6),
        "provenance": Provenance.ESTIMATED.value,
    }


def chipro_tweak_constraints(
    sdc_text: str,
    target_freq_mhz: float,
    input_delay_ns: float = 0.5,
    output_delay_ns: float = 0.5,
) -> dict[str, Any]:
    """Safely parse and update clock period and I/O delays in SDC constraints."""
    if target_freq_mhz <= 0:
        raise ValueError(f"target_freq_mhz must be positive, got {target_freq_mhz}")

    period_ns = round(1000.0 / float(target_freq_mhz), 6)
    in_delay = round(float(input_delay_ns), 6)
    out_delay = round(float(output_delay_ns), 6)

    clocks_updated = 0
    in_delays_updated = 0
    out_delays_updated = 0

    lines = sdc_text.splitlines() if sdc_text else []
    updated_lines: list[str] = []

    period_re = re.compile(r"(-period\s+)([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")
    io_delay_re = re.compile(
        r"^(\s*set_(?:input|output)_delay\b(?:\s+-[A-Za-z_]+\s+(?:\S+|\{[^}]*\}|\[[^\]]*\]))*?\s+)"
        r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
        r"(\s+.*)$"
    )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("create_clock") and "-period" in line:
            new_line, count = period_re.subn(rf"\g<1>{period_ns}", line, count=1)
            clocks_updated += count
            updated_lines.append(new_line)
        elif stripped.startswith("set_input_delay"):
            new_line, count = io_delay_re.subn(rf"\g<1>{in_delay}\g<3>", line, count=1)
            in_delays_updated += count
            updated_lines.append(new_line)
        elif stripped.startswith("set_output_delay"):
            new_line, count = io_delay_re.subn(rf"\g<1>{out_delay}\g<3>", line, count=1)
            out_delays_updated += count
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)

    if clocks_updated == 0:
        updated_lines.append(f"create_clock -name clk -period {period_ns} [get_ports clk]")
        clocks_updated = 1
    if in_delays_updated == 0:
        updated_lines.append(f"set_input_delay {in_delay} -clock clk [all_inputs]")
        in_delays_updated = 1
    if out_delays_updated == 0:
        updated_lines.append(f"set_output_delay {out_delay} -clock clk [all_outputs]")
        out_delays_updated = 1

    return {
        "ok": True,
        "sdc_text": "\n".join(updated_lines) + "\n",
        "period_ns": period_ns,
        "target_freq_mhz": float(target_freq_mhz),
        "input_delay_ns": in_delay,
        "output_delay_ns": out_delay,
        "clocks_updated": clocks_updated,
        "input_delays_updated": in_delays_updated,
        "output_delays_updated": out_delays_updated,
    }


def chipro_open_klayout(gds_path: str) -> dict[str, Any]:
    """Check KLayout and GDS availability and launch safely when GUI display exists."""
    klayout_bin = shutil.which("klayout")
    klayout_available = klayout_bin is not None
    gds_exists = bool(gds_path) and os.path.isfile(gds_path)
    command = f"klayout {shlex.quote(gds_path)}"

    launched = False
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if klayout_available and gds_exists and has_display:
        try:
            subprocess.Popen(
                [klayout_bin, gds_path],  # type: ignore[list-item]
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            launched = True
        except OSError:
            launched = False

    return {
        "launched": launched,
        "klayout_available": klayout_available,
        "gds_exists": gds_exists,
        "command": command,
        "gds_path": gds_path,
    }
