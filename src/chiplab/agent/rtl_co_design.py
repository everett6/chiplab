"""MAGE / VLSI-AGENT multi-agent RTL + textual-waveform debugging and CHIA cyclic co-design."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from chiplab.models.materials.custom_circuit import CustomCircuit
from chiplab.models.materials.process_nodes import ProcessCorner, get_process_node
from chiplab.schemas.common import Provenance, TrackedValue


# ---------------------------------------------------------------------------
# MAGE / VLSI-AGENT: Static AST & Lint Check
# ---------------------------------------------------------------------------

_MODULE_HEADER_RE = re.compile(
    r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\([^)]*\)\s*)?\((.*?)\)\s*;",
    re.DOTALL,
)
_PORT_DECL_RE = re.compile(
    r"\b(input|output|inout)\s+(?:(wire|reg|logic)\s+)?(?:(signed)\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*)?([A-Za-z_][A-Za-z0-9_$,\s]*)"
)
_ALWAYS_RE = re.compile(
    r"\balways(?:_ff|_comb|_latch)?\s*(?:@\s*\(([^)]*)\))?",
    re.IGNORECASE,
)


def _extract_ports(rtl_code: str) -> list[dict[str, Any]]:
    """Extract port declarations from Verilog RTL (ANSI or non-ANSI style)."""
    ports: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Check ANSI header ports first, then body declarations
    search_regions: list[str] = []
    hdr = _MODULE_HEADER_RE.search(rtl_code)
    if hdr:
        search_regions.append(hdr.group(2))
    search_regions.append(rtl_code)

    for region in search_regions:
        for line in region.splitlines():
            clean = re.sub(r"//.*", "", line).strip()
            if not clean:
                continue
            m = _PORT_DECL_RE.search(clean)
            if not m:
                continue
            direction = m.group(1)
            is_signed = bool(m.group(3))
            msb_s, lsb_s = m.group(4), m.group(5)
            msb = int(msb_s) if msb_s is not None else 0
            lsb = int(lsb_s) if lsb_s is not None else 0
            width = abs(msb - lsb) + 1 if msb_s is not None else 1

            names_raw = m.group(6).split(";")[0]
            for raw_name in names_raw.split(","):
                name = raw_name.strip()
                # Stop if another direction keyword slipped in
                if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_$]*$", name):
                    continue
                if name in seen:
                    continue
                seen.add(name)
                ports.append(
                    {
                        "name": name,
                        "direction": direction,
                        "width": width,
                        "msb": msb,
                        "lsb": lsb,
                        "signed": is_signed,
                    }
                )
    return ports


def mage_static_ast_check(rtl_code: str) -> dict[str, Any]:
    """Extract module name, ports, bit widths, always blocks, and report lint issues."""
    lint_issues: list[dict[str, Any]] = []
    lines = rtl_code.splitlines()

    mod_match = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)", rtl_code)
    module_name = mod_match.group(1) if mod_match else None
    if module_name is None:
        lint_issues.append(
            {
                "severity": "error",
                "rule": "missing_module",
                "message": "No Verilog 'module' declaration found.",
                "line": 1,
            }
        )

    if not re.search(r"\bendmodule\b", rtl_code):
        lint_issues.append(
            {
                "severity": "error",
                "rule": "missing_endmodule",
                "message": "Missing 'endmodule' keyword.",
                "line": len(lines),
            }
        )

    # Check begin/end balance
    stripped_no_comments = re.sub(r"//.*", "", rtl_code)
    stripped_no_comments = re.sub(r"/\*.*?\*/", "", stripped_no_comments, flags=re.DOTALL)
    n_begin = len(re.findall(r"\bbegin\b", stripped_no_comments))
    n_end = len(re.findall(r"\bend\b", stripped_no_comments))
    if n_begin != n_end:
        lint_issues.append(
            {
                "severity": "error",
                "rule": "unbalanced_begin_end",
                "message": f"Unbalanced begin/end blocks ({n_begin} begin vs {n_end} end).",
                "line": 1,
            }
        )

    ports = _extract_ports(rtl_code)
    if module_name and not ports:
        lint_issues.append(
            {
                "severity": "warning",
                "rule": "no_ports",
                "message": f"Module '{module_name}' has no parsed input/output ports.",
                "line": 1,
            }
        )

    # Extract always blocks and check blocking vs non-blocking conventions
    always_blocks: list[dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        clean_line = re.sub(r"//.*", "", line)
        m = _ALWAYS_RE.search(clean_line)
        if not m:
            continue
        sens = (m.group(1) or "*").strip()
        is_ff = bool(re.search(r"\b(posedge|negedge)\b", sens)) or "always_ff" in clean_line
        kind = "ff" if is_ff else "comb"

        # Grab block snippet until end or next always/endmodule
        snippet_lines = [clean_line]
        depth = 1 if re.search(r"\bbegin\b", clean_line) else 0
        for j in range(idx, len(lines)):
            nxt = re.sub(r"//.*", "", lines[j])
            snippet_lines.append(nxt)
            depth += len(re.findall(r"\bbegin\b", nxt))
            depth -= len(re.findall(r"\bend\b", nxt))
            if depth <= 0 and (";" in nxt or re.search(r"\bend\b", nxt)):
                break
        snippet = "\n".join(snippet_lines)
        has_nonblocking = "<=" in snippet
        # Look for blocking '=' excluding <=, ==, !=, >=
        has_blocking = bool(re.search(r"(?<![<!>=])=(?!=)", snippet))

        if kind == "ff" and has_blocking and not has_nonblocking:
            lint_issues.append(
                {
                    "severity": "warning",
                    "rule": "blocking_in_ff",
                    "message": "Blocking assignment '=' used inside sequential always block; prefer '<='.",
                    "line": idx,
                }
            )
        if kind == "comb" and has_nonblocking:
            lint_issues.append(
                {
                    "severity": "warning",
                    "rule": "nonblocking_in_comb",
                    "message": "Non-blocking assignment '<=' used inside combinational always block; prefer '='.",
                    "line": idx,
                }
            )
        if kind == "comb" and re.search(r"\bif\b", snippet) and not re.search(r"\belse\b", snippet):
            lint_issues.append(
                {
                    "severity": "warning",
                    "rule": "inferred_latch",
                    "message": "Combinational always block has 'if' without 'else'; may infer a latch.",
                    "line": idx,
                }
            )

        always_blocks.append(
            {
                "kind": kind,
                "sensitivity": sens,
                "line": idx,
                "uses_blocking": has_blocking,
                "uses_nonblocking": has_nonblocking,
            }
        )

    # Check undriven outputs
    for p in ports:
        if p["direction"] == "output":
            pname = p["name"]
            driven = bool(
                re.search(rf"\b{re.escape(pname)}\s*(?:<=|=)", stripped_no_comments)
                or re.search(rf"\bassign\s+{re.escape(pname)}\b", stripped_no_comments)
            )
            if not driven:
                lint_issues.append(
                    {
                        "severity": "warning",
                        "rule": "undriven_output",
                        "message": f"Output port '{pname}' appears undriven.",
                        "line": 1,
                    }
                )

    has_errors = any(issue["severity"] == "error" for issue in lint_issues)
    return {
        "passed": not has_errors,
        "module_name": module_name,
        "ports": ports,
        "always_blocks": always_blocks,
        "lint_issues": lint_issues,
        "error_count": sum(1 for i in lint_issues if i["severity"] == "error"),
        "warning_count": sum(1 for i in lint_issues if i["severity"] == "warning"),
    }


# ---------------------------------------------------------------------------
# MAGE / VLSI-AGENT: Textual-Waveform Testbench Generator
# ---------------------------------------------------------------------------


def _find_clk_rst_ports(ports: list[dict[str, Any]]) -> tuple[str | None, str | None, bool]:
    """Identify clock port, reset port, and whether reset is active-low."""
    clk_port: str | None = None
    rst_port: str | None = None
    rst_active_low = False
    for p in ports:
        if p["direction"] != "input":
            continue
        low = p["name"].lower()
        if clk_port is None and low in ("clk", "clock", "i_clk"):
            clk_port = p["name"]
        elif rst_port is None and low in ("rst", "reset", "rst_n", "reset_n", "aresetn", "i_rst"):
            rst_port = p["name"]
            rst_active_low = low.endswith("_n") or low.endswith("n") and "reset" in low
    return clk_port, rst_port, rst_active_low


def _default_vectors_for_ports(ports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate deterministic default stimulus vectors for a port list."""
    inputs = [p for p in ports if p["direction"] == "input"]
    clk_port, rst_port, rst_active_low = _find_clk_rst_ports(ports)
    data_inputs = [p for p in inputs if p["name"] not in (clk_port, rst_port)]

    vectors: list[dict[str, Any]] = []
    if rst_port is not None:
        v_rst: dict[str, Any] = {rst_port: 0 if rst_active_low else 1}
        for p in data_inputs:
            v_rst[p["name"]] = 0
        vectors.append(v_rst)

    sample_vals = [(1, 2), (3, 4), (5, 7), (8, 3)]
    for step_idx, (va, vb) in enumerate(sample_vals):
        vec: dict[str, Any] = {}
        if rst_port is not None:
            vec[rst_port] = 1 if rst_active_low else 0
        for idx, p in enumerate(data_inputs):
            mask = (1 << p["width"]) - 1
            if p["width"] == 1:
                vec[p["name"]] = (step_idx + idx) % 2
            else:
                base = va if idx == 0 else (vb if idx == 1 else step_idx + 1)
                vec[p["name"]] = base & mask
        vectors.append(vec)
    return vectors


def mage_generate_textual_testbench(
    top_module: str,
    ports: list[dict[str, Any]],
    vectors: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a Verilog testbench emitting MAGE-style WAVE checkpoints and MISMATCH markers."""
    clk_port, rst_port, _ = _find_clk_rst_ports(ports)
    vecs = vectors if vectors is not None else _default_vectors_for_ports(ports)

    lines: list[str] = [
        "`timescale 1ns/1ps",
        f"module tb_{top_module};",
    ]

    for p in ports:
        width = int(p.get("width", 1))
        decl_type = "reg" if p["direction"] == "input" else "wire"
        range_str = f"[{width - 1}:0] " if width > 1 else ""
        lines.append(f"  {decl_type} {range_str}{p['name']};")

    # Instantiate DUT
    port_conns = ", ".join(f".{p['name']}({p['name']})" for p in ports)
    lines.append(f"  {top_module} dut ({port_conns});")

    if clk_port:
        lines.append(f"  initial {clk_port} = 0;")
        lines.append(f"  always #5 {clk_port} = ~{clk_port};")

    # Build WAVE $display format
    wave_fmt_parts = ["WAVE t=%0t"]
    wave_args = ["$time"]
    for p in ports:
        fmt = "%b" if int(p.get("width", 1)) == 1 else "%0d"
        wave_fmt_parts.append(f"{p['name']}={fmt}")
        wave_args.append(p["name"])
    wave_fmt = " ".join(wave_fmt_parts)
    wave_args_str = ", ".join(wave_args)

    output_names = {p["name"] for p in ports if p["direction"] == "output"}
    input_names = [p["name"] for p in ports if p["direction"] == "input" and p["name"] != clk_port]

    lines.append("  initial begin")
    for name in input_names:
        lines.append(f"    {name} = 0;")

    for vec in vecs:
        for name in input_names:
            if name in vec:
                lines.append(f"    {name} = {int(vec[name])};")
        if clk_port:
            lines.append(f"    @(posedge {clk_port}); #1;")
        else:
            lines.append("    #5;")
        lines.append(f'    $display("{wave_fmt}", {wave_args_str});')

        # Expected checks
        expected_map: dict[str, Any] = {}
        if isinstance(vec.get("expected"), dict):
            expected_map.update(vec["expected"])
        for out_name in output_names:
            if out_name in vec:
                expected_map[out_name] = vec[out_name]
            elif f"expected_{out_name}" in vec:
                expected_map[out_name] = vec[f"expected_{out_name}"]

        for sig, exp_val in expected_map.items():
            if sig in output_names:
                lines.append(
                    f'    if ({sig} !== {int(exp_val)}) '
                    f'$display("MISMATCH t=%0t signal={sig} expected=%0d actual=%0d", $time, {sig});'
                )

    lines.append("    $finish;")
    lines.append("  end")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Deterministic Python Cycle-Level Verilog Evaluator (fallback if no iverilog)
# ---------------------------------------------------------------------------


def _eval_expr(expr: str, env: dict[str, int], widths: dict[str, int]) -> int:
    """Safely evaluate a Verilog-like expression without eval()."""
    expr = expr.strip()
    # Strip Verilog sized literals like 4'd0, 8'hFF, 1'b0
    expr = re.sub(r"\d+'b([01_]+)", lambda m: str(int(m.group(1).replace("_", ""), 2)), expr)
    expr = re.sub(r"\d+'h([0-9a-fA-F_]+)", lambda m: str(int(m.group(1).replace("_", ""), 16)), expr)
    expr = re.sub(r"\d+'d([0-9_]+)", lambda m: str(int(m.group(1).replace("_", ""), 10)), expr)

    # Tokenize: identifiers, numbers, operators, parens
    token_spec = [
        ("OP", r"===|!==|==|!=|<=|>=|<<|>>|&&|\|\||[+\-*&|^~<>!()]"),
        ("NUM", r"\b\d+\b"),
        ("ID", r"\b[A-Za-z_][A-Za-z0-9_$]*\b"),
        ("SKIP", r"\s+"),
    ]
    master_re = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in token_spec))
    tokens: list[tuple[str, str]] = []
    for m in master_re.finditer(expr):
        kind = m.lastgroup
        if kind != "SKIP":
            tokens.append((kind or "", m.group()))

    pos = 0

    def peek() -> tuple[str, str] | None:
        return tokens[pos] if pos < len(tokens) else None

    def consume() -> tuple[str, str]:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def parse_or() -> int:
        val = parse_and()
        while peek() and peek()[1] in ("||", "|"):
            op = consume()[1]
            rhs = parse_and()
            val = int(bool(val) or bool(rhs)) if op == "||" else (val | rhs)
        return val

    def parse_and() -> int:
        val = parse_xor()
        while peek() and peek()[1] in ("&&", "&"):
            op = consume()[1]
            rhs = parse_xor()
            val = int(bool(val) and bool(rhs)) if op == "&&" else (val & rhs)
        return val

    def parse_xor() -> int:
        val = parse_eq()
        while peek() and peek()[1] == "^":
            consume()
            val = val ^ parse_eq()
        return val

    def parse_eq() -> int:
        val = parse_rel()
        while peek() and peek()[1] in ("==", "!=", "===", "!=="):
            op = consume()[1]
            rhs = parse_rel()
            val = int(val == rhs) if op in ("==", "===") else int(val != rhs)
        return val

    def parse_rel() -> int:
        val = parse_shift()
        while peek() and peek()[1] in ("<", ">", "<=", ">="):
            op = consume()[1]
            rhs = parse_shift()
            if op == "<":
                val = int(val < rhs)
            elif op == ">":
                val = int(val > rhs)
            elif op == "<=":
                val = int(val <= rhs)
            else:
                val = int(val >= rhs)
        return val

    def parse_shift() -> int:
        val = parse_add()
        while peek() and peek()[1] in ("<<", ">>"):
            op = consume()[1]
            rhs = parse_add()
            val = (val << rhs) if op == "<<" else (val >> rhs)
        return val

    def parse_add() -> int:
        val = parse_mul()
        while peek() and peek()[1] in ("+", "-"):
            op = consume()[1]
            rhs = parse_mul()
            val = (val + rhs) if op == "+" else (val - rhs)
        return val

    def parse_mul() -> int:
        val = parse_unary()
        while peek() and peek()[1] == "*":
            consume()
            val = val * parse_unary()
        return val

    def parse_unary() -> int:
        tok = peek()
        if tok and tok[1] in ("!", "~", "-", "+"):
            op = consume()[1]
            v = parse_unary()
            if op == "!":
                return int(not v)
            if op == "~":
                return ~v
            if op == "-":
                return -v
            return v
        return parse_primary()

    def parse_primary() -> int:
        tok = peek()
        if tok is None:
            return 0
        if tok[1] == "(":
            consume()
            val = parse_or()
            if peek() and peek()[1] == ")":
                consume()
            return val
        consume()
        if tok[0] == "NUM":
            return int(tok[1])
        return int(env.get(tok[1], 0))

    if not tokens:
        return 0
    return parse_or()


def _exec_verilog_statements(
    stmts: list[str],
    env: dict[str, int],
    widths: dict[str, int],
    next_env: dict[str, int] | None = None,
) -> None:
    """Execute simple if/else and assignment statements."""
    i = 0
    while i < len(stmts):
        line = stmts[i].strip()
        if not line or line in ("begin", "end"):
            i += 1
            continue

        # Handle if (...) stmt [else stmt]
        if_m = re.match(r"^if\s*\((.+)\)\s*(.*)$", line)
        if if_m:
            cond_expr = if_m.group(1)
            rest = if_m.group(2).strip()
            cond_val = _eval_expr(cond_expr, env, widths) != 0

            then_stmts: list[str] = []
            else_stmts: list[str] = []

            if rest and rest != "begin":
                then_stmts.append(rest)
            elif rest == "begin":
                i += 1
                depth = 1
                while i < len(stmts) and depth > 0:
                    s = stmts[i].strip()
                    if s == "begin":
                        depth += 1
                    elif s == "end":
                        depth -= 1
                        if depth == 0:
                            break
                    then_stmts.append(s)
                    i += 1

            # Check for else on next line
            if i + 1 < len(stmts) and stmts[i + 1].strip().startswith("else"):
                i += 1
                else_rest = re.sub(r"^else\s*", "", stmts[i].strip()).strip()
                if else_rest and else_rest != "begin":
                    else_stmts.append(else_rest)
                elif else_rest == "begin":
                    i += 1
                    depth = 1
                    while i < len(stmts) and depth > 0:
                        s = stmts[i].strip()
                        if s == "begin":
                            depth += 1
                        elif s == "end":
                            depth -= 1
                            if depth == 0:
                                break
                        else_stmts.append(s)
                        i += 1

            chosen = then_stmts if cond_val else else_stmts
            _exec_verilog_statements(chosen, env, widths, next_env=next_env)
            i += 1
            continue

        # Handle assignment: lhs <= rhs; or lhs = rhs;
        assign_m = re.match(r"^([A-Za-z_][A-Za-z0-9_$]*)\s*(<=|=)\s*(.+?)\s*;?$", line)
        if assign_m:
            lhs, op, rhs = assign_m.group(1), assign_m.group(2), assign_m.group(3)
            val = _eval_expr(rhs, env, widths)
            w = widths.get(lhs, 32)
            val &= (1 << w) - 1
            if op == "<=" and next_env is not None:
                next_env[lhs] = val
            else:
                env[lhs] = val
                if next_env is not None:
                    next_env[lhs] = val
        i += 1


def _simulate_rtl_python(
    rtl_code: str,
    ports: list[dict[str, Any]],
    vectors: list[dict[str, Any]],
) -> str:
    """Deterministic Python cycle-level evaluator producing MAGE WAVE & MISMATCH logs."""
    clean = re.sub(r"//.*", "", rtl_code)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)

    widths: dict[str, int] = {p["name"]: int(p.get("width", 1)) for p in ports}
    # Also capture internal reg/wire widths
    for m in re.finditer(r"\b(?:reg|wire|logic)\s+(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s+)?([A-Za-z_][A-Za-z0-9_$]*)", clean):
        msb = int(m.group(1)) if m.group(1) else 0
        lsb = int(m.group(2)) if m.group(2) else 0
        widths.setdefault(m.group(3), abs(msb - lsb) + 1 if m.group(1) else 1)

    env: dict[str, int] = {name: 0 for name in widths}
    clk_port, _, _ = _find_clk_rst_ports(ports)
    output_names = [p["name"] for p in ports if p["direction"] == "output"]
    input_names = [p["name"] for p in ports if p["direction"] == "input" and p["name"] != clk_port]

    # Parse continuous assigns
    assigns: list[tuple[str, str]] = []
    for m in re.finditer(r"\bassign\s+([A-Za-z_][A-Za-z0-9_$]*)\s*=\s*([^;]+);", clean):
        assigns.append((m.group(1), m.group(2)))

    # Parse always blocks
    seq_blocks: list[list[str]] = []
    comb_blocks: list[list[str]] = []
    lines = clean.splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        m = _ALWAYS_RE.search(line)
        if m:
            sens = (m.group(1) or "*").strip()
            is_ff = bool(re.search(r"\b(posedge|negedge)\b", sens)) or "always_ff" in line
            body_lines: list[str] = []
            if "begin" in line:
                depth = 1
                idx += 1
                while idx < len(lines) and depth > 0:
                    s = lines[idx].strip()
                    if s == "begin" or re.search(r"\bbegin\b", s):
                        depth += len(re.findall(r"\bbegin\b", s))
                    if re.search(r"\bend\b", s):
                        depth -= len(re.findall(r"\bend\b", s))
                        if depth <= 0:
                            break
                    body_lines.append(s)
                    idx += 1
            else:
                idx += 1
                if idx < len(lines):
                    body_lines.append(lines[idx].strip())
            if is_ff:
                seq_blocks.append(body_lines)
            else:
                comb_blocks.append(body_lines)
        idx += 1

    log_lines: list[str] = []
    t_ns = 0
    for vec in vectors:
        for name in input_names:
            if name in vec:
                env[name] = int(vec[name]) & ((1 << widths.get(name, 32)) - 1)

        if clk_port:
            t_ns += 5
            env[clk_port] = 1
            next_env = dict(env)
            for block in seq_blocks:
                _exec_verilog_statements(block, env, widths, next_env=next_env)
            env.update(next_env)
            t_ns += 1
        else:
            t_ns += 5

        for block in comb_blocks:
            _exec_verilog_statements(block, env, widths, next_env=None)
        for lhs, rhs in assigns:
            w = widths.get(lhs, 32)
            env[lhs] = _eval_expr(rhs, env, widths) & ((1 << w) - 1)

        wave_parts = [f"WAVE t={t_ns}"]
        for p in ports:
            val = env.get(p["name"], 0)
            if p["name"] == clk_port:
                wave_parts.append(f"{p['name']}={val & 1}")
            else:
                wave_parts.append(f"{p['name']}={val}")
        log_lines.append(" ".join(wave_parts))

        # Check expectations
        expected_map: dict[str, Any] = {}
        if isinstance(vec.get("expected"), dict):
            expected_map.update(vec["expected"])
        for out_name in output_names:
            if out_name in vec:
                expected_map[out_name] = vec[out_name]
            elif f"expected_{out_name}" in vec:
                expected_map[out_name] = vec[f"expected_{out_name}"]

        for sig, exp_val in expected_map.items():
            if sig in output_names:
                actual = env.get(sig, 0)
                if actual != int(exp_val):
                    log_lines.append(
                        f"MISMATCH t={t_ns} signal={sig} expected={int(exp_val)} actual={actual}"
                    )

        if clk_port:
            t_ns += 4
            env[clk_port] = 0

    return "\n".join(log_lines) + ("\n" if log_lines else "")


# ---------------------------------------------------------------------------
# MAGE / VLSI-AGENT: Waveform-Guided Debug & Patching
# ---------------------------------------------------------------------------


def _parse_waveform_log(waveform_log: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse WAVE and MISMATCH lines from a simulation log."""
    waves: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for line_no, raw in enumerate(waveform_log.splitlines(), start=1):
        line = raw.strip()
        if line.startswith("WAVE "):
            entry: dict[str, Any] = {"line": line_no}
            for kv in line[5:].split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        entry[k] = int(v)
                    except ValueError:
                        entry[k] = v
            waves.append(entry)
        elif line.startswith("MISMATCH "):
            mm: dict[str, Any] = {"line": line_no}
            for kv in line[9:].split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        mm[k] = int(v)
                    except ValueError:
                        mm[k] = v
            mismatches.append(mm)
    return waves, mismatches


def mage_waveform_debug(rtl_code: str, waveform_log: str) -> dict[str, Any]:
    """Parse textual waveform checkpoints, identify root cause, and produce a patched RTL."""
    waves, mismatches = _parse_waveform_log(waveform_log)
    if not mismatches:
        return {
            "has_mismatches": False,
            "mismatches": [],
            "failing_time": None,
            "root_cause": "No waveform mismatches detected.",
            "patch_suggestion": "No patch required.",
            "patched_rtl": rtl_code,
        }

    first = mismatches[0]
    fail_t = first.get("t")
    sig = str(first.get("signal", ""))
    expected = first.get("expected")
    actual = first.get("actual")

    # Find matching WAVE state at fail_t
    wave_at_fail = next((w for w in waves if w.get("t") == fail_t), {})
    patched_rtl = rtl_code
    root_cause = f"Signal '{sig}' mismatch at t={fail_t}: expected {expected}, got {actual}."
    patch_suggestion = "Inspect assignment driving the failing output signal."

    # Heuristic 1: Operator mismatch on assignment to `sig` (e.g. a - b instead of a + b)
    op_swaps = [
        ("-", "+", lambda a, b: a + b),
        ("+", "-", lambda a, b: a - b),
        ("&", "|", lambda a, b: a | b),
        ("|", "&", lambda a, b: a & b),
        ("^", "+", lambda a, b: a + b),
    ]
    patched = False
    if sig and isinstance(expected, int) and isinstance(actual, int):
        assign_pat = re.compile(
            rf"(\b{re.escape(sig)}\s*(?:<=|=)\s*)([A-Za-z_0-9]+)\s*([+\-&|^])\s*([A-Za-z_0-9]+)(\s*;)"
        )
        for m in assign_pat.finditer(rtl_code):
            lhs_prefix, op1_name, cur_op, op2_name, suffix = m.groups()
            v1 = wave_at_fail.get(op1_name) if not op1_name.isdigit() else int(op1_name)
            v2 = wave_at_fail.get(op2_name) if not op2_name.isdigit() else int(op2_name)
            if isinstance(v1, int) and isinstance(v2, int):
                for old_op, new_op, fn in op_swaps:
                    if cur_op == old_op and fn(v1, v2) == expected:
                        replacement = f"{lhs_prefix}{op1_name} {new_op} {op2_name}{suffix}"
                        patched_rtl = rtl_code[: m.start()] + replacement + rtl_code[m.end() :]
                        root_cause = (
                            f"Wrong operator '{old_op}' in assignment to '{sig}' at t={fail_t}: "
                            f"{op1_name}({v1}) {old_op} {op2_name}({v2}) = {actual}, expected {expected}."
                        )
                        patch_suggestion = f"Replace '{old_op}' with '{new_op}' in '{sig}' assignment."
                        patched = True
                        break
            if patched:
                break

    # Heuristic 2: Inverted reset polarity (e.g., if (!rst) vs if (rst))
    if not patched and isinstance(expected, int) and expected == 0 and isinstance(actual, int) and actual != 0:
        if re.search(r"if\s*\(\s*!\s*(rst|reset)\s*\)", rtl_code):
            patched_rtl = re.sub(r"if\s*\(\s*!\s*(rst|reset)\s*\)", r"if (\1)", rtl_code, count=1)
            root_cause = f"Inverted active-high reset polarity at t={fail_t} causing '{sig}'={actual} instead of 0."
            patch_suggestion = "Remove '!' from reset condition."
            patched = True

    # Heuristic 3: Wrong reset constant (e.g., sig <= 1; inside reset branch when expected == 0)
    if not patched and isinstance(expected, int) and isinstance(actual, int):
        const_pat = re.compile(rf"(\b{re.escape(sig)}\s*(?:<=|=)\s*)(\d+)(\s*;)")
        for m in const_pat.finditer(rtl_code):
            if int(m.group(2)) == actual:
                patched_rtl = rtl_code[: m.start()] + f"{m.group(1)}{expected}{m.group(3)}" + rtl_code[m.end() :]
                root_cause = f"Incorrect constant {actual} assigned to '{sig}' at t={fail_t}; expected {expected}."
                patch_suggestion = f"Update constant for '{sig}' from {actual} to {expected}."
                patched = True
                break

    return {
        "has_mismatches": True,
        "mismatches": mismatches,
        "failing_time": fail_t,
        "root_cause": root_cause,
        "patch_suggestion": patch_suggestion,
        "patched_rtl": patched_rtl,
    }


def mage_rtl_pipeline(
    rtl_code: str,
    top_module: str = "top",
    test_vectors: list[dict[str, Any]] | None = None,
    max_fix_rounds: int = 3,
) -> dict[str, Any]:
    """Orchestrate Static AST Check -> Textual-Waveform Testbench -> Simulation -> Waveform-Guided Debug."""
    current_rtl = rtl_code
    agent_trace: list[dict[str, Any]] = []
    iverilog_bin = shutil.which("iverilog")
    vvp_bin = shutil.which("vvp")

    lint_report = mage_static_ast_check(current_rtl)
    effective_top = lint_report.get("module_name") or top_module
    ports = lint_report.get("ports", [])
    vecs = test_vectors if test_vectors is not None else _default_vectors_for_ports(ports)
    testbench_v = mage_generate_textual_testbench(effective_top, ports, vecs)
    waveform_log = ""
    passed = False
    rounds = 0

    for round_idx in range(1, max(1, max_fix_rounds) + 1):
        rounds = round_idx
        lint_report = mage_static_ast_check(current_rtl)
        ports = lint_report.get("ports", [])
        effective_top = lint_report.get("module_name") or top_module
        testbench_v = mage_generate_textual_testbench(effective_top, ports, vecs)

        if iverilog_bin and vvp_bin and lint_report["passed"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                rtl_path = Path(tmpdir) / f"{effective_top}.v"
                tb_path = Path(tmpdir) / f"tb_{effective_top}.v"
                vvp_path = Path(tmpdir) / "sim.vvp"
                rtl_path.write_text(current_rtl, encoding="utf-8")
                tb_path.write_text(testbench_v, encoding="utf-8")
                comp = subprocess.run(
                    [iverilog_bin, "-o", str(vvp_path), str(rtl_path), str(tb_path)],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    shell=False,
                )
                if comp.returncode == 0:
                    sim = subprocess.run(
                        [vvp_bin, str(vvp_path)],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        shell=False,
                    )
                    waveform_log = sim.stdout
                else:
                    waveform_log = _simulate_rtl_python(current_rtl, ports, vecs)
        else:
            waveform_log = _simulate_rtl_python(current_rtl, ports, vecs)

        debug_res = mage_waveform_debug(current_rtl, waveform_log)
        agent_trace.append(
            {
                "round": round_idx,
                "lint_passed": lint_report["passed"],
                "mismatches": len(debug_res["mismatches"]),
                "root_cause": debug_res["root_cause"],
                "patch_suggestion": debug_res["patch_suggestion"],
            }
        )

        if lint_report["passed"] and not debug_res["has_mismatches"]:
            passed = True
            break

        if round_idx < max_fix_rounds and debug_res["patched_rtl"] != current_rtl:
            current_rtl = debug_res["patched_rtl"]
        else:
            break

    return {
        "passed": passed,
        "rounds": rounds,
        "lint_report": lint_report,
        "testbench_v": testbench_v,
        "waveform_log": waveform_log,
        "final_rtl": current_rtl,
        "agent_trace": agent_trace,
    }


# ---------------------------------------------------------------------------
# CHIA Cyclic Hardware/Software Co-Design Loop
# ---------------------------------------------------------------------------


def _evaluate_chia_ppa(
    node_nm: float,
    corner: ProcessCorner,
    gate_count: int,
    pipeline_stages: int,
    base_logic_depth_fo4: float,
    activity_factor: float,
    vdd_scale: float,
    target_freq_mhz: float,
) -> dict[str, Any]:
    """Evaluate PPA and critical-path slack using ProcessNodeParams + CustomCircuit."""
    pnode = get_process_node(node_nm=node_nm, corner=corner)
    circ = CustomCircuit(name="chia_crit_path", node_nm=node_nm, corner=corner)

    # Model a representative critical-path RC wire + stage delay
    wire_len_um = max(5.0, 20.0 * (node_nm / 28.0))
    circ.add_wire_segment("crit_wire", "in_n", "out_n", length_um=wire_len_um)
    wire_r = pnode.wire_r_ohm_per_um.value * wire_len_um
    wire_c_f = pnode.wire_c_ff_per_um.value * 1e-15 * wire_len_um
    wire_delay_ps = 0.69 * wire_r * wire_c_f * 1e12

    eff_vdd = pnode.vdd.value * vdd_scale
    vth = pnode.vth_n.value
    # Alpha-power delay scaling with Vdd
    nom_drive = max(0.15, pnode.vdd.value - vth)
    eff_drive = max(0.12, eff_vdd - vth)
    vdd_delay_factor = (nom_drive / eff_drive) ** 1.3

    depth_per_stage = max(4.0, base_logic_depth_fo4 / max(1, pipeline_stages))
    stage_delay_ps = (depth_per_stage * pnode.fo4_delay_ps.value * vdd_delay_factor) + wire_delay_ps
    flop_overhead_ps = 2.0 * pnode.fo4_delay_ps.value

    crit_path_ps = TrackedValue(
        value=float(stage_delay_ps + flop_overhead_ps),
        provenance=Provenance.ESTIMATED,
        metadata={"unit": "ps", "description": "Critical path stage delay"},
    )

    achieved_freq_mhz = 1e6 / max(1.0, crit_path_ps.value)
    target_period_ns = 1000.0 / max(1e-3, target_freq_mhz)
    achieved_period_ns = crit_path_ps.value / 1000.0
    timing_slack_ns = target_period_ns - achieved_period_ns

    # Area: logic gates + pipeline register overhead
    flop_gates = int(gate_count * 0.06 * max(0, pipeline_stages - 1))
    total_gates = gate_count + flop_gates
    density_kgates_mm2 = max(1.0, pnode.gate_density_kgates_mm2.value)
    # 1 kgate/mm^2 = 1e-3 gates/um^2
    area_um2 = (total_gates / (density_kgates_mm2 * 1e3)) * 1e6

    # Power: dynamic + static leakage
    dyn_e_fj = pnode.dynamic_energy_fj_per_gate.value * (vdd_scale ** 2)
    dyn_power_mw = total_gates * activity_factor * dyn_e_fj * 1e-15 * achieved_freq_mhz * 1e6 * 1e3
    leak_nw = pnode.leakage_nw_per_gate.value * vdd_scale
    static_power_mw = total_gates * leak_nw * 1e-6
    total_power_mw = dyn_power_mw + static_power_mw

    return {
        "node_nm": float(node_nm),
        "corner": corner.value,
        "pipeline_stages": int(pipeline_stages),
        "vdd_v": round(eff_vdd, 4),
        "critical_path_ps": round(crit_path_ps.value, 3),
        "achieved_freq_mhz": round(achieved_freq_mhz, 2),
        "timing_slack_ns": round(timing_slack_ns, 4),
        "area_um2": round(area_um2, 2),
        "power_mw": round(total_power_mw, 4),
        "dynamic_power_mw": round(dyn_power_mw, 4),
        "static_power_mw": round(static_power_mw, 4),
        "total_gates": total_gates,
    }


def chia_codesign_loop(
    design_spec: dict[str, Any] | None = None,
    node_nm: float = 28.0,
    target_freq_mhz: float = 500.0,
    max_area_um2: float = 50000.0,
    max_power_mw: float = 50.0,
    max_iters: int = 5,
) -> dict[str, Any]:
    """Run CHIA cyclic hardware/software co-design optimization until PPA targets are met."""
    spec = dict(design_spec or {})
    gate_count = int(spec.get("gate_count", 25000))
    pipeline_stages = int(spec.get("pipeline_stages", 2))
    base_logic_depth_fo4 = float(spec.get("logic_depth_fo4", 60.0))
    activity_factor = float(spec.get("activity_factor", 0.15))
    cur_node_nm = float(node_nm)
    corner = ProcessCorner(str(spec.get("corner", "TT")).upper())
    vdd_scale = float(spec.get("vdd_scale", 1.0))

    history: list[dict[str, Any]] = []
    converged = False
    final_metrics: dict[str, Any] = {}

    for it in range(1, max(1, max_iters) + 1):
        metrics = _evaluate_chia_ppa(
            node_nm=cur_node_nm,
            corner=corner,
            gate_count=gate_count,
            pipeline_stages=pipeline_stages,
            base_logic_depth_fo4=base_logic_depth_fo4,
            activity_factor=activity_factor,
            vdd_scale=vdd_scale,
            target_freq_mhz=target_freq_mhz,
        )

        timing_ok = metrics["achieved_freq_mhz"] >= target_freq_mhz
        area_ok = metrics["area_um2"] <= max_area_um2
        power_ok = metrics["power_mw"] <= max_power_mw

        if timing_ok and area_ok and power_ok:
            bottleneck = "none"
            action = "targets_met"
            converged = True
        elif not timing_ok:
            bottleneck = "timing"
            if pipeline_stages < 8:
                pipeline_stages += 2
                action = f"increase_pipeline_stages_to_{pipeline_stages}"
            elif corner != ProcessCorner.FF:
                corner = ProcessCorner.FF
                action = "switch_corner_to_FF"
            else:
                cur_node_nm = max(2.0, round(cur_node_nm * 0.65, 2))
                action = f"scale_node_to_{cur_node_nm}nm"
        elif not area_ok:
            bottleneck = "area"
            cur_node_nm = max(2.0, round(cur_node_nm * 0.6, 2))
            action = f"scale_node_to_{cur_node_nm}nm"
        else:
            bottleneck = "power"
            if vdd_scale > 0.8:
                vdd_scale = round(max(0.75, vdd_scale - 0.12), 3)
                action = f"reduce_vdd_scale_to_{vdd_scale}"
            else:
                cur_node_nm = max(2.0, round(cur_node_nm * 0.7, 2))
                action = f"scale_node_to_{cur_node_nm}nm"

        step_record = {
            "iteration": it,
            "metrics": metrics,
            "bottleneck": bottleneck,
            "action": action,
            "targets_met": converged,
        }
        history.append(step_record)
        final_metrics = metrics
        if converged:
            break

    return {
        "converged": converged,
        "iterations": len(history),
        "final_metrics": final_metrics,
        "history": history,
    }
