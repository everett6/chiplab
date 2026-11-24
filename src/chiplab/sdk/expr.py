"""Sandboxed expression language for component models.

Custom components describe area, power, delay, validation rules and digital behaviour with
expressions such as ``4 * W * L + 0.2`` or ``(a + b) & mask(width)``. They are parsed with
Python's ``ast`` and evaluated by a whitelist interpreter: no attribute access, no subscripts,
no lambdas, no imports, no calls except the functions listed in FUNCS, bounded expression
length, bounded integer size and bounded exponent. Nothing user- or agent-written is ever
executed as Python code.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

MAX_LEN = 600
MAX_INT_BITS = 4096


class ExprError(ValueError):
    pass


def _clog2(x: float) -> int:
    return 0 if x <= 1 else math.ceil(math.log2(x))


def _mask(n: float) -> int:
    n = int(n)
    if not 0 <= n <= MAX_INT_BITS:
        raise ExprError("mask width out of range")
    return (1 << n) - 1


def _quantize(v: float, bits: float, vref: float = 1.0) -> int:
    levels = (1 << int(bits)) - 1
    return int(max(0, min(levels, round(v / vref * levels))))


FUNCS = {
    "min": min, "max": max, "abs": abs, "round": round, "int": int, "float": float,
    "ceil": math.ceil, "floor": math.floor, "sqrt": math.sqrt, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "clog2": _clog2, "mask": _mask, "quantize": _quantize,
    "bit": lambda v, i: (int(v) >> int(i)) & 1, "popcount": lambda v: bin(int(v)).count("1"),
    "select": lambda c, a, b: a if c else b,
}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.LShift: operator.lshift, ast.RShift: operator.rshift, ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_, ast.BitXor: operator.xor}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert, ast.Not: operator.not_}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge}


def _check(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v.bit_length() > MAX_INT_BITS:
        raise ExprError("integer too large")
    if isinstance(v, float) and (math.isnan(v) or abs(v) > 1e300):
        raise ExprError("float out of range")
    return v


def parse(src: str) -> ast.Expression:
    if not isinstance(src, str):
        raise ExprError("expression must be a string")
    if len(src) > MAX_LEN:
        raise ExprError(f"expression longer than {MAX_LEN} characters")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"syntax error in {src!r}: {e.msg}") from e
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
                               ast.Name, ast.Constant, ast.Call, ast.Load, ast.And, ast.Or,
                               *_BIN, *_UN, *_CMP))
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            ok = False
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in FUNCS
                                           or node.keywords):
            ok = False
        if not ok:
            raise ExprError(f"not allowed in expressions: {type(node).__name__} in {src!r}")
    return tree


def names(src: str) -> set[str]:
    return {n.id for n in ast.walk(parse(src)) if isinstance(n, ast.Name) and n.id not in FUNCS}


def evaluate(src: str | int | float, env: dict[str, Any]) -> Any:
    if isinstance(src, (int, float)) and not isinstance(src, bool):
        return src
    tree = parse(str(src))

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise ExprError(f"unknown name {n.id!r}")
            return env[n.id]
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 64:
                raise ExprError("exponent too large")
            if isinstance(n.op, ast.LShift) and b > MAX_INT_BITS:
                raise ExprError("shift too large")
            try:
                return _check(_BIN[type(n.op)](a, b))
            except ZeroDivisionError as e:
                raise ExprError("division by zero") from e
        if isinstance(n, ast.UnaryOp):
            return _check(_UN[type(n.op)](ev(n.operand)))
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, ast.Call):
            return _check(FUNCS[n.func.id](*[ev(a) for a in n.args]))
        raise ExprError(f"unsupported node {type(n).__name__}")

    return ev(tree)
