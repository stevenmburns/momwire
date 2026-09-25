"""No arithmetic `reduction` clause on an `omp simd` directive — momwire#1193.

The house rule is momwire#781's: `omp simd reduction(+ : ...)` LICENSES the
compiler to reassociate the sum, so the reduction tree follows whatever
vectorization it picks per function. The answer then belongs to the
compiler, not the source: an unrelated edit to the translation unit, or a
compiler upgrade, can move it at the ulp level, and two kernels written
identically can disagree (#781's arm64 case). A sum whose order matters is
spelled in the source instead — serially, or in fixed lanes combined in a
written order (`_near_interface_accel.cpp`'s `column_member`, #1193).

`reduction(max)` / `reduction(min)` are exact in any order and stay allowed
(`_accel_mw568.cpp`). `-` and `*` are refused with `+`: `-` is a sum by
another name, and a product reassociates the same way.

A prose rule did not hold: #781 dropped the `+` clauses on 2026-09-01, and
the near-interface column kernel (#899) brought one back four days later,
where it stayed until #1193. So this reads the sources, in every spelling a clause can reach `omp simd`
through: a `#pragma` line, a `_Pragma("...")` string, a `#define` body, and
the arguments of any function-like macro whose body expands to `omp simd`
(`MW_OMP_SIMD`, `MW_NI_SIMD`, and any added later).
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "momwire"

# `reduction(<modifier>, <op> : list)` — capture everything before the colon.
_REDUCTION = re.compile(r"\breduction\s*\(([^:()]*):")
_ARITH = re.compile(r"[+\-*]")
_OMP_SIMD = re.compile(r"\bomp\b.*\bsimd\b", re.DOTALL)
_DEFINE = re.compile(r"^\s*#\s*define\s+(\w+)(\([^)]*\))?(.*)$")
_PRAGMA = re.compile(r"^\s*#\s*pragma\s+(.*)$")
_PRAGMA_OP = re.compile(r'_Pragma\s*\(\s*"((?:[^"\\]|\\.)*)"\s*\)')


def _strip_comments(src: str) -> str:
    """Comments out, string and char literals kept, newlines kept (so line
    numbers survive). Prose citing the forbidden clause — as the #781
    comments do — must not trip the gate."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif c == "/" and src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * src.count("\n", i, j))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i : j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _logical_lines(src: str):
    """(first physical line number, text) with backslash continuations
    joined, so a multi-line `#pragma` or `#define` is one unit."""
    phys = src.split("\n")
    i = 0
    while i < len(phys):
        start, text = i + 1, phys[i]
        while text.endswith("\\") and i + 1 < len(phys):
            i += 1
            text = text[:-1] + " " + phys[i]
        yield start, text
        i += 1


def _balanced_args(src: str, open_at: int) -> str:
    depth = 0
    for j in range(open_at, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[open_at + 1 : j]
    return src[open_at + 1 :]


def _offending(text: str) -> bool:
    return any(_ARITH.search(m.group(1)) for m in _REDUCTION.finditer(text))


def scan(sources: dict[str, str]):
    """Every simd directive the sources can spell, and the ones carrying an
    arithmetic reduction. Returns (offenders, n_directives, simd_macros)."""
    stripped = {name: _strip_comments(s) for name, s in sources.items()}
    # Function-like macros expanding to `omp simd`, from ALL sources first:
    # `_accel_common.h` defines the macro the .cpp files invoke.
    simd_macros = set()
    for s in stripped.values():
        for _, line in _logical_lines(s):
            m = _DEFINE.match(line)
            if m and m.group(2) and _OMP_SIMD.search(m.group(3)):
                simd_macros.add(m.group(1))
    offenders, n_directives = [], 0
    for name, s in stripped.items():
        for lineno, line in _logical_lines(s):
            texts = []
            m = _PRAGMA.match(line)
            if m and _OMP_SIMD.search(m.group(1)):
                texts.append(m.group(1))
            m = _DEFINE.match(line)
            if m and _OMP_SIMD.search(m.group(3)):
                texts.append(m.group(3))
            texts += [p for p in _PRAGMA_OP.findall(line) if _OMP_SIMD.search(p)]
            for t in texts:
                n_directives += 1
                if _offending(t):
                    offenders.append(f"{name}:{lineno}: {line.strip()}")
        if simd_macros:
            call = re.compile(r"\b(" + "|".join(sorted(simd_macros)) + r")\s*\(")
            for m in call.finditer(s):
                line_start = s.rfind("\n", 0, m.start()) + 1
                if re.fullmatch(r"\s*#\s*define\s+", s[line_start : m.start()]):
                    continue  # the macro's own definition, not a use
                n_directives += 1
                if _offending(_balanced_args(s, m.end() - 1)):
                    lineno = s.count("\n", 0, m.start()) + 1
                    offenders.append(f"{name}:{lineno}: {m.group(1)}(...)")
    return offenders, n_directives, simd_macros


def _tree_sources():
    files = sorted(PKG.glob("*.cpp")) + sorted(PKG.glob("*.h"))
    return {f.name: f.read_text() for f in files}


def test_no_arithmetic_reduction_on_any_omp_simd_directive():
    offenders, n_directives, macros = scan(_tree_sources())
    # Not vacuous: the scan must actually see the tree's simd sites and both
    # simd macros, or a rename/relocation would pass this by finding nothing.
    assert {"MW_OMP_SIMD", "MW_NI_SIMD"} <= macros, macros
    # 50 at #1193; the floor only has to tell "found them" from "found none".
    assert n_directives >= 25, n_directives
    assert not offenders, (
        "`omp simd reduction(+|-|*)` lets the compiler choose the summation "
        "order (momwire#781, #1193). Spell the sum's order in the source "
        "instead:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "src",
    [
        "#pragma omp simd reduction(+ : a)\n",
        "#pragma omp parallel for simd reduction(+:a)\n",
        "#pragma omp simd \\\n    reduction(-: a)\n",
        "#pragma omp simd reduction(inscan, * : a)\n",
        '_Pragma("omp simd reduction(+:a, b)")\n',
        '#define RED _Pragma("omp simd reduction(+:x)")\n',
        "#define S(c) P_(omp simd c)\nvoid f() {\n  S(reduction(+ : sr, si))\n}\n",
        "#define S(c) P_(omp simd c)\nvoid f() {\n  S(aligned(p) reduction ( + : s))\n}\n",
        "#define S(c) P_(omp simd c)\n#define T S(reduction(+ : s))\n",
    ],
)
def test_the_scan_catches_each_spelling(src):
    offenders, _, _ = scan({"x.cpp": src})
    assert offenders, src


@pytest.mark.parametrize(
    "src",
    [
        "#pragma omp simd reduction(max : a) reduction(min : b)\n",
        "#pragma omp parallel for reduction(+ : a)\n",  # not simd: out of scope
        "// #pragma omp simd reduction(+ : a)\n",
        "/* MW_NI_SIMD(reduction(+ : sr, si)) */\n#define MW_NI_SIMD(c) P_(omp simd c)\n",
        "#define S(c) P_(omp simd c)\nvoid f() {\n  S()\n}\n",
    ],
)
def test_the_scan_passes_what_the_rule_allows(src):
    offenders, _, _ = scan({"x.cpp": src})
    assert not offenders, offenders
