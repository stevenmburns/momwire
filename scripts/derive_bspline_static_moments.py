"""Pre-derive the same-edge static-kernel moment integrals with sympy and
dump them as inline numpy + C++ source.

Two families come out of this one run.

J_pq — the REDUCED kernel's static part
---------------------------------------
The B-spline solver (`momwire.bspline.BSplineSolver`) needs closed-form
∫∫ (s-α)^p (s'-A)^q / √((s-s')²+a²) ds' ds for p, q ∈ {0..d} on every
same-edge segment pair. Sympy can integrate this in seconds, but doing so
at every module import is slow (8s+) and adds sympy as a runtime dep.

D_pq — the EXTENDED kernel's static CORRECTION (momwire#249)
------------------------------------------------------------
`extended_kernel=True` multiplies the shared mixed-potential kernel by NEC
Eq 89's `1 + T1·C2 − T2·C1`; on the coaxial equal-radius pairs momwire
extends (design §4.2) that collapses to a scalar factor of R alone, whose
k → 0 limit adds

    Δg(ξ) = −a²/(2R³) + 3a⁴/(4R⁵),      R = √(ξ² + a²),  ξ = s − s'

to the reduced kernel's 1/R. `D_pq = ∫∫ (s-α)^p (s'-A)^q Δg(s-s') ds' ds`
is the second generated family.

    THE INVARIANT: Δg is ONE integrand. Never derive ∫∫…/R³ and ∫∫…/R⁵ as
    two families and add them. Both are individually O(1) while their sum
    is O(a²) — on the diagonal pair at α=A=0, β=B=h they are −h and +h to
    leading order, so a two-family spelling would ship the answer as the
    O(a²) residue of a catastrophic cancellation (momwire#205 class).

The derivation route is by parts, not brute force. With
`H(ξ) = −a²/(4√(ξ²+a²))` one has `H″ = Δg` exactly, so for `Q(t) = (t−A)^q`
with q ≤ 2 (hence constant `Q″ = q(q−1)`),

    ∫_A^B Q(t) H″(s−t) dt = −[Q H′]_A^B − [Q′ H]_A^B + q(q−1)·∫_A^B H(s−t) dt

and therefore

    D_pq = −∫ P(s)[Q H′]_A^B ds − ∫ P(s)[Q′ H]_A^B ds − q(q−1)·(a²/4)·J_p0

i.e. boundary terms in `H` and `H′ = a²ξ/(4R³)` — 1-D integrals of a
polynomial against an elementary antiderivative, a second each — plus a
scaled copy of the already-derived, already-audited J family. Handing the
combined Δg straight to two nested `sp.integrate` calls also works and
returns the identical D₀₀, but costs 553 s for that one moment alone.

Every emitted D expression carries an explicit `a²` factor at the top
level, so the a → 0 collapse onto the reduced kernel is structural. (It is
*exactly* 0.0 in IEEE only where the bracket is finite: D₀₀ everywhere with
distinct corners, and any moment on a pair whose segments do not touch. The
q ≥ 1 moments inherit the J family's `asinh(ξ/a)` spelling, which is ±inf at
a = 0 for the same reason J itself is — a = 0 is not a supported input to
either family.)

This script runs sympy ONCE and emits:
  * `src/momwire/_bspline_static_moments.py` — inline numpy expressions
    (CPU fallback when the C++ extension isn't available, and the
    canonical reference for the C++ side).
  * `src/momwire/_bspline_static_moments_inline.h` — inline C++ functions
    `J_static_pq_PP_QQ(alpha, beta, A, B, a)` for each (p, q). Included
    by `_accelerators.cpp` to power the static-moments fast path.
  * `src/momwire/_bspline_ek_moments.py` — `D_ek_moment(p, q, alpha, beta,
    A, B, a)`, the numpy twin of the above for the EK correction.
  * `src/momwire/_bspline_ek_moments_inline.h` — inline C++ functions
    `D_ek_pq_PP_QQ(alpha, beta, A, B, a)`. Not yet included by
    `_accelerators.cpp`: the C++ EK twins are momwire#249 follow-up B.

The four outputs are run through `ruff format` so a re-run is byte-clean
against the tree (CI enforces `ruff format --check`).

Re-run this script when you need to extend `max_d`.

Usage:
    .venv/bin/python scripts/derive_bspline_static_moments.py
"""

import pathlib
import shutil
import subprocess

import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter
from sympy.printing.numpy import NumPyPrinter

MAX_D = 2

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_PATH_PY = REPO_ROOT / "src" / "momwire" / "_bspline_static_moments.py"
OUT_PATH_H = REPO_ROOT / "src" / "momwire" / "_bspline_static_moments_inline.h"
OUT_PATH_EK_PY = REPO_ROOT / "src" / "momwire" / "_bspline_ek_moments.py"
OUT_PATH_EK_H = REPO_ROOT / "src" / "momwire" / "_bspline_ek_moments_inline.h"

# Placeholder standing in for the J_p0 call in the q = 2 by-parts term. It is
# a plain Symbol rather than a sympy Function so both printers emit it
# verbatim; the per-language call text is substituted for it afterwards.
J_CALL = sp.Symbol("J_STATIC_P0_CALL")


def derive_all(max_d):
    s, sp_dummy, alpha, beta, A, B, a = sp.symbols(
        "s sp_dummy alpha beta A B a", real=True, positive=True
    )
    np_printer = NumPyPrinter()
    cxx_printer = CXX11CodePrinter()
    entries = []
    for p in range(max_d + 1):
        for q in range(max_d + 1):
            integrand = (
                (s - alpha) ** p
                * (sp_dummy - A) ** q
                / sp.sqrt((s - sp_dummy) ** 2 + a**2)
            )
            inner = sp.integrate(integrand, (sp_dummy, A, B))
            outer = sp.integrate(inner, (s, alpha, beta))
            np_code = np_printer.doprint(outer).replace("numpy.", "np.")
            cxx_code = cxx_printer.doprint(outer)
            entries.append((p, q, np_code, cxx_code))
    return entries


def emit_py(entries, path, max_d):
    body = [
        '"""Inline closed-form J_pq^static = ∫∫ (s-α)^p (s\'-A)^q / √((s-s\')²+a²)',
        f"on a single straight edge, for p, q ∈ {{0..{max_d}}}.",
        "",
        "Generated by scripts/derive_bspline_static_moments.py (do not edit).",
        '"""',
        "",
        "import numpy as np",
        "",
        "",
        "def J_static_moment(p, q, alpha, beta, A, B, a):",
        '    """One static moment integral (without the 1/(4π) prefactor).',
        "    alpha/beta/A/B/a broadcast as numpy arrays.",
        '    """',
    ]
    for idx, (p, q, np_code, _) in enumerate(entries):
        prefix = "if" if idx == 0 else "elif"
        body.append(f"    {prefix} (p, q) == ({p}, {q}):")
        body.append(f"        return {np_code}")
    body.append("    else:")
    body.append(
        f"        raise ValueError(f'(p, q) = ({{p}}, {{q}}) not in [0, {max_d}]²')"
    )
    body.append("")
    path.write_text("\n".join(body))


def emit_h(entries, path, max_d):
    body = [
        f"// Inline closed-form J_pq^static for p, q in {{0..{max_d}}}.",
        "// Generated by scripts/derive_bspline_static_moments.py (do not edit).",
        "// Included from _accelerators.cpp.",
        "#pragma once",
        "#include <cmath>",
        "",
    ]
    for p, q, _, cxx_code in entries:
        body.append(
            f"static inline double J_static_pq_{p}_{q}("
            "double alpha, double beta, double A, double B, double a) {"
        )
        body.append(f"    return {cxx_code};")
        body.append("}")
        body.append("")
    path.write_text("\n".join(body))


def derive_ek_all(max_d):
    """The EK static-correction family D_pq, by parts (module docstring).

    Returns the same (p, q, np_code, cxx_code) tuples `derive_all` does.
    """
    alpha, beta, A, B, a = sp.symbols("alpha beta A B a", real=True, positive=True)
    # `xi` is the axial offset s - c of an observer point from ONE endpoint c
    # of the source segment, and `d` = c - alpha shifts the observer-side
    # polynomial into that frame: (xi + d)^p = (s - alpha)^p. Both are signed,
    # so neither may inherit the positive=True the geometry symbols carry.
    xi, d = sp.symbols("xi d", real=True)
    R = sp.sqrt(xi**2 + a**2)

    # Antiderivatives of the two boundary integrands, with the shared a²/4
    # stripped off so it can be factored to the front of every emitted
    # expression: f0 belongs to H = -(a²/4)/R and f1 to H' = (a²/4)·ξ/R³.
    # Indefinite-then-substitute, deliberately: sympy returns the definite
    # form of the f1 integrals unevaluated for p ≥ 1.
    f0, f1 = {}, {}
    for p in range(max_d + 1):
        poly = sp.expand((xi + d) ** p)
        f0[p] = sp.simplify(sp.integrate(poly / R, xi))
        f1[p] = sp.simplify(sp.integrate(poly * xi / R**3, xi))

    def corner(F, p, c):
        """[F_p]_{s=alpha}^{s=beta} about the source endpoint c."""
        shift = {d: c - alpha}
        return F[p].subs({xi: beta - c, **shift}) - F[p].subs({xi: alpha - c, **shift})

    np_printer = NumPyPrinter()
    cxx_printer = CXX11CodePrinter()
    entries = []
    for p in range(max_d + 1):
        for q in range(max_d + 1):
            # Q(t) = (t-A)^q and its derivative, evaluated at the two source
            # endpoints. 0^0 = 1, so Q(A) survives only at q = 0 and Q'(A)
            # only at q = 1; Q'' = q(q-1) is the constant that scales the
            # reused J_p0 double integral.
            q_at_b = (B - A) ** q
            q_at_a = sp.Integer(1 if q == 0 else 0)
            qp_at_b = q * (B - A) ** (q - 1) if q >= 1 else sp.Integer(0)
            qp_at_a = sp.Integer(1 if q == 1 else 0)
            bracket = -q_at_b * corner(f1, p, B) + q_at_a * corner(f1, p, A)
            bracket += qp_at_b * corner(f0, p, B) - qp_at_a * corner(f0, p, A)
            bracket += -sp.Integer(q * (q - 1)) * J_CALL
            expr = sp.Rational(1, 4) * a**2 * bracket
            np_code = (
                np_printer.doprint(expr)
                .replace("numpy.", "np.")
                .replace(J_CALL.name, f"J_static_moment({p}, 0, alpha, beta, A, B, a)")
            )
            cxx_code = cxx_printer.doprint(expr).replace(
                J_CALL.name, f"J_static_pq_{p}_0(alpha, beta, A, B, a)"
            )
            entries.append((p, q, np_code, cxx_code))
    return entries


def emit_ek_py(entries, path, max_d):
    body = [
        "\"\"\"Inline closed-form D_pq^EK = ∫∫ (s-α)^p (s'-A)^q Δg(s-s') ds' ds,",
        "the extended-kernel correction to the static moments on a single",
        f"straight edge, for p, q ∈ {{0..{max_d}}} (momwire#249).",
        "",
        "    Δg(ξ) = -a²/(2R³) + 3a⁴/(4R⁵),   R = √(ξ² + a²)",
        "",
        "ONE integrand, never the two families separately — see the generator's",
        "docstring for why (they cancel to O(a²)). Add this to J_static_moment",
        "to get the extended kernel's static part; the explicit a² prefactor is",
        "what makes the a → 0 collapse structural rather than numerical.",
        "",
        "Generated by scripts/derive_bspline_static_moments.py (do not edit).",
        '"""',
        "",
        "import numpy as np",
        "",
        "from ._bspline_static_moments import J_static_moment",
        "",
        "",
        "def D_ek_moment(p, q, alpha, beta, A, B, a):",
        '    """One EK static-correction moment (without the 1/(4π) prefactor).',
        "    alpha/beta/A/B/a broadcast as numpy arrays.",
        '    """',
    ]
    for idx, (p, q, np_code, _) in enumerate(entries):
        prefix = "if" if idx == 0 else "elif"
        body.append(f"    {prefix} (p, q) == ({p}, {q}):")
        body.append(f"        return {np_code}")
    body.append("    else:")
    body.append(
        f"        raise ValueError(f'(p, q) = ({{p}}, {{q}}) not in [0, {max_d}]²')"
    )
    body.append("")
    path.write_text("\n".join(body))


def emit_ek_h(entries, path, max_d):
    body = [
        f"// Inline closed-form D_pq^EK for p, q in {{0..{max_d}}} (momwire#249).",
        "// Generated by scripts/derive_bspline_static_moments.py (do not edit).",
        "// NOT included from _accelerators.cpp yet — the C++ extended-kernel",
        "// twins are momwire#249 follow-up B; this is the reference they need.",
        "#pragma once",
        "#include <cmath>",
        '#include "_bspline_static_moments_inline.h"',
        "",
    ]
    for p, q, _, cxx_code in entries:
        body.append(
            f"static inline double D_ek_pq_{p}_{q}("
            "double alpha, double beta, double A, double B, double a) {"
        )
        body.append(f"    return {cxx_code};")
        body.append("}")
        body.append("")
    path.write_text("\n".join(body))


def format_outputs(paths):
    """Run `ruff format` over the emitted python so a re-run is byte-clean.

    The committed generators' output has always been formatted (CI runs
    `ruff format --check`), so without this step re-running the pipeline
    produces a diff that is pure line-wrapping. Best-effort: a missing ruff
    warns rather than failing the derivation that just cost minutes.
    """
    ruff = shutil.which("ruff")
    if ruff is None:
        print("WARNING: ruff not on PATH — emitted files are UNFORMATTED")
        return
    subprocess.run([ruff, "format", *[str(p) for p in paths]], check=True)


def main():
    print(f"Deriving J_pq^static for p, q ∈ {{0..{MAX_D}}} (this takes ~10s)…")
    entries = derive_all(MAX_D)
    emit_py(entries, OUT_PATH_PY, MAX_D)
    emit_h(entries, OUT_PATH_H, MAX_D)
    print(f"Wrote {OUT_PATH_PY}")
    print(f"Wrote {OUT_PATH_H}")

    print(f"Deriving D_pq^EK for p, q ∈ {{0..{MAX_D}}} (by parts, ~2s)…")
    ek_entries = derive_ek_all(MAX_D)
    emit_ek_py(ek_entries, OUT_PATH_EK_PY, MAX_D)
    emit_ek_h(ek_entries, OUT_PATH_EK_H, MAX_D)
    print(f"Wrote {OUT_PATH_EK_PY}")
    print(f"Wrote {OUT_PATH_EK_H}")

    format_outputs([OUT_PATH_PY, OUT_PATH_EK_PY])


if __name__ == "__main__":
    main()
