"""Compare two builds' object code function by function.

    python scripts/core_split/objdiff.py DIR_A DIR_B
    python scripts/core_split/objdiff.py --pair A.o B.o [--pair ...]
        [--rename REGEX=REPL] [--report REGEX]

Each object is disassembled (`objdump -d -r -C --no-show-raw-insn`) and each
function's instruction stream normalised: addresses and in-section branch
targets dropped, symbolic targets and relocations kept, `.LC`/`_omp_fn`
numbering erased. Identical streams mean identical arithmetic: the change
cannot have moved a bit of that function's output.

Where streams differ, the FP-op sequence (SSE/AVX arithmetic, conversions,
sign/abs masks, and libm/libmvec calls) is compared next, then its multiset
(the same operations in another schedule). A different multiset is a change
to look at; a different schedule of the same multiset is not proof of
anything either way, which is why the trace comparison (cmp_traces.py) is
the bit-identity gate and this is the static half.

DIR_A DIR_B: every object present in both directories (e.g. two builds'
`build/temp.*/_accelerators_avx2/src/momwire`), matched by file name and by
full demangled function name.

--pair: one object against another of a different name, e.g. a pre-split
kernel TU against its core TU. Functions are matched by BARE name (no
namespaces, return type or parameter list, which a split changes), after
--rename on side A's names. --report REGEX prints the per-function verdict
for every matched function whose bare name matches.
"""

import argparse
import collections
import glob
import os
import re
import subprocess

FN = re.compile(r"^[0-9a-f]+ <(.*)>:$")
INSN = re.compile(r"^\s*[0-9a-f]+:\s*(.*)$")
RELOC = re.compile(r"^\s*[0-9a-f]+:\s*(R_\S+)\s+(.*)$")
HEX_TARGET = re.compile(r"\b[0-9a-f]+ <")
LC = re.compile(r"\.LC\d+")
OMP = re.compile(r"\._omp_fn\.\d+")
FPOP = re.compile(
    r"^(v?(add|sub|mul|div|sqrt|min|max|fmadd|fmsub|fnmadd|fnmsub|fmaddsub|"
    r"fmsubadd|hadd|hsub|round|cvt\w*)\w*(sd|pd|ss|ps)|v?xorp[sd]|v?andp[sd]|"
    r"v?andnp[sd])\b"
)
# A libm/libmvec call target: a bare C symbol, matched whole (demangled C++
# names are full of words like `const` that a substring search would take).
MATHCALL = re.compile(
    r"_ZGV\w+|(?:sin|cos|tan|exp|expm1|log|log1p|sqrt|cbrt|pow|atan2?|asin|acos|"
    r"sinh|cosh|tanh|asinh|acosh|atanh|hypot|sincos|erfc?|lgamma|tgamma|[jy][01n]|"
    r"fma|frexp|ldexp|c(?:exp|log|sqrt|abs|pow|sin|cos|tan|arg)|__(?:mul|div)[sdxt]c3)f?"
)
ADDEND = re.compile(r"[+-]0x[0-9a-f]+$")


def functions(obj):
    """{demangled name: normalised instruction lines} for one object."""
    text = subprocess.run(
        ["objdump", "-d", "-r", "-C", "--no-show-raw-insn", obj],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fns = collections.OrderedDict()
    cur = None
    for line in text.splitlines():
        m = FN.match(line)
        if m:
            name = OMP.sub("._omp_fn.N", m.group(1))
            key, i = name, 1
            while key in fns:
                i += 1
                key = f"{name}#{i}"
            cur = key
            fns[cur] = []
            continue
        if cur is None:
            continue
        r = RELOC.match(line)
        if r:
            tgt = re.sub(
                r"^(\.text[.\w]*|\.rodata[.\w]*)[+-]0x[0-9a-f]+", r"\1", r.group(2)
            )
            fns[cur].append(
                "RELOC " + r.group(1) + " " + LC.sub(".LC", OMP.sub("._omp_fn.N", tgt))
            )
            continue
        m = INSN.match(line)
        if m:
            s = HEX_TARGET.sub("<", m.group(1))
            s = re.sub(r"#\s*[0-9a-f]+ <", "# <", s)
            fns[cur].append(LC.sub(".LC", OMP.sub("._omp_fn.N", s)).strip())
    return fns


def fp_ops(lines):
    """The FP-op and math-call sequence of one function's stream."""
    out = []
    for line in lines:
        if line.startswith("RELOC"):
            _, kind, target = line.split(" ", 2)
            target = ADDEND.sub("", target)
            if ("PLT32" in kind or "GOTPCREL" in kind) and MATHCALL.fullmatch(target):
                out.append(target)
            continue
        mnemonic = line.split(None, 1)[0] if line else ""
        if FPOP.match(mnemonic):
            out.append(mnemonic)
    return out


def bare(name):
    """`ns::f<T>(args) [clone ._omp_fn.N]` -> `f<T> [clone ._omp_fn.N]`."""
    name = name.replace("(anonymous namespace)::", "")
    clone = ""
    m = re.search(r"( \[clone [^\]]*\])+(#\d+)?$", name)
    if m:
        clone, name = name[m.start() :], name[: m.start()]
    depth, cut = 0, len(name)
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            cut = i
            break
    head = name[:cut]
    depth, start = 0, 0
    for i, ch in enumerate(head):  # drop a return type: last depth-0 token
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == " " and depth == 0:
            start = i + 1
    head = head[start:]
    depth, start = 0, 0
    for i in range(len(head) - 1):  # drop namespaces/classes at depth 0
        ch = head[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif head[i : i + 2] == "::" and depth == 0:
            start = i + 2
    return head[start:] + clone


def glue(name):
    return "pybind11" in name or "PyInit" in name or name.startswith("std::")


def verdict(a, b):
    if a == b:
        return "same instructions"
    fa, fb = fp_ops(a), fp_ops(b)
    if fa == fb:
        return f"same FP-op sequence ({len(fa)} ops; {len(a)} vs {len(b)} insns)"
    ca, cb = collections.Counter(fa), collections.Counter(fb)
    if ca == cb:
        return f"same FP-op multiset, rescheduled ({len(fa)} ops)"
    delta = {k: cb[k] - ca[k] for k in sorted(set(ca) | set(cb)) if ca[k] != cb[k]}
    return f"FP ops DIFFER ({len(fa)} vs {len(fb)}): {delta}"


def compare(fa, fb, label, report=None):
    same = [n for n in fa if n in fb and fa[n] == fb[n]]
    diff = [n for n in fa if n in fb and fa[n] != fb[n]]
    fp_diff = [n for n in diff if fp_ops(fa[n]) != fp_ops(fb[n])]
    only_a = [n for n in fa if n not in fb and not glue(n)]
    only_b = [n for n in fb if n not in fa and not glue(n)]
    print(
        f"== {label}: {len(same)} same, {len(diff)} differ "
        f"({len(fp_diff)} with a different FP-op sequence), "
        f"{len(only_a)} only in A, {len(only_b)} only in B (pybind11/std glue not counted)"
    )
    for n in fp_diff[:40]:
        print("   FP-DIFF", n[:160], "--", verdict(fa[n], fb[n]))
    for n in only_a[:15]:
        print("   ONLY-A ", n[:160])
    for n in only_b[:15]:
        print("   ONLY-B ", n[:160])
    if report is not None:
        for n in fa:
            if n in fb and re.search(report, n):
                print(f"   {n[:100]:100s} {verdict(fa[n], fb[n])}")
    return len(fp_diff)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="*", help="DIR_A DIR_B")
    p.add_argument("--pair", nargs=2, action="append", metavar=("A.o", "B.o"))
    p.add_argument("--rename", action="append", default=[], metavar="REGEX=REPL")
    p.add_argument("--report", metavar="REGEX")
    args = p.parse_args()
    total = 0
    if args.dirs:
        a, b = args.dirs
        for oa in sorted(glob.glob(os.path.join(a, "*.o"))):
            ob = os.path.join(b, os.path.basename(oa))
            if os.path.exists(ob):
                total += compare(
                    functions(oa), functions(ob), os.path.basename(oa), args.report
                )
            else:
                print("== only in A:", os.path.basename(oa))
        for ob in sorted(glob.glob(os.path.join(b, "*.o"))):
            if not os.path.exists(os.path.join(a, os.path.basename(ob))):
                print("== only in B:", os.path.basename(ob))
    renames = [r.split("=", 1) for r in args.rename]
    for oa, ob in args.pair or []:
        fa = collections.OrderedDict()
        for n, lines in functions(oa).items():
            k = bare(n)
            for pat, repl in renames:
                k = re.sub(pat, repl, k)
            fa.setdefault(k, lines)
        fb = collections.OrderedDict()
        for n, lines in functions(ob).items():
            fb.setdefault(bare(n), lines)
        label = f"{os.path.basename(oa)} vs {os.path.basename(ob)} (bare names)"
        total += compare(fa, fb, label, args.report)
    print("functions with a different FP-op sequence:", total)


if __name__ == "__main__":
    main()
