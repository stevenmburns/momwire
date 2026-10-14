"""Compare the machine code of two builds' object files, function by function.

    python objdiff.py <tempdirA> <tempdirB> [--tu NAME ...] [--map A.o=B.o]

Each object is disassembled (objdump -d -r -C --no-show-raw-insn); each
function's instruction stream is normalised (addresses and in-section branch
targets dropped, symbolic targets and relocations kept, .LC numbering erased)
and compared. Identical streams mean identical arithmetic: the build change
cannot have moved a bit of that function's output.
"""

import collections
import glob
import os
import re
import subprocess
import sys

FN = re.compile(r"^[0-9a-f]+ <(.*)>:$")
INSN = re.compile(r"^\s*[0-9a-f]+:\s*(.*)$")
RELOC = re.compile(r"^\s*[0-9a-f]+:\s*(R_\S+)\s+(.*)$")
HEX_TARGET = re.compile(r"\b[0-9a-f]+ <")
LC = re.compile(r"\.LC\d+")
OMP = re.compile(r"\._omp_fn\.\d+")


def funcs(obj):
    out = subprocess.run(
        ["objdump", "-d", "-r", "-C", "--no-show-raw-insn", obj],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fns = collections.OrderedDict()
    cur = None
    for line in out.splitlines():
        m = FN.match(line)
        if m:
            cur = OMP.sub("._omp_fn.N", m.group(1))
            k, i = cur, 1
            while k in fns:
                i += 1
                k = f"{cur}#{i}"
            cur = k
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
            s = m.group(1)
            s = HEX_TARGET.sub("<", s)
            s = re.sub(r"#\s*[0-9a-f]+ <", "# <", s)
            s = LC.sub(".LC", OMP.sub("._omp_fn.N", s))
            fns[cur].append(s.strip())
    return fns


FPOP = re.compile(
    r"^(v?(add|sub|mul|div|sqrt|min|max|fmadd|fmsub|fnmadd|fnmsub|fmaddsub|fmsubadd|hadd|hsub|round|cvt\w*)\w*(sd|pd|ss|ps)|v?xorp[sd]|v?andp[sd]|v?andnp[sd])\b"
)


MATHCALL = re.compile(
    r"\b(_ZGV\w+|(sin|cos|tan|exp|expm1|log|log1p|sqrt|cbrt|pow|atan2?|asin|acos|sinh|cosh|tanh|asinh|acosh|atanh|hypot|sincos|erf|erfc|lgamma|tgamma|j0|j1|y0|y1|jn|yn|fma|frexp|ldexp|c\w+|__\w+c3)f?)\b"
)


def fpseq(lines):
    out = []
    for l in lines:
        if l.startswith("RELOC"):
            if ("PLT32" in l or "GOTPCREL" in l) and MATHCALL.search(l):
                out.append(l)
            continue
        mn = l.split(None, 1)[0] if l else ""
        if FPOP.match(mn):
            out.append(mn)
    return out


def glue(n):
    return "pybind11" in n or "PyInit" in n or "pybind" in n or n.startswith("std::")


def main():
    a, b = sys.argv[1], sys.argv[2]
    pairs = []
    maps = [x.split("=", 1)[1] for x in sys.argv[3:] if x.startswith("--map=")]
    for p in maps:
        x, y = p.split(",")
        pairs.append((os.path.join(a, x), os.path.join(b, y)))
    if not maps:
        for oa in sorted(glob.glob(os.path.join(a, "*.o"))):
            ob = os.path.join(b, os.path.basename(oa))
            if os.path.exists(ob):
                pairs.append((oa, ob))
    tot = collections.Counter()
    for oa, ob in pairs:
        fa, fb = funcs(oa), funcs(ob)
        same = [n for n in fa if n in fb and fa[n] == fb[n]]
        diff = [n for n in fa if n in fb and fa[n] != fb[n]]
        only_a = [n for n in fa if n not in fb]
        only_b = [n for n in fb if n not in fa]
        kd = [n for n in diff if fpseq(fa[n]) != fpseq(fb[n])]
        print(
            f"== {os.path.basename(oa)} vs {os.path.basename(ob)}: {len(same)} same, "
            f"{len(diff)} differ ({len(kd)} with a different FP-op/call sequence), "
            f"{len(only_a)} only-A, {len(only_b)} only-B"
        )
        for n in kd[:40]:
            print("   DIFF", n[:160], f"({len(fa[n])} vs {len(fb[n])} insns)")
        for n in [x for x in only_a if not glue(x)][:15]:
            print("   ONLY-A", n[:160])
        for n in [x for x in only_b if not glue(x)][:15]:
            print("   ONLY-B", n[:160])
        tot["same"] += len(same)
        tot["diff"] += len(diff)
        tot["kdiff"] += len(kd)
    print("TOTAL", dict(tot))


if __name__ == "__main__":
    main()


def show(oa, ob, name, ctx=3):
    import difflib

    fa, fb = funcs(oa), funcs(ob)
    ks = [k for k in fa if k.startswith(name)]
    for k in ks:
        if k in fb and fa[k] != fb[k]:
            print("####", k[:150])
            for l in list(difflib.unified_diff(fa[k], fb[k], lineterm="", n=ctx))[:80]:
                print(l)


def showfp(oa, ob, prefix, maxl=40):
    fa, fb = funcs(oa), funcs(ob)
    for k in fa:
        if k.startswith(prefix) and k in fb and fpseq(fa[k]) != fpseq(fb[k]):
            print("####", k[:150], len(fpseq(fa[k])), len(fpseq(fb[k])))
            sa, sb = fpseq(fa[k]), fpseq(fb[k])
            import collections

            ca, cb = collections.Counter(sa), collections.Counter(sb)
            print(
                "   op-count delta:",
                {m: cb[m] - ca[m] for m in set(ca) | set(cb) if cb[m] != ca[m]},
            )
            return
