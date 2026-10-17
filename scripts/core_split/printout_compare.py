"""The C++ NEC-5 printout for razor-2p (`_nec5_razor_printout_core.cpp`, unit
3a) against `momwire-nec5-razor-2p`'s, deck by deck, whole file against
whole file.

    python scripts/core_split/printout_compare.py OUT.jsonl DECK_OR_DIR...
    python scripts/core_split/printout_compare.py OUT.jsonl --texts HARVEST.jsonl
    python scripts/core_split/printout_compare.py OUT.jsonl --golden GOLDEN_REPO

(`--texts` reads deck texts as JSON strings, one per line, as
`nec5_harvest_plugin.py` records them; `--golden` compares against the
stored razor-2p printouts of a checkout that carries
`scripts/golden_printouts.py` (momwire#1473), engine stamp
masked as they mask it, instead of a fresh Python render.)

The C++ side is `nec5_razor_printout_driver` (built here from the core TUs
with setup.py's Linux flags, as `tests/test_razor_core_standalone.py` builds
it, unless `--driver EXE` names one): deck file in, printout bytes out, no
Python. The Python side is what the launcher writes: `_shell.render(text,
basis="razor-2p")` (the default, Python route), or the last line of defence's
INTERNAL ERROR printout when it raises, through `write_printout`'s CRLF and
latin-1 transliteration. Line 2's stamp fields are Python's own, handed to
the driver.

Each deck is counted by the C++ route:

* ``printed`` -- the whole printout must be Python's, byte for byte; counted
  by what the structure half answered (served / refused / raised) and by
  whether it came through the INTERNAL ERROR frame;
* ``declined`` -- counted per reason, never compared.

One JSON line per deck; a summary on stdout; exit status 1 on any mismatch.
Threads are whatever OMP_NUM_THREADS / OPENBLAS_NUM_THREADS say (the same on
both sides: the driver inherits the environment).
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import sysconfig
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import momwire_serve_client

# AVX-512 off before numpy loads, in this process (its in-process python
# reference) and in every child (the launcher, the executable's fallback):
# exe_compare / stream_compare hold all of them to one byte, and GitHub's
# runners can have AVX-512, where the launcher pins itself and a bare
# in-process render would not.
momwire_serve_client.pin_cpu_env()

import numpy as np  # noqa: E402 - after the AVX-512 pin

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "src" / "momwire"
SPLIT = ROOT / "scripts" / "core_split"
BASIS = "razor-2p"
DRIVER = "nec5_razor_printout_driver"
# momwire#1517 step 1: `--basis bspline` builds bs2's driver,
# which also links the bspline front, fill and engine TUs.
BSPLINE_SOURCES = (
    "_nec5_bspline_seam_core.cpp",
    "_bspline_front_core.cpp",
    "_bspline_fill_core.cpp",
    "_nec5_bspline_printout_core.cpp",
)


def _decks(args):
    out = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            out += sorted(
                q for q in p.rglob("*") if q.is_file() and q.suffix in (".nec", ".deck")
            )
        else:
            out.append(p)
    return out


def scipy_openblas():
    import scipy

    found = sorted(
        glob.glob(str(Path(scipy.__file__).parent.parent / "scipy.libs" / "*openblas*"))
    )
    return found[0] if found else None


def numpy_openblas():
    """numpy's bundled ILP64 OpenBLAS: the library its matmul and
    `linalg.solve` call."""
    import numpy

    found = sorted(
        glob.glob(
            str(Path(numpy.__file__).parent.parent / "numpy.libs" / "*openblas64*")
        )
    )
    return found[0] if found else None


def stamp_fields():
    """Line 2's version, variant and overlay marker, as this process stamps them."""
    from momwire.eznec import _printout

    version = _printout._stamp_field(_printout._momwire_version()) or "unknown"
    variant = _printout._stamp_field(_printout._accelerator_variant()) or "none"
    return version, variant, _printout._overlay_marker(version)


ARRAYS = ("y", "coeffs", "v_applied", "v_gap", "i_port", "i_source")


def python_printout(text: str, basis: str = BASIS, runs: list | None = None) -> bytes:
    """`momwire-nec5-<basis>`'s printout file for this deck text, as bytes.

    With ``runs`` (a list), every run `_serve._finish` finishes appends its
    arrays there, in execution order (`ARRAYS`: the solve's Y, the
    coefficients `_finish` reads its currents from, `_port_state`'s four
    vectors)."""
    from momwire.eznec import _serve, _shell

    orig_finish, orig_state = _serve._finish, _serve._port_state
    if runs is not None:
        seen = {}

        def port_state(*args, **kwargs):
            seen["state"] = state = orig_state(*args, **kwargs)
            return state

        def finish(deck, prepared):
            data = orig_finish(deck, prepared)
            st = seen.pop("state")
            coeffs = prepared.solution.coeffs @ (
                _serve._transform(prepared.mesh) @ st.v_gap
            )
            runs.append(
                dict(
                    y=prepared.solution.y,
                    coeffs=coeffs,
                    v_applied=st.v_applied,
                    v_gap=st.v_gap,
                    i_port=st.i_port,
                    i_source=st.i_source,
                )
            )
            return data

        _serve._port_state, _serve._finish = port_state, finish
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = _shell.render(text, basis=basis)
            except Exception as exc:  # noqa: BLE001 - `_shell.main`'s last line of defence
                out = _shell.seam_failure(text, exc, basis=basis)[0]
    finally:
        _serve._port_state, _serve._finish = orig_state, orig_finish
    return out.replace("\n", "\r\n").encode("latin-1", errors=_shell._PRINTOUT_ERRORS)


def _standalone_test_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_standalone_test", ROOT / "tests" / "test_razor_core_standalone.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_driver(out_dir: Path, basis: str = "razor-2p") -> Path:
    """The driver, from the core TUs alone, with the loaded variant's flags
    (the standalone test's compile)."""
    from momwire import _accel

    t = _standalone_test_module()
    compile_args, link_args, avx2_only = t._linux_flags()
    if _accel.acc is not None and _accel.acc.__name__.endswith("_sse2"):
        compile_args = [a for a in compile_args if a not in avx2_only]
    cxx = shlex.split(os.environ.get("CXX", "g++"))
    base = [
        *cxx,
        *shlex.split(sysconfig.get_config_var("CFLAGS") or ""),
        *shlex.split(sysconfig.get_config_var("CCSHARED") or ""),
        "-fvisibility=hidden",
        "-g0",
        *compile_args,
        "-I",
        str(PKG),
        "-I",
        str(ROOT / "extern" / "xsf" / "include"),
    ]
    driver = DRIVER
    extra = ()
    if basis == "bspline":
        driver, extra = "nec5_bspline_printout_driver", BSPLINE_SOURCES
    sources = (
        [PKG / s for s in t.CORE_SOURCES]
        + [PKG / s for s in extra]
        + [SPLIT / f"{driver}.cpp"]
    )
    procs, objects = [], []
    for src in sources:
        obj = out_dir / (src.stem + ".o")
        cmd = [*base, "-c", str(src), "-o", str(obj)]
        procs.append((cmd, subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)))
        objects.append(str(obj))
    for cmd, proc in procs:
        _, err = proc.communicate()
        if proc.returncode:
            raise SystemExit(f"{shlex.join(cmd)}\n{err}")
    exe = out_dir / driver
    cmd = [*cxx, *objects, *link_args, "-ldl", "-o", str(exe)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode:
        raise SystemExit(f"{shlex.join(cmd)}\n{run.stderr}")
    return exe


def run_driver(exe, decks, out_dir, basis=BASIS, stamp=None, arrays=False):
    """The C++ answers: ``[dict(route, seam, raised, reason, printout)]``,
    and with ``arrays`` each finished run's `ARRAYS` bytes (``runs``)."""
    version, variant, marker = stamp or stamp_fields()
    scipy_lib, numpy_lib = scipy_openblas(), numpy_openblas()
    if scipy_lib is None or numpy_lib is None:
        raise SystemExit(
            "no bundled OpenBLAS beside SciPy / numpy (not a wheel install)"
        )
    answers = []
    # in chunks: one process per few hundred decks keeps argv short
    for start in range(0, len(decks), 200):
        chunk = decks[start : start + 200]
        sub = Path(out_dir) / str(start)
        sub.mkdir(parents=True, exist_ok=True)
        run = subprocess.run(
            [
                str(exe),
                scipy_lib,
                numpy_lib,
                basis,
                version,
                variant,
                marker or "-",
                str(sub),
                *map(str, chunk),
            ],
            capture_output=True,
            env=dict(os.environ, MW_PRINTOUT_ARRAYS="1" if arrays else "0"),
        )
        if run.returncode:
            raise SystemExit(run.stderr.decode(errors="replace"))
        blocks = run.stdout.decode("ascii").split("== ")[1:]
        assert len(blocks) == len(chunk), (len(blocks), len(chunk))
        for i, (path, block) in enumerate(zip(chunk, blocks, strict=True)):
            lines = block.split("\n")
            assert lines[0] == str(path), (lines[0], path)
            field = {
                line.split(" ", 1)[0]: line.split(" ", 1)[1] if " " in line else ""
                for line in lines[1:6]
                if line
            }
            out = sub / f"{i}.out"
            answers.append(
                dict(
                    route=field["route"],
                    seam=field["seam"],
                    raised=field["raised"],
                    reason=bytes.fromhex(field["reason"]).decode("utf-8"),
                    printout=out.read_bytes() if field["route"] == "printed" else None,
                    runs=[
                        {a: (sub / f"{i}_{k}_{a}.bin").read_bytes() for a in ARRAYS}
                        for k in range(int(field.get("arrays", "0")))
                    ],
                )
            )
    return answers


def first_difference(a: bytes, b: bytes) -> str:
    al, bl = a.split(b"\r\n"), b.split(b"\r\n")
    for k, (x, y) in enumerate(zip(al, bl)):
        if x != y:
            return f"line {k + 1}: python {x!r} / c++ {y!r}"
    return f"line counts {len(al)} / {len(bl)}"


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    out_path = Path(args.pop(0))
    driver = None
    if "--driver" in args:
        k = args.index("--driver")
        driver = Path(args[k + 1])
        del args[k : k + 2]
    golden = None
    basis = BASIS
    if "--basis" in args:
        k = args.index("--basis")
        basis = args[k + 1]
        del args[k : k + 2]
    arrays = "--arrays" in args
    if arrays:
        args.remove("--arrays")
    below_bridge = "--below-bridge" in args
    if below_bridge:
        args.remove("--below-bridge")
    work = Path(tempfile.mkdtemp(prefix="printout_compare_"))
    if args and args[0] == "--texts":
        decks, labels = [], []
        for n, line in enumerate(Path(args[1]).read_text().splitlines()):
            text = json.loads(line)
            try:
                data = text.encode("latin-1")
            except UnicodeEncodeError:
                continue  # not a deck FILE: the launcher reads latin-1 bytes
            p = work / "texts" / f"{n}.nec"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            decks.append(p)
            labels.append(f"{args[1]}:{n + 1}")
    elif args and args[0] == "--golden":
        golden = Path(args[1]).resolve()
        sys.path.insert(0, str(golden / "scripts"))
        import golden_printouts as gp  # noqa: PLC0415 - the generator's own case list

        decks = [golden / "tests" / "fixtures" / d for d in gp.DECKS]
        labels = list(gp.DECKS)
    else:
        decks = _decks(args)
        labels = [str(d) for d in decks]
    if driver is None:
        (work / "build").mkdir()
        driver = build_driver(work / "build", basis)
    # `--below-bridge` (momwire#1512, until the below table is ported): the
    # Python route renders every deck FIRST, and the below/below grids it
    # filled are handed to the driver (`MW_TEST_BELOW_TABLES`), so a buried
    # deck is printed by C++ end to end on Python's tables.
    if below_bridge:
        sys.path.insert(0, str(SPLIT))
        import buried_z_compare as bz  # noqa: PLC0415 - the gate's own dumper

        for d in decks:  # fills the grids; the comparison renders again
            python_printout(d.read_bytes().decode("latin-1"), basis=basis)
        bz.dump_below_grids(work / "bridge")
        os.environ["MW_TEST_BELOW_TABLES"] = str(work / "bridge")
    answers = run_driver(driver, decks, work / "out", basis=basis, arrays=arrays)
    runs_compared = 0

    counts, declined, mismatches = Counter(), Counter(), 0
    with open(out_path, "w") as out:
        for label, deck, ans in zip(labels, decks, answers, strict=True):
            rec = {"deck": label, "route": ans["route"], "seam": ans["seam"]}
            if ans["route"] == "declined":
                declined[ans["reason"]] += 1
                counts["declined"] += 1
                rec["reason"] = ans["reason"]
            else:
                frame = f"|internal-error:{ans['raised']}" if ans["raised"] else ""
                key = f"printed:{ans['seam']}{frame}"
                if golden is not None:
                    from golden_printouts import golden_path, mask_stamp  # noqa: PLC0415

                    ref = golden_path(label, basis).read_bytes().decode("latin-1")
                    mine = mask_stamp(
                        ans["printout"].decode("latin-1").replace("\r\n", "\n"), basis
                    )
                    same = ref == mine
                    if not same:
                        rec["difference"] = first_difference(
                            ref.replace("\n", "\r\n").encode("latin-1"),
                            mine.replace("\n", "\r\n").encode("latin-1"),
                        )
                else:
                    py_runs = [] if arrays else None
                    ref = python_printout(
                        deck.read_bytes().decode("latin-1"), basis=basis, runs=py_runs
                    )
                    same = ref == ans["printout"]
                    if not same:
                        rec["difference"] = first_difference(ref, ans["printout"])
                    if arrays and ans["raised"] == "":
                        # Every run either side finished, array for array.
                        if len(py_runs) != len(ans["runs"]):
                            same = False
                            rec["arrays"] = f"runs {len(py_runs)} / {len(ans['runs'])}"
                        for k, (pr, cr) in enumerate(zip(py_runs, ans["runs"])):
                            runs_compared += 1
                            for a in ARRAYS:
                                mine = np.ascontiguousarray(pr[a], dtype=np.complex128)
                                if mine.tobytes() != cr[a]:
                                    same = False
                                    rec["arrays"] = f"run {k}: {a} differs"
                        rec["runs"] = len(ans["runs"])
                rec["same"] = same
                if same:
                    counts[key] += 1
                else:
                    mismatches += 1
                    counts[key + "|MISMATCH"] += 1
            out.write(json.dumps(rec) + "\n")
    print(
        f"decks {len(decks)}: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    )
    if declined:
        print(
            "declined by reason: "
            + ", ".join(f"{k} {v}" for k, v in declined.most_common())
        )
    if arrays:
        print(f"runs whose arrays were compared ({', '.join(ARRAYS)}): {runs_compared}")
    print(f"mismatches {mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
