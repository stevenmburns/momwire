"""razor's core TUs build and run with no Python: the core/binding split's contract.

`_razor_core.cpp` and `_razor_t2_core.cpp` hold razor's kernels behind
`mw::view` signatures, and `_accel_razor.cpp` / `_accel_razor_t2.cpp` are only
their pybind11 bindings. The split is worth something only while the core
needs nothing from Python, and nothing in the extension build would notice
the day a core TU picked up a pybind11 include: setuptools hands every TU of
the extension the Python and pybind11 include paths.

So this test compiles the core TUs itself, with setup.py's own flags for the
loaded variant and the same compiler command setuptools uses, but WITHOUT
those include paths, and checks the dependency list besides. It then links
`scripts/core_split/razor_chain_driver.cpp` against them, runs a real razor
row window (committed in tests/fixtures, captured by
`scripts/core_split/capture_window.py --nsegs 12`) through
seg_moments -> assemble_t1 -> t2_rows in C++ alone, and requires every byte
of M0, M1, T1 and the Z rows to equal what the bound kernels return for the
same inputs. Same flags, same sources, so the same bits: a difference means
the extension's core objects and the standalone ones are not the same code.

Linux only: the flags are read from setup.py's Linux branch, and the macOS and
Windows lanes build the extension through their own toolchains.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import shlex
import subprocess
import sys
import sysconfig

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "src" / "momwire"
SPLIT = ROOT / "scripts" / "core_split"
FIXTURE = ROOT / "tests" / "fixtures" / "razor_window_n12.npz"
# `_razor_solve_core.cpp` is the whole-solve prototype's driver of the other
# two, and `_razor_front_core.cpp` its front half (construction, geometry,
# the prepared tables, the port columns); both are compiled and linked here
# so that they, too, stay Python-free.
CORE_SOURCES = (
    "_razor_core.cpp",
    "_razor_t2_core.cpp",
    "_accel_razor_cplx.cpp",
    "_razor_solve_core.cpp",
    "_razor_front_core.cpp",
    "_razor_buried_core.cpp",
    # The exact ring kernel (momwire#1479): razor's plan and correction, and
    # the point moments' Python-free twin of the extension's TU.
    "_razor_exact_core.cpp",
    "_exact_kernel_point_core.cpp",
    # The B-spline basis' exact kernel: the coaxial pair moments and their
    # row windows (momwire#1517 step 0), the extension's 1410 text.
    "_exact_kernel_coaxial_core.cpp",
    # The B-spline kernels (momwire#1517 step 0): the extension's bspline
    # TU's text over `_mw_nd.h`'s arrays.
    "_bspline_core.cpp",
    # The NEC-5 deck reader (unit 2a): the executable's first step, card text
    # to the parsed deck, linked into `nec5_deck_driver.cpp` below.
    "_nec5_deck_core.cpp",
    # The nec2 / nec4 deck reader (momwire#1607 A1): the NEC-4.2-dialect
    # executables' first step, linked into `nec4_deck_driver.cpp` below.
    "_nec4_deck_core.cpp",
    # The NEC-4.2 seam front over it (momwire#1607 A2), linked into
    # `nec4_seam_driver.cpp`.
    "_nec4_seam_core.cpp",
    # The seam's structure half (unit 2b): the parsed deck to RazorSolver's
    # arguments, linked into `nec5_razor_driver.cpp` (deck text -> currents);
    # the engine-neutral seam and razor-2p's engine over it.
    "_nec5_seam_core.cpp",
    "_nec5_razor_seam_core.cpp",
    # The run half and the printout writer (unit 3a): deck text to the
    # printout's bytes, linked into `nec5_razor_printout_driver.cpp`.
    "_nec5_printout_core.cpp",
    "_nec5_razor_printout_core.cpp",
    # Wire loading (LD 5 / 2 / 3): the vendored xsf's Bessel ratio.
    "_razor_wire_loading_core.cpp",
    # The composing Sommerfeld ground (GN 0 / GN 2): the six-integral
    # kernel, the grid, and the remainder's projection and moments.
    "_somm_kernel_core.cpp",
    "_somm_grid_core.cpp",
    "_somm_proj_core.cpp",
    "_somm_below_proj_core.cpp",
    "_somm_below_grid_core.cpp",
    "_somm_below_fill_core.cpp",
    "_near_interface_core.cpp",
    "_near_interface_eval.cpp",
    "_razor_crossing_core.cpp",
    # The near-field remainder over a finite ground (S4).
    "_somm_field_core.cpp",
    # The B-spline remainder block's kernel (momwire#1517 step 3).
    "_somm_remainder_core.cpp",
    "_somm_pairs_core.cpp",
    # bs2's NEC-5 seam, front half, fill and run half (momwire#1517 step 1):
    # deck text to printout bytes for the B-spline basis, linked into
    # `nec5_bspline_printout_driver.cpp`.
    "_nec5_bspline_seam_core.cpp",
    "_bspline_front_core.cpp",
    "_bspline_fill_core.cpp",
    "_bspline_crossing_core.cpp",
    "_nec5_bspline_printout_core.cpp",
)
DECK = ROOT / "tests" / "fixtures" / "razor_front_deck.txt"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="flags are read from setup.py's Linux branch"
)


def _linux_flags():
    """(compile args, link args, AVX2-only flags) from setup.py, without
    importing it: the `else:` branch of the win32/darwin/else chain."""
    tree = ast.parse((ROOT / "setup.py").read_text())
    compile_args = link_args = avx2_only = None
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and "win32" in ast.unparse(node.test)
            and isinstance(node.orelse[0], ast.If)
        ):
            for stmt in node.orelse[0].orelse:
                if isinstance(stmt, ast.Assign) and isinstance(
                    stmt.targets[0], ast.Name
                ):
                    if stmt.targets[0].id == "extra_compile_args":
                        compile_args = ast.literal_eval(stmt.value)
                    elif stmt.targets[0].id == "extra_link_args":
                        link_args = ast.literal_eval(stmt.value)
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_AVX2_ONLY_FLAGS"
        ):
            avx2_only = ast.literal_eval(node.value)
    assert compile_args and link_args and avx2_only, "setup.py's Linux flags moved"
    return compile_args, link_args, avx2_only


def _capture_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_capture_window", SPLIT / "capture_window.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def accel():
    from momwire import _accel

    if _accel.acc is None:
        pytest.skip("no compiled accelerator loaded")
    return _accel.acc


@pytest.fixture(scope="module")
def driver(accel, tmp_path_factory):
    """The chain driver, built from the core TUs alone with the loaded
    variant's flags; and the dependency files the compile wrote.

    The command is the extension build's, as setuptools spells it for a
    Pybind11Extension: CXX, Python's CFLAGS and CCSHARED, pybind11's two
    additions, then setup.py's flags. Only the include paths are missing.
    """
    compile_args, link_args, avx2_only = _linux_flags()
    if accel.__name__.endswith("_sse2"):
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
    out = tmp_path_factory.mktemp("razor_core")
    drivers = (
        "razor_chain_driver",
        "razor_front_driver",
        "nec5_deck_driver",
        "nec4_deck_driver",
        "nec4_seam_driver",
        "nec5_razor_driver",
        "nec5_razor_printout_driver",
        "nec5_bspline_printout_driver",
        "exact_kernel_coaxial_driver",
        "bspline_core_driver",
    )
    sources = [PKG / s for s in CORE_SOURCES] + [SPLIT / f"{d}.cpp" for d in drivers]
    objects, deps, procs = {}, [], []
    for src in sources:  # concurrently: the -O3 core TUs dominate the cost
        obj, dep = out / (src.stem + ".o"), out / (src.stem + ".d")
        cmd = [*base, "-MD", "-MF", str(dep), "-c", str(src), "-o", str(obj)]
        procs.append((cmd, subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)))
        objects[src.stem] = str(obj)
        deps.append(dep)
    for cmd, proc in procs:
        _, err = proc.communicate()
        assert proc.returncode == 0, f"{shlex.join(cmd)}\n{err}"
    core = [objects[pathlib.Path(s).stem] for s in CORE_SOURCES]
    exes = {}
    for d in drivers:
        exes[d] = out / d
        cmd = [*cxx, *core, objects[d], *link_args, "-ldl", "-o", str(exes[d])]
        run = subprocess.run(cmd, capture_output=True, text=True)
        assert run.returncode == 0, f"{shlex.join(cmd)}\n{run.stderr}"
    return exes, deps


def test_the_core_includes_no_python_header(driver):
    """No header the core TUs or the driver pulled in is Python's or
    pybind11's (the compile above had neither include path, so this is
    belt and braces against a system-wide pybind11 install)."""
    _, deps = driver
    seen = " ".join(d.read_text() for d in deps)
    assert "_razor_core.h" in seen, "the dependency files are not what they seem"
    assert "_razor_solve_core.h" in seen, "the solve prototype's core was not built"
    assert "_razor_front_core.h" in seen, "the front half's core was not built"
    assert "_nec5_deck_core.h" in seen, "the deck reader's core was not built"
    assert "_nec4_deck_core.h" in seen, "the nec2 / nec4 reader's core was not built"
    assert "_nec4_seam_core.h" in seen, "the NEC-4.2 seam's core was not built"
    assert "_nec5_razor_seam_core.h" in seen, "the seam's core was not built"
    assert "_bspline_core.h" in seen, "the B-spline kernels' core was not built"
    for marker in ("Python.h", "pybind11", "numpy"):
        assert marker not in seen, f"a core TU includes {marker}"


def test_the_driver_links_no_python(driver):
    exes, _ = driver
    for exe in exes.values():
        libs = subprocess.run(["ldd", str(exe)], capture_output=True, text=True).stdout
        assert "libpython" not in libs, libs


@pytest.mark.parametrize("dialect", ["nec4", "nec2"])
def test_the_nec4_deck_reader_in_cxx_alone_is_pythons(driver, dialect):
    """momwire#1607 A1: the nec2 / nec4 reader, built from the core TUs alone
    (`nec4_deck_driver`), reads every NEC deck in tests/fixtures as
    `momwire.deck.parse(text, dialect)` does, field for field (the
    extension-bound twin of this check is `tests/test_nec4_deck_core.py`).
    Only the three deliberately non-finite / oversized fixtures decline."""
    spec = importlib.util.spec_from_file_location(
        "core_split_nec4_deck_compare", SPLIT / "nec4_deck_compare.py"
    )
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)
    decks = [
        p
        for p in sorted((ROOT / "tests" / "fixtures").rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]
    counts, mismatches = nc.compare(decks, dialect, driver[0]["nec4_deck_driver"])
    assert not mismatches, mismatches[:3]
    assert counts["parsed"] >= 200 and counts["refused"] >= 100, counts
    assert counts["declined"] <= 3, counts


@pytest.mark.parametrize("basis", ["sinusoidal", "bspline"])
def test_the_nec4_seam_in_cxx_alone_is_pythons(driver, basis):
    """momwire#1607 A2: the NEC-4.2 slot's front (`_nec4.render` up to the
    solve), built from the core TUs alone (`nec4_seam_driver`), serves and
    refuses every NEC deck in tests/fixtures as the slot does, the printout's
    view of the cards field for field (the extension-bound twin of this
    check is `tests/test_nec4_seam_core.py`)."""
    spec = importlib.util.spec_from_file_location(
        "core_split_nec4_seam_compare", SPLIT / "nec4_seam_compare.py"
    )
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)
    decks = [
        p
        for p in sorted((ROOT / "tests" / "fixtures").rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]
    counts, mismatches = nc.compare(decks, basis, driver[0]["nec4_seam_driver"])
    assert not mismatches, mismatches[:3]
    assert counts["served"] >= 200 and counts["refused"] >= 100, counts
    assert counts["declined"] <= 3, counts


@pytest.mark.parametrize("prefix", ["", "ek_"], ids=["reduced", "extended_kernel"])
def test_the_cxx_chain_is_the_python_chain_bit_for_bit(driver, accel, prefix, tmp_path):
    exe = driver[0]["razor_chain_driver"]
    cap = _capture_module()
    w = np.load(FIXTURE)
    ek = w[prefix + "sm_group_i"].size > 0
    assert ek == (prefix == "ek_"), "the fixture's windows are not the ones named"
    M0, M1, T1, rows = cap.python_chain(accel, w, prefix)
    cap.write_driver_inputs(tmp_path, w, prefix)
    run = subprocess.run([str(exe), str(tmp_path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    for name, ref, order in (
        ("M0", M0, "C"),
        ("M1", M1, "C"),
        ("t1", T1, "C"),
        ("rows", rows, "F"),  # Z's own layout, column-major
    ):
        assert ref.size > 0
        mine = (tmp_path / f"drv_{name}.bin").read_bytes()
        assert mine == np.ravel(ref, order=order).tobytes(), (
            f"{name}: the C++ chain and the Python chain disagree"
        )


# --- geometry -> currents, no Python -------------------------------------


def _scipy_openblas():
    """SciPy's bundled OpenBLAS: the library whose zgetrf/zgetrs the Python
    route reaches (`cython_lapack`)."""
    import glob

    import scipy

    found = sorted(
        glob.glob(
            str(
                pathlib.Path(scipy.__file__).parent.parent / "scipy.libs" / "*openblas*"
            )
        )
    )
    if not found:
        pytest.skip("no bundled OpenBLAS beside SciPy (not a wheel install)")
    return found[0]


# One BLAS / OpenMP thread on both sides of a byte comparison. OpenBLAS's LU
# is deterministic at a given thread count but not ACROSS thread counts once
# the matrix is big enough to thread (measured: a 104-unknown deck gives
# other bits at 1 and 16 threads, the same on the Python route and in the
# driver), and the driver inherits the ENVIRONMENT while the in-process
# library was sized when it loaded -- which can disagree, because a test
# module may export a width at import (tests/test_momwire.py exports 8).
_ONE_THREAD = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def _one_thread():
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=1)


@pytest.fixture(autouse=True)
def _one_blas_thread_everywhere(monkeypatch):
    """Every byte comparison here at one thread on BOTH sides: the driver's
    environment (it inherits ours, and tests/test_momwire.py exports a width of
    8 when it is collected) and this process's library."""
    for k, v in _ONE_THREAD.items():
        monkeypatch.setenv(k, v)
    with _one_thread():
        yield


def _read_deck(path):
    """The fixture deck as RazorSolver's keyword arguments (razor-2p)."""
    kw = dict(
        wires=[],
        n_per_edge_per_wire=[],
        feeds=[],
        node_gaps=[],
        nec5_quadrature=True,
        extended_kernel=False,
        exact_kernel="auto",
    )
    for line in path.read_text().splitlines():
        line = line.split("#")[0].split()
        if not line:
            continue
        key, rest = line[0], line[1:]
        if key == "wavelength":
            kw["wavelength"] = float(rest[0])
        elif key == "radius":
            r = [float(v) for v in rest]
            kw["wire_radius"] = r[0] if len(r) == 1 else r
        elif key == "ground":
            kw["ground_z"] = float(rest[0])
        elif key == "ek":
            kw["extended_kernel"] = rest[0] != "0"
        elif key == "exact":
            kw["exact_kernel"] = {"auto": "auto", "on": True}.get(rest[0], False)
        elif key == "wire":
            cut = rest.index(":")
            kw["n_per_edge_per_wire"].append([int(c) for c in rest[:cut]])
            xyz = [float(v) for v in rest[cut + 1 :]]
            kw["wires"].append([tuple(xyz[i : i + 3]) for i in range(0, len(xyz), 3)])
        elif key == "feed":
            arc = None if rest[1] == "mid" else float(rest[1])
            kw["feeds"].append(
                (int(rest[0]), arc, complex(float(rest[2]), float(rest[3])))
            )
        elif key == "gap":
            kw["node_gaps"].append(
                (int(rest[0]), rest[1], complex(float(rest[2]), float(rest[3])))
            )
    return kw


def test_geometry_to_currents_in_cxx_alone_is_the_python_route(driver, tmp_path):
    """The front half, the fill and the LU, from a deck file, in a process
    with no Python: Z and the coefficients are the Python route's bytes."""
    from scipy.linalg import cython_blas, cython_lapack

    from momwire import RazorSolver, razor

    lib = _scipy_openblas()
    # The library the driver loads IS the one the Python route calls: SciPy's
    # cython_blas / cython_lapack (whose capsules the binding reads) resolve
    # their BLAS and LAPACK against it.
    for module in (cython_blas, cython_lapack):
        deps = subprocess.run(["ldd", module.__file__], capture_output=True, text=True)
        assert os.path.realpath(lib) in {
            os.path.realpath(w)
            for w in deps.stdout.split()
            if w.startswith("/") and "openblas" in w
        }, deps.stdout

    run = subprocess.run(
        [str(driver[0]["razor_front_driver"]), str(DECK), lib, str(tmp_path)],
        capture_output=True,
        text=True,
        env=dict(os.environ, **_ONE_THREAD),
    )
    assert run.returncode == 0, run.stderr
    meta = (tmp_path / "front_meta.txt").read_text().split()
    assert meta[0].isdigit(), f"the driver did not solve: {meta}"
    n, m, info = (int(v) for v in meta)
    assert info == 0

    seen = {}
    orig = razor._bspline._lu_solve

    def lu(Z, rhs, **k):
        seen["Z"] = np.array(Z, order="F", copy=True)
        return orig(Z, rhs, **k)

    razor._bspline._lu_solve = lu
    saved = razor._RAZOR_CXX
    razor._RAZOR_CXX = False
    try:
        with _one_thread():
            sol = RazorSolver(**_read_deck(DECK)).compute_port_solution()
    finally:
        razor._bspline._lu_solve = orig
        razor._RAZOR_CXX = saved
    assert sol.coeffs.shape == (n, m) and m == 3
    assert (tmp_path / "front_Z.bin").read_bytes() == np.ravel(
        seen["Z"], order="F"
    ).tobytes()
    assert (tmp_path / "front_X.bin").read_bytes() == np.ravel(
        sol.coeffs, order="F"
    ).tobytes()


# --- deck text -> the parsed deck, no Python --------------------------------


def _deck_compare_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_deck_compare", SPLIT / "deck_compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_deck_reader_in_cxx_alone_is_the_python_reader(driver):
    """Every NEC deck in tests/fixtures through the reader in a process with
    no Python: the parse and the seam's front (NX structures, XQ runs, the
    refusal it prints) are the Python reader's, field for field."""
    dc = _deck_compare_module()
    decks = [
        p
        for p in sorted((ROOT / "tests" / "fixtures").rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]
    assert len(decks) >= 250
    run = subprocess.run(
        [str(driver[0]["nec5_deck_driver"]), *map(str, decks)],
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    chunks = run.stdout.decode("ascii").split("== ")[1:]
    assert len(chunks) == len(decks)
    outcomes = set()
    for path, chunk in zip(decks, chunks, strict=True):
        name, body = chunk.split("\n", 1)
        assert name == str(path)
        parse, stream = body.split("--\n", 1)
        text = path.read_bytes().decode("latin-1")
        ref = dc.py_parse(text)
        assert dc.first_difference(ref, parse) is None, (
            f"{path}: {dc.first_difference(ref, parse)}"
        )
        ref = dc.py_stream(text)
        assert dc.first_difference(ref, stream) is None, (
            f"{path}: {dc.first_difference(ref, stream)}"
        )
        outcomes.add("refused" if parse.startswith("parse !") else "parsed")
    assert outcomes == {"parsed", "refused"}


# --- deck text -> currents, no Python ----------------------------------------


def _seam_compare_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_seam_compare", SPLIT / "seam_compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_deck_text_to_currents_in_cxx_alone_is_the_python_seam(driver, tmp_path):
    """Every NEC deck in tests/fixtures through the reader, the seam's
    structure half, the front half, the fill and the LU, in a process with no
    Python: each deck the C++ path answers is answered as the Python seam
    answers it -- the same route, the same refusal sentence or exception, the
    same number of preparations, and every solved coefficient the same bytes
    (both sides on one BLAS thread: `_ONE_THREAD`). A deck it declines is
    counted, never compared."""
    sc = _seam_compare_module()
    lib = _scipy_openblas()
    decks = [
        p
        for p in sorted((ROOT / "tests" / "fixtures").rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]
    run = subprocess.run(
        [str(driver[0]["nec5_razor_driver"]), lib, str(tmp_path), *map(str, decks)],
        capture_output=True,
        env=dict(os.environ, **_ONE_THREAD),
    )
    assert run.returncode == 0, run.stderr
    chunks = run.stdout.decode("ascii").split("== ")[1:]
    assert len(chunks) == len(decks)
    routes = {}
    for i, (path, chunk) in enumerate(zip(decks, chunks, strict=True)):
        lines = chunk.splitlines()
        assert lines[0] == str(path)
        route = lines[1].split(" ", 1)[1]
        kind = lines[2].split(" ", 1)[1] if " " in lines[2] else ""
        message = bytes.fromhex(lines[3].split(" ", 1)[1] if " " in lines[3] else "")
        prepared = [line.split() for line in lines[4:]]
        routes[route] = routes.get(route, 0) + 1
        if route == "declined":
            continue
        with _one_thread():
            py = sc.python_seam(path.read_bytes().decode("latin-1"))
        if py["outcome"] == "printout-raised":
            # Python raised WRITING a printout (the echo's own card read),
            # after the preparations it had reached: the seam's answer is
            # "served", those preparations a prefix of the C++ seam's, and
            # the frame is the printout test's to compare (below).
            py = dict(py, outcome="served", error_kind="", message="")
            prepared = prepared[: len(py["prepared"])]
        assert (route, kind, message.decode("utf-8")) == (
            py["outcome"],
            py["error_kind"],
            py["message"],
        ), path
        assert len(prepared) == len(py["prepared"]), path
        for k, (row, ref) in enumerate(zip(prepared, py["prepared"], strict=True)):
            n, m, info = int(row[2]), int(row[3]), int(row[4])
            assert info == 0 and ref["X"].shape == (n, m), path
            x = (tmp_path / f"{i}_{k}_X.bin").read_bytes()
            assert x == np.ravel(ref["X"], order="F").tobytes(), path
    # Not vacuous: the C++ path answers decks of every kind it covers.
    assert routes.get("served", 0) >= 20 and routes.get("refused", 0) >= 5, routes


# --- deck text -> the printout, no Python ------------------------------------


def _printout_compare_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_printout_compare", SPLIT / "printout_compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRINTOUT_DECKS = ROOT / "tests" / "fixtures" / "razor_printout_3a"
LD_DECKS = ROOT / "tests" / "fixtures" / "razor_printout_ld"
RP_DECKS = ROOT / "tests" / "fixtures" / "razor_printout_rp"
NE_DECKS = ROOT / "tests" / "fixtures" / "razor_printout_ne"
WL_DECKS = ROOT / "tests" / "fixtures" / "razor_printout_wl"
SOMM_DECKS = ROOT / "tests" / "fixtures" / "razor_printout_somm"
NET_DECKS = ROOT / "tests" / "fixtures" / "razor_printout_net"
EK_DECKS = ROOT / "tests" / "fixtures" / "razor_printout_ek"


def test_deck_text_to_printout_in_cxx_alone_is_the_launchers(driver, tmp_path):
    """Every NEC deck in tests/fixtures through the whole chain -- the
    reader, the seam's structure half, the front half, the fill and the LU,
    the run half and the printout writer -- in a process with no Python:
    every deck the C++ path prints is, byte for byte, the file
    `momwire-nec5-razor-2p` writes for it (results, refusals and the
    INTERNAL ERROR frame alike; both sides on one BLAS thread). Below the
    printed digits, every run either side finishes carries the same bytes:
    the solve's Y (`cols.T @ X`, numpy's zgemm / zdotu), the coefficients
    `_finish` reads (`X @ (T @ v_gap)`), and `_port_state`'s four vectors.
    A deck it declines is counted by reason, never compared."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    decks = [
        p
        for p in sorted((ROOT / "tests" / "fixtures").rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]
    saved = {k: os.environ.get(k) for k in _ONE_THREAD}
    os.environ.update(_ONE_THREAD)
    try:
        answers = pc.run_driver(
            driver[0]["nec5_razor_printout_driver"], decks, tmp_path, arrays=True
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    counts, reasons = {}, set()
    with _one_thread():
        for path, ans in zip(decks, answers, strict=True):
            if ans["route"] == "declined":
                reasons.add(ans["reason"])
                continue
            key = ans["seam"] + ("|frame" if ans["raised"] else "")
            counts[key] = counts.get(key, 0) + 1
            py_runs = []
            ref = pc.python_printout(path.read_bytes().decode("latin-1"), runs=py_runs)
            assert ans["printout"] == ref, (
                f"{path}: {pc.first_difference(ref, ans['printout'])}"
            )
            if ans["raised"]:
                continue
            assert len(ans["runs"]) == len(py_runs), path
            for k, (cx, py) in enumerate(zip(ans["runs"], py_runs, strict=True)):
                for name in pc.ARRAYS:
                    mine = np.ascontiguousarray(py[name], dtype=np.complex128)
                    assert cx[name] == mine.tobytes(), f"{path} run {k}: {name}"
                counts["runs"] = counts.get("runs", 0) + 1
                ports = int(round(len(cx["y"]) ** 0.5 / 4))  # 16 bytes a cell
                if ports > 1:
                    counts["multi-port"] = counts.get("multi-port", 0) + 1
    # Not vacuous: results, refusals and the last line of defence all print,
    # and the three run-half declines are reached by name.
    assert counts.get("served", 0) >= 25 and counts.get("refused", 0) >= 50, counts
    assert counts.get("served|frame", 0) >= 2, counts
    assert counts.get("runs", 0) >= 30 and counts.get("multi-port", 0) >= 5, counts
    # The run half declines nothing by name any more: LD 0 / 1 / 4, RP and
    # NE / NH all print (what is left declines in the seam, before it).
    assert not any(r.endswith("not yet in C++") for r in reasons), reasons
    # momwire#1602: a deck read as a program prints, compared above
    assert "sequence" not in reasons, reasons


SEQUENCE_DECKS = [
    ROOT / "tests" / "fixtures" / "nec5_sequence" / "rp_ne_two_runs.nec",
    ROOT / "tests" / "fixtures" / "nec5_sequence" / "rp_ne_one_xq.nec",
    ROOT / "tests" / "fixtures" / "nec5_sequence" / "fr_after_xq.nec",
    ROOT / "tests" / "fixtures" / "eznec_fr_step_1508" / "p07_two_fr.nec",
    ROOT / "tests" / "fixtures" / "eznec_fr_step_1508" / "p11_ex_change.nec",
    ROOT / "tests" / "fixtures" / "eznec_fr_step_1508" / "p18_two_rp.nec",
    ROOT / "tests" / "fixtures" / "eznec_nec42" / "probes" / "p25_xq_rp_ne_free.nec",
]


def test_the_sequence_decks_print_as_python_does(driver, tmp_path):
    """momwire#1602 / #1590: a deck read as a program (momwire 0.79.2) --
    RP then NE before each XQ of a two-excitation deck (a feed-to-probe
    transfer), RP then NE before one XQ, two RP before one XQ, an FR after
    an XQ, a stepping first run followed by another, a request after a
    no-op XQ -- prints through the C++ core, which declined every one of
    them in 0.80.0, byte for byte as `momwire-nec5-razor-2p` writes it."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    with _one_thread():
        answers = pc.run_driver(
            driver[0]["nec5_razor_printout_driver"], SEQUENCE_DECKS, tmp_path
        )
        for path, ans in zip(SEQUENCE_DECKS, answers, strict=True):
            assert (ans["route"], ans["seam"], ans["raised"]) == (
                "printed",
                "served",
                "",
            ), path
            ref = pc.python_printout(path.read_bytes().decode("latin-1"))
            assert ans["printout"] == ref, (
                f"{path}: {pc.first_difference(ref, ans['printout'])}"
            )


def test_the_printout_decks_take_the_paths_they_name(driver, tmp_path):
    """The committed 3a decks print (none declines), so each path they were
    written for -- a dense node-gap column through zgemm, numpy's zgesv
    under two set currents, a complex EX 4 scale, three complex volts over a
    GD, an NX chain ending on a refusal, a three-run deck sharing one
    preparation, the echo's two raises -- is compared above, not skipped."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    decks = sorted(PRINTOUT_DECKS.glob("*.nec"))
    assert len(decks) == 7
    answers = pc.run_driver(driver[0]["nec5_razor_printout_driver"], decks, tmp_path)
    routes = {
        p.stem: (a["route"], a["seam"], a["raised"])
        for p, a in zip(decks, answers, strict=True)
    }
    assert routes == {
        "chain_then_refusal": ("printed", "refused", ""),
        "echo_raises_inf_field": ("printed", "served", "OverflowError"),
        "echo_raises_nxq": ("printed", "served", "DeckError"),
        "multi_run_three_ports": ("printed", "served", ""),
        "node_gap_ex4_complex": ("printed", "served", ""),
        "node_gap_two_sources": ("printed", "served", ""),
        "three_voltages_gd": ("printed", "served", ""),
    }, routes
    chain = answers[0]["printout"]
    assert chain.count(b"- - - ANTENNA INPUT PARAMETERS - - -") == 2
    assert b"NEC ERROR - this deck carries no FR card" in chain


def test_the_lumped_load_decks_print_their_loads(driver, tmp_path):
    """LD 0 / 1 / 4 print through the C++ run half (`load_impedance`, the
    loads in series inside their ports, the loaded admittance of a
    multi-source drive, the WIRE LOSS watts, the loading table). Each
    committed LD deck prints (none declines) with its loading rows, so each
    path -- a series RLC in the driven segment, parallel traps, a fixed
    reactance under two volts, two loaded current sources, EZNEC's 1e-10 V
    probe at a loaded node, an all-zero parallel card (an open) over a
    perfect ground -- is compared byte for byte by the test above."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    decks = sorted(LD_DECKS.glob("*.nec"))
    assert len(decks) == 6
    answers = pc.run_driver(driver[0]["nec5_razor_printout_driver"], decks, tmp_path)
    for path, ans in zip(decks, answers, strict=True):
        assert (ans["route"], ans["seam"], ans["raised"]) == (
            "printed",
            "served",
            "",
        ), path
        assert b"THIS STRUCTURE IS NOT LOADED" not in ans["printout"], path
        assert ans["printout"] == pc.python_printout(
            path.read_bytes().decode("latin-1")
        ), path


def test_the_pattern_decks_print_their_patterns(driver, tmp_path):
    """RP prints through the C++ run half (razor-2p's element moments, the
    moment sums in numpy's einsum order, the PEC image or the image weighted
    by the GD medium's Fresnel coefficients, the hypot gains, the dust floor
    and the polarisation ellipse, the average gain, the range form). Each
    committed RP deck prints (none declines) with its patterns -- a GD
    medium at a range (the LOWER MEDIUM line) with the average gain, rows
    past the horizon over a perfect ground, a whole sphere and a second card
    in free space, a loaded inverted V at a range with negative theta -- and
    is compared byte for byte by the corpus test above.

    The two-card deck is a "sequence" (momwire 0.80.0): its second RP runs
    on the standing solution and NEC-5 (x13) echoes it between the two
    patterns, which the C++ core prints as Python does since momwire#1602
    (it declined it in 0.80.0, momwire#1590)."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    every = sorted(RP_DECKS.glob("*.nec"))
    assert len(every) == 4
    answers = pc.run_driver(driver[0]["nec5_razor_printout_driver"], every, tmp_path)
    for path, ans in zip(every, answers, strict=True):
        assert (ans["route"], ans["seam"], ans["raised"]) == (
            "printed",
            "served",
            "",
        ), path
        assert b"RADIATION PATTERNS" in ans["printout"], path
        assert ans["printout"] == pc.python_printout(
            path.read_bytes().decode("latin-1")
        ), path


def test_the_near_field_decks_print_their_fields(driver, tmp_path):
    """NE / NH print through the C++ run half over free space and GN 1:
    the grid (rectangular, or spherical with its axes exact), razor-2p's
    current resampled eight ways along each edge (numpy's linspace and
    interp), the mixed-potential E (current and node-charge sums) or the H
    curl, the PEC image, the zero floor. Each committed deck prints with its
    table -- a free-space grid, a spherical NH over GN 1 through the horizon
    and the nadir, a near field beside a pattern on a loaded V, two driven
    verticals over GN 1 -- and is compared byte for byte above."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    decks = sorted(NE_DECKS.glob("*.nec"))
    assert len(decks) == 4
    answers = pc.run_driver(driver[0]["nec5_razor_printout_driver"], decks, tmp_path)
    for path, ans in zip(decks, answers, strict=True):
        assert (ans["route"], ans["seam"], ans["raised"]) == (
            "printed",
            "served",
            "",
        ), path
        assert b"- - - NEAR " in ans["printout"], path
        assert ans["printout"] == pc.python_printout(
            path.read_bytes().decode("latin-1")
        ), path


def test_the_network_decks_print_their_networks(driver, tmp_path):
    """TL / NT print through the C++ run half: the reducer (`flat_network`,
    the chain-matrix TL stamps with glibc's complex sinh / cosh, the NT and
    shunt admittances, the terminations), SciPy's `zgesvx` on the MNA system,
    `reduced_ports`, the multi-source columns through the same reducer, the
    NETWORK DATA tables, the connection points and the gyrator readout.
    Each committed deck prints with its network -- a crossed TL with shunts
    at NEC's zero length, a TL table then an NT table, EZNEC's current source
    (a gyrator on a parked wire), two loaded current sources through lines
    from a parked node, two voltage sources across an NT -- and is compared
    byte for byte by the corpus test above."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    decks = sorted(NET_DECKS.glob("*.nec"))
    assert len(decks) == 5
    answers = pc.run_driver(driver[0]["nec5_razor_printout_driver"], decks, tmp_path)
    for path, ans in zip(decks, answers, strict=True):
        assert (ans["route"], ans["seam"], ans["raised"]) == (
            "printed",
            "served",
            "",
        ), path
        assert b"- - - NETWORK DATA - - -" in ans["printout"], path
        assert ans["printout"] == pc.python_printout(
            path.read_bytes().decode("latin-1")
        ), path


BURIED_DECKS = [
    ROOT / "tests" / "fixtures" / rel
    for rel in (
        "eznec_crossing_source_1282/dan_end1.nec",
        "eznec_crossing_source_1282/dan_end1_via6.nec",
        "eznec_crossing_source_1282/dan_seg1end2.nec",
        "eznec_nec42/probes/p9_above_and_buried_gn2.nec",
        "eznec_phantom_1139/ground_rod_efhw_wa7ark.nec",
        "eznec_split_1281/dan_through.nec",
        "eznec_split_1281/thr_n21.nec",
    )
] + [
    # NX chains of buried structures (momwire#1512): kept out of
    # tests/fixtures, whose every deck the walks above run with the anchors on
    # `auto` -- a chain swept in frequency engages them, and the driver and
    # this process see different histories. Here the tables are exact.
    SPLIT / "nx_decks" / name
    for name in ("ward_many_nx_cut5.nec", "simnec_stream_2281440.nec")
]


def test_the_buried_decks_print_with_their_tables_filled_in_cxx(
    driver, tmp_path, monkeypatch
):
    """A buried deck through the C++ path end to end (momwire#1512 B1-B6, B5):
    the crossing and split seams, the two media's fills, the cross blocks and
    node charges, the below/below table filled in C++ (`_somm_below_fill_core`,
    the integrand `_somm_below_six_inline.h` shared with the extension), and
    the run half's split rows and transmitted far field -- each deck printed,
    none declined, and each printout the launcher's bytes. Exact tables
    (MOMWIRE_BELOW_ANCHORS=off): the anchors' sweep state lives per process,
    and the driver and this process see different histories."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    monkeypatch.setenv("MOMWIRE_BELOW_ANCHORS", "off")
    answers = pc.run_driver(
        driver[0]["nec5_razor_printout_driver"], BURIED_DECKS, tmp_path
    )
    served = 0
    for path, ans in zip(BURIED_DECKS, answers, strict=True):
        assert ans["route"] == "printed", (path, ans.get("reason"))
        served += ans["seam"] == "served"
        assert ans["printout"] == pc.python_printout(
            path.read_bytes().decode("latin-1")
        ), path
    assert served >= 5


def test_the_sommerfeld_decks_print_refuse_or_decline_by_name(driver, tmp_path):
    """GN 0 / GN 2 through the C++ path (momwire#1479 S3): the composing fill
    -- the C2-weighted image, its scaled charge term and the smooth remainder
    over the grid -- prints served decks byte for byte (an elevated dipole, a
    base-fed vertical's grounded row, a dipole low enough to raise the
    remainder's source order, a two-radius junction under the extended
    kernel); the refusals print too (a near field at a grounded base, a wire
    in the plane, a spherical grid reaching under a GD); a near field over the
    composing ground prints too (S4: `C2 image + remainder` through
    `_somm_field_core`); and a buried radial screen prints its refusal
    (momwire#1512 B1)."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    decks = sorted(SOMM_DECKS.glob("*.nec"))
    assert len(decks) == 11
    answers = pc.run_driver(driver[0]["nec5_razor_printout_driver"], decks, tmp_path)
    routes = {
        p.stem: (a["route"], a["seam"], a["reason"])
        for p, a in zip(decks, answers, strict=True)
    }
    served = ("printed", "served", "")
    assert routes == {
        "gd_ne_dipole": served,
        "gd_nh_spherical_below": ("printed", "refused", ""),
        "gn0_buried_radial": ("printed", "refused", ""),
        "gn0_inverted_vee_junction": served,
        "gn0_nh_dipole": served,
        "gn0_ne_at_contact": ("printed", "refused", ""),
        "gn0_served_dipole": served,
        "gn2_in_plane_wire": ("printed", "refused", ""),
        "gn2_low_dipole_grazing": served,
        "gn2_near_field": served,
        "gn2_vertical_contact": served,
    }, routes
    for path, ans in zip(decks, answers, strict=True):
        if ans["route"] == "printed":
            assert ans["printout"] == pc.python_printout(
                path.read_bytes().decode("latin-1")
            ), path


def test_the_wire_loading_decks_print_their_loading(driver, tmp_path):
    """WIRE loading prints through the C++ path: LD 5's internal impedance
    (the vendored xsf's `cyl_bessel_ie`, which is scipy.special.ive's own
    code), LD 2 / LD 3's per-metre RLC, the loading stencil added after the
    fill and before the LU, WIRE LOSS (libm's hypot, momwire#1501) and the
    WIRE / PER METER rows. Each committed deck answers as Python does --
    LD 5 on one wire of two, LD 5 and LD 2 on one wire, LD 3 over GN 1, LD 5
    on every wire under two sources with a pattern, LD 2's inductance alone
    beside a lumped load, LD 3's capacitance refused by name, and three
    free-space twins of captured decks with LD 5 on every wire (tag 0) and
    a network (0208's three TLs, 0215's NT pins, 0185's NT gyrator) -- the
    purely imaginary j*omega*mu*sigma whose square root is libm's csqrt
    (numpy's), not CPython's cmath.sqrt."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    decks = sorted(WL_DECKS.glob("*.nec"))
    assert len(decks) == 9
    answers = pc.run_driver(driver[0]["nec5_razor_printout_driver"], decks, tmp_path)
    for path, ans in zip(decks, answers, strict=True):
        refused = path.stem.endswith("_refused")
        want = ("printed", "refused" if refused else "served", "")
        assert (ans["route"], ans["seam"], ans["raised"]) == want, path
        if not refused:
            assert b"THIS STRUCTURE IS NOT LOADED" not in ans["printout"], path
        assert ans["printout"] == pc.python_printout(
            path.read_bytes().decode("latin-1")
        ), path


def test_the_exact_kernel_decks_print_their_correction(driver, tmp_path):
    """The exact ring kernel (momwire#1433) through the C++ path: "auto"
    engages it on segments under three radii, `build_exact_plan` replays
    `_razor_exact_plan.build_plan` and `add_exact_correction` the
    correction (the point moments are the extension's own inline text), after
    the free block and before the image and the loading. Each committed deck
    prints -- a fat dipole, a fat monopole's grounded base row under its
    image, two lines of two radii (the EK tube per observer), a bent loaded
    wire over a ground (one group per leg), two collinear wires as one group
    -- and a radius step on one line (the two-radius cross jobs after the
    groups, momwire#1438), each compared byte for byte by the corpus test
    above."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    decks = sorted(EK_DECKS.glob("*.nec"))
    assert len(decks) == 6
    answers = pc.run_driver(driver[0]["nec5_razor_printout_driver"], decks, tmp_path)
    for path, ans in zip(decks, answers, strict=True):
        assert (ans["route"], ans["seam"], ans["raised"]) == (
            "printed",
            "served",
            "",
        ), path
        assert ans["printout"] == pc.python_printout(
            path.read_bytes().decode("latin-1")
        ), path


def test_the_printout_spells_outside_latin1_as_python_does(driver):
    """`write_printout`'s transliteration: every character momwire's own
    sentences carry outside latin-1 (and the two `str.upper` takes a latin-1
    letter to), the em dash's spacing rule, and a character no table names,
    through the C++ writer and through Python's codec, the same bytes."""
    from momwire.eznec import _shell

    chars = set()
    for p in (ROOT / "src" / "momwire").rglob("*.py"):
        chars |= {c for c in p.read_text(encoding="utf-8") if ord(c) >= 0x100}
    chars |= {"Μ", "Ÿ", "一"}
    text = "".join(sorted(chars)) + "\na—b a — b —x\n—"
    run = subprocess.run(
        [str(driver[0]["nec5_razor_printout_driver"]), "--transliterate"],
        input=text.encode("utf-8"),
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    ref = text.replace("\n", "\r\n").encode("latin-1", errors=_shell._PRINTOUT_ERRORS)
    assert run.stdout == ref


def test_the_printout_cells_are_pythons_format(driver):
    """Every float cell width the writer uses, over the edges (NaN of both
    signs, the infinities, signed zeros, subnormals, three-digit exponents,
    exact ties) and 20,000 random bit patterns: printf in the C locale is
    Python's format, NaN spelled as Python spells it."""
    import struct

    rng = np.random.default_rng(1479)
    values = [
        float("nan"),
        -float("nan"),
        float("inf"),
        -float("inf"),
        0.0,
        -0.0,
        5e-324,
        1e-310,
        1e100,
        -9.99995e99,
        0.125,
        2.5,
        1.000005,
        123456.78905,
        *rng.standard_normal(5000) * 10.0 ** rng.integers(-30, 30, 5000),
    ]
    bits = [struct.unpack("<Q", struct.pack("<d", v))[0] for v in values]
    bits += [int(b) for b in rng.integers(0, 2**63, 15000, dtype=np.int64)]
    run = subprocess.run(
        [str(driver[0]["nec5_razor_printout_driver"]), "--format"],
        input="".join(f"{b:016x}\n" for b in bits).encode(),
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    lines = run.stdout.decode().splitlines()
    assert len(lines) == len(bits)
    for b, line in zip(bits, lines, strict=True):
        x = struct.unpack("<d", struct.pack("<Q", b))[0]
        ref = "|".join(
            (
                f"{x:13.5E}",
                f"{x:11.4E}",
                f"{x:12.4E}",
                f"{x:10.3E}",
                f"{x:10.4f}",
                f"{x:7.2f}",
                f"{x:7.3f}",
            )
        )
        assert line == ref, hex(b)


@pytest.mark.parametrize("symmetric", [True, False], ids=["symmetric", "numpy_key"])
def test_the_coaxial_exact_kernel_in_cxx_alone_is_the_extensions(
    driver, accel, symmetric, tmp_path
):
    """The B-spline basis' exact kernel (`_exact_kernel_coaxial_core.h`,
    momwire#1517 step 0) built from the core TUs alone: a graded, mixed-
    direction coaxial group's row windows (the second window over a warm
    cache), every pair's moments by each route, and the ring kernel and its
    split, byte for byte the extension's."""
    from momwire import _exact_kernel as ek

    exe = driver[0]["exact_kernel_coaxial_driver"]
    n, a, nd = 24, 1e-3, 3
    k = 2.0 * np.pi / 10.0
    h = 0.05 * (1.0 + 0.15 * np.arange(n) ** 1.3 / n)
    sgn = np.where(np.arange(n) % 5 == 3, -1.0, 1.0)
    ends = np.concatenate([[0.0], np.cumsum(h)])
    x0 = np.where(sgn > 0, ends[:-1], ends[1:])
    q = 1e-11 * float(np.max(h))
    r0, r1 = 9, n
    xg, wg = ek._gl01(ek._N_GL)
    xs, ws = ek._gl01(ek._N_SPLIT)
    wls = ek._gl01_log(ek._N_SPLIT)
    z = np.concatenate([np.linspace(-0.3, 0.3, 41), [1e-7, 2.5e-3, 5.9e-3, 6.1e-3]])
    for name, v in (
        ("x0", x0),
        ("sgn", sgn),
        ("h", h),
        ("xg", xg),
        ("wg", wg),
        ("xs", xs),
        ("ws", ws),
        ("wls", wls),
        ("z", z),
    ):
        np.ascontiguousarray(v, dtype=np.float64).tofile(tmp_path / f"{name}.bin")
    (tmp_path / "params.txt").write_text(
        f"{a!r} {k!r} {nd} -1 {q!r} {int(symmetric)} 1 {r0} {r1}\n"
    )
    run = subprocess.run([str(exe), str(tmp_path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    rules = accel.ExactKernelRules(xg, wg, xs, ws, wls)
    rows = accel.ExactKernelCoaxialRows(
        x0, sgn, h, a, k, nd, -1, q, symmetric, rules, 1
    )
    first = np.empty((nd, nd, r0, n), dtype=np.complex128)
    second = np.empty((nd, nd, r1 - r0, n), dtype=np.complex128)
    rows.rows(0, r0, first)
    rows.rows(r0, r1, second)
    ref_rows = first.tobytes() + second.tobytes()
    ref_pm = b"".join(
        accel.exact_kernel_pair_moments(
            h[i], h[j], x0[j] - x0[i], sgn[i], sgn[j], a, k, nd, rules, far
        ).tobytes()
        for i in range(n)
        for j in range(n)
        for far in (-1, 0, 1)
    )
    near = z[np.abs(z) < 6.0 * a]
    split_a, split_b = accel.exact_kernel_ring_split(near, a, k)
    for name, ref in (
        ("rows", ref_rows),
        ("pm", ref_pm),
        ("ring", accel.exact_kernel_ring(z, a, k).tobytes()),
        ("split_a", split_a.tobytes()),
        ("split_b", split_b.tobytes()),
    ):
        assert len(ref) > 0
        assert (tmp_path / f"drv_{name}.bin").read_bytes() == ref, name
    counts = (tmp_path / "drv_counts.txt").read_text().split()
    assert counts == [str(rows.evaluated), str(rows.cached)]


def test_the_bspline_kernels_in_cxx_alone_are_the_extensions(driver, accel, tmp_path):
    """The B-spline kernels' core (`_bspline_core.h`, momwire#1517 step 0)
    built from the core TUs alone, on the default host (memory from `new`,
    no GIL): a bent two-wire segment set's off-edge moments, by the lane path
    where this variant builds it and by the per-pair walk, and a uniform
    edge's static table, byte for byte the extension's."""
    exe = driver[0]["bspline_core_driver"]
    a, k, max_d, h, N = 1e-3, 2.0 * np.pi / 7.0, 2, 0.11, 9
    t = np.linspace(0.0, 1.0, 8)[:, None]
    leg1 = np.array([0.0, 0.0, 0.0]) + t * np.array([0.7, 0.0, 0.0])
    leg2 = np.array([0.7, 0.0, 0.0]) + t * np.array([0.3, 0.0, 0.6])
    pts = np.concatenate([leg1, leg2[1:]])
    seg_l = np.ascontiguousarray(pts[:-1])
    seg_r = np.ascontiguousarray(pts[1:])
    gl_t, gl_w = np.polynomial.legendre.leggauss(4)
    gl_t = 0.5 * (gl_t + 1.0)
    gl_w = 0.5 * gl_w
    for name, v in (("seg_l", seg_l), ("seg_r", seg_r), ("gl_t", gl_t), ("gl_w", gl_w)):
        np.ascontiguousarray(v, dtype=np.float64).tofile(tmp_path / f"{name}.bin")
    (tmp_path / "params.txt").write_text(f"{a!r} {k!r} {max_d} {h!r} {N}\n")
    run = subprocess.run([str(exe), str(tmp_path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    def full(reference):
        return accel.seg_seg_full_moments_bspline(
            seg_l, seg_r, seg_l, seg_r, a * a, k, max_d, gl_t, gl_w, reference
        )

    for name, ref in (
        ("full", full(False)),
        ("full_ref", full(True)),
        ("static", accel.seg_seg_static_moments_bspline_uniform(h, a, N, max_d)),
    ):
        assert ref.size > 0
        assert (tmp_path / f"drv_{name}.bin").read_bytes() == ref.tobytes(), name


BSPLINE_KCL_SYM_DECKS = ROOT / "tests" / "fixtures" / "bspline_printout_kcl_sym"
BSPLINE_CHUNKED_DECKS = SPLIT / "bspline_chunked_decks"
BSPLINE_EXACT_DECKS = SPLIT / "bspline_exact_decks"
BSPLINE_BURIED_DECKS = ROOT / "tests" / "fixtures" / "bspline_printout_buried"


def _bspline_printouts(driver, decks, tmp_path):
    """bs2's driver against `_shell.render(text, basis="bspline")` over
    `decks`, both on one BLAS thread: every printed deck byte for byte, and
    every finished run's arrays. Returns (counts, decline reasons)."""
    pc = _printout_compare_module()
    if pc.scipy_openblas() is None or pc.numpy_openblas() is None:
        pytest.skip("no bundled OpenBLAS beside SciPy / numpy (not a wheel install)")
    saved = {k: os.environ.get(k) for k in _ONE_THREAD}
    os.environ.update(_ONE_THREAD)
    try:
        answers = pc.run_driver(
            driver[0]["nec5_bspline_printout_driver"],
            decks,
            tmp_path,
            basis="bspline",
            arrays=True,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    counts, reasons = {}, {}
    with _one_thread():
        for path, ans in zip(decks, answers, strict=True):
            if ans["route"] == "declined":
                reasons[ans["reason"]] = reasons.get(ans["reason"], 0) + 1
                continue
            key = ans["seam"] + ("|frame" if ans["raised"] else "")
            counts[key] = counts.get(key, 0) + 1
            py_runs = []
            ref = pc.python_printout(
                path.read_bytes().decode("latin-1"), basis="bspline", runs=py_runs
            )
            assert ans["printout"] == ref, (
                f"{path}: {pc.first_difference(ref, ans['printout'])}"
            )
            if ans["raised"]:
                continue
            assert len(ans["runs"]) == len(py_runs), path
            for k, (cx, py) in enumerate(zip(ans["runs"], py_runs, strict=True)):
                for name in pc.ARRAYS:
                    mine = np.ascontiguousarray(py[name], dtype=np.complex128)
                    assert cx[name] == mine.tobytes(), f"{path} run {k}: {name}"
                counts["runs"] = counts.get("runs", 0) + 1
    return counts, reasons


def test_deck_text_to_printout_in_cxx_alone_is_bsplines(driver, tmp_path):
    """momwire#1517 step 1: every NEC deck in tests/fixtures through bs2's
    whole chain in a process with no Python -- the reader, the seam with the
    cut spelling, the front half, the dense fill, the KCL Schur solve, the
    spline readouts and sampler, the run half and the writer. Every deck it
    prints is, byte for byte, what `_shell.render(text, basis="bspline")`
    prints, and every finished run carries Python's Y, coefficients and port
    vectors -- wire loading (LD 5 / 2 / 3) and the exact ring-kernel
    correction (step 2), a perfect ground's image term (step 3a) and the
    Sommerfeld composition with its graded raised pairs (steps 3b, 3c) and a
    wholly-buried structure (step 4a) included. What it declines is a deck
    with wires on both sides of the interface and no crossing node (the
    transmitted blocks, step 4c), a geometry Python refuses by medium
    (its sentence not yet ported) and the exact kernel's radius-step cross
    job, by name, and nothing else. A one-radius crossing node is served
    (step 4b), a node gap on one included (its KCL row, momwire#1282 /
    #1569), and a two-radius one (antennaknobs plan U5, 0.79.2)."""
    # The wholly-buried decks have their own test below (their Python fills
    # are the slow ones), so this one keeps its pre-step-4 time.
    decks = [
        p
        for p in sorted((ROOT / "tests" / "fixtures").rglob("*"))
        if p.is_file()
        and p.suffix in (".nec", ".deck")
        and BSPLINE_BURIED_DECKS not in p.parents
    ]
    counts, reasons = _bspline_printouts(driver, decks, tmp_path)
    # Not vacuous: results, refusals and the INTERNAL ERROR frame all print.
    assert counts.get("served", 0) >= 80 and counts.get("refused", 0) >= 100, counts
    assert counts.get("raised|frame", 0) >= 3, counts
    assert counts.get("runs", 0) >= 100, counts
    assert set(reasons) <= {
        "front:bspline-buried-two-radius-spread",
        "front:bspline-buried-two-radius-nodes",
        "front:bspline-buried-mixed",
        "front:bspline-buried-refusal-crossing",
        "front:bspline-buried-refusal-medium",
        "front:bspline-buried-refusal-contact",
        "front:bspline-buried-refusal-unearned",
        "front:bspline-exact-kernel-radius-step",
    }, reasons


def test_the_wholly_buried_decks_print_as_python(driver, tmp_path):
    """momwire#1517 step 4a: every wire below a Sommerfeld ground. The plan's
    below/below extents, the direct and A_m-weighted image blocks at the
    medium's complex k (the subset windows, the complex same-edge moments as
    numpy's einsum sums them), and the below/below remainder over the table
    filled in C++, field form, each unordered pair once through the shared
    assembler (`_field_galerkin_inline.h`) and projection
    (`_somm_below_remainder_inline.h`) -- loaded, multi-frequency, bent and
    hubbed, a pattern, and a wire long enough that the remainder's pairs
    reach past the cap."""
    decks = sorted(BSPLINE_BURIED_DECKS.glob("*.nec"))
    assert len(decks) == 6
    counts, reasons = _bspline_printouts(driver, decks, tmp_path)
    assert counts.get("served", 0) == 6 and not reasons, (counts, reasons)
    assert counts.get("runs", 0) == 7, counts


BSPLINE_CROSSING_DECKS = [
    ROOT / "tests" / "fixtures" / rel
    for rel in (
        "eznec_split_1281/dan_through.nec",
        "eznec_split_1281/thr_n21.nec",
        "eznec_fr_step_1508/b01_buried_xq.nec",
        "eznec_fr_step_1508/b02_buried_rp.nec",
        "bspline_printout_crossing/aca_vertical_screen.nec",
    )
]


def test_the_crossing_decks_print_as_python(driver, tmp_path):
    """momwire#1517 step 4b: a one-radius crossing node. The crossing serve
    (`_bspline_crossing_core.cpp`) -- the split's cluster partition, its
    direct groups through one memo, the far blocks' ACA (the screen with a
    vertical through the plane takes it), the ends and the designed corner,
    and the self completions -- composed into Z as the buried fill composes
    it, with the above blocks and both remainders around it."""
    counts, reasons = _bspline_printouts(driver, BSPLINE_CROSSING_DECKS, tmp_path)
    assert counts.get("served", 0) == len(BSPLINE_CROSSING_DECKS) and not reasons, (
        counts,
        reasons,
    )


def test_an_exactly_symmetric_kcl_system_takes_scipys_symmetric_solver(
    driver, tmp_path
):
    """`scipy.linalg.solve` detects an exactly symmetric complex system and
    solves it with zsytrf + zsytrs rather than getrf + getrs: two decks whose
    KCL Schur system comes out bit-symmetric print as Python's."""
    decks = sorted(BSPLINE_KCL_SYM_DECKS.glob("*.nec"))
    assert len(decks) == 2
    counts, reasons = _bspline_printouts(driver, decks, tmp_path)
    assert counts.get("served", 0) == 2 and not reasons, (counts, reasons)


@pytest.mark.slow
def test_the_chunked_fill_decks_print_as_python(driver, tmp_path):
    """Past the dense tensor's `swept_mem_mb` budget (1,402 and 1,500
    segments) both sides take the chunked fill (`_compute_Z_dense_chunked`):
    observer windows into the windowed assembler, then each edge's same-edge
    correction in windows. One deck carries a near-field request."""
    decks = sorted(BSPLINE_CHUNKED_DECKS.glob("*.nec"))
    assert len(decks) == 2
    counts, reasons = _bspline_printouts(driver, decks, tmp_path)
    assert counts.get("served", 0) == 2 and not reasons, (counts, reasons)


@pytest.mark.slow
def test_the_exact_kernel_windows_print_as_python(driver, tmp_path):
    """momwire#1517 step 2: fat wires long enough that the exact-kernel
    correction walks several observer windows (1,000 segments) and follows
    the chunked fill (1,500): CoaxialRows minus the BaseRows replay, its
    Toeplitz gathers and ladder ties, window by window."""
    decks = sorted(BSPLINE_EXACT_DECKS.glob("*.nec"))
    assert len(decks) == 2
    counts, reasons = _bspline_printouts(driver, decks, tmp_path)
    assert counts.get("served", 0) == 2 and not reasons, (counts, reasons)
