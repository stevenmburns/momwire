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
    ]
    out = tmp_path_factory.mktemp("razor_core")
    drivers = ("razor_chain_driver", "razor_front_driver")
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
    for marker in ("Python.h", "pybind11", "numpy"):
        assert marker not in seen, f"a core TU includes {marker}"


def test_the_driver_links_no_python(driver):
    exes, _ = driver
    for exe in exes.values():
        libs = subprocess.run(["ldd", str(exe)], capture_output=True, text=True).stdout
        assert "libpython" not in libs, libs


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
    route reaches (`cython_lapack`) and whose ddot equals numpy's norm."""
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
