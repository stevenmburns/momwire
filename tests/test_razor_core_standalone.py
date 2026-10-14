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
CORE_SOURCES = ("_razor_core.cpp", "_razor_t2_core.cpp", "_accel_razor_cplx.cpp")

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
    sources = [PKG / s for s in CORE_SOURCES] + [SPLIT / "razor_chain_driver.cpp"]
    objects, deps, procs = [], [], []
    for src in sources:  # concurrently: the -O3 core TUs dominate the cost
        obj, dep = out / (src.stem + ".o"), out / (src.stem + ".d")
        cmd = [*base, "-MD", "-MF", str(dep), "-c", str(src), "-o", str(obj)]
        procs.append((cmd, subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)))
        objects.append(str(obj))
        deps.append(dep)
    for cmd, proc in procs:
        _, err = proc.communicate()
        assert proc.returncode == 0, f"{shlex.join(cmd)}\n{err}"
    exe = out / "razor_chain_driver"
    cmd = [*cxx, *objects, *link_args, "-o", str(exe)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    assert run.returncode == 0, f"{shlex.join(cmd)}\n{run.stderr}"
    return exe, deps


def test_the_core_includes_no_python_header(driver):
    """No header the core TUs or the driver pulled in is Python's or
    pybind11's (the compile above had neither include path, so this is
    belt and braces against a system-wide pybind11 install)."""
    _, deps = driver
    seen = " ".join(d.read_text() for d in deps)
    assert "_razor_core.h" in seen, "the dependency files are not what they seem"
    for marker in ("Python.h", "pybind11", "numpy"):
        assert marker not in seen, f"a core TU includes {marker}"


def test_the_driver_links_no_python(driver):
    exe, _ = driver
    libs = subprocess.run(["ldd", str(exe)], capture_output=True, text=True).stdout
    assert "libpython" not in libs, libs


@pytest.mark.parametrize("prefix", ["", "ek_"], ids=["reduced", "extended_kernel"])
def test_the_cxx_chain_is_the_python_chain_bit_for_bit(driver, accel, prefix, tmp_path):
    exe, _ = driver
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
