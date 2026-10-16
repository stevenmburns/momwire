"""momwire-razor2p (standalone/, unit 4): the launcher's contract, at the edges.

The corpus gate (`scripts/core_split/exe_compare.py`) holds every deck's
printout to `momwire-nec5-razor-2p`'s. These tests hold the rest of the
contract, where the executable must behave as the launcher does without a
deck to compare: the stdin form, `-version`, the argument errors, SimNEC's
streaming call, a deck asking for a `.GMO` file, the version check, and the
rule that CPython is loaded, never linked.

The executable needs a static OpenBLAS (`standalone/build_openblas.sh`) and a
CMake build, which no CI lane makes, so this module runs only when
MOMWIRE_RAZOR2P_EXE names a built executable whose momwire environment is
this one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
EXE = os.environ.get("MOMWIRE_RAZOR2P_EXE")

pytestmark = pytest.mark.skipif(
    not EXE, reason="MOMWIRE_RAZOR2P_EXE does not name a built momwire-razor2p"
)

DIPOLE = FIXTURES / "razor_printout_3a" / "node_gap_two_sources.nec"
EXAMPLE3 = FIXTURES / "4nec2_example3" / "EXAMPLE3.inp"


def _launcher() -> str:
    name = "momwire-nec5-razor-2p" + (".exe" if sys.platform == "win32" else "")
    found = shutil.which("momwire-nec5-razor-2p") or str(
        Path(sys.executable).parent / name
    )
    assert Path(found).exists(), found
    return found


def _run(argv, *, cwd, stdin=subprocess.DEVNULL, env=None, routes=None):
    env = dict(os.environ if env is None else env)
    if routes is not None:
        env["MW_EXE_ROUTE_LOG"] = str(routes)
    return subprocess.run(
        argv, cwd=cwd, stdin=stdin, env=env, capture_output=True, check=False
    )


def _both(argv_tail, tmp_path, *, stdin_text=None):
    """The executable's and the launcher's run of the same command line, each
    in its own copy of the working directory: (exe result, launcher result,
    exe dir, launcher dir)."""
    out = []
    for name, prog in (("exe", EXE), ("launcher", _launcher())):
        d = tmp_path / name
        d.mkdir()
        for f in (DIPOLE, EXAMPLE3):
            shutil.copy(f, d / f.name)
        # 4nec2 writes no EN; the two-path form refuses such a deck as
        # truncated (and writes no .GMO), so the two-path case closes it.
        (d / "EXAMPLE3_EN.nec").write_bytes(EXAMPLE3.read_bytes() + b"EN\n")
        stdin = subprocess.DEVNULL
        if stdin_text is not None:
            (d / "stdin.txt").write_text(stdin_text)
            stdin = open(d / "stdin.txt", "rb")  # noqa: SIM115 - closed below
        try:
            out.append(_run([prog, *argv_tail], cwd=d, stdin=stdin))
        finally:
            if stdin_text is not None:
                stdin.close()
    return out[0], out[1], tmp_path / "exe", tmp_path / "launcher"


def test_python_is_loaded_never_linked():
    if sys.platform != "linux":
        pytest.skip("ldd is the Linux check")
    ldd = subprocess.run(["ldd", EXE], capture_output=True, text=True, check=True)
    assert "libpython" not in ldd.stdout, ldd.stdout


def test_a_covered_deck_never_starts_python(tmp_path):
    routes = tmp_path / "routes"
    r = _run([EXE, str(DIPOLE), str(tmp_path / "x.out")], cwd=tmp_path, routes=routes)
    assert r.returncode == 0, r.stderr
    assert routes.read_text().split(" ", 1)[0].startswith("cxx"), routes.read_text()
    ref = _run([_launcher(), str(DIPOLE), str(tmp_path / "ref.out")], cwd=tmp_path)
    assert ref.returncode == 0
    assert (tmp_path / "x.out").read_bytes() == (tmp_path / "ref.out").read_bytes()


def test_a_deck_asking_for_a_gmo_file_writes_it_as_the_launcher_does(tmp_path):
    exe, ref, de, dl = _both(["EXAMPLE3_EN.nec", "EXAMPLE3.out"], tmp_path)
    assert exe.returncode == ref.returncode == 0
    assert (de / "EXAMPLE3.out").read_bytes() == (dl / "EXAMPLE3.out").read_bytes()
    assert (dl / "EXAMPLE3.GMO").exists()
    assert (de / "EXAMPLE3.GMO").read_bytes() == (dl / "EXAMPLE3.GMO").read_bytes()


def test_the_stdin_form(tmp_path):
    exe, ref, de, dl = _both(
        [], tmp_path, stdin_text=f"{EXAMPLE3.name}\nEXAMPLE3.out\n"
    )
    assert exe.returncode == ref.returncode == 0
    assert (de / "EXAMPLE3.out").read_bytes() == (dl / "EXAMPLE3.out").read_bytes()
    assert (de / "EXAMPLE3.GMO").read_bytes() == (dl / "EXAMPLE3.GMO").read_bytes()


@pytest.mark.parametrize(
    "argv",
    [["only-one.nec"], ["a.nec", "b.out", "c"], ["-", "-"], ["-version"]],
    ids=["one argument", "three arguments", "streaming", "version"],
)
def test_the_command_line_edges_answer_as_the_launcher(argv, tmp_path):
    exe, ref, _de, _dl = _both(argv, tmp_path)
    assert (exe.returncode, exe.stdout) == (ref.returncode, ref.stdout)
    if argv == ["-", "-"]:
        assert exe.stderr.strip() == ref.stderr.strip()


def test_another_release_beside_it_is_refused_in_the_printout(tmp_path):
    """A momwire environment whose installed release is not the executable's:
    every deck is refused, by name, in the printout; nothing is solved."""
    home = tmp_path / "env"
    if sys.platform == "win32":
        sp = home / "Lib" / "site-packages"
    else:
        sp = home / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}"
        sp = sp / "site-packages"
    meta = sp / "momwire-0.0.1.dist-info"
    meta.mkdir(parents=True)
    (meta / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: momwire\nVersion: 0.0.1\n\n"
    )
    env = dict(os.environ, MOMWIRE_PYTHON_HOME=str(home))
    r = _run([EXE, str(DIPOLE), str(tmp_path / "x.out")], cwd=tmp_path, env=env)
    assert r.returncode == 0
    text = (tmp_path / "x.out").read_bytes().decode("latin-1")
    assert "the momwire installed beside it is 0.0.1" in text, text
    assert "NEC ERROR" in text
