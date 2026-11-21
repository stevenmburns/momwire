"""The dash-free basis names, and the deprecated spellings (momwire#2091).

A SHORT name is the basis alone (``bs2``, ``bs1``, ``sg``, ``razor2p``); a
LONG name is dialect then basis (``nec5-bs2``). The spellings that shipped
before the rename stay accepted as deprecated aliases and answer exactly as
their canonical name does. The decisive gate is the equivalence below: the
same deck through the old name and the new one, through every front door the
basis is served by, prints the same bytes but for the one deprecation note.
"""

from __future__ import annotations

import importlib.util
import io
import re
import shutil
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import pytest

import momwire_eznec_client as eznec_client
import momwire_nec2c_client as nec2c_client
import momwire_serve_client as mech
from momwire import _basis_names, _features
from momwire.deck import BASES, NEC2_BASES, build_solver, parse
from momwire.deck._solver import basis_entry
from momwire.dropin.nec2c import _portal, engine_scope
from momwire.dropin.nec5 import _printout, _serve, _shell

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PYPROJECT = ROOT / "pyproject.toml"

_spec = importlib.util.spec_from_file_location(
    "eznec_freeze_build", ROOT / "scripts" / "eznec_freeze" / "build.py"
)
FREEZE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FREEZE)

ALIASES = dict(_basis_names.BASIS_ALIASES)
OLD_NAMES = tuple(ALIASES)

# One small served deck per dialect.
NEC5_DECK = FIXTURES / "eznec" / "decks" / "0036_dipole-in-free-space.nec"
NEC4_DECK = FIXTURES / "eznec_nec42" / "0228_dipole-10m-free.nec"
NEC2_DECK = FIXTURES / "nec_portal" / "catalog_dipoles_invvee.deck"

# The NEC-4.2 slot (`golden_printouts._CLOCK`, FILL= / FACTOR= / RUN TIME =
# seconds) and the NEC-2 printout (FILL: / FACTOR: msec) print wall-clock
# readings; nothing else differs run to run.
_CLOCK = re.compile(
    r"(FILL=|FACTOR=|^ RUN TIME =|FILL:|FACTOR:)( *)\d+(?:\.\d+)?", re.M
)


def _note(old: str) -> str:
    return f" (renamed from {old})"


def _without_note(printout: str, old: str) -> str:
    """``printout`` with the deprecation note taken off every line carrying
    it; fails when no line carries it, so a missing note is not a pass."""
    note = _note(old)
    assert note in printout, f"no deprecation note for {old!r} in the printout"
    return printout.replace(note, "")


# --- the table --------------------------------------------------------------


def test_the_roster_is_canonical_and_dash_free():
    assert set(BASES) == {
        "bs2",
        "bs1",
        "hmatrix",
        "arrayblock",
        "sinusoidal",
        "sg",
        "razor2p",
        "pulse",
    }
    assert not any("-" in name for name in BASES)
    assert not set(ALIASES) & set(BASES)
    assert set(ALIASES.values()) <= set(BASES)


@pytest.mark.parametrize("old", OLD_NAMES)
def test_an_alias_warns_naming_the_canonical_name(old):
    new = ALIASES[old]
    with pytest.warns(DeprecationWarning, match=rf"'{re.escape(old)}'.*use '{new}'"):
        assert basis_entry(old) == basis_entry(new)


@pytest.mark.parametrize("new", sorted(BASES))
def test_a_canonical_name_does_not_warn(new):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        basis_entry(new)


def test_the_long_form_canonicalises_its_basis_part():
    assert _basis_names.canonical_engine("nec5-razor-2p", warn=False) == "nec5-razor2p"
    assert _basis_names.canonical_engine("nec42-bspline", warn=False) == "nec42-bs2"
    assert _basis_names.canonical_engine("nec2c-bspline-d1", warn=False) == "nec2c-bs1"
    assert _basis_names.canonical_engine("nec42-sinusoidal") == "nec42-sinusoidal"


@pytest.mark.parametrize("old", OLD_NAMES)
def test_the_feature_tables_read_an_alias_as_its_canonical_name(old):
    assert _features.basis_solver(old) == _features.basis_solver(ALIASES[old])


def test_the_nec2_client_restates_the_table():
    """The stdlib client cannot import momwire; its copy is pinned here."""
    assert nec2c_client.BASIS_ALIASES == {
        old: new for old, new in ALIASES.items() if new in NEC2_BASES
    }
    assert tuple(sorted(NEC2_BASES)) == nec2c_client.BASIS_NAMES


# --- the decisive gate: old name and new name, same bytes -------------------


def _nec2_aliases() -> list[str]:
    return [old for old, new in ALIASES.items() if new in NEC2_BASES]


@pytest.mark.parametrize("old", _nec2_aliases())
def test_python_api_answers_an_alias_as_its_canonical_name(old):
    model = parse(NEC2_DECK.read_text())
    with pytest.warns(DeprecationWarning):
        built_old = build_solver(model, basis=old)
    built_new = build_solver(model, basis=ALIASES[old])
    assert type(built_old.solver) is type(built_new.solver)
    z_old = built_old.solver.compute_port_solution()
    z_new = built_new.solver.compute_port_solution()
    assert repr(z_old) == repr(z_new)


@pytest.mark.parametrize("old", ["razor-2p"])
def test_python_api_answers_a_razor_alias_as_razor2p(old):
    """Razor serves the NEC-5 dialect only; its Python door there is
    `_serve.serve`."""
    from momwire.deck import parse_nec5

    deck = parse_nec5(_shell.read_deck(NEC5_DECK))
    with pytest.warns(DeprecationWarning):
        old_z = _serve.serve(deck, basis=old)
    new_z = _serve.serve(deck, basis="razor2p")
    assert repr(old_z) == repr(new_z)


def _launcher_render(prog: str, deck: Path, tmp_path: Path) -> str:
    """The printout a launcher NAMED ``prog`` writes: the thin client's own
    name rules (`momwire_eznec_client.main`) pick basis and dialect, and the
    engine answers in process, as the resident server's seam does."""
    seen = {}

    def served(deck_bytes, basis, idle, dialect):
        seen["route"] = (basis, dialect)
        seam = _shell.seam(
            basis=_serve.BASIS if basis is None else basis, dialect=dialect
        )
        return seam.answer(deck_bytes.decode("latin-1"), "")[0].encode("latin-1")

    out = tmp_path / f"{prog}.out"
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(eznec_client, "_served_bytes", served)
        mp.setattr("sys.argv", [prog, str(deck), str(out)])
        assert eznec_client.main(None) == 0
    finally:
        mp.undo()
    assert seen, "the in-process server never answered"
    return out.read_bytes().decode("latin-1")


@pytest.mark.parametrize("old", OLD_NAMES)
def test_a_nec5_launcher_named_by_an_alias_answers_as_the_canonical_one(old, tmp_path):
    new = ALIASES[old]
    printed_old = _launcher_render(f"momwire-nec5-{old}", NEC5_DECK, tmp_path)
    printed_new = _launcher_render(f"momwire-nec5-{new}", NEC5_DECK, tmp_path)
    assert "NEC ERROR" not in printed_new
    assert printed_new.split("\n")[1].split()[2] == new
    assert _without_note(printed_old, old) == printed_new


@pytest.mark.parametrize(
    "old", [old for old, new in ALIASES.items() if new in NEC2_BASES]
)
def test_a_nec42_launcher_named_by_an_alias_answers_as_the_canonical_one(old, tmp_path):
    new = ALIASES[old]
    printed_old = _launcher_render(f"momwire-nec42-{old}", NEC4_DECK, tmp_path)
    printed_new = _launcher_render(f"momwire-nec42-{new}", NEC4_DECK, tmp_path)
    assert "NEC ERROR" not in printed_new
    assert _CLOCK.sub(r"\1", _without_note(printed_old, old)) == _CLOCK.sub(
        r"\1", printed_new
    )


@pytest.mark.parametrize("old", ["razor-2p"])
def test_a_razor_alias_in_the_nec42_slot_refuses_as_razor2p_does(old, tmp_path):
    printed_old = _launcher_render(f"momwire-nec42-{old}", NEC4_DECK, tmp_path)
    printed_new = _launcher_render("momwire-nec42-razor2p", NEC4_DECK, tmp_path)
    assert "NEC ERROR" in printed_new and "'razor2p'" in printed_new
    assert _without_note(printed_old, old) == printed_new


def _nec2_render(prog: str, tmp_path: Path) -> str:
    out = tmp_path / f"{prog}.out"
    with engine_scope():
        code = _portal.main(
            [str(NEC2_DECK), str(out)],
            stdin=io.StringIO(""),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            prog=prog,
        )
    assert code == 0
    return out.read_bytes().decode("latin-1")


@pytest.mark.parametrize("old", _nec2_aliases())
def test_a_nec2_launcher_named_by_an_alias_answers_as_the_canonical_one(old, tmp_path):
    new = ALIASES[old]
    printed_old = _nec2_render(f"momwire-nec2c-{old}", tmp_path)
    printed_new = _nec2_render(f"momwire-nec2c-{new}", tmp_path)
    assert "ERROR" not in printed_new
    assert _CLOCK.sub(r"\1", _without_note(printed_old, old)) == _CLOCK.sub(
        r"\1", printed_new
    )


@pytest.mark.parametrize("old", _nec2_aliases())
def test_the_nec2_basis_flag_takes_an_alias(old):
    out = io.StringIO()
    with engine_scope():
        rest, code = _portal.configure_engine(["--basis", old], out, prog="x")
        assert code is None, out.getvalue()
        assert _portal._active_basis_name == ALIASES[old]
    assert nec2c_client.resolve_engine(["--basis", old], prog="x")[1] is None


@pytest.mark.parametrize("old", ["razor-2p"])
def test_the_nec2_portal_refuses_a_razor_alias_as_a_razor_basis(old):
    out = io.StringIO()
    with engine_scope():
        _rest, code = _portal.configure_engine([], out, prog=f"momwire-nec2c-{old}")
    assert code == 3
    assert "does not serve the nec2 dialect" in out.getvalue()


def test_line2_stamps_the_canonical_name_and_notes_the_spelling():
    line = _printout.engine_stamp("razor-2p")
    fields = line.split()
    assert fields[2] == "razor2p"
    assert line.endswith("(renamed from razor-2p)")
    assert len(line) <= 79
    assert "renamed" not in _printout.engine_stamp("razor2p")


# --- razor-nec5: retired, refused by name on every route ---------------------

RETIRED = "basis 'razor-nec5' was renamed and is no longer accepted; use 'razor2p'"


def test_razor_nec5_is_not_an_alias():
    assert "razor-nec5" not in ALIASES and "razor-nec5" not in BASES
    assert _basis_names.RETIRED_BASES == {"razor-nec5": "razor2p"}


def test_razor_nec5_refuses_on_the_python_api():
    model = parse(NEC2_DECK.read_text())
    with pytest.raises(ValueError, match=re.escape(RETIRED)):
        build_solver(model, basis="razor-nec5")
    with pytest.raises(ValueError, match=re.escape(RETIRED)):
        basis_entry("razor-nec5")


@pytest.mark.parametrize(
    ("prog", "deck"),
    [("momwire-nec5-razor-nec5", NEC5_DECK), ("momwire-nec42-razor-nec5", NEC4_DECK)],
)
def test_razor_nec5_refuses_in_the_printout_of_a_nec5_or_nec42_launcher(
    prog, deck, tmp_path
):
    printed = _launcher_render(prog, deck, tmp_path)
    assert "NEC ERROR" in printed
    assert RETIRED in printed
    assert "INPUT PARAMETERS" not in printed


@pytest.mark.parametrize("argv", [["--basis", "razor-nec5"], []])
def test_razor_nec5_refuses_at_the_nec2_probe(argv):
    out = io.StringIO()
    with engine_scope():
        _rest, code = _portal.configure_engine(
            argv, out, prog="momwire-nec2c-razor-nec5"
        )
    assert code == 3
    assert RETIRED in out.getvalue()
    _engine, error = nec2c_client.resolve_engine(argv, prog="momwire-nec2c-razor-nec5")
    assert error is not None and RETIRED in error


# --- the integration lane: the real processes --------------------------------


def _venv_script(name: str) -> Path:
    path = Path(sys.executable).with_name(name)
    if not path.exists():
        pytest.skip(f"{name} is not installed beside {sys.executable}")
    return path


def _process_render(prog: str, deck: Path, out: Path) -> str:
    """The printout a process answers for a launcher named ``prog``.

    NEC-5 and NEC-4.2: the thin client's own name rules, then its stock
    one-shot child (`momwire_eznec_client._one_shot`, the rung every
    resident failure falls to), so no resident server outlives the test.
    NEC-2: a copy of the console script itself, which is the engine."""
    if "nec2c" in prog:
        copy = out.with_name(prog)
        shutil.copy(_venv_script("momwire-nec2c"), copy)
        subprocess.run(
            [sys.executable, str(copy), str(deck), str(out)], check=True, timeout=600
        )
    else:
        is_nec4, basis = mech.filename_nec4(prog)
        if not is_nec4:
            basis = mech.filename_basis(prog, "nec5-")
        eznec_client._one_shot(
            str(deck), str(out), basis, "nec4" if is_nec4 else "nec5"
        )
    return out.read_bytes().decode("latin-1")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("family", "old", "deck"),
    [
        ("momwire-nec5", "razor-2p", NEC5_DECK),
        ("momwire-nec5", "bspline", NEC5_DECK),
        ("momwire-nec42", "sinusoidal-galerkin", NEC4_DECK),
        ("momwire-nec2c", "bspline-d1", NEC2_DECK),
    ],
)
def test_a_process_named_by_an_alias_answers_as_the_canonical_one(
    family, old, deck, tmp_path
):
    """The filename alias is read by the parser, not shipped as a script."""
    new = ALIASES[old]
    printed = {
        name: _process_render(f"{family}-{name}", deck, tmp_path / f"{name}.out")
        for name in (old, new)
    }
    clock = (lambda t: _CLOCK.sub(r"\1", t)) if family != "momwire-nec5" else str
    assert "ERROR" not in printed[new]
    assert clock(_without_note(printed[old], old)) == clock(printed[new])


# --- nothing shipped carries an old name -------------------------------------

_OLD_TOKEN = re.compile(
    r"(?<![\w])(razor-2p|razor-nec5|bspline-d1|sinusoidal-galerkin|bspline)(?![\w-])"
)


# The deprecated console scripts pip still installs for the deprecation window
# (pyproject's DEPRECATED block): exactly these, and nowhere but pip. Removing
# the aliases removes this set and that block together.
DEPRECATED_SCRIPTS = frozenset(
    f"momwire-{dialect}-{old}"
    for dialect, olds in (
        ("nec5", ("razor-2p", "bspline", "bspline-d1", "sinusoidal-galerkin")),
        ("nec42", ("bspline", "bspline-d1", "sinusoidal-galerkin")),
        ("nec2c", ("bspline", "bspline-d1", "sinusoidal-galerkin")),
    )
    for old in olds
)


def _console_scripts() -> dict[str, str]:
    return tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]


def test_pip_installs_exactly_the_deprecated_scripts_each_beside_its_canonical_one():
    scripts = _console_scripts()
    old = {n for n in scripts if _OLD_TOKEN.search(n)}
    assert old == DEPRECATED_SCRIPTS
    for name in DEPRECATED_SCRIPTS:
        dialect, basis = name.removeprefix("momwire-").split("-", 1)
        canonical = f"momwire-{dialect}-{ALIASES[basis]}"
        assert scripts[name] == scripts[canonical], (name, canonical)


def _shipped_names() -> dict[str, str]:
    scripts = _console_scripts()
    names = {
        name: "pyproject console script"
        for name in scripts
        if name not in DEPRECATED_SCRIPTS
    }
    names.update({stem: "drop-in launcher" for stem in FREEZE.launcher_stems()})
    names[FREEZE.ENGINE_NAME] = "drop-in engine"
    cmake = (ROOT / "standalone" / "CMakeLists.txt").read_text()
    for target in re.findall(r"set\(MW_EXE (\S+)\)", cmake):
        names[target] = "standalone CMake target"
    for define in re.findall(
        r'MW_EXE_(?:PROGRAM|LAUNCHER|BASIS_NAME)="([^"]+)"', cmake
    ):
        names[define] = "standalone -version / line 2 name"
    for path in (ROOT / "standalone" / "release").iterdir():
        names[path.name] = "standalone release file"
    return names


def test_no_shipped_name_carries_an_old_basis_name():
    names = _shipped_names()
    assert len(names) > 40, "the census found too few names to mean anything"
    stale = {n: why for n, why in names.items() if _OLD_TOKEN.search(n)}
    assert not stale, stale


def test_the_drop_in_signs_one_file_per_launcher_and_the_engine():
    stems = FREEZE.launcher_stems()
    assert len(stems) == len(set(stems)) == 6
    assert not any(_OLD_TOKEN.search(stem) for stem in stems)


# --- the census: no new old name in what momwire prints or documents ----------
#
# What momwire PRINTS is its string literals (Python constants, C++ string
# literals outside comments) and the README text the freeze generates; what it
# DOCUMENTS is the user-facing pages below. Comments and dated records are
# history, not output, and are not read. A bare `bspline` is the solver
# family's name as well as the old basis name, so only the basis-name shapes
# of it count: the whole literal, quoted or backticked, `--basis bspline`,
# `basis="bspline"`, and the long forms.

_OLD_IN_TEXT = re.compile(
    r"(?<![\w.])(?:razor-2p|razor-nec5|bspline-d1|sinusoidal-galerkin)(?![\w-])"
    r"|(?<![\w])(?:nec5|nec42|nec4|nec2c|eznec|dropin)-bspline(?![\w])"
    r"|--basis[ =]bspline(?![\w-])|basis\s*=\s*['\"]bspline['\"]"
    r"|`bspline`|^bspline$|['\"]bspline['\"]|momwire \S+ bspline "
)


def _py_strings(path: Path):
    import ast

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant):
                    yield part.value


def _cxx_strings(path: Path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        code = line.split("//", 1)[0]
        yield from re.findall(r'"((?:[^"\\]|\\.)*)"', code)


def _doc_lines(path: Path):
    yield from path.read_text(encoding="utf-8", errors="replace").splitlines()


def _census_sources():
    for path in sorted((ROOT / "src").rglob("*.py")):
        yield path, _py_strings
    yield ROOT / "scripts" / "eznec_freeze" / "build.py", _py_strings
    for pattern in ("src/momwire/*.cpp", "src/momwire/*.h", "standalone/*.cpp"):
        for path in sorted(ROOT.glob(pattern)):
            yield path, _cxx_strings
    for pattern in ("standalone/*.h", "standalone/*.inc"):
        for path in sorted(ROOT.glob(pattern)):
            yield path, _cxx_strings
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "capability-matrix.md",
        ROOT / "docs" / "razor-solver.md",
        ROOT / "support" / "hierarchy.toml",
    ]
    docs += sorted((ROOT / "site" / "src" / "content" / "docs").rglob("*.md"))
    docs += sorted(
        p for p in (ROOT / "standalone" / "release").iterdir() if p.is_file()
    )
    for path in docs:
        yield path, _doc_lines


def _census() -> dict[str, int]:
    found: dict[str, int] = {}
    for path, reader in _census_sources():
        for text in reader(path):
            n = len(_OLD_IN_TEXT.findall(text))
            if n:
                key = path.relative_to(ROOT).as_posix()
                found[key] = found.get(key, 0) + n
    return found


# Every occurrence that is meant to be there, by file: the alias table and its
# restated copies, the deprecation notes that must name the old spelling,
# the solver FAMILY name `bspline` (`Feature.solvers`, `FEATURE_SOLVER`), and
# history (names a release shipped and later removed).
ALLOWED = {
    "README.md": 1,  # the aliases, named as aliases
    "docs/razor-solver.md": 4,  # history: names shipped and retired
    "scripts/eznec_freeze/build.py": 8,  # the README's rename and REMOVED tables
    "site/src/content/docs/reference/eznec-nec5.md": 8,  # the rename and REMOVED tables
    "site/src/content/docs/reference/portal-usage.md": 2,  # history: retired nec2c razor names
    "src/momwire/_basis_names.py": 6,  # the alias table
    "src/momwire/_features.py": 18,  # the solver family `bspline`
    "src/momwire/bspline.py": 1,  # FEATURE_SOLVER, the family name
    "src/momwire/dropin/nec5/_printout.py": 1,  # the deprecation note's example
    "src/momwire/dropin/nec5/_serve.py": 1,  # a dated measurement in the module docstring
    "src/momwire_nec2c_client.py": 4,  # the alias table and retired name, restated
    "standalone/nec2c_cli.inc": 6,  # the alias table, restated; razor-nec5 refused by name
}


def test_no_new_old_name_in_what_momwire_prints_or_documents():
    found = _census()
    grown = {
        path: (n, ALLOWED.get(path, 0))
        for path, n in found.items()
        if n > ALLOWED.get(path, 0)
    }
    assert not grown, (
        "an old basis name (momwire#2091) appeared in printed text or docs; "
        "use the canonical name, or, for a deliberate one, raise ALLOWED: "
        f"{grown}"
    )


def test_the_census_allowlist_is_tight():
    """A count that fell is a stale allowance, which would hide a new one."""
    found = _census()
    assert {p: n for p, n in ALLOWED.items() if found.get(p, 0) != n} == {}
