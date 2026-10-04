"""The drop-in bundle's launcher roster and README, without a freeze.

`scripts/eznec_freeze/build.py` names every launcher the Windows bundle ships
and writes the README.txt beside them.  The freeze itself runs only on the
Windows lane (`eznec-dropin.yml`), so the roster and the README's table are
gated here, from the same constants the copy loop reads (momwire#1295
phase 4: `momwire-nec5*` / `momwire-nec4*` primary, `momwire-eznec*`
deprecated but still shipped).
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

from momwire.deck._solver import (
    BASES,
    NEC2_BASES,
    basis_from_program_name,
    dialect_from_program_name,
    nec4_basis_from_program_name,
)
from momwire.eznec import _serve

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "scripts" / "eznec_freeze" / "build.py"


def _build():
    spec = importlib.util.spec_from_file_location("eznec_freeze_build_readme", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _console_scripts() -> dict[str, str]:
    return tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]


def _resolved(stem: str) -> tuple[str, str]:
    """(dialect, basis) a launcher stem answers with, the way the launcher
    and `entry.py` read it."""
    dialect = dialect_from_program_name(stem)
    if dialect == "nec4":
        suffix = nec4_basis_from_program_name(stem)
    elif "eznec-" in stem:
        suffix = basis_from_program_name(stem, "eznec-")
    else:
        suffix = basis_from_program_name(stem, "nec5-")
    return dialect, _serve.BASIS if suffix is None else suffix


def test_the_bundle_ships_seven_launchers_in_three_families():
    build = _build()
    stems = build.launcher_stems()
    assert len(stems) == len(set(stems)) == 7
    assert set(build.NEC5_LAUNCHERS) == {"momwire-nec5", "momwire-nec5-razor-2p"}
    assert set(build.NEC4_LAUNCHERS) == {"momwire-nec4", "momwire-nec4-sinusoidal"}
    deprecated = set(stems) - set(build.NEC5_LAUNCHERS) - set(build.NEC4_LAUNCHERS)
    assert deprecated == {
        "momwire-eznec",
        "momwire-eznec-razor-2p",
        "momwire-eznec-razor-nec5",
    }
    assert build.ENGINE_NAME not in stems


def test_every_launcher_answers_the_slot_and_basis_its_family_claims():
    build = _build()
    for stem in build.NEC5_LAUNCHERS:
        dialect, basis = _resolved(stem)
        assert dialect == "nec5" and basis in BASES, stem
    for stem in build.NEC4_LAUNCHERS:
        dialect, basis = _resolved(stem)
        assert dialect == "nec4" and basis in NEC2_BASES, stem
    assert _resolved("momwire-nec4") == ("nec4", _serve.BASIS)
    assert _resolved("momwire-nec4-sinusoidal") == ("nec4", "sinusoidal")


def test_a_deprecated_name_answers_exactly_as_its_successor():
    """Deprecated in prose only: each old name resolves to the same dialect
    and the same solver entry as the primary name the README equates it
    with (``razor-nec5`` and ``razor-2p`` are two spellings of one entry)."""
    build = _build()
    pairs = {build.NAME: build.DEPRECATED_SUCCESSORS[""]}
    pairs.update(
        {
            f"{build.NAME}-{b}": build.DEPRECATED_SUCCESSORS[b]
            for b in build.SHIPPED_VARIANTS
        }
    )
    assert set(build.DEPRECATED_SUCCESSORS) == {"", *build.SHIPPED_VARIANTS}
    for old, new in pairs.items():
        assert new in build.NEC5_LAUNCHERS, (old, new)
        old_dialect, old_basis = _resolved(old)
        new_dialect, new_basis = _resolved(new)
        assert old_dialect == new_dialect == "nec5", (old, new)
        assert BASES[old_basis] == BASES[new_basis], (old, new)


def test_the_primary_names_are_also_console_scripts():
    """The bundle and `pip install momwire` spell the primary names alike."""
    scripts = _console_scripts()
    build = _build()
    for stem in (*build.NEC5_LAUNCHERS, *build.NEC4_LAUNCHERS):
        assert scripts.get(stem) == "momwire_eznec_client:main", stem


def test_the_readmes_known_bases_are_the_rosters():
    """The README's per-slot lists are restated in build.py (it must not
    import what it freezes); held here to the console-script families, which
    `test_eznec_nec42_1295` and `test_nec5_command_family_1239` hold to the
    roster in turn."""
    build = _build()
    scripts = _console_scripts()
    nec5 = {
        basis_from_program_name(k, "nec5-")
        for k in scripts
        if k.startswith("momwire-nec5")
    } - {None}
    nec4 = {
        nec4_basis_from_program_name(k)
        for k in scripts
        if dialect_from_program_name(k) == "nec4"
    } - {None}
    assert set(build.NEC5_KNOWN_BASES) == nec5
    assert set(build.NEC4_KNOWN_BASES) == nec4 == set(NEC2_BASES)


def test_the_readme_table_has_one_row_per_shipped_executable():
    build = _build()
    text = build.readme_text("9.9.9", "SIGNING: this build is unsigned.", ".exe")
    rows = [
        line.split()[0]
        for line in text.splitlines()
        if line.startswith("  momwire-") and line.split()[0].endswith(".exe")
    ]
    assert sorted(rows) == sorted(
        f"{s}.exe" for s in (*build.launcher_stems(), build.ENGINE_NAME)
    )
    assert "momwire-eznec 9.9.9 " in text
    assert "DEPRECATED" in text
    for stem in ("momwire-eznec", "momwire-eznec-razor-2p", "momwire-eznec-razor-nec5"):
        (row,) = [ln for ln in text.splitlines() if ln.startswith(f"  {stem}.exe ")]
        assert row.split()[1] == "=", row
    for basis in (*build.NEC5_KNOWN_BASES, *build.NEC4_KNOWN_BASES):
        assert basis in text, basis
    # The table and the lists are generated: hold them to the README's own
    # ~70-column wrap.
    table = build.launcher_table(".exe")
    assert max(len(line) for line in table.splitlines()) <= 72
