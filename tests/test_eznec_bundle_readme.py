"""The drop-in bundle's launcher roster and README, without a freeze.

`scripts/eznec_freeze/build.py` names every launcher the Windows bundle ships
and writes the README.txt beside them.  The freeze itself runs only on the
Windows lane (`eznec-dropin.yml`), so the roster and the README's table are
gated here, from the same constants the copy loop reads (momwire#1295
phase 4: `momwire-nec5*` / `momwire-nec42*`; the deprecated `momwire-eznec*`
names were dropped in momwire#1583).
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
SITE_PAGE = REPO / "site" / "src" / "content" / "docs" / "reference" / "eznec-nec5.md"


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


def test_the_bundle_ships_four_launchers_in_two_families():
    """momwire#1583 dropped the three deprecated `momwire-eznec*` names; the
    signing post-condition's "all N executables" is this count plus one."""
    build = _build()
    stems = build.launcher_stems()
    assert len(stems) == len(set(stems)) == 4
    assert set(build.NEC5_LAUNCHERS) == {"momwire-nec5", "momwire-nec5-razor-2p"}
    assert set(build.NEC4_LAUNCHERS) == {"momwire-nec42", "momwire-nec42-sinusoidal"}
    assert set(stems) == {*build.NEC5_LAUNCHERS, *build.NEC4_LAUNCHERS}
    assert not any("eznec" in stem for stem in stems)
    assert build.ENGINE_NAME == "momwire-engine"
    assert build.ENGINE_NAME not in stems
    assert build.NAME == "momwire"
    assert build.LAUNCHER in stems
    # The smoke's gate 4 reads SHIPPED_VARIANTS as the copies of LAUNCHER.
    assert build.SHIPPED_VARIANTS == ("razor-2p",)


def test_every_launcher_answers_the_slot_and_basis_its_family_claims():
    build = _build()
    for stem in build.NEC5_LAUNCHERS:
        dialect, basis = _resolved(stem)
        assert dialect == "nec5" and basis in BASES, stem
    for stem in build.NEC4_LAUNCHERS:
        dialect, basis = _resolved(stem)
        assert dialect == "nec4" and basis in NEC2_BASES, stem
    assert _resolved("momwire-nec42") == ("nec4", _serve.BASIS)
    assert _resolved("momwire-nec42-sinusoidal") == ("nec4", "sinusoidal")


def test_the_engine_name_selects_no_basis():
    """``momwire-engine`` carries neither marker, so the frozen engine started
    under its own name serves the default; the launchers say ``--basis``."""
    build = _build()
    assert _resolved(build.ENGINE_NAME) == ("nec5", _serve.BASIS)


def test_the_primary_names_are_also_console_scripts():
    """The bundle and `pip install momwire` spell the primary names alike."""
    scripts = _console_scripts()
    build = _build()
    for stem in (*build.NEC5_LAUNCHERS, *build.NEC4_LAUNCHERS):
        assert scripts.get(stem) == "momwire_eznec_client:main", stem


def _site_known_bases() -> dict[str, set[str]]:
    """The per-slot lists of bases a copy can name, read from the reference
    page's ``text`` block (momwire#1593 moved them out of the README, which
    now links the page)."""
    text = SITE_PAGE.read_text()
    block = text.split("```text\nNEC-5 slot:", 1)[1].split("```", 1)[0]
    slots: dict[str, set[str]] = {}
    slot = "NEC-5"
    for line in ("NEC-5 slot:" + block).splitlines():
        for label in ("NEC-5 slot:", "NEC-4.2 slot:"):
            if line.startswith(label):
                slot = label.removesuffix(" slot:")
                line = line.removeprefix(label)
        slots.setdefault(slot, set()).update(line.split())
    return slots


def test_the_reference_pages_known_bases_are_the_rosters():
    """The reference page's per-slot lists, held to the console-script
    families, which `test_eznec_nec42_1295` and
    `test_nec5_command_family_1239` hold to the roster in turn.  The NEC-5
    list leaves out the bases that refuse every deck in that dialect
    (`sinusoidal`, `pulse`: centre-fed only) and the deprecated `razor-nec5`
    spelling; the NEC-4.2 list is `NEC2_BASES`."""
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
    slots = _site_known_bases()
    assert slots["NEC-5"] == nec5
    assert slots["NEC-4.2"] == nec4 == set(NEC2_BASES)


def test_the_readme_is_install_steps_and_a_link():
    """momwire#1593 (Dan AC6LA): the README says how to install under EZNEC,
    SimNEC and 4nec2 and points at the reference page for the rest; the
    reference material it used to repeat lives only there."""
    build = _build()
    for target in build.TARGETS:
        text = build.readme_text("9.9.9", "SIGNING: unsigned.", ".exe", target=target)
        assert text.rstrip().endswith("https://momwire.dev/reference/eznec-nec5/")
        for gone in (".GMO joins", "REPRODUCTION IS NOT ACCURACY", "MAKING ANOTHER"):
            assert gone not in text, gone
        for kept in ("INSTALLING FOR 4NEC2", "SIMNEC.", "External NEC-5"):
            assert kept in text, kept
    page = SITE_PAGE.read_text()
    for moved in ("Ward AE6TY's port draws ends", "Reproduction is not accuracy"):
        assert moved in page, moved


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
    assert text.startswith("momwire 9.9.9 ")
    # Not an EZNEC-only bundle: every host it serves is named up front.
    head = text.split("SIGNING:", 1)[0]
    for host in ("EZNEC", "SimNEC", "4nec2"):
        assert host in head, host
    # The table is generated: hold them to the README's own
    # ~70-column wrap.
    table = build.launcher_table(".exe")
    assert max(len(line) for line in table.splitlines()) <= 72


def test_only_the_win7_readme_states_the_win7_prerequisites():
    """momwire#1456: the Windows 7 zip names what the PC needs; the release
    zip does not, and an unknown target refuses rather than defaulting."""
    build = _build()
    args = ("9.9.9", "SIGNING: this build is unsigned.", ".exe")
    standard = build.readme_text(*args)
    assert standard == build.readme_text(*args, "standard")
    assert standard == build.readme_text(*args, "standard", True)
    assert "WINDOWS 7 BUILD" not in standard
    for bundled in (False, True):
        win7 = build.readme_text(*args, "win7", bundled)
        block = build.win7_prerequisites(bundled)
        for needed in (
            "Windows 7 SP1, 64-bit",
            "KB2533623",
            "KB2999226",
            "2015-2022 x64",
        ):
            assert needed in win7, (bundled, needed)
        # The Win7 README is the standard one plus the prerequisites, nothing else.
        assert win7.replace(block, "", 1) == standard
        assert max(len(line) for line in block.splitlines()) <= 72
    # The UCRT line says which this bundle is, never both.
    assert "carries its own copy" in build.win7_prerequisites(True)
    assert "does not carry its own copy" in build.win7_prerequisites(False)
    try:
        build.readme_text(*args, "win8")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown bundle target must refuse")


def test_the_readme_says_which_names_changed():
    """momwire#1583: anyone with an old engine path finds the new name."""
    build = _build()
    text = build.readme_text("9.9.9", "SIGNING: this build is unsigned.", ".exe")
    assert "NAMES CHANGED IN 0.80.0" in text
    for old, new in (
        ("momwire-eznec-windows.zip", "momwire-windows.zip"),
        ("momwire-eznec-windows-win7.zip", "momwire-windows-win7.zip"),
        ("momwire-eznec-engine.exe", "momwire-engine.exe"),
        ("momwire-eznec.exe", "momwire-nec5.exe"),
        ("momwire-eznec-razor-2p.exe", "momwire-nec5-razor-2p.exe"),
        ("momwire-eznec-razor-nec5.exe", "momwire-nec5-razor-2p.exe"),
    ):
        (row,) = [ln for ln in text.splitlines() if f" {old} " in ln]
        assert row.rstrip().endswith(new), row
    # Outside that paragraph the old names do not appear at all.
    rest = text.split("The momwire-nec5 and momwire-nec42 launcher names", 1)[1]
    assert "momwire-eznec" not in rest


def test_the_first_run_note_reads_as_dan_worded_it():
    """momwire#1583 item 4, line for line."""
    build = _build()
    lines = build.readme_text("9.9.9", "SIGNING: unsigned.", ".exe").splitlines()
    at = lines.index('FIRST RUN (if you did not "unblock" as described above):')
    assert lines[at + 1] == 'Windows will show "Windows protected your PC", and you'
    assert lines[at + 2] == (
        'have to click "More info" and then "Run anyway". Nothing is wrong'
    )
