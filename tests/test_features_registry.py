"""The experimental-feature registry (`momwire._features`) and the support
hierarchy (`support/hierarchy.toml`).

What is held here:

- every feature is off by default, named ``MOMWIRE_X_<NAME>``, with a reach
  inside the hierarchy and a kind the split rule knows;
- the TOML and the module carry the same levels in the same order, and every
  CI job is either under a level or listed as unassigned;
- the keyword wins over the environment, and an alias is honoured;
- `current_level` / `certified` place each context where the hierarchy says;
- a native feature outside its reach refuses, a Python one runs;
- line 2 of a printout names the features that are on, and ONLY then: with
  every flag off the printout is the one the golden corpus froze;
- a standalone executable refuses any feature, in the registry's sentence.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

from momwire import _features as F
from momwire.dropin.nec5 import _printout

ROOT = Path(__file__).resolve().parents[1]
HIERARCHY = ROOT / "support" / "hierarchy.toml"
WORKFLOWS = ROOT / ".github" / "workflows"
EXE_SOURCE = ROOT / "standalone" / "momwire_razor2p.cpp"


@pytest.fixture
def no_features(monkeypatch):
    """Every feature variable (and alias) unset."""
    for f in F.FEATURES.values():
        for var in (f.env, *f.aliases):
            monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _load_toml():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: the floor, not a CI lane
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(HIERARCHY.read_text(encoding="utf-8"))


# --- the registry -------------------------------------------------------------


def test_every_default_is_off(no_features):
    assert F.active() == []
    for name in F.FEATURES:
        assert F.enabled(name) is False


def test_the_env_names_follow_the_pattern():
    for name, f in F.FEATURES.items():
        assert f.name == name
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name)
        assert f.env == F.ENV_PREFIX + name.upper()
        for alias in f.aliases:
            assert alias.startswith("MOMWIRE_") and not alias.startswith(F.ENV_PREFIX)


def test_reach_kind_and_dates_are_valid():
    import datetime

    for f in F.FEATURES.values():
        assert 1 <= f.reach <= len(F.LEVELS), f
        assert f.kind in F.KINDS, f
        assert f.issue > 0
        datetime.date.fromisoformat(f.since)
        assert f.stable == (f.reach == len(F.LEVELS))


def test_every_offered_feature_carries_its_front_end_fields():
    # momwire#1869: a front end derives its switch from these, so an offered
    # feature must say what to show and how far it moves the answer.
    import math

    for f in F.FEATURES.values():
        assert isinstance(f.offer, bool), f
        assert math.isfinite(f.drift) and f.drift >= 0.0, f
        if f.offer:
            assert f.label and f.summary and f.kwarg, f
            assert "(faster)" not in f.label, f


def test_every_feature_is_below_the_executables():
    # The executables' refusal is a generic scan, which is right only while
    # no feature reaches an executable's level.
    first_exe = F._level("exe-nec5-bspline-razor-2p-windows")
    assert all(f.reach < first_exe for f in F.FEATURES.values())


def test_the_kwarg_overrides_the_env(no_features):
    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "1")
    assert F.enabled("offset_reuse") is True
    assert F.enabled("offset_reuse", False) is False
    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "0")
    assert F.enabled("offset_reuse") is False
    assert F.enabled("offset_reuse", True) is True


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_the_on_values(no_features, value):
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", value)
    assert F.active() == ["pair_reuse"]


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "2"])
def test_the_off_values(no_features, value):
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", value)
    assert F.active() == []


def test_an_alias_is_honoured_and_the_canonical_name_wins(no_features):
    no_features.setenv("MOMWIRE_EXACT_KERNEL_SERIES", "1")
    assert "exact_kernel_series" in F.active()
    no_features.setenv("MOMWIRE_X_EXACT_KERNEL_SERIES", "0")
    assert "exact_kernel_series" not in F.active()


def test_offset_reuse_reads_the_env_through_the_registry(no_features):
    from momwire.bspline import BSplineSolver

    wire = [[[0.0, -2.5, 10.0], [0.0, 2.5, 10.0]]]
    kw = dict(wires=wire, n_per_edge_per_wire=[[21]], wire_radius=1e-3)
    assert BSplineSolver(**kw)._offset_reuse is False
    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "1")
    assert BSplineSolver(**kw)._offset_reuse is True
    assert BSplineSolver(**kw, offset_reuse=False)._offset_reuse is False


# --- the hierarchy ------------------------------------------------------------


def test_the_toml_and_the_module_carry_the_same_levels():
    levels = _load_toml()["level"]
    assert [(lv["id"], lv["description"]) for lv in levels] == [
        (lv.id, lv.description) for lv in F.LEVELS
    ]
    for lv in levels:
        assert set(lv) == {"id", "description", "jobs"}, lv["id"]
        # ak-linux-sse2 names none: no job runs the Python suite on the sse2
        # variant yet (the nightly's sse2 rotation, rollout step 5).
        assert lv["jobs"] or lv["id"] == "ak-linux-sse2", lv["id"]


def _workflow_jobs():
    """Every ``<file>:<job id>`` under .github/workflows, read off the
    two-space-indented keys of each file's top-level ``jobs:`` block."""
    jobs = set()
    for path in sorted(WORKFLOWS.glob("*.yml")):
        in_jobs = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.match(r"^\S", line):
                in_jobs = line.rstrip() == "jobs:"
                continue
            m = re.match(r"^  ([A-Za-z0-9_-]+):\s*(#.*)?$", line)
            if in_jobs and m:
                jobs.add(f"{path.name}:{m.group(1)}")
    return jobs


def test_every_ci_job_is_placed_once_or_unassigned():
    data = _load_toml()
    assigned = [job for lv in data["level"] for job in lv["jobs"]]
    unassigned = set(data["unassigned"])
    jobs = _workflow_jobs()
    assert jobs, "no workflow jobs read"
    assert set(assigned) | unassigned == jobs
    assert not set(assigned) & unassigned
    # A job sits under two levels only when it is a matrix over platforms.
    twice = {j for j in assigned if assigned.count(j) > 1}
    assert twice <= {"wheels.yml:wheel_smoke"}, twice


@pytest.mark.parametrize(
    ("context", "platform", "basis", "variant", "level"),
    [
        ("python", "linux", None, "avx2", 1),
        ("python", "linux", None, "sse2", 2),
        ("python", "linux", None, "legacy", 2),
        ("python", "linux", None, None, 2),  # no accelerator: pure Python
        ("python", "win32", None, "avx2", 3),
        ("python", "win32", None, "sse2", 3),
        ("python", "darwin", None, "legacy", 5),
        ("dropin", "linux", None, "avx2", 4),
        ("dropin", "linux", None, "sse2", 4),
        ("dropin", "win32", None, None, 4),
        ("dropin", "darwin", None, None, 5),
        ("exe", "win32", "razor2p", None, 6),
        ("exe", "win32", "bspline_nec5", None, 6),
        ("exe", "linux", "razor2p", None, 7),
        ("exe", "darwin", "bspline_nec5", None, 7),
        ("exe", "linux", "bspline_nec42", None, 8),
        ("exe", "win32", "sin42", None, 9),
        # the long forms (momwire#1831)
        ("exe", "win32", "nec5-razor-2p", None, 6),
        ("exe", "linux", "nec5-bspline", None, 7),
        ("exe", "darwin", "nec42-bspline", None, 8),
        ("exe", "linux", "nec42-sinusoidal", None, 9),
    ],
)
def test_current_level(context, platform, basis, variant, level):
    kw = dict(basis=basis, platform=platform, variant=variant)
    assert F.current_level(context, **kw) == level
    assert F.certified("offset_reuse", context, **kw) == (level <= 1)


def test_the_level_numbers_follow_the_ids():
    assert [F._level(lv.id) for lv in F.LEVELS] == list(range(1, len(F.LEVELS) + 1))


def test_the_default_variant_is_the_loaded_one(monkeypatch):
    from momwire import _accel

    for loaded, level in (("avx2", 1), ("sse2", 2), (None, 2)):
        monkeypatch.setattr(_accel, "VARIANT", loaded)
        assert F.current_level("python", platform="linux") == level


# --- the split rule -------------------------------------------------------------


def test_a_native_feature_outside_its_reach_refuses(no_features, monkeypatch):
    monkeypatch.setitem(
        F.FEATURES,
        "probe",
        F.Feature("probe", 1, "MOMWIRE_X_PROBE", "native", 1, "2026-11-06"),
    )
    no_features.setenv("MOMWIRE_X_PROBE", "1")
    # The drop-in is past reach 1 on every platform.
    with pytest.raises(F.FeatureNotCertified, match=r"^probe is not certified"):
        F.enabled("probe", context="dropin")
    assert F.enabled("probe", False, context="dropin") is False


def test_a_python_feature_outside_its_reach_runs(no_features):
    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "1")
    assert F.FEATURES["offset_reuse"].kind == "python"
    assert F.enabled("offset_reuse", context="dropin") is True


# --- the stamp ------------------------------------------------------------------


def _bare_stamp(basis):
    """Line 2 as main writes it: no feature tail."""
    version = _printout._momwire_version()
    variant = _printout._accelerator_variant()
    return f" momwire {version} {basis} {variant}"


def test_the_stamp_shows_only_when_a_feature_is_on(no_features):
    assert F.stamp() == "" and F.stamp("dropin") == ""
    assert _printout.engine_stamp("bspline") == _bare_stamp("bspline")
    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "1")
    # The drop-in is past reach 1 everywhere, so the platform is named.
    with F.context_scope("dropin"):
        assert F.stamp() == "+offset_reuse uncertified on this platform"
        assert _printout.engine_stamp("bspline") == (
            _bare_stamp("bspline") + " +offset_reuse uncertified on this platform"
        )
    # Outside a drop-in entry the route is Python's: level 1 on Linux avx2.
    plain = "" if F.certified("offset_reuse") else " uncertified on this platform"
    assert F.stamp() == "+offset_reuse" + plain
    no_features.setenv("MOMWIRE_X_PAIR_REUSE", "1")
    assert F.stamp("dropin") == "+pair_reuse+offset_reuse uncertified on this platform"


def test_the_dropin_entry_runs_as_the_dropin(no_features):
    from momwire.dropin.nec5 import _shell

    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "1")
    deck = (ROOT / "tests" / "fixtures" / _GOLDEN_CASES[0][0]).read_text("latin-1")
    line2 = _shell.render(deck, basis="bspline").split("\n")[1]
    assert (
        line2 == _bare_stamp("bspline") + " +offset_reuse uncertified on this platform"
    )


def test_a_long_basis_keeps_the_feature_tail(no_features):
    no_features.setenv("MOMWIRE_X_OFFSET_REUSE", "1")
    with F.context_scope("dropin"):
        line = _printout.engine_stamp("b" * 200)
    assert len(line) <= 79
    assert line.endswith(" +offset_reuse uncertified on this platform")


# The flags-off printout: the golden corpus (scripts/golden_printouts.py),
# whose byte gate masks line 2, plus line 2 itself held to main's form.
_GOLDEN_CASES = [
    ("eznec/decks/0036_dipole-in-free-space.nec", "bspline"),
    ("eznec/decks/0036_dipole-in-free-space.nec", "razor-2p"),
]


def _golden_module():
    spec = importlib.util.spec_from_file_location(
        "golden_printouts", ROOT / "scripts" / "golden_printouts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(("deck", "basis"), _GOLDEN_CASES)
def test_the_flag_off_printout_is_unchanged(no_features, deck, basis):
    from golden_compare import undefined_blanked
    from threadpoolctl import threadpool_limits

    from momwire.dropin.nec5 import _shell

    G = _golden_module()
    assert (deck, basis) in G.CASES
    text = _shell.read_deck(G.FIXTURES / deck)
    with threadpool_limits(limits=1):
        printout = _shell.render(text, basis=basis)
    line2 = printout.split("\n")[1]
    assert line2 == _bare_stamp(basis)
    assert "+" not in line2
    golden = G.golden_path(deck, basis).read_bytes().decode("latin-1")
    masked = G.mask_stamp(printout, basis)
    assert masked == golden or undefined_blanked(masked) == undefined_blanked(golden)


# --- the executables --------------------------------------------------------------


def test_the_exe_refusal_sentence_and_aliases_match_the_registry():
    src = EXE_SOURCE.read_text(encoding="utf-8")
    # The word is spelled P%cthon there: the no-Python build carries no
    # "python" bytes (tests/test_standalone_nopython.py).
    sentence = F.EXE_REFUSAL.format(name="%s").replace("Python", "P%cthon")
    assert f"\"{sentence}\\n\",\n                 name.c_str(), 'y');" in src
    assert 'prefix = "MOMWIRE_X_"' in src
    table = re.search(r"FEATURE_ALIASES\[\]\[2\] = \{(.*?)\n\};", src, re.S)
    assert table is not None
    pairs = set(re.findall(r'\{"(MOMWIRE_[A-Z0-9_]+)", "([a-z0-9_]+)"\}', table[1]))
    assert pairs == {
        (alias, f.name) for f in F.FEATURES.values() for alias in f.aliases
    }
    # Both entry points refuse before anything else runs.
    assert src.count("if (const int refused = feature_refusal()) return refused;") == 2


EXE = os.environ.get("MOMWIRE_RAZOR2P_STANDALONE_EXE")
DECK = ROOT / "tests" / "fixtures" / "eznec" / "decks" / "0036_dipole-in-free-space.nec"


@pytest.mark.skipif(
    not EXE, reason="MOMWIRE_RAZOR2P_STANDALONE_EXE does not name a built executable"
)
@pytest.mark.parametrize(
    ("var", "name"),
    [
        ("MOMWIRE_X_OFFSET_REUSE", "offset_reuse"),
        ("MOMWIRE_X_PAIR_REUSE", "pair_reuse"),
        ("MOMWIRE_X_NOT_YET_REGISTERED", "not_yet_registered"),
        ("MOMWIRE_RAZOR_CXX", "razor_cxx"),
    ],
)
def test_the_exe_refuses_a_feature(tmp_path, var, name):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(F.ENV_PREFIX)
        and k not in {a for f in F.FEATURES.values() for a in f.aliases}
    }
    env[var] = "1"
    out = tmp_path / "out.txt"
    r = subprocess.run(
        [EXE, str(DECK), str(out)], env=env, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 2
    assert r.stderr == F.EXE_REFUSAL.format(name=name) + "\n"
    assert not out.exists()
    # Off is off: the same call with the variable at 0 answers the deck.
    env[var] = "0"
    r = subprocess.run(
        [EXE, str(DECK), str(out)], env=env, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr
    assert out.read_bytes()


# --- the launchers' warm servers ------------------------------------------------


def test_the_warm_server_keys_carry_the_features_on(no_features):
    # A server inherits its spawning client's environment, so a warm one
    # started with no feature must not answer a call that asks for one.
    import momwire_eznec_client as ez
    import momwire_nec2c_client as n2
    import momwire_serve_client as mech

    k_ez, k_n2 = ez.config_key(None, 900.0), n2.config_key([], 900.0)
    no_features.setenv("MOMWIRE_X_CROSSING_TABLE", "0")
    assert (ez.config_key(None, 900.0), n2.config_key([], 900.0)) == (k_ez, k_n2)
    no_features.setenv("MOMWIRE_X_CROSSING_TABLE", " On ")
    on_ez = ez.config_key(None, 900.0)
    assert on_ez != k_ez and n2.config_key([], 900.0) != k_n2
    no_features.setenv("MOMWIRE_RAZOR_CXX", "1")
    assert ez.config_key(None, 900.0) not in (k_ez, on_ez)
    assert mech.feature_identity() == ["feature=crossing_table", "feature=razor_cxx"]
    # The clients' alias list is the registry's.
    assert mech.FEATURE_ALIASES == {
        a: f.name for f in F.FEATURES.values() for a in f.aliases
    }
    assert mech.FEATURE_ENV_PREFIX == F.ENV_PREFIX
