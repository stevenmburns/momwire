"""The Skylake nightly's harness (scripts/nightly): its corpus, its cell
rotation, its known list, and the severity triage proposes. The nightly itself
runs on Skylake only (scripts/nightly/README.md); this is what a PR can check."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NIGHTLY = ROOT / "scripts" / "nightly"
sys.path.insert(0, str(NIGHTLY))

import audit  # noqa: E402
import nightly  # noqa: E402
import triage  # noqa: E402


def test_every_corpus_line_names_a_directory_of_decks():
    decks = audit.read_corpus(ROOT, None)
    groups = {d["group"] for d in decks}
    for ln in (NIGHTLY / "corpus.txt").read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            dialects, rel = ln.split()
            assert set(dialects.split(",")) <= {"nec5", "nec42"}, ln
            assert rel in groups, f"{rel} holds no *.nec / *.deck"
    assert not any(d.get("local") for d in decks)


def test_every_exe_has_decks():
    decks = audit.read_corpus(ROOT, None)
    for name, (_, _, _, dialect) in audit.EXES.items():
        assert sum(dialect in d["dialects"] for d in decks) > 100, name


def test_the_rotation_covers_the_space_every_five_nights():
    space = {f"{v}-T{t}" for v in ("avx2", "sse2") for t in (1, 2, 4)}
    assert set(nightly.ROTATION) == space - {nightly.EVERY_NIGHT}
    day = dt.date(2026, 11, 6)
    for k in range(10):
        five = {nightly.rotation_cell(day + dt.timedelta(days=k + j)) for j in range(5)}
        assert five == set(nightly.ROTATION)


def test_the_threaded_copy_runs_every_night():
    """PA3(a): the OpenMP copy at one thread beside the serial copy, every
    night, and a cell spelling audit.py reads as MW_EXE_COPY=threads."""
    assert nightly.EVERY_NIGHT_THREADS == nightly.EVERY_NIGHT + "-threads"
    assert audit.parse_cell("avx2-T1") == ("avx2", 1, "policy")
    assert audit.parse_cell("avx2-T1-threads") == ("avx2", 1, "threads")
    assert audit.parse_cell("sse2-T4-threads") == ("sse2", 4, "threads")
    with pytest.raises(SystemExit):
        audit.parse_cell("avx2-T1-serial")


def test_the_anchors_phase_names_a_test_that_exists():
    """PA3(b): the nightly's `anchors-auto` phase runs a module that exists."""
    import inspect

    src = inspect.getsource(nightly.run_anchors_auto)
    assert "tests/test_standalone_anchors_auto.py" in src
    assert (ROOT / "tests" / "test_standalone_anchors_auto.py").is_file()


def test_known_list_entries_are_well_formed():
    kinds = {
        "mismatch", "exe-declines", "exe-no-answer", "py-error", "variant",
        "py-drift", "sweep-outcome", "sweep-z", "sweep-mem", "sweep-time",
        "copy-route", "copy-differs",
    }  # fmt: skip
    for k in triage.load_known(NIGHTLY / "known.json"):
        assert k["kind"] in kinds and k.get("note"), k


def _row(exe, deck, **kw):
    r = dict(
        exe=exe,
        deck=deck,
        group="g",
        py_variant="avx2",
        route="cxx:serial",
        exe_outcome="served",
        py_outcome="served",
        identical_masked=True,
        py_sha="a",
        exe_sha="a",
    )
    r.update(kw)
    return r


def _night(d: Path, rows, sha="1" * 40):
    d.mkdir()
    (d / "night.json").write_text(
        json.dumps(dict(sha=sha, date="2026-11-06", ak_sha="x", errors={}))
    )
    (d / "audit-avx2-T1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


@pytest.fixture
def known(tmp_path):
    p = tmp_path / "known.json"
    p.write_text(
        json.dumps({"known": [{"kind": "exe-declines", "match": "k*", "note": "n"}]})
    )
    return p


def test_triage_severity(tmp_path, known):
    base = _night(
        tmp_path / "base",
        [
            _row("e", "served-then-declined"),
            _row("e", "old-decline", route="refused:x"),
        ],
    )
    night = _night(
        tmp_path / "night",
        [
            _row("e", "same"),
            _row("e", "differs", identical_masked=False, diff_line=7),
            _row("e", "served-then-declined", route="refused:why"),
            _row("e", "old-decline", route="refused:x"),
            _row("e", "known-decline", route="refused:x"),
            _row("e", "both-refuse", route="refused:x", py_outcome="refused"),
            _row("e", "sse", py_variant="sse2"),
        ],
    )
    r = triage.triage(night, base, known)
    sev = {f["id"]: f["severity"] for f in r["findings"]}
    assert sev == {
        "mismatch|avx2-T1|e|differs": "severe",
        "exe-declines|avx2-T1|e|served-then-declined": "severe",
        "exe-declines|avx2-T1|e|old-decline": "moderate",
        "exe-declines|avx2-T1|e|known-decline": "known",
        "variant|avx2-T1|e|sse": "severe",
    }
    assert r["status"] == "red"


def test_the_threaded_cell_is_held_to_the_serial_copy_and_its_route(tmp_path):
    """`copy-differs`: the threaded copy's printout is not the plain cell's;
    `copy-route`: the -threads cell served through the serial copy, so it
    measured nothing. Both severe; a refusal in either cell is neither."""
    night = _night(
        tmp_path / "n",
        [
            _row("e", "same"),
            _row("e", "differs", exe_sha="s"),
            _row("e", "serial-ran"),
            _row("e", "refused", route="refused:x", py_outcome="refused"),
        ],
    )
    (night / "audit-avx2-T1-threads.jsonl").write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in (
                _row("e", "same", route="cxx:threads"),
                _row("e", "differs", route="cxx:threads", exe_sha="t"),
                _row("e", "serial-ran", route="cxx:serial"),
                _row("e", "refused", route="refused:x", py_outcome="refused"),
            )
        )
    )
    r = triage.triage(night, None, tmp_path / "none.json")
    sev = {f["id"]: f["severity"] for f in r["findings"]}
    assert sev == {
        "copy-differs|avx2-T1-threads|e|differs": "severe",
        "copy-route|avx2-T1-threads|e|serial-ran": "severe",
    }


def test_a_night_of_known_findings_is_green(tmp_path, known):
    night = _night(tmp_path / "n", [_row("e", "k1", route="refused:x")])
    assert triage.triage(night, None, known)["status"] == "green"


def test_sweep_z_bar(tmp_path):
    def sweep(d, z):
        d.joinpath("sweep.jsonl").write_text(
            json.dumps(
                dict(
                    id="c",
                    outcome="ok",
                    z=[[z, 0.0]],
                    mode="par1",
                    ru_maxrss_mb=1,
                    wall_s=1,
                )
            )
            + "\n"
        )

    base, night = _night(tmp_path / "b", []), _night(tmp_path / "n", [])
    sweep(base, 50.0)
    for z, want in (
        (50.0, None),
        (50.0 * (1 + 1e-13), "moderate"),
        (50.0 * (1 + 1e-11), "severe"),
    ):
        sweep(night, z)
        f = triage.triage(night, base, tmp_path / "none.json")["findings"]
        assert (f[0]["severity"] if f else None) == want, z


def _printout(night: Path, key: str, text: str) -> None:
    d = night / "py-avx2-T1"
    d.mkdir(exist_ok=True)
    (d / f"{key}.py").write_text(text)


_BODY = " NEC\n momwire {v} {basis} avx2\n IMPEDANCE {z}\n"


def _drift(tmp_path, *, base_label, label, v0, v1, basis0, basis1, z0, z1):
    base = _night(tmp_path / "base", [_row(base_label, "d", key="kb", py_sha="b")])
    night = _night(tmp_path / "night", [_row(label, "d", key="kn", py_sha="n")])
    _printout(base, "kb", _BODY.format(v=v0, basis=basis0, z=z0))
    _printout(night, "kn", _BODY.format(v=v1, basis=basis1, z=z1))
    return [
        f
        for f in triage.triage(night, base, tmp_path / "none.json")["findings"]
        if f["kind"] == "py-drift"
    ]


def test_py_drift_masks_the_identity_line(tmp_path):
    """A release reads as 'a number moved 8.42e-02' on every deck when line 2's
    version is compared as numbers (0.87.1 -> 0.95.1, the 11-07..11-21 reds)."""
    f = _drift(
        tmp_path, base_label="e", label="e", v0="0.87.1", v1="0.95.1",
        basis0="razor-2p", basis1="razor2p", z0="7.1521E+01", z1="7.1521E+01",
    )  # fmt: skip
    assert f == []


def test_py_drift_still_sees_a_number_off_the_identity_line(tmp_path):
    f = _drift(
        tmp_path, base_label="e", label="e", v0="0.87.1", v1="0.95.1",
        basis0="bspline", basis1="bspline", z0="7.1521E+01", z1="7.2000E+01",
    )  # fmt: skip
    assert [x["severity"] for x in f] == ["severe"]


def test_a_base_from_before_the_rename_pairs_with_the_new_label(tmp_path):
    """momwire#2091: without the label map the base has no row for
    nec5-razor2p and the drift check compares nothing, in silence."""
    f = _drift(
        tmp_path, base_label="nec5-razor-2p", label="nec5-razor2p",
        v0="0.95.1", v1="0.96.0", basis0="razor-2p", basis1="razor2p",
        z0="7.1521E+01", z1="7.2000E+01",
    )  # fmt: skip
    assert [(x["exe"], x["severity"]) for x in f] == [("nec5-razor2p", "severe")]


def test_accept_moves_the_base_and_records_why(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly, "ROOT", tmp_path)
    monkeypatch.setattr(nightly, "STATE", tmp_path / "state.json")
    night = _night(tmp_path / "red", [_row("e", "d")], sha="2" * 40)
    (night / "triage.json").write_text(json.dumps({"counts": {"severe": 3}}))
    state = {"last_green": {"sha": "1" * 40, "date": "2026-11-07", "dir": "old"}}
    assert nightly.accept(night, "reviewed: intended", state) == 0
    s = json.loads((tmp_path / "state.json").read_text())
    assert s["last_green"] == {"sha": "2" * 40, "date": "2026-11-06", "dir": str(night)}
    (rec,) = map(json.loads, (tmp_path / "accepted.jsonl").read_text().splitlines())
    assert rec["note"] == "reviewed: intended"
    assert rec["replaces"]["date"] == "2026-11-07" and rec["counts"] == {"severe": 3}


def test_the_nightlys_temporaries_live_under_its_root(tmp_path, monkeypatch):
    assert nightly.TMP.is_relative_to(nightly.ROOT)
    assert nightly.PYTEST_BASETEMP.is_relative_to(nightly.ROOT)
    monkeypatch.setattr(nightly, "TMP", tmp_path / "tmp")
    assert nightly.env_base()["TMPDIR"] == str(tmp_path / "tmp")
    assert (tmp_path / "tmp").is_dir()


def test_the_sweep_times_a_cold_sommerfeld_cache_of_its_own(tmp_path, monkeypatch):
    """The Sommerfeld disk cache is keyed on the build, so a night on a new
    commit was cold and a repeat night warm (wire.beverage 1.40 s against
    0.18 s), and the sweep compared the two as a regression."""
    seen = {}
    monkeypatch.setattr(nightly, "TMP", tmp_path / "tmp")
    monkeypatch.setattr(nightly, "MW", tmp_path / "mw")
    (tmp_path / "mw").mkdir()
    monkeypatch.setattr(nightly, "git", lambda *a: "f" * 40)

    def fake_sh(cmd, **kw):
        cache = Path(kw["env"]["MOMWIRE_SOMM_CACHE_DIR"])
        seen.update(kw, empty=cache.is_dir() and not any(cache.iterdir()))

    monkeypatch.setattr(nightly, "sh", fake_sh)
    night = tmp_path / "night"
    stale = night / "somm-cache" / "old.npz"
    stale.parent.mkdir(parents=True)
    stale.write_text("x")
    nightly.run_sweep(night, tmp_path / "harness", None)
    cache = Path(seen["env"]["MOMWIRE_SOMM_CACHE_DIR"])
    assert cache == night.resolve() / "somm-cache"
    assert seen["empty"]  # the sweep ran on an empty cache
    assert not cache.exists()  # and it is gone after the sweep


import report_tools  # noqa: E402


def test_the_report_agent_can_only_read(tmp_path):
    cmd = nightly.agent_command(tmp_path / "n", tmp_path / "b", "c" * 40, ["avx2-T1"])
    allowed = cmd[cmd.index("--allowedTools") + 1 : cmd.index("--disallowedTools")]
    denied = cmd[cmd.index("--disallowedTools") + 1 :]
    # the allowed list is a whitelist only in the default mode (an account
    # in auto mode would otherwise approve more)
    assert cmd[cmd.index("--permission-mode") + 1] == "default"
    assert not {"Write", "Edit", "NotebookEdit"} & set(allowed)
    assert {"Write", "Edit", "NotebookEdit"} <= set(denied)
    bash = [t for t in allowed if t.startswith("Bash(")]
    tools = nightly.HERE / "report_tools.py"
    assert sorted(bash) == sorted(
        [f"Bash(git -C {nightly.MW} {v}:*)" for v in ("log", "show", "diff")]
        + [f"Bash(python3 {tools}:*)"]
    )
    prompt = cmd[cmd.index("-p") + 1]
    assert "{" not in prompt.replace("{%", "")  # every placeholder filled
    assert str(tmp_path / "n") in prompt and "c" * 40 in prompt


def _fake_claude(monkeypatch, bindir: Path, body: str) -> None:
    """An absolute path, so env_base()'s ~/.local/bin can never put the real
    CLI first: a test must not start a real agent."""
    bindir.mkdir()
    exe = bindir / "claude"
    exe.write_text(f"#!/bin/sh\n{body}\n")
    exe.chmod(0o755)
    monkeypatch.setattr(nightly, "CLAUDE", str(exe))


def test_the_report_agent_writes_report_md(tmp_path, monkeypatch):
    _fake_claude(monkeypatch, tmp_path / "bin", "echo '# no regression'")
    monkeypatch.setattr(nightly, "TMP", tmp_path / "tmp")
    monkeypatch.setattr(nightly, "log", lambda *_: None)
    night = tmp_path / "night"
    night.mkdir()
    nightly.agent_report(night, tmp_path, "c" * 40, ["avx2-T1"])
    assert (night / "report.md").read_text() == "# no regression\n"


def test_the_report_agent_never_fails_the_night(tmp_path, monkeypatch):
    _fake_claude(monkeypatch, tmp_path / "bin", "echo boom >&2; exit 3")
    monkeypatch.setattr(nightly, "TMP", tmp_path / "tmp")
    monkeypatch.setattr(nightly, "log", lambda *_: None)
    night = tmp_path / "night"
    night.mkdir()
    nightly.agent_report(night, tmp_path, "c" * 40, ["avx2-T1"])  # no raise
    assert not (night / "report.md").exists()
    assert "boom" in (night / "report-agent.log").read_text()
    monkeypatch.setenv(nightly.AGENT_ENV, "0")
    _fake_claude(monkeypatch, tmp_path / "bin2", "echo '# should not run'")
    nightly.agent_report(night, tmp_path, "c" * 40, ["avx2-T1"])
    assert not (night / "report.md").exists()


def test_report_tools_read_and_never_write(tmp_path):
    base = _night(tmp_path / "base", [_row("e", "d", key="kb", py_sha="b")])
    night = _night(tmp_path / "night", [_row("e", "d", key="kn", py_sha="n")])
    _printout(base, "kb", _BODY.format(v="0.87.1", basis="bspline", z="7.1521E+01"))
    _printout(night, "kn", _BODY.format(v="0.95.1", basis="bspline", z="7.2000E+01"))
    (night / "triage.json").write_text(
        json.dumps(
            dict(status="red", base="1" * 40, base_date="2026-11-06",
                 counts={"severe": 1},
                 findings=[triage.finding("py-drift", "severe", "x", cell="avx2-T1",
                                          exe="e", deck="d")])
        )
    )  # fmt: skip
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    s = report_tools.summary(night)
    assert "[1] kind=py-drift severity=severe exe=e cell=avx2-T1" in s
    d = report_tools.diff(night, base, "e", "d")
    assert "-" + " IMPEDANCE 7.1521E+01" in d and "+ IMPEDANCE 7.2000E+01" in d
    assert "0.87.1" not in d and "0.95.1" not in d  # line 2 masked
    m = report_tools.masked(night, base)
    assert "differ only on line 2: 0" in m and "differ elsewhere: 1" in m
    lines = report_tools.nights(tmp_path).splitlines()
    assert [x.split()[0] for x in lines] == ["base", "night"]
    assert "red" in lines[1] and '"severe": 1' in lines[1]
    after = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after


def test_printouts_read_a_packed_night_in_one_pass(tmp_path):
    import tarfile

    night = tmp_path / "n"
    (night / "py-avx2-T1").mkdir(parents=True)
    for k in ("a", "b", "c"):
        (night / "py-avx2-T1" / f"{k}.py").write_text(f"printout {k}\n")
    with tarfile.open(night / "py-avx2-T1.tar.xz", "w:xz") as tf:
        tf.add(night / "py-avx2-T1", arcname="py-avx2-T1")
    for k in ("a", "b", "c"):
        (night / "py-avx2-T1" / f"{k}.py").unlink()
    (night / "py-avx2-T1").rmdir()
    p = triage.Printouts(night)
    assert [p.get(k) for k in ("c", "a", "b", "zz")] == [
        b"printout c\n",
        b"printout a\n",
        b"printout b\n",
        None,
    ]


# ------------------------------------------- bisect probes of older trees
# The 2026-11-22 and 11-23 bisects passed the dash-free names (momwire#2091)
# to 276e88fb, which predates them: Python answered "unknown basis 'bs2'",
# the executable step re-ran the night's own binary, and the bisect reported
# "reproduces at the last green commit: not a regression" from that.


def test_the_legacy_spellings_are_the_alias_table_inverted():
    from momwire._basis_names import BASIS_ALIASES

    assert audit.LEGACY_BASIS == {new: old for old, new in BASIS_ALIASES.items()}
    assert {b for _, _, b, _ in audit.EXES.values()} <= set(BASIS_ALIASES.values()) | {
        "sinusoidal"
    }


@pytest.mark.parametrize(
    ("renamed", "want"),
    [(True, ["bs2", "razor2p", "bs2", "sinusoidal"]),
     (False, ["bspline", "razor-2p", "bspline", "sinusoidal"])],
)  # fmt: skip
def test_python_is_asked_in_the_trees_spelling(monkeypatch, renamed, want):
    import momwire.dropin.nec5 as nec5

    asked = []

    class Shell:
        @staticmethod
        def render_files(text, **kw):
            asked.append(kw["basis"])
            return [b"ok"]

        printout_bytes = staticmethod(lambda out: out)

    monkeypatch.setattr(nec5, "_shell", Shell)
    monkeypatch.setattr(audit, "_renamed", lambda: renamed)
    for _, _, basis, dialect in audit.EXES.values():
        assert audit._python("deck", basis, dialect) == (b"ok", None)
    assert asked == want


def test_a_basis_the_tree_does_not_know_is_a_harness_finding(tmp_path):
    assert audit.unrunnable(b"NEC ERROR - unknown basis 'bs2' for EZNEC", "bs2")
    assert audit.unrunnable(b"NEC ERROR - unknown basis 'pulse2'", "bs2") is None
    base = _night(tmp_path / "base", [_row("e", "d")])
    night = _night(
        tmp_path / "night",
        [_row("e", "d", py_sha="b", py_outcome="refused", identical_masked=False,
              py_unrunnable="unknown basis 'bs2'")],
    )  # fmt: skip
    found = triage.triage_audit(night, base)
    assert [(f["kind"], f["severity"]) for f in found] == [("harness", "severe")]


HELP = """The following are some of the valid targets for this Makefile:
... all (the default if no target is given)
... clean
... mw_core
... {target}
... home/x/mw/src/momwire/_bspline_core.o
"""


@pytest.mark.parametrize(
    "target",
    [
        "momwire-nec42-standalone",  # 276e88fb, before momwire#1831
        "momwire-nec42-bspline-standalone",  # before momwire#2091
        "momwire-nec42-bs2-standalone",
    ],
)
def test_the_exe_target_is_the_one_the_tree_defines(target):
    assert nightly.standalone_target(HELP.format(target=target)) == target


def test_a_build_dir_without_one_standalone_target_fails():
    with pytest.raises(RuntimeError):
        nightly.standalone_target(HELP.format(target="mw_core_serial"))


def _fake_cmake(monkeypatch, tmp_path, tree_target, makes):
    monkeypatch.setattr(nightly, "EXES", tmp_path / "exes")

    class P:
        stdout = HELP.format(target=tree_target)

    def fake_sh(cmd, **kw):
        cmd = list(map(str, cmd))
        if cmd[-2:] == ["--target", "help"]:
            return P
        if "--build" in cmd and makes:
            Path(cmd[2], cmd[-1]).write_text("built")
        return P

    monkeypatch.setattr(nightly, "sh", fake_sh)
    b = tmp_path / "exes" / "bspline_nec42-avx2"
    b.mkdir(parents=True)
    for stale in ("momwire-nec42-bs2-standalone", tree_target):
        (b / stale).write_text("the night's binary")
    return b


def test_a_probe_never_runs_the_nights_own_binary(monkeypatch, tmp_path):
    """make calls a target the tree does not define "up to date" when a file
    of that name is in the shared build dir."""
    old = "momwire-nec42-standalone"
    b = _fake_cmake(monkeypatch, tmp_path, old, makes=True)
    targets = {}
    nightly.build_exes([("nec42-bs2", "avx2")], tmp_path, tmp_path / "log", targets)
    assert targets == {"nec42-bs2": old}
    assert (b / old).read_text() == "built"
    _fake_cmake(monkeypatch, tmp_path / "2", old, makes=False)
    with pytest.raises(RuntimeError, match="made no"):
        nightly.build_exes([("nec42-bs2", "avx2")], tmp_path, tmp_path / "log")


def test_the_audit_runs_the_target_the_probe_built(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(nightly, "sh", lambda cmd, **kw: seen.setdefault("cmd", cmd))
    nightly.run_cell(
        "avx2-T1", tmp_path, exe_targets={"nec42-bs2": "momwire-nec42-standalone"}
    )
    cmd = list(map(str, seen["cmd"]))
    assert cmd[cmd.index("--exe-target") + 1] == "nec42-bs2=momwire-nec42-standalone"


DRIFT = triage.finding("py-drift", "severe", "x", cell="avx2-T1", exe="e", deck="d")


def _probe(monkeypatch, tmp_path, row):
    """probe() with the builds faked and the audit writing `row`."""
    for name in ("build_python", "build_exes", "openblas", "pack", "log"):
        monkeypatch.setattr(nightly, name, lambda *a, **k: None)
    monkeypatch.setattr(nightly, "git", lambda *a: "")
    monkeypatch.setattr(nightly, "load_state", dict)

    def fake_cell(cell, d, **kw):
        (d / f"audit-{cell}.jsonl").write_text(json.dumps(row) + "\n")

    monkeypatch.setattr(nightly, "run_cell", fake_cell)
    base = _night(tmp_path / "base", [_row("e", "d")])
    return nightly.probe("a" * 40, [DRIFT], base, tmp_path / "work")


def test_a_probe_that_cannot_run_python_is_untested(monkeypatch, tmp_path):
    row = _row("e", "d", py_sha="b", py_outcome="refused", identical_masked=False,
               py_unrunnable="unknown basis 'bs2'", cell="avx2-T1")  # fmt: skip
    assert _probe(monkeypatch, tmp_path, row) == nightly.UNTESTED
    why = (tmp_path / "work" / ("a" * 12) / "untested.txt").read_text()
    assert "unknown basis 'bs2'" in why


@pytest.mark.parametrize(("py_sha", "want"), [("a", "good"), ("b", "bad")])
def test_a_probe_that_ran_is_good_or_bad(monkeypatch, tmp_path, py_sha, want):
    row = _row("e", "d", py_sha=py_sha, cell="avx2-T1")
    assert _probe(monkeypatch, tmp_path, row) == want


def _bisect(monkeypatch, tmp_path, verdicts):
    """bisect() over commits c1..c4 (c4 tonight), each probe's verdict from
    `verdicts`; returns (bisect.json, the commit `mw` was left on)."""
    commits = ["c1", "c2", "c3", "c4"]
    head = {"sha": "c4"}

    def fake_git(*a):
        if a[0] == "rev-parse":
            return head["sha"] if a[1] == "HEAD" else a[1]
        if a[0] == "rev-list":
            return " ".join(commits)
        if a[0] == "checkout":
            head["sha"] = a[-1]
        return "subject"

    def fake_probe(sha, *a):
        head["sha"] = sha
        v = verdicts[sha]
        if isinstance(v, Exception):
            raise v
        return v

    monkeypatch.setattr(nightly, "git", fake_git)
    monkeypatch.setattr(nightly, "probe", fake_probe)
    monkeypatch.setattr(nightly, "build_python", lambda *a, **k: None)
    monkeypatch.setattr(nightly, "log", lambda *a: None)
    monkeypatch.setattr(nightly, "MW", tmp_path / "mw")
    monkeypatch.setattr(nightly, "TMP", tmp_path / "tmp")
    night = _night(tmp_path / "night", [], sha="c4")
    base = _night(tmp_path / "base", [], sha="c0")
    result = dict(findings=[DRIFT])
    try:
        return nightly.bisect(night, base, result), head["sha"]
    except RuntimeError:
        return None, head["sha"]


def test_an_untested_green_commit_is_no_verdict(monkeypatch, tmp_path):
    out, left = _bisect(monkeypatch, tmp_path, {"c0": nightly.UNTESTED})
    assert out["result"].startswith("could not test the last green commit")
    assert "not a regression" not in out["result"]
    assert left == "c4"


def test_an_untested_commit_inside_the_range_is_stepped_over(monkeypatch, tmp_path):
    v = {"c0": "good", "c1": "good", "c2": nightly.UNTESTED, "c3": "bad"}
    out, left = _bisect(monkeypatch, tmp_path, v)
    assert out["first_bad"] == "c3" and out["untested"] == ["c2"]
    assert not out["exact"] and left == "c4"


@pytest.mark.parametrize("verdict", ["good", "bad", RuntimeError("probe died")])
def test_the_bisect_puts_mw_back(monkeypatch, tmp_path, verdict):
    v = {"c0": verdict, "c1": "good", "c2": "good", "c3": "good"}
    _, left = _bisect(monkeypatch, tmp_path, v)
    assert left == "c4"
