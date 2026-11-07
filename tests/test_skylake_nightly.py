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


def test_known_list_entries_are_well_formed():
    kinds = {
        "mismatch", "exe-declines", "exe-no-answer", "py-error", "variant",
        "py-drift", "sweep-outcome", "sweep-z", "sweep-mem", "sweep-time",
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
