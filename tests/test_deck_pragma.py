"""Per-deck feature pragmas: ``CM momwire: <name>`` (`_features.deck_pragmas`).

A deck may turn on a feature that cannot change its answer for itself, one
feature a comment card, as that feature's environment variable set on would
for that deck alone. Held here:

- the reader: the card is CM in either case after blanks or tabs, its text
  starts with ``momwire:`` in any case, the name is the identifier after it,
  lowered, and the rest of the line is ignored; one name a line, each name
  once, in the order they first appear; a card EZNEC wrapped from a long
  description is read line by line, so a pragma that starts a continuation
  line counts and one in the middle of a line does not;
- the registry: only features marked ``pragma`` (today pair_reuse and
  rotational_symmetry) are taken, each within DEFAULT_DRIFT_MAX and off by
  default; the executables' list and the warm-server clients' list are the
  registry's;
- the scope: a pragma reads as the variable set on (`_env_value`), the
  variable set either way wins, a name the registry does not take is named
  on line 2 (`pragma_note`), a scope inside another is the outer one's, and
  a deck with no pragma card changes nothing.

The executables' copy of the reader (`deck_pragmas` in
standalone/momwire_razor2p.cpp) is held to this one through their printouts
(tests/test_standalone_nopython.py, test_standalone_nec42.py,
test_standalone_nec2c.py); the drop-ins' behaviour is
tests/test_deck_pragma_dropin.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from momwire import _features as F

ROOT = Path(__file__).resolve().parents[1]
EXE_MAIN = ROOT / "standalone" / "momwire_razor2p.cpp"


@pytest.fixture
def no_features(monkeypatch):
    for f in F.FEATURES.values():
        for var in (f.env, *f.aliases):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(F, "_PRAGMA_WITHHELD", frozenset())
    return monkeypatch


# --- the reader -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("CM momwire: pair_reuse\n", ["pair_reuse"]),
        # either case, everywhere but the order
        ("cm MOMWIRE: Pair_Reuse\n", ["pair_reuse"]),
        ("Cm MoMwIrE:ROTATIONAL_SYMMETRY\n", ["rotational_symmetry"]),
        # blanks and tabs before the card, after it and after the colon
        ("  \tCM \t momwire:\t  pair_reuse\n", ["pair_reuse"]),
        # EZNEC's own shape for a description line: CM, a tab, the text
        ("CM \tmomwire: pair_reuse\n", ["pair_reuse"]),
        ("CMmomwire:pair_reuse\n", ["pair_reuse"]),
        # the rest of the line is ignored, so ONE name a line
        ("CM momwire: pair_reuse rotational_symmetry\n", ["pair_reuse"]),
        ("CM momwire: pair_reuse, please\n", ["pair_reuse"]),
        ("CM momwire: pair_reuse.\n", ["pair_reuse"]),
        ("CM momwire: pair_reuse-x\n", ["pair_reuse"]),
        # unknown and ineligible names are read; the scope sorts them
        ("CM momwire: bogus\n", ["bogus"]),
        ("CM momwire: mixed_lu\n", ["mixed_lu"]),
        ("CM momwire: 42x\n", ["42x"]),
        # no name, nothing
        ("CM momwire:\n", []),
        ("CM momwire:   \n", []),
        ("CM momwire: ??? pair_reuse\n", []),
        # not at the start of the comment text: prose, not a pragma
        ("CM uses momwire: pair_reuse\n", []),
        ("CM ! momwire: pair_reuse\n", []),
        # a space inside the prefix is no prefix
        ("CM momwire : pair_reuse\n", []),
        ("CM mom wire: pair_reuse\n", []),
        # only a CM card carries one
        ("CE momwire: pair_reuse\n", []),
        ("GW momwire: pair_reuse\n", []),
        ("momwire: pair_reuse\n", []),
        # several lines: each name once, in the order they first appear
        (
            "CM momwire: rotational_symmetry\nCM momwire: pair_reuse\n"
            "CM momwire: Rotational_Symmetry\n",
            ["rotational_symmetry", "pair_reuse"],
        ),
        # every line end: LF, CRLF, CR alone, none at the end
        ("CM a\r\nCM momwire: pair_reuse\r\nCE\r\n", ["pair_reuse"]),
        ("CM a\rCM momwire: pair_reuse\rCE\r", ["pair_reuse"]),
        ("CE\nCM momwire: pair_reuse", ["pair_reuse"]),
        ("", []),
    ],
)
def test_the_reader(text, names):
    assert F.deck_pragmas(text) == names


def test_a_wrapped_description_is_read_line_by_line():
    # EZNEC writes a model's description as CM cards of about 76 columns,
    # so a long paragraph wraps: a pragma the user put on its own line, or
    # one the wrap happened to put at the start of a continuation line,
    # starts a card, and one inside a line does not.
    deck = (
        "CM This model is a long description of an antenna that wraps across\n"
        "CM momwire: pair_reuse is what the user typed on a line of its own\n"
        "CM and here the text goes on, mentioning momwire: rotational_symmetry\n"
        "CM in the middle of a line, which is prose and not a pragma.\n"
        "CM ! Written by EZNEC/Pro+ v. 7.0 in NEC-5 format.\n"
        "CE\n"
    )
    assert F.deck_pragmas(deck) == ["pair_reuse"]


def test_a_non_ascii_name_character_ends_the_name():
    # Latin-1 decks: only ASCII letters, digits and underscores make a name,
    # as in the executables' reader (`str.isalnum` would take more).
    assert F.deck_pragmas("CM momwire: pair\xe9reuse\n") == ["pair"]
    assert F.deck_pragmas("CM momwire: \xb2pair_reuse\n") == []


# --- the registry -----------------------------------------------------------------

PRAGMA = {"pair_reuse", "rotational_symmetry"}


def test_the_pragma_features_are_the_ones_that_cannot_change_the_answer():
    marked = {n for n, f in F.FEATURES.items() if f.pragma}
    assert marked == PRAGMA
    for name in marked:
        f = F.FEATURES[name]
        assert f.drift <= F.DEFAULT_DRIFT_MAX, name
        assert not f.default, name
        assert f.kind == "python", name
        assert f.part_of is None, name
        assert F.pragma_eligible(name)
    for name, f in F.FEATURES.items():
        if not f.pragma:
            assert not F.pragma_eligible(name), name
    assert not F.pragma_eligible("no_such_feature")


def _exe_pragma_list() -> list[str]:
    m = re.search(
        r"const char \*const PRAGMA_FEATURES\[\] = \{([^}]*)\};", EXE_MAIN.read_text()
    )
    assert m, "PRAGMA_FEATURES not found in the executables' main"
    return re.findall(r'"([^"]+)"', m[1])


def test_the_executables_pragma_list_is_the_registrys():
    names = _exe_pragma_list()
    assert sorted(names) == sorted(PRAGMA) and len(names) == len(PRAGMA)


def test_the_executables_take_pair_reuse_only():
    # Owner decision (momwire#1338): rotational symmetry is the drop-ins'
    # alone, so the executables carry no row for it and a pragma for it
    # there is ignored (named on line 2), never refused. pair_reuse reaches
    # every executable.
    src = EXE_MAIN.read_text()
    table = re.search(r"const ExeFeature EXE_FEATURES\[\] = \{(.*?)\n\};", src, re.S)
    carried = set(re.findall(r'\{"([a-z_0-9]+)",', table[1]))
    assert "pair_reuse" in carried
    assert "rotational_symmetry" not in carried


# --- the scope --------------------------------------------------------------------

DECK = "CM momwire: pair_reuse\nCM momwire: bogus\nCE\n"
PAIR = F.FEATURES["pair_reuse"]


def test_a_pragma_reads_as_the_variable_set_on(no_features):
    assert F._env_value(PAIR) is None
    with F.pragma_scope(DECK):
        assert F._env_value(PAIR) is True
        assert F.enabled("pair_reuse")
        assert "pair_reuse" in F.active()
        assert F.stamp(solver="razor") == "+pair_reuse"
        # a part reads the whole's, as under the variable
        assert F.enabled("razor_pair_reuse")
        assert F.enabled("offset_reuse")
    assert F._env_value(PAIR) is None
    assert not F.enabled("pair_reuse")
    assert F.stamp(solver="razor") == ""


@pytest.mark.parametrize("value", ["0", "off", "no", "false"])
def test_the_variable_off_wins(no_features, value):
    no_features.setenv(PAIR.env, value)
    with F.pragma_scope(DECK):
        assert F._env_value(PAIR) is False
        assert not F.enabled("pair_reuse")
        assert F.stamp(solver="razor") == ""
        # killed, not ignored: line 2 notes only the unknown name
        assert F.pragma_note() == "(pragma ignored: bogus)"


def test_the_variable_on_is_as_today(no_features):
    no_features.setenv(PAIR.env, "1")
    with F.pragma_scope("CE\n"):
        assert F.enabled("pair_reuse") and F.pragma_note() == ""


def test_a_part_turned_off_still_decides_under_the_pragma(no_features):
    no_features.setenv("MOMWIRE_X_RAZOR_PAIR_REUSE", "0")
    with F.pragma_scope(DECK):
        assert F.enabled("pair_reuse")
        assert not F.enabled("razor_pair_reuse")
        assert F.stamp(solver="razor") == ""


def test_the_keyword_still_wins(no_features):
    with F.pragma_scope(DECK):
        assert not F.enabled("pair_reuse", False)


def test_ignored_names_are_noted_in_order_once(no_features):
    deck = (
        "CM momwire: mixed_lu\nCM momwire: pair_reuse\nCM momwire: bogus\n"
        "CM momwire: MIXED_LU\nCM momwire: threads\n"
    )
    with F.pragma_scope(deck):
        assert F.pragmas_on() == {"pair_reuse"}
        assert F.pragma_note() == "(pragma ignored: mixed_lu, bogus, threads)"
        # an ineligible feature named in a pragma does not turn on
        assert not F.enabled("mixed_lu")


def test_no_pragma_card_changes_nothing(no_features):
    with F.pragma_scope("CM a plain comment\nCE\n"):
        assert F.pragmas_on() == frozenset()
        assert F.pragma_note() == ""
        assert F.stamp() == ""
    assert F.pragma_note() == "" and F.pragmas_on() == frozenset()


def test_a_scope_inside_another_is_the_outer_ones(no_features):
    with F.pragma_scope(DECK):
        with F.pragma_scope("CM momwire: rotational_symmetry\n"):
            assert F.pragmas_on() == {"pair_reuse"}
        with F.pragma_scope("CE\n"):
            assert F.pragmas_on() == {"pair_reuse"}


def test_a_withheld_pragma_is_ignored_and_noted(no_features):
    # An executable's embedded launcher withholds the pragma features the
    # executable does not carry (rotational_symmetry), so its Python answer
    # ignores them as the C++ route does.
    F.withhold_pragmas(["rotational_symmetry"])
    deck = "CM momwire: rotational_symmetry\nCM momwire: pair_reuse\n"
    with F.pragma_scope(deck):
        assert F.pragmas_on() == {"pair_reuse"}
        assert not F.enabled("rotational_symmetry")
        assert F.pragma_note() == "(pragma ignored: rotational_symmetry)"


def test_the_decorator_scopes_its_first_argument(no_features):
    @F.pragma_scoped
    def entry(text, *, extra=None):
        return F.pragmas_on(), F.pragma_note(), extra

    assert entry(DECK, extra=1) == (
        frozenset({"pair_reuse"}),
        "(pragma ignored: bogus)",
        1,
    )
    assert F.pragmas_on() == frozenset()


def test_the_scope_is_per_thread(no_features):
    # The resident server answers decks concurrently, each on its own
    # thread: one deck's pragma must not reach another's solve.
    import threading

    seen = []
    with F.pragma_scope(DECK):
        t = threading.Thread(target=lambda: seen.append(F.pragmas_on()))
        t.start()
        t.join()
    assert seen == [frozenset()]
