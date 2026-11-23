"""momwire#1964: a NEC-5 number with an exponent in more than 20 characters.

The licensed NEC-5 engine reads such a token wrongly, sometimes silently
(measured black-box on an NT card's Y12, read back from its NETWORK DATA
echo: ``-5.79364253832523e-06``, 21 characters, read as -5.7936, and
``-1.234567890123456e-1``, 21, stopped on INVALID NUMBER; every 20-character
spelling read as written, and a token with no exponent read fine at 26).
momwire reads every one of them as written, which is what the deck means,
so it does not refuse: the NEC-5 seam raises a :class:`LongExponentToken`
advisory naming the tokens (the launchers show it on stderr when
``MOMWIRE_ADVISORIES`` asks), and the printout is unchanged.

The C++ reader's copy (``_nec5_deck_core.cpp``'s ``long_exponent_note``,
through ``text_probe``) says the same sentence, byte for byte, over the
cases below and every deck under ``tests/fixtures``.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import pytest

from momwire.deck._nec5 import (
    LONG_EXPONENT_LIMIT,
    LongExponentToken,
    long_exponent_note,
    long_exponent_tokens,
)
from momwire.dropin.nec5 import _shell

FIXTURES = Path(__file__).parent / "fixtures"

# The issue's table: (token, characters, flagged).
TABLE = [
    ("-5.7936425383252e-06", False),
    ("5.79364253832523e-06", False),
    ("-5.79364253832523e-6", False),
    ("-5.79364253832523e-06", True),
    ("-5.79364253832523e+06", True),
    ("-1.234567890123456e-1", True),
    ("-0.12345678901234567890123", False),  # no exponent, 26 characters
]


def _deck(volts: str) -> str:
    """A synthetic dipole whose source's real part is spelled ``volts``."""
    return (
        "CM long-exponent probe (momwire#1964)\n"
        "CE\n"
        "GW 1 11 0 0 -0.25 0 0 0.25 0.001\n"
        "GE 0\n"
        f"EX 0 1 6 0 {volts} 0\n"
        "FR 0 1 0 0 299.7925 0\n"
        "XQ\n"
        "EN\n"
    )


@pytest.mark.parametrize(("token", "flagged"), TABLE, ids=[t for t, _ in TABLE])
def test_the_issues_tokens(token, flagged):
    assert (len(token) > LONG_EXPONENT_LIMIT and "e" in token.lower()) is flagged
    assert long_exponent_tokens(f"NT 1 6 2 1 0 {token} 0 0 0\n") == (
        (token,) if flagged else ()
    )


def test_what_counts_as_a_token():
    long_e = "-5.79364253832523e-06"
    long_d = "-5.79364253832523D-06"
    text = "\n".join(
        [
            f"CM a comment quoting {long_e} is not a field",
            f"CE {long_e}",
            f"GW1,11,0,0,{long_d},0,0,.25,.001",  # fused first field, commas
            f"GN 2 0 0 0 13 .005 {'e' * 25}",  # a table filename, not a number
            f"EX 0 1 6 0 {long_e}",
            "LD 5 1 0 0 5.8e7 -1.234567890123456789",  # long, no exponent
            "SY X=1.2345678901234567890e5",
        ]
    )
    assert long_exponent_tokens(text) == (long_d, long_e)


def test_the_sentence():
    one = long_exponent_note(("-5.79364253832523e-06",))
    assert one == (
        "the deck writes a number with an exponent in more than 20 characters "
        "(-5.79364253832523e-06); momwire reads it as written, but the licensed "
        "NEC-5 engine misreads such a number, sometimes silently (momwire#1964)"
    )
    many = long_exponent_note(tuple(f"-5.7936425383252{k}e-06" for k in range(5)))
    assert many.startswith(
        "the deck writes 5 numbers with an exponent in more than 20 characters "
        "(-5.79364253832520e-06, -5.79364253832521e-06, -5.79364253832522e-06 and "
        "2 more); momwire reads each as written"
    )
    assert long_exponent_note(()) is None


def test_the_seam_warns_and_prints_as_before():
    """The advisory, never a refusal; the printout carries no note."""
    with pytest.warns(LongExponentToken, match=r"momwire#1964") as caught:
        out = _shell.render(_deck("1.000000000000000e+00"), basis="razor2p")
    assert "(1.000000000000000e+00)" in str(caught[0].message)
    assert "NEC ERROR" not in out
    assert "misreads" not in out
    with warnings.catch_warnings():
        warnings.simplefilter("error", LongExponentToken)
        short = _shell.render(_deck("1.00000000000000e+00"), basis="razor2p")
    assert "NEC ERROR" not in short
    # One field's spelling apart (the card echo), the two decks print the
    # same printout: the same lines, the same results.
    assert len(out.splitlines()) == len(short.splitlines())
    differ = [
        (a, b)
        for a, b in zip(out.splitlines(), short.splitlines(), strict=True)
        if a != b
    ]
    assert all("RUN TIME" in a or "EX " in a or "TIME" in a for a, _b in differ), differ


# --- the C++ reader's copy -------------------------------------------------


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "nec5_text_probe"):
        pytest.skip("no compiled accelerator with the NEC-5 deck reader")
    return _accel.acc


def _py(text: str) -> str:
    note = long_exponent_note(long_exponent_tokens(text))
    if note is None:
        return "probe = None\n"
    return f"probe = str {note.encode('utf-8').hex()}\n"


def _cases() -> list[str]:
    cases = [f"NT 1 6 2 1 0 {t} 0 0 0\n" for t, _ in TABLE]
    cases.append(_deck("1.000000000000000e+00"))
    cases.append("".join(f"EX 0 1 6 0 -5.7936425383252{k}e-06\n" for k in range(7)))
    rng = random.Random(1964)
    alphabet = "0123456789eEdD+-.,_ \t\xa0\xb5\x85abcGWNTEXCMinfa"
    for _ in range(1500):
        head = rng.choice(["GW", "NT", "EX", "cm", "CE", "SY", "GN", "LD", "gw1"])
        body = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 60)))
        cases.append(f"{head} {body}\n")
    return cases


def test_the_cxx_reader_says_the_same(acc):
    bad = []
    for text in _cases():
        got = acc.nec5_text_probe("long_exponent_note", text.encode("latin-1"))
        if got != _py(text):
            bad.append((text, _py(text), got))
    assert not bad, f"{len(bad)} differ; first: {bad[0]!r}"
    flagged = sum(_py(t) != "probe = None\n" for t in _cases())
    assert flagged >= 3


def test_the_cxx_reader_says_the_same_on_every_fixture_deck(acc):
    decks = [
        p
        for p in sorted(FIXTURES.rglob("*"))
        if p.is_file() and p.suffix in (".nec", ".deck")
    ]
    assert len(decks) >= 400
    bad = []
    for path in decks:
        raw = path.read_bytes()
        text = raw.decode("latin-1")
        got = acc.nec5_text_probe("long_exponent_note", raw)
        if got != _py(text):
            bad.append(path.name)
    assert not bad, bad[:5]


def test_a_streamed_structure_cut_at_nx_warns_too():
    """SimNEC drives the NEC-5 slot as a stream (``- -``), each structure cut
    at NX answered alone; the advisory reaches those pieces as it reaches a
    file's whole text."""
    first = _deck("1.000000000000000e+00").replace("EN\n", "NX\n")
    with pytest.warns(LongExponentToken, match=r"momwire#1964"):
        out, go_on = _shell._stream_piece(first, basis="razor2p")
    assert go_on and "NEC ERROR" not in out
    with warnings.catch_warnings():
        warnings.simplefilter("error", LongExponentToken)
        _shell._stream_piece(
            _deck("1.00000000000000e+00").replace("EN\n", "NX\n"), basis="razor2p"
        )
