"""A stepping FR card on the NEC-5 seam — momwire#1508.

EZNEC and SimNEC call the engine once per frequency, so the seam refused any
FR asking for more than one. A deck that hands the engine a whole sweep now
gets every frequency, solved one after another in this process, printed in
the licensed NEC-5's stepped layout. The gates:

  * the frequency list is NEC-2's (linear and multiplicative, accumulated a
    step at a time), and it is the list NEC-5 printed on every probe;
  * the LAYOUT is the licensed one line for line, numbers masked, on every
    served probe, on razor-2p and bspline;
  * each frequency block is exactly the printout a one-frequency deck at that
    frequency gives, and the impedances agree with NEC-5's on razor-nec5;
  * the decks NEC-5 answers in a way this seam does not reproduce are refused
    by name, and a one-frequency deck never takes this path.

``tests/fixtures/eznec_fr_step_1508/`` (README there) holds the probes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from momwire.deck._cards import DeckError
from momwire.deck._nec5 import _stepped_frequencies, parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_fr_step_1508"
FREE_SPACE = (
    "p02_lin2",
    "p03_lin5",
    "p04_mul5",
    "p05_neg3",
    "p08_rp",
    "p09_rp_xq",
    "p10d_ne_xq",
    "p10e_nh_rp",
    "p14_xnda1003",
    "p15_fr_first",
    "p17_ld5",
)
# Over GN 2 with buried radials: a cold Sommerfeld fill per frequency, ~6 s.
BURIED = tuple(
    pytest.param(n, marks=pytest.mark.slow) for n in ("b01_buried_xq", "b02_buried_rp")
)
SERVED = (*FREE_SPACE, *BURIED)
ONE_FREQUENCY = ("p01_lin1", "p06_zero", "p10b_ne_single")
TWIN = "razor-nec5"
_NUMBER = re.compile(r"[-+]?\d+(\.\d+)?(E?[-+]\d+)?")


def _deck(name):
    return (FIXTURES / f"{name}.nec").read_text()


def _oracle(name):
    lines = (FIXTURES / f"{name}.out").read_text(encoding="latin-1").splitlines()
    # NEC-5's ground-table cache chatter: a blank, then two lines.
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip() and i + 1 < len(lines) and "GMPINO" in lines[i + 1]:
            i += 3
            continue
        out.append(lines[i])
        i += 1
    return out


def _ours(lines):
    # momwire's own interface-split note (momwire#1281) and the blank after
    # it, which NEC-5 does not print.
    out, i = [], 0
    while i < len(lines):
        if "(momwire#1281)" in lines[i]:
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return out


def _mask(line):
    return " ".join("#" if _NUMBER.fullmatch(t) else t for t in line.split())


def _from_structure(lines):
    i = next(i for i, ln in enumerate(lines) if "STRUCTURE SPECIFICATION" in ln)
    return [_mask(ln) for ln in lines[i:]]


def _printed_frequencies(lines):
    return [float(ln.split("=")[1].split()[0]) for ln in lines if "FREQUENCY=" in ln]


def _impedances(lines):
    out = []
    for block in "\n".join(lines).split("ANTENNA INPUT PARAMETERS")[1:]:
        row = next(
            ln.split()
            for ln in block.splitlines()
            if re.match(r"^\s+\d+\s+\d+\s+1\s", ln)
        )
        out.append(complex(float(row[7]), float(row[8])))
    return out


# -- the frequency list -------------------------------------------------------


def test_linear_and_multiplicative_steps_accumulate():
    assert _stepped_frequencies(28.0, 1.0, 3, multiplicative=False) == (
        28.0,
        29.0,
        30.0,
    )
    assert _stepped_frequencies(30.0, -1.0, 3, multiplicative=False) == (
        30.0,
        29.0,
        28.0,
    )
    got = _stepped_frequencies(26.0, 1.05, 5, multiplicative=True)
    want = [26.0]
    for _ in range(4):
        want.append(want[-1] * 1.05)
    assert got == tuple(want)


def test_a_step_is_added_to_the_last_frequency_not_multiplied_by_the_index():
    """NEC-2 steps its loop by adding; 0.1 is not exact in binary, so the two
    spellings part in the last bit (here at the third step)."""
    got = _stepped_frequencies(7.0, 0.1, 4, multiplicative=False)
    loop = [7.0]
    for _ in range(3):
        loop.append(loop[-1] + 0.1)
    assert got == tuple(loop)
    assert got != tuple(7.0 + n * 0.1 for n in range(4))


@pytest.mark.parametrize("name", SERVED)
def test_the_list_is_the_one_nec5_printed(name):
    deck = parse_nec5(_deck(name))
    printed = _printed_frequencies(_oracle(name))
    assert len(deck.frequencies_mhz) == len(printed) > 1
    for ours, theirs in zip(deck.frequencies_mhz, printed, strict=True):
        assert abs(ours - theirs) <= 5e-5 * abs(theirs)


@pytest.mark.parametrize("name", ONE_FREQUENCY)
def test_one_frequency_is_not_a_step(name):
    deck = parse_nec5(_deck(name))
    assert deck.frequencies_mhz == ()
    assert deck.frequency_mhz == 28.0


def test_a_stepping_type_nec_does_not_define_is_refused():
    with pytest.raises(DeckError, match="stepping type 2"):
        parse_nec5(_deck("p02_lin2").replace("FR 0 2", "FR 2 2"))


# -- the printout ---------------------------------------------------------------


@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
@pytest.mark.parametrize("name", SERVED)
def test_the_layout_is_the_licensed_one(name, basis):
    ours = _from_structure(_ours(render(_deck(name), basis=basis).splitlines()))
    assert ours == _from_structure(_oracle(name))


@pytest.mark.parametrize("name", FREE_SPACE)
def test_the_impedances_are_the_licensed_ones(name):
    """razor-nec5 is NEC-5's formulation twin; measured 3.0e-5 worst."""
    ours = _impedances(render(_deck(name), basis=TWIN).splitlines())
    theirs = _impedances(_oracle(name))
    assert len(ours) == len(theirs) > 1
    assert max(abs(a - b) / abs(b) for a, b in zip(ours, theirs, strict=True)) < 1e-4


def _blocks(text):
    """The per-frequency bodies: from each FREQUENCY heading to the next, the
    last one to the closing echoes; MATRIX TIMING's seconds masked."""
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if "- - FREQUENCY - -" in ln]
    tail = max(i for i, ln in enumerate(lines) if "INPUT LINE" in ln)
    bounds = [*heads, tail]
    out = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        body = [ln for ln in lines[a:b] if not ln.startswith("ALLOCATE CM:")]
        body = [_mask(ln) if "FILL=" in ln else ln for ln in body]
        while body and not body[-1].strip():
            body.pop()
        out.append(body)
    return out


def _stepped_and_singles(text, basis):
    """The stepped printout's frequency blocks, and the blocks the same deck
    prints as one one-frequency deck per step."""
    deck = parse_nec5(text)
    stepped = _blocks(render(text, basis=basis))
    lines = text.splitlines()
    k = next(i for i, ln in enumerate(lines) if ln[:2].upper() == "FR")
    singles = []
    for f in deck.frequencies_mhz:
        one = "\n".join([*lines[:k], f"FR 0 1 0 0 {f!r} 0", *lines[k + 1 :]]) + "\n"
        singles += _blocks(render(one, basis=basis))
    return stepped, singles


@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
@pytest.mark.parametrize(
    "name",
    [
        "p04_mul5",
        "p10d_ne_xq",
        "p08_rp",
        "p17_ld5",
        pytest.param("b01_buried_xq", marks=pytest.mark.slow),
    ],
)
def test_each_block_is_the_one_frequency_printout(name, basis):
    """Byte for byte (timing seconds masked): the shared fill half
    (`_serve._port_solution`) changes nothing a one-point deck prints."""
    stepped, singles = _stepped_and_singles(_deck(name), basis)
    assert stepped == singles


CORPUS = sorted((Path(__file__).parent / "fixtures" / "eznec" / "decks").glob("*.nec"))


@pytest.mark.integration
@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
def test_every_corpus_deck_steps_to_its_one_point_printouts(basis):
    """Every EZNEC corpus deck, its FR turned into three steps 1 % apart:
    each block is that frequency's one-point printout. The decks whose last
    request is a lone NE are refused, as NEC-5 runs no frequency for them."""
    served = 0
    for path in CORPUS:
        text = path.read_text(encoding="latin-1")
        lines = text.splitlines()
        k = next(i for i, ln in enumerate(lines) if ln[:2].upper() == "FR")
        f = float(lines[k].replace(",", " ").split()[5])
        lines[k] = f"FR 0 3 0 0 {f!r} {f * 0.01!r}"
        stepped_text = "\n".join(lines) + "\n"
        if "NEC ERROR" in render(stepped_text, basis=basis):
            assert "needs an XQ or RP after it" in render(stepped_text, basis=basis)
            continue
        stepped, singles = _stepped_and_singles(stepped_text, basis)
        assert stepped == singles, path.name
        served += 1
    assert served >= 70


@pytest.mark.parametrize(
    ("basis", "method"),
    [("razor-2p", "_assemble_Z_prepare"), ("bspline", "_same_edge_prep")],
)
def test_the_wavenumber_independent_half_is_built_once(monkeypatch, basis, method):
    from momwire.bspline import BSplineSolver
    from momwire.razor import RazorSolver

    cls = RazorSolver if basis == "razor-2p" else BSplineSolver
    calls = []
    real = getattr(cls, method)

    def spy(self, *a, **kw):
        calls.append(1)
        return real(self, *a, **kw)

    monkeypatch.setattr(cls, method, spy)
    render(_deck("p03_lin5"), basis=basis)
    assert len(calls) == 1


def test_every_frequency_is_solved_in_this_process(monkeypatch):
    seen = []
    real = _serve.serve

    def spy(deck, **kw):
        seen.append(deck.frequency_mhz)
        return real(deck, **kw)

    monkeypatch.setattr(_serve, "serve", spy)
    render(_deck("p03_lin5"), basis="razor-2p")
    assert seen == [26.0, 27.0, 28.0, 29.0, 30.0]


@pytest.mark.parametrize(
    ("name", "says"),
    [
        ("p10_ne", "needs an XQ or RP after it"),
        ("p10c_ne_fr_first", "needs an XQ or RP after it"),
        ("p16_no_xq", "needs an XQ or RP after it"),
        ("p11_ex_change", "served in a deck with one run"),
        ("p18_two_rp", "RP follows the card that runs"),
    ],
)
def test_what_nec5_answers_differently_is_refused_by_name(name, says):
    out = render(_deck(name), basis="razor-2p")
    assert "NEC ERROR" in out
    assert says in out


@pytest.mark.parametrize("name", ONE_FREQUENCY)
def test_a_one_frequency_deck_prints_as_it_always_did(name):
    text = _deck(name)
    assert "- - FREQUENCY - -" in render(text, basis="razor-2p")
    assert render(text, basis="razor-2p").count("- - FREQUENCY - -") == 1
