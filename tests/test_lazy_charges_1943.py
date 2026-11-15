"""momwire#1943 lever 5: the charge table is computed when it is read.

`DeckSolver.solve_group` computed every segment's charge density at every
frequency, and only a `PQ` card (or the NEC-4 dialect's charge rows) reads
it: 0.18 s of the sym4 ELT whip's 13 steps on Skylake, printed nowhere. The
gates: a deck with no `PQ` never computes it, a deck that prints it computes
it once per solve, and the value read is the one the eager solve returned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from momwire.dropin.nec2c import _portal

FIXTURES = Path(__file__).parent / "fixtures" / "nec_portal"


def _count(monkeypatch):
    calls = []
    real = _portal.DeckSolver._segment_charges

    def counted(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(_portal.DeckSolver, "_segment_charges", counted)
    return calls


def test_a_deck_without_pq_never_computes_the_charges(monkeypatch):
    calls = _count(monkeypatch)
    text = (FIXTURES / "dipole_fr_sweep.deck").read_text()
    assert "PQ" not in text
    _portal.render_deck(text)
    assert calls == []


def test_a_printed_table_is_computed_once_per_solve(monkeypatch):
    calls = _count(monkeypatch)
    text = (FIXTURES / "dipole_pq_charges.deck").read_text()
    deck = _portal.parse_deck(text)
    solver = _portal.DeckSolver(deck, basis="sinusoidal")
    group = next(g for g in deck.groups if g is not None)
    result = solver.solve_group(group, group.freqs_mhz[0])
    assert calls == []
    first = result.segment_charges
    assert result.segment_charges is first  # read once, then held
    assert len(calls) == 1
    eager = solver._segment_charges(result.solver, result.coeffs, _omega(result))
    assert first.dtype == eager.dtype and first.tobytes() == eager.tobytes()


def _omega(result):
    return 2.0 * np.pi * _portal.C_LIGHT / float(result.wavelength)
