"""momwire#1420 step 3: the per-run finish (element geometry, element
currents and charges, razor's knot currents and slopes) without per-element
and per-wire numpy round trips.

Speed changes that must move no bit. Each gate captures the REAL arguments
the production seam (``_shell.render``) hands ``_element_geometry`` and
``_element_currents_and_charges``, and compares what the new code returns
against verbatim copies of the walks it replaced (``_ref_*`` below), bit for
bit — on decks chosen to reach every branch: plain junctions, a hub of more
than eight ends (a pairwise rather than sequential sum), grounded ends,
one-segment wires, a card split at z = 0, and phantom elements.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from momwire.eznec import _serve
from momwire.eznec._shell import render
from momwire.razor import RazorSolver

FIXTURES = Path(__file__).parent / "fixtures"


# ---- verbatim copies of the replaced walks --------------------------------
def _ref_element_geometry(structure, wavelength):
    centres, lengths, tags = [], [], []
    for wire, points in zip(structure.wires, structure.points, strict=True):
        for k in range(wire.segment_count):
            a = np.asarray(points[k], dtype=float)
            b = np.asarray(points[k + 1], dtype=float)
            centres.append(0.5 * (a + b) / wavelength)
            lengths.append(float(np.linalg.norm(b - a)) / wavelength)
            tags.append(wire.tag)
    return centres, lengths, tags


def _ref_razor_knots(solver, coeffs):
    coeffs = np.asarray(coeffs)
    geom = solver._build_geometry()
    per_wire = geom["per_wire"]
    seg_offsets = geom["seg_offsets"]
    basis_offsets = geom["basis_offsets"]
    n_interior = geom["n_basis_interior"]
    wing_seg, wing_rise, wing_sigma = (
        geom["wing_seg"],
        geom["wing_rise"],
        geom["wing_sigma"],
    )
    out = []
    for w_idx, pw in enumerate(per_wire):
        n_knots = pw["arc_at_knot"].shape[0]
        I = np.zeros(n_knots, dtype=np.complex128)  # noqa: E741
        I[1:-1] = coeffs[basis_offsets[w_idx] : basis_offsets[w_idx + 1]]
        out.append(I)
    n_basis_total = wing_seg.shape[0]
    if n_basis_total > n_interior:
        j_seg = wing_seg[n_interior:].reshape(-1)
        j_rise = wing_rise[n_interior:].reshape(-1)
        j_sigma = wing_sigma[n_interior:].reshape(-1)
        j_coeff = np.repeat(coeffs[n_interior:], 2)
        for w_idx in range(len(per_wire)):
            start_seg = seg_offsets[w_idx]
            end_seg = seg_offsets[w_idx + 1] - 1
            at_start = (j_seg == start_seg) & ~j_rise
            if at_start.any():
                out[w_idx][0] = (j_sigma[at_start] * j_coeff[at_start]).sum()
            at_end = (j_seg == end_seg) & j_rise
            if at_end.any():
                out[w_idx][-1] = (j_sigma[at_end] * j_coeff[at_end]).sum()
    return out


def _ref_razor_slopes(solver, coeffs, s_array):
    per_wire = solver._build_geometry()["per_wire"]
    knot_currents = _ref_razor_knots(solver, coeffs)
    out = []
    for w_idx, pw in enumerate(per_wire):
        arc = pw["arc_at_knot"]
        slope = np.diff(knot_currents[w_idx]) / np.diff(arc)
        s_eval = (
            arc if s_array is None else np.asarray(s_array[w_idx], dtype=np.float64)
        )
        if s_eval.shape[0] == 0:
            out.append(np.zeros(0, dtype=np.complex128))
            continue
        s_eval = np.clip(s_eval, arc[0], arc[-1])
        span = np.clip(
            np.searchsorted(arc, s_eval, side="right") - 1, 0, slope.shape[0] - 1
        )
        out.append(np.asarray(slope[span], dtype=np.complex128))
    return out


def _ref_element_currents_and_charges(solver, mesh, coeffs, omega):
    knot_currents = solver.currents_at_knots(coeffs)
    centres_per_piece = []
    for piece in mesh.pieces:
        step = piece.length / piece.n_elements
        centres_per_piece.append((np.arange(piece.n_elements) + 0.5) * step)
    for split in mesh.split_elements.values():
        centres_per_piece[split.centre_piece] = np.append(
            centres_per_piece[split.centre_piece], split.centre_s
        )
    slopes = solver.current_slopes(coeffs, centres_per_piece)
    currents, charges = [], []
    for index, (piece_index, element) in enumerate(mesh.element_of):
        if piece_index == _serve._NO_PIECE:
            currents.append(0j)
            charges.append(0j)
            continue
        split = mesh.split_elements.get(index)
        if split is not None:
            first = np.asarray(knot_currents[split.lower])[0]
            last = np.asarray(knot_currents[split.upper])[-1]
            currents.append(0.5 * (first + last))
            charges.append(-slopes[split.centre_piece][-1] / (1j * omega))
            continue
        knots = np.asarray(knot_currents[piece_index])
        currents.append(0.5 * (knots[element] + knots[element + 1]))
        charges.append(-slopes[piece_index][element] / (1j * omega))
    return currents, charges


# ---- bitwise comparison ----------------------------------------------------
def _same(a, b):
    if isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _same(x, y)
        return
    x, y = np.asarray(a), np.asarray(b)
    assert type(a) is type(b), (type(a), type(b))
    assert x.dtype == y.dtype and x.shape == y.shape
    assert x.tobytes() == y.tobytes()


def _hub_deck(n_radials):
    """A vertical over perfect ground with ``n_radials`` elevated radials at
    its base, two one-segment stubs and a one-segment wire joined at both
    ends: a junction of n_radials + 1 ends, so side A sums n_radials wings."""
    rad = "".join(
        f"GW {k + 2} 5 0 0 1 {float(4 * np.cos(a))!r} {float(4 * np.sin(a))!r} 1 1e-3\n"
        for k, a in enumerate(np.linspace(0, 2 * np.pi, n_radials, endpoint=False))
    )
    t = n_radials + 2
    return (
        "CM hub\nCE\nGW 1 9 0 0 1 0 0 6 1e-3\n"
        + rad
        + f"GW {t} 1 0 0 6 0.3 0 6 1e-3\nGW {t + 1} 1 0.3 0 6 0.3 0 6.3 1e-3\n"
        + f"GW {t + 2} 3 0 0 0 0 0 1 1e-3\n"
        + "GE 1\nGN 1\nFR 0 1 0 0 14\nEX 0 1 1 0 1 0\nPQ 0\nXQ\nEN\n"
    )


DECKS = {
    "hub4": _hub_deck(4),
    "hub11": _hub_deck(11),
    "split": (FIXTURES / "eznec_split_1281" / "dan_through.nec").read_text(
        encoding="latin-1"
    ),
    "phantom": next((FIXTURES / "eznec_phantom_1139").glob("*.nec")).read_text(
        encoding="latin-1"
    ),
}


@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
@pytest.mark.parametrize("name", sorted(DECKS))
def test_finish_matches_the_walks_at_the_seam(monkeypatch, name, basis):
    seen = {"geometry": 0, "elements": 0, "razor": 0}
    geometry, elements = _serve._element_geometry, _serve._element_currents_and_charges

    def check_geometry(structure, wavelength):
        got = geometry(structure, wavelength)
        _same(list(got), list(_ref_element_geometry(structure, wavelength)))
        seen["geometry"] += 1
        return got

    def check_elements(solver, mesh, coeffs, omega):
        got = elements(solver, mesh, coeffs, omega)
        _same(
            list(got),
            list(_ref_element_currents_and_charges(solver, mesh, coeffs, omega)),
        )
        if (
            isinstance(solver, RazorSolver)
            and getattr(solver, "_wire_dedup", None) is None
        ):
            seen["razor"] += 1
            _same(solver.currents_at_knots(coeffs), _ref_razor_knots(solver, coeffs))
            centres = [np.linspace(-0.1, p.length + 0.1, 7) for p in mesh.pieces]
            _same(
                solver.current_slopes(coeffs, centres),
                _ref_razor_slopes(solver, coeffs, centres),
            )
            _same(
                solver.current_slopes(coeffs), _ref_razor_slopes(solver, coeffs, None)
            )
        seen["elements"] += 1
        return got

    monkeypatch.setattr(_serve, "_element_geometry", check_geometry)
    monkeypatch.setattr(_serve, "_element_currents_and_charges", check_elements)
    render(DECKS[name], basis=basis)
    assert seen["geometry"] >= 1 and seen["elements"] >= 1
    if basis == "razor-2p":
        assert seen["razor"] >= 1


def test_the_hub_deck_reaches_a_pairwise_sum():
    """The hub11 deck must put more than eight wings on one end, or the gate
    above never compares a pairwise (rather than sequential) sum."""
    seen = []

    real = RazorSolver._end_wings

    def spy(self, geom):
        table = real(self, geom)
        seen.extend(len(w) for _, _, w in table)
        return table

    with pytest.MonkeyPatch.context() as m:
        m.setattr(RazorSolver, "_end_wings", spy)
        render(DECKS["hub11"], basis="razor-2p")
    assert max(seen) > 8


def test_a_seeded_one_bit_change_fails_the_finish_gate(monkeypatch):
    """Red control: a length one ulp long fails the seam gate."""
    geometry = _serve._element_geometry

    def seeded(structure, wavelength):
        centres, lengths, tags = geometry(structure, wavelength)
        lengths[0] = float(np.nextafter(lengths[0], np.inf))
        return centres, lengths, tags

    monkeypatch.setattr(_serve, "_element_geometry", seeded)
    with pytest.raises(AssertionError):
        test_finish_matches_the_walks_at_the_seam(monkeypatch, "hub4", "razor-2p")
