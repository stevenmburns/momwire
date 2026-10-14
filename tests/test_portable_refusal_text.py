"""The shared refusals spell numbers, shapes and wire ends portably.

They used to embed numpy's array printing (``got [ 0.001 -0.002]``), a
tuple shape (``got shape (3,)``) and Python's string repr (``wire 0
'start'``). The razor-2p C++ front half (momwire#1479) reproduces these
messages word for word, and copying another language's printing into C++ is
the wrong dependency, so the text is fixed here: ``%.12g`` numbers,
comma-separated in brackets, shapes as ``3`` / ``2x3``, and wire ends as
"the start of wire 0".
"""

import pytest

from momwire import _wire_spec
from momwire.razor import RazorSolver


def test_numbers_and_shapes_are_spelled_portably():
    assert _wire_spec.format_numbers([0.001, -0.002, 1e-12]) == "[0.001, -0.002, 1e-12]"
    assert _wire_spec.format_numbers([0.1 + 0.2]) == "[0.3]"
    assert _wire_spec.format_shape((3,)) == "3"
    assert _wire_spec.format_shape((2, 3)) == "2x3"
    assert _wire_spec.format_shape(()) == "scalar"


def test_the_radius_refusals_word_for_word():
    with pytest.raises(ValueError) as exc:
        _wire_spec.normalize_wire_radius([0.001, -0.002], 2)
    assert str(exc.value) == (
        "wire_radius entries must be positive and finite, got [0.001, -0.002]"
    )
    with pytest.raises(ValueError) as exc:
        _wire_spec.normalize_wire_radius([0.001, 0.002, 0.003], 2)
    assert str(exc.value) == (
        "wire_radius: expected a scalar or a length-2 sequence "
        "(one entry per wire), got shape 3"
    )


TEE = [
    [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
    [(1.0, 0.0, 1.0), (2.0, 0.0, 1.0)],
    [(1.0, 0.0, 1.0), (1.0, 0.0, 2.0)],
]


def _solver(**kw):
    return RazorSolver(wires=TEE, nsegs=7, wire_radius=1e-3, wavelength=10.0, **kw)


def test_the_node_gap_refusals_name_the_end_in_words():
    with pytest.raises(ValueError) as exc:
        _solver(node_gaps=[(0, "start", 1.0)])
    assert "node_gaps[0]: the start of wire 0 is not a member" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        _solver(node_gaps=[(0, "end", 1.0), (0, "end", 1.0)])
    assert "node_gaps[1]: the end of wire 0 is listed twice" in str(exc.value)
    for msg in (str(exc.value),):
        assert "'end'" not in msg and "'start'" not in msg


def test_the_inert_wire_refusal_names_the_basis_without_a_repr():
    """The seam's inert one-segment wire refusal (momwire#608) names the
    basis plainly, not as a Python string repr, so the C++ seam (#1479) says
    the same words."""
    from momwire.eznec import _serve
    from momwire.deck._nec5 import parse_nec5

    deck = parse_nec5(
        "CM one-segment GW, junctioned at neither end\nCE\n"
        "GW 1,8,0.,-2.,0.,0.,2.,0.,.001\n"
        "GW 2,1,0.5,-0.25,0.,0.5,0.25,0.,.001\n"
        "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,4,0,1.,0.\nXQ\nEN\n"
    )
    with pytest.raises(_serve.ServeRefusal) as exc:
        _serve.serve(deck, basis="razor-2p")
    assert str(exc.value) == (
        "wire 2 is one segment long and meets no other wire at either end, so "
        "basis razor-2p has no basis function to put on it and it would "
        "scatter nothing (momwire#608); this deck declares 1 such GW card"
    )
