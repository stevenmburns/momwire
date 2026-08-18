"""Constructor-side normalisation of the wire spec, shared by every solver.

The formulations disagree about basis, testing rule, kernel and ground —
they do not disagree about what `wire_radius=[…]` means. The audit
momwire#429 measured 464 of 749 constructor code lines sitting in verbatim
clones across the four roots; this module is where those land as they are
extracted, starting with the radius (rank 3, with momwire#425).

Nothing here knows a formulation. A function that would have to branch on
one belongs in the solver.
"""

import numpy as np


def normalize_wire_radius(value, n_wires, *, per_wire_refusal=None):
    """`wire_radius` → ``((n_wires,) float array, uniform value or None)``.

    A scalar applies to every wire; a length-n_wires sequence gives each
    wire (polyline) its own conductor radius (stevenmburns/momwire#147).
    Every entry must be positive and finite — a radius of zero is not a
    thin wire, it is the singular kernel the reduced form exists to avoid.

    The second return value is the SCALAR FAST PATH: the common radius when
    every wire shares one (including a uniform array), else None. It is
    what keeps the historical scalar code paths — and the single-`a` C++
    kernels — bit-identical whenever a model is uniform, however the caller
    spelled it.

    `per_wire_refusal` is for a formulation whose kernel takes exactly one
    `a`: pass the refusal prose and a non-scalar value raises
    `NotImplementedError` with it instead of being accepted. That is one
    spelling of the refusal rather than one per solver, and dropping the
    argument is the whole of what "this formulation gained per-wire radii"
    means at the constructor.
    """
    if per_wire_refusal is not None and not np.isscalar(value):
        raise NotImplementedError(per_wire_refusal)
    radius = np.asarray(value, dtype=float)
    if radius.ndim == 0:
        radius = np.full(n_wires, float(radius))
    elif radius.shape != (n_wires,):
        raise ValueError(
            f"wire_radius: expected a scalar or a length-{n_wires} sequence "
            f"(one entry per wire), got shape {radius.shape}"
        )
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError(
            f"wire_radius entries must be positive and finite, got {radius}"
        )
    uniform = float(radius[0]) if np.all(radius == radius[0]) else None
    return radius, uniform
