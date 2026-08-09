"""The public multi-port solve result (`stevenmburns/momwire#232`).

Every solver family already builds one right-hand-side column per port, runs
ONE fill and ONE factorisation over all of them, reads the port currents off
the resulting columns — and then returns only the admittance matrix, dropping
the columns on the floor. Consumers that need the columns (a NEC-protocol
front end sending several `EX` sets against one geometry, a field evaluation
that must not re-solve) were reaching into private per-family internals to get
them back. `PortSolution` is what `compute_port_solution()` returns instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PortSolution:
    """Everything one multi-port solve produced, from one fill + factorisation.

    Ports are ordered exactly as the solver's own port list: the configured
    gap feeds first, then `junction_ports`, then (where the family has them)
    `node_ports`. That is the same order `compute_y_matrix` indexes.

    Attributes
    ----------
    y:
        `(n_ports, n_ports)` complex short-circuit admittance matrix. Bit
        identical to `compute_y_matrix()` — the latter is implemented as
        `compute_port_solution().y`, so the two cannot drift apart.
    coeffs:
        `(n_dof, n_ports)` complex. Column *j* is the solution vector for a
        1 V drive at port *j* with every other port shorted, in the basis this
        solver family solves in. Any other excitation is `coeffs @ V` with no
        second fill — that is the point of the class. `n_dof` is the family's
        own degree-of-freedom count (segment amplitudes for the sinusoidal
        families, B-spline amplitudes for the spline families, plus each
        family's port/enrichment blocks where those are active); it is NOT in
        general the number of wire segments.
    port_currents:
        `(n_ports, n_ports)` complex — the current at port *i* under the unit
        drive of port *j*. This IS the admittance matrix; it is carried as its
        own field so a consumer can assert the identity rather than assume it
        (`sol.port_currents is sol.y` holds today).
    basis:
        Opaque per-solve handle carrying whatever context this family needs to
        interpret a column of `coeffs` — geometry tables, the segment view, the
        basis polynomials. **Do not introspect it**: its type and contents are
        private to the solver family and change without notice. Its stability
        contract is exactly this: stable across the ports OF ONE SOLUTION, and
        not stable across solves. Re-solving (a new frequency, a changed
        geometry, a second `compute_port_solution()` call) invalidates it, and
        pairing a `coeffs` column with a `basis` from a different solution is
        undefined. For turning a column into currents on wires today, use the
        solver's public `currents_at_knots(coeffs[:, j])`.
    """

    y: np.ndarray
    coeffs: np.ndarray
    port_currents: np.ndarray
    basis: Any

    @property
    def n_ports(self) -> int:
        """Number of ports — `y.shape[0]`, and `coeffs`' column count."""
        return int(self.y.shape[0])
