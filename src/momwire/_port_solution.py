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

from contextlib import contextmanager
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


class _SweptPortSolutions:
    """The swept port entry points, thinned over ONE per-k core (#252).

    `compute_y_matrix_swept` used to carry its own copy of each family's port
    algebra — the drive columns, the junction-port split, the readout — beside
    the copy in `compute_port_solution`. That is exactly the drift #232 closed
    at single k, reopening one level up, and the swept side is where it would
    have gone unnoticed longest (nothing in the tree consumed swept port
    columns). Both public swept port entry points now read the family's
    `_port_solutions_swept` generator and do nothing else.

    A family supplies:

    * `_port_solutions_swept(k_array)` — a generator yielding one
      `PortSolution` per wavenumber, in `k_array` order, whose per-k core is
      the family's own `compute_port_solution` wherever the family has no
      batched accelerator to preserve. It must put the frequency triple back
      the way it found it.
    * `_port_count()` — how many ports `compute_port_solution` would return,
      answered without solving (so an empty sweep still gets its shape).

    The generator is what bounds memory: `compute_y_matrix_swept` keeps only
    the Y blocks, so a batched family's per-chunk solution columns die with
    their chunk instead of accumulating across the sweep.
    """

    def _port_solutions_swept(self, k_array):
        """Per-k `PortSolution` generator — implemented by each family."""
        raise NotImplementedError

    def _port_count(self):
        """Port count without solving — implemented by each family."""
        raise NotImplementedError

    def _set_k(self, kk):
        """Rebind the frequency triple the sweeps mutate in lockstep."""
        self.k = float(kk)
        self.omega = self.k * self.c
        self.wavelength = self.c / (self.omega / (2 * np.pi))

    @contextmanager
    def _k_restored(self):
        """Run a per-k loop with the frequency triple put back on the way out
        — including on an exception, and on an abandoned generator (CPython
        closes it, which raises `GeneratorExit` through this `finally`)."""
        saved = (self.k, self.wavelength, self.omega)
        try:
            yield
        finally:
            self.k, self.wavelength, self.omega = saved

    def compute_port_solution_swept(self, k_array):
        """One `PortSolution` per wavenumber, in `k_array` order (#252).

        The swept twin of `compute_port_solution`: `[i]` is what that method
        returns at `k_array[i]`, same port order, same column conventions,
        same opaque-`basis` rules (each entry's handle belongs to its own
        solve and to no other).

        This materialises every solution column of the whole sweep — for a
        large model that is `n_k · n_dof · n_ports` complex numbers. Callers
        that only want the admittances should use `compute_y_matrix_swept`,
        which streams the same core and keeps only the Y blocks.
        """
        return list(self._port_solutions_swept(np.asarray(k_array, dtype=float)))

    def compute_y_matrix_swept(self, k_array) -> np.ndarray:
        """Per-frequency Y matrices; returns `(n_k, n_ports, n_ports)`.

        `[i]` is `compute_y_matrix()` at `k_array[i]` — this is
        `compute_port_solution_swept`'s `y` field and nothing else, so the two
        cannot drift (#252). Streams the per-k core, so the solution columns
        never accumulate across the sweep.
        """
        k_array = np.asarray(k_array, dtype=float)
        n_p = self._port_count()
        out = np.zeros((k_array.shape[0], n_p, n_p), dtype=np.complex128)
        for i, sol in enumerate(self._port_solutions_swept(k_array)):
            out[i] = sol.y
        return out
