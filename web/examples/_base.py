"""Base types for the antenna-example registry.

An `AntennaExample` bundles everything the web layer needs to serve one
geometry: parameter schema, pysim solve/sweep, and pynec build/solve. Any
callable left as None signals the backend doesn't support that operation
for this geometry.

Keeping these as plain callables (rather than a class hierarchy) means
each example module is a flat file of functions plus one EXAMPLE = ...
assignment at the bottom — easy to read, easy to delete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# A pynec "build" returns the dict-of-context-plus-derived-geom that both
# solve() and pattern() consume — see web/pynec_backend.py for the shape.
PynecBuildFn = Callable[[dict], dict]

# pysim solve / pynec solve: take a request dict, return the response dict
# the frontend renders.
SolveFn = Callable[[dict], dict]

# pysim sweep: take a request dict + frequency list, return parallel
# (real, imag) lists of input impedance per frequency.
SweepFn = Callable[[dict, list[float]], tuple[list[float], list[float]]]


@dataclass(frozen=True)
class ParamSpec:
    """One UI-exposed parameter for a geometry.

    Reserved for the upcoming `GET /examples` endpoint that lets the
    frontend render parameter controls generically. The current pilot
    keeps the field empty; sliders still come from the hand-written
    `DEFAULT_BACKEND_OPTS` in App.tsx until the schema cutover lands.
    """

    name: str
    label: str
    default: Any
    kind: str = "float"  # float | int | bool | enum
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    sweepable: bool = False


@dataclass(frozen=True)
class AntennaExample:
    name: str
    label: str
    pysim_solve: SolveFn
    pysim_sweep: SweepFn
    pynec_build: Optional[PynecBuildFn] = None
    pynec_solve: Optional[SolveFn] = None
    param_schema: tuple[ParamSpec, ...] = field(default_factory=tuple)
