"""Physical constants shared across the package.

Single owner for values that more than one module needs spelled identically.
Before momwire#456 the speed of light was defined four times over (here,
``_sommerfeld``, ``portal._portal``, ``deck._solver``, plus the arriving
network core); four literals cannot disagree usefully, and one of them
drifting is a silent wavelength error rather than a failure. Modules keep
their own local names — ``_C_LIGHT = C_LIGHT`` — so no call site changes.
"""

from __future__ import annotations

# Exact by SI definition of the metre, not a measurement: no more digits exist.
C_LIGHT = 299_792_458.0
