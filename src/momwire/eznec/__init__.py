"""``momwire.eznec`` is :mod:`momwire.dropin.nec5` (momwire#1831).

The NEC-5 / NEC-4.2 drop-in first shipped under this name. It stays
importable, submodules included, as the same module objects;
``python -m momwire.eznec`` still runs the engine. New code imports
``momwire.dropin.nec5``.
"""

from momwire.dropin import _legacy

_legacy.install(__name__)
