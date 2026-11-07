"""``momwire.portal`` is :mod:`momwire.dropin.nec2c` (momwire#1831).

The SimNEC NEC-2 drop-in first shipped under this name. It stays importable,
submodules included, as the same module objects; ``python -m momwire.portal``
still runs the engine. New code imports ``momwire.dropin.nec2c``.
"""

from momwire.dropin import _legacy

_legacy.install(__name__)
