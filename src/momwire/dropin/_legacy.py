"""The old package names, ``momwire.portal`` and ``momwire.eznec``, as aliases.

momwire#1831 moved them to :mod:`momwire.dropin.nec2c` and
:mod:`momwire.dropin.nec5`.  antennaknobs and the scripts import the old
names, including private submodules (``momwire.eznec._serve``,
``momwire.portal._portal``), and some of them monkeypatch what they import.
So an alias here is the SAME module object, never a copy:

* importing the old package puts the new package in ``sys.modules`` under the
  old name, so ``momwire.eznec is momwire.dropin.nec5``;
* :class:`_AliasFinder` answers ``momwire.eznec.<sub>`` with the module
  ``momwire.dropin.nec5.<sub>``, imported once under its new name.

``python -m momwire.eznec`` and ``python -m momwire.portal`` still work:
``runpy`` finds ``__main__`` through the new package's path, and its relative
imports resolve through the finder.  ``__main__`` is therefore never aliased.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys

ALIASES = {
    "momwire.portal": "momwire.dropin.nec2c",
    "momwire.eznec": "momwire.dropin.nec5",
}


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec):
        module = importlib.import_module(self._target)
        self._spec = module.__spec__
        return module

    def exec_module(self, module) -> None:
        # The import system stamps the alias's spec on the module it was
        # handed; put the module's own back, so it still names its real home.
        module.__spec__ = self._spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for old, new in ALIASES.items():
            if not fullname.startswith(old + "."):
                continue
            rest = fullname[len(old) :]
            if rest.rsplit(".", 1)[-1] == "__main__":
                return None
            return importlib.machinery.ModuleSpec(fullname, _AliasLoader(new + rest))
        return None


_FINDER = _AliasFinder()


def install(old: str) -> None:
    """Make ``old`` (a key of :data:`ALIASES`) an alias of its new package."""
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FINDER)
    sys.modules[old] = importlib.import_module(ALIASES[old])
