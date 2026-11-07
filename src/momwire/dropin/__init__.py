"""momwire as a drop-in engine: the Python route a host launches.

A host program (SimNEC, EZNEC, 4nec2) launches an external NEC engine by its
file name and talks to it through a fixed contract.  This package is momwire
standing in for that engine, in Python: installed from PyPI or frozen into
the Windows bundle, backed by the resident warm server, and serving every
deck momwire can solve.  The C++ executables under ``standalone/`` are the
other route: no Python, a subset served, the rest refused by name.  Both are
engines a host launches, with the same argv and stdin contracts
(momwire#1831).

One module per engine contract:

* :mod:`momwire.dropin.nec2c` -- the NEC-2 dialect under nec2c's contract:
  one long-lived process, decks streamed over stdin, printouts on stdout,
  the ``-version`` handshake and ``NX`` framing (``momwire-nec2c*``).
* :mod:`momwire.dropin.nec5` -- the NEC-5 dialect and the NEC-4.2 slot under
  the console engines' contract: one launch per calculation,
  ``engine <deck> <printout>``, exit 0, refusals written into the printout
  (``momwire-nec5*`` and ``momwire-nec42*``).  The NEC-4.2 slot is a mode
  inside it (``_nec4.py``); it can become its own module later.

``momwire.portal`` and ``momwire.eznec``, the names these first shipped
under, stay importable as shims onto the modules above (:mod:`._legacy`).

The isolation rule (#846 §4) carries over: ``momwire.dropin`` may import
momwire's public solver API and :mod:`momwire.deck`, and **nothing outside
``momwire/dropin/`` may import from it**, bar the two shims.  No host
protocol detail may become load-bearing for a solver.
``tests/test_portal.py`` enforces it at the source.
"""
