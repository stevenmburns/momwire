"""PyInstaller entry point for the packaged EZNEC drop-in executable.

The frozen program is :mod:`momwire.eznec`'s process shell and nothing else:
``momwire-eznec <deck> <printout>``, the same two-argument protocol EZNEC
Pro+ v7 uses to launch its external NEC-5 console engine.  A separate script
(rather than ``-m momwire.eznec``) exists only because PyInstaller wants a
file to trace from.
"""

import sys

from momwire.eznec._shell import main

if __name__ == "__main__":
    sys.exit(main())
