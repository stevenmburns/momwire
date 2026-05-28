# pysim

A pure-Python method-of-moments antenna simulator with optional C++ accelerators (pybind11).

Extracted from [antenna_designer](https://github.com/stevenmburns/antenna_designer).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Test

```bash
pip install pytest numpy scipy matplotlib icecream scikit-rf
pytest tests/
```

## Optional: PyNEC backend

The web UI can run either the in-tree triangular MoM solver or NEC2 via [PyNEC](https://github.com/tmolteno/python-necpp), useful for cross-validation and for the ~5–10× faster single-frequency solves NEC2 delivers. PyNEC is vendored as a git submodule because the PyPI wheel builds are broken on current Python versions.

```bash
git submodule update --init --recursive
pip install swig          # SWIG goes into .venv via pip
sudo apt install autoconf automake libtool m4    # one-time system deps
scripts/build_pynec.sh
```

After the build, `from PyNEC import nec_context` works in `.venv`. The web UI's "solver" tab in the simulation section toggles between pysim and PyNEC at runtime. If the build is skipped the UI silently falls back to pysim.
