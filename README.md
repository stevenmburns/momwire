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
