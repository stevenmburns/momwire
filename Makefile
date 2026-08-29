# Named development lanes, mirroring antennaknobs' Makefile (its issue #733).
# The point is the same: each lane is a command you can type, not an
# incantation you must remember — and each pytest invocation below is copied
# VERBATIM from .github/workflows/ci.yml, with the line cited, so "make test
# passed" and "the PR gate will pass" mean the same thing.
#
# momwire needs this more than antennaknobs does, because it has six distinct
# marker expressions rather than one. Retyping `-m "slow and not memgate and
# not crossgate"` from memory is how a lane silently runs the wrong set, and
# `memgate` additionally needs `-n0` — without it the peak-RSS gates share a
# box with xdist workers and measure something else entirely.
#
# WHICH LANES GATE WHAT (ci.yml job conditions):
#   PR + push : lint, test, pynec
#   push only : integration, slow, crossgate, memgate, macos
# So `make gates` is what a merge to main will check, and `make test` is what
# blocks your PR.
#
# Lane economics (measured 2026-08-29, this 8-core box, post-#687 split):
#   make build    ~45 s cold; 1.1 s when nothing changed; 12 s after editing a
#                 small TU and 28 s after editing _accel_bspline.cpp, WITH
#                 ccache. Without ccache every edit is ~31 s (see below).
#   make lint     ~2 s
#   make test     ~2m33s — 4213 tests, and exactly what the PR gate runs.
#   integration / slow / crossgate / memgate are push-lane gates; not timed
#   here because they are run deliberately, not per edit.
#
# BUILD + ccache. setuptools' build_ext has no per-object staleness check: it
# recompiles EVERY source in an extension whenever one is newer. With the
# accelerator split into five TUs that means five compiles per edit, and
# ccache is what makes the four untouched ones ~free. `build` wires it the way
# ci.yml does (CC/CXX prefixed), and falls back silently when ccache is absent
# so a fresh clone still builds. Measured here: 48 s without, 12-28 s with,
# depending on which section you touched.
#
# This file deliberately does NOT reimplement the compile. setup.py carries
# ~250 lines of platform-conditional flags across win32/darwin/linux and two
# extensions with different flag sets; a second copy would drift, and drift in
# compile flags is SILENT. cibuildwheel and pip go through setup.py regardless.

# Interpreter resolution. Bare `python` is NOT assumed on PATH: momwire is
# usually checked out as a submodule of antennaknobs and shares that repo's
# .venv, which is often not activated. Order: an activated venv, a repo-local
# .venv, the parent repo's .venv, then python3. Absolute paths deliberately —
# invoking `../.venv/bin/python` relatively emits a sys.prefix RuntimeWarning
# on every call. Override with `make PYTHON=/path/to/python <target>`.
ifdef VIRTUAL_ENV
  PYTHON ?= $(VIRTUAL_ENV)/bin/python
else
  PYTHON ?= $(firstword $(wildcard $(abspath .venv/bin/python)) \
                        $(wildcard $(abspath ../.venv/bin/python)) python3)
endif
PY = $(PYTHON) -m

# ccache if it is on PATH, otherwise nothing. The base compiler follows the
# platform, matching ci.yml (gcc/g++ on Linux, clang/clang++ on macOS).
CCACHE := $(shell command -v ccache 2>/dev/null)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CC_BASE = clang
  CXX_BASE = clang++
else
  CC_BASE = gcc
  CXX_BASE = g++
endif
ifneq ($(CCACHE),)
  BUILD_ENV = CC="ccache $(CC_BASE)" CXX="ccache $(CXX_BASE)"
else
  BUILD_ENV =
endif

.PHONY: build test pynec macos-set integration slow crossgate memgate lint gates ccache-stats

# Rebuild the C++ extensions in place, through ccache when available.
build:
	$(BUILD_ENV) $(PYTHON) setup.py build_ext --inplace

# The PR gate's pytest, verbatim from ci.yml:129. Explicit `-m` rather than
# relying on pyproject's addopts default, for the same reason ci.yml is
# explicit: a lane must not change meaning when addopts is edited.
test:
	$(PY) pytest tests/ -m "not slow and not memgate and not integration"

# ci.yml:472 — the PyNEC cross-check. Also a PR gate; needs PyNEC installed.
pynec:
	$(PY) pytest tests/test_pynec_backend.py -v

# ci.yml:209 — the macOS lane's set (it keeps integration, which the linux PR
# lane excludes: subprocess/socket/CLI is exactly where platforms differ).
# Provided so every ci.yml pytest command has a lane; on linux it simply runs
# a superset of `test`.
macos-set:
	$(PY) pytest tests/ -m "not slow and not memgate"

# ci.yml:250 — subprocess/socket/CLI/printout seams. Push lane.
integration:
	$(PY) pytest tests/ -m "integration and not slow and not memgate"

# ci.yml:309 — the >couple-seconds tests. Push lane.
slow:
	$(PY) pytest tests/ -m "slow and not memgate and not crossgate"

# ci.yml:380 — cross-engine certification. Inherits addopts' xdist: these
# measure impedances, not memory, and parallelize fine.
crossgate:
	$(PY) pytest tests/ -m crossgate

# ci.yml:421 — memory certification. `-n0` is load-bearing: these gates
# measure peak RSS and cannot share a box with concurrent xdist workers.
memgate:
	$(PY) pytest tests/ -m memgate -n0

# ci.yml:62,66 — ruff is pinned to 0.15.21 in CI; a newer local ruff can
# disagree. `format --check` is separate because formatter drift is invisible
# without it.
lint:
	$(PY) ruff check
	$(PY) ruff format --check

# Everything a merge to main will check, locally.
gates: lint test integration slow crossgate memgate

# Did ccache actually help? Hits should be (TUs - 1) after a single-file edit.
ccache-stats:
	@ccache --show-stats 2>/dev/null || echo "ccache not installed (make build still works)"
