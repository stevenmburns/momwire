# Named development lanes, mirroring antennaknobs' Makefile (its issue #733).
# The point is the same: each lane is a command you can type, not an
# incantation you must remember — and each pytest invocation below is copied
# VERBATIM from the ci.yml job named beside it, so "make test passed" and
# "the PR gate will pass" mean the same thing. That claim is ENFORCED, not
# asserted: tests/test_makefile_lanes.py checks every lane against its own
# ci.yml job (and the reverse) — the full rationale for the lane taxonomy,
# the memgate -n0 rule, and the pin-not-generate design lives there, once.
#
# WHICH LANES GATE WHAT (ci.yml job conditions):
#   PR + push : lint, test, pynec
#   push only : integration, slow, crossgate, memgate, macos
# `make gates` is what a merge to main will check, and `make test` + `make
# pynec` + `make lint` are what block your PR.
#
# BUILD + ccache. setuptools' build_ext has no per-object staleness check: it
# recompiles EVERY source in an extension whenever one is newer. With the
# accelerator split into five TUs (momwire#687) ccache is what makes the
# untouched ones ~free — `build` wires CC/CXX the way ci.yml does and falls
# back silently when ccache is absent. Timing measurements live in
# momwire#715/#716, where they carry their dates.
#
# This file deliberately does NOT reimplement the compile. setup.py carries
# ~250 lines of platform-conditional flags across win32/darwin/linux and two
# extensions with different flag sets; a second copy would drift, and drift in
# compile flags is SILENT. cibuildwheel and pip go through setup.py regardless.

# Interpreter resolution. Bare `python` is NOT assumed on PATH: momwire is
# usually checked out as a submodule of antennaknobs and shares that repo's
# .venv, which is often not activated. Order: an activated venv (existence-
# checked — a stale exported VIRTUAL_ENV from a dead shell must fall through,
# not break every lane), a repo-local .venv, the parent repo's .venv, then
# python3. Absolute paths deliberately — invoking `../.venv/bin/python`
# relatively emits a sys.prefix RuntimeWarning on every call. An exported or
# command-line PYTHON wins over all of this (`?=` semantics), which is the
# supported override: `make PYTHON=/path/to/python <target>`.
PYTHON ?= $(firstword \
    $(if $(VIRTUAL_ENV),$(wildcard $(VIRTUAL_ENV)/bin/python)) \
    $(wildcard $(abspath .venv/bin/python)) \
    $(wildcard $(abspath ../.venv/bin/python)) \
    python3)
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

# Never parallelize the lanes themselves: every pytest lane except memgate
# already saturates the box via addopts' `-n auto` (oversubscription measured
# SLOWER in momwire#400), and memgate certifies peak RSS, which is only a
# measurement when nothing else is running. `make -j gates` without this
# would run the RSS gates beside five parallel suites — invalid numbers, and
# a breach of the 16GB box's heavy-process budget.
.NOTPARALLEL:

.PHONY: build test pynec macos-set integration slow crossgate memgate lint gates ccache-stats

# Rebuild the C++ extensions in place, through ccache when available.
# MOMWIRE_REQUIRE_ACCEL=1 makes a compile failure FAIL the lane: setup.py's
# OptionalBuildExt otherwise downgrades toolchain errors to a warning (the
# sdist graceful path), which locally means exit 0 and the suite silently
# running a STALE .so — the repo's documented measurement hazard.
#
# It guards ONE way in, though: a compile that FAILS. It cannot see a compile
# that never happened, which is what momwire#824 turned out to be — an edited
# header absent from `depends=`, so build_ext recompiled nothing, relinked the
# old object and exited 0. Both routes end at a stale .so and only this one
# announces itself. tests/test_accel_header_deps.py covers the other.
build:
	$(BUILD_ENV) MOMWIRE_REQUIRE_ACCEL=1 $(PYTHON) setup.py build_ext --inplace

# The PR gate's pytest, verbatim from ci.yml's `test` job. Explicit `-m`
# rather than relying on pyproject's addopts default, for the same reason
# ci.yml is explicit: a lane must not change meaning when addopts is edited.
test:
	$(PY) pytest tests/ -m "not slow and not memgate and not integration"

# ci.yml `test-pynec` job — the PyNEC cross-check. Also a PR gate. Skips
# cleanly (importorskip) when PyNEC isn't installed, so it costs nothing in
# `gates` on a box without the wheel.
pynec:
	$(PY) pytest tests/test_pynec_backend.py -v

# ci.yml `test-macos` job's set (it keeps integration, which the linux PR
# lane excludes: subprocess/socket/CLI is exactly where platforms differ).
# Provided so every ci.yml pytest command has a lane; on linux its SELECTION
# is exactly `test` + `integration`, so `gates` covers it without re-running.
macos-set:
	$(PY) pytest tests/ -m "not slow and not memgate"

# ci.yml `test-integration` job — subprocess/socket/CLI/printout seams.
integration:
	$(PY) pytest tests/ -m "integration and not slow and not memgate"

# ci.yml `test-slow` job — the >couple-seconds tests.
slow:
	$(PY) pytest tests/ -m "slow and not memgate and not crossgate"

# ci.yml `test-crossgate` job — cross-engine certification. Inherits addopts'
# xdist: these measure impedances, not memory, and parallelize fine.
crossgate:
	$(PY) pytest tests/ -m crossgate

# ci.yml `test-memgate` job — memory certification. `-n0` is load-bearing:
# these gates measure peak RSS and cannot share a box with xdist workers.
memgate:
	$(PY) pytest tests/ -m memgate -n0

# ci.yml `lint` job. CI pins ruff==0.16.5; the guard below makes a missing
# local ruff a clear one-liner instead of a mid-lane stack trace, and names
# the pin so a version disagreement with CI has a first place to look.
lint:
	@$(PY) ruff --version >/dev/null 2>&1 || { \
	  echo "ruff is not installed in $(PYTHON)'s environment."; \
	  echo "CI pins it: pip install ruff==0.16.5    (ci.yml lint job)"; \
	  exit 1; }
	$(PY) ruff check
	$(PY) ruff format --check

# Everything a merge to main will check, locally. pynec is included (it is a
# PR + push gate; it self-skips without PyNEC). macos-set is not a separate
# prerequisite because on linux its selection is exactly test + integration,
# both already here — see the lane's own comment.
gates: lint test pynec integration slow crossgate memgate

# Did ccache actually help? Hits should be (TUs - 1) after a single-file edit.
ccache-stats:
	@ccache --show-stats 2>/dev/null || echo "ccache not installed (make build still works)"
