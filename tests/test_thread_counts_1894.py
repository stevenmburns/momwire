"""Table builds and the times line take the thread count that was asked for
(momwire#1894). A thread count may change scheduling only."""

from __future__ import annotations

import collections

import numpy as np
import pytest

from momwire import _near_interface as ni
from momwire import _times


@pytest.fixture
def physical(monkeypatch):
    """A 16-core box, whatever this one is."""
    monkeypatch.setattr(ni, "_physical_cpu_count", lambda: 16)


@pytest.mark.parametrize(
    ("env", "want"),
    [
        (None, 16),
        ("", 16),
        ("junk", 16),
        ("0", 16),
        ("-3", 16),
        ("1", 1),
        ("2", 2),
        ("8", 8),
        ("64", 16),
    ],
)
def test_the_table_thread_count_is_held_to_the_request(
    monkeypatch, physical, env, want
):
    if env is None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", env)
    assert ni._table_threads() == want


def test_the_crossing_table_build_asks_for_the_requested_workers(monkeypatch, physical):
    seen = []

    def spy(*args, **kw):
        seen.append(args[7])  # the n_threads slot of crossing_table_build
        return np.zeros(1)

    monkeypatch.setattr(ni._nia, "crossing_table_build", spy, raising=False)
    monkeypatch.setattr(ni, "_CROSSING_TABLE_CACHE", collections.OrderedDict())
    stats = dict(ni._CROSSING_TABLE_STATS)
    monkeypatch.setattr(ni, "_CROSSING_TABLE_STATS", stats)
    k_p = 2.0 * np.pi * 3.6e6 / 299792458.0
    for env, want in (("2", 2), ("1", 1)):
        monkeypatch.setenv("OMP_NUM_THREADS", env)
        ni._CROSSING_TABLE_CACHE.clear()
        ni._crossing_table_vals(k_p, complex(0.5, -0.1), ni._LAM_MULT)
        assert seen[-1] == want
    assert len(seen) == 2


def test_the_times_line_reports_the_fill_threads(monkeypatch):
    monkeypatch.delenv("MOMWIRE_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert _times.fill_threads() == 3
    with _times.answering("test") as t:
        line = t.line()
    assert " threads=3 " in line
