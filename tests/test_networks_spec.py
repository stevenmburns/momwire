"""The circuit spec layer: validation, the formal `Cable`, per-branch stamps
(momwire#456 ws2, B2).

Where ``test_networks_reducer.py`` is the antennaknobs MNA battery ported
whole, this file covers what the MOVE made new or newly load-bearing:

* the flat `Network`'s validation — the checks that survived the surgery, and
  the ``branch_paths`` field that stopped being derived and became passable;
* `TL.from_cable`'s formal `Cable` type, which no longer resolves names
  because momwire ships no catalog;
* per-branch stamp oracles for the branches whose antennaknobs tests were
  wrapped around engines or a Touchstone parser neither of which came along —
  `Admittance`, `TouchstoneLoad`/`TouchstoneTwoPort` through a hand-rolled
  `AdmittanceData`, `Transformer`, `Shunt`, `DrivenCurrent`.

Every antenna Y here is synthetic, so a failure names a stamp, not a design.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from momwire.networks import (
    TL,
    Admittance,
    Autotransformer,
    BalancedLine,
    C_LIGHT,
    Cable,
    Driven,
    DrivenCurrent,
    FloatingBalun,
    Load,
    Network,
    NetworkReducer,
    PortOnWire,
    PortOnWireFloating,
    PortVirtual,
    Shunt,
    TouchstoneLoad,
    TouchstoneTwoPort,
    Transformer,
    TwoPort,
)
from momwire.networks._reducer import _BudgetEntry
from momwire.networks._spec import _branch_port_refs

FREQ_MHZ = 28.0
WL = C_LIGHT / (FREQ_MHZ * 1e6)
OMEGA = 2.0 * np.pi * FREQ_MHZ * 1e6


def synth_y(n, seed):
    """Reciprocal, diagonally-dominant complex antenna Y (well conditioned)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    y = 0.004 * (a + a.T) / 2.0
    return y + np.eye(n) * (0.02 + 0.008j)


def reducer(net):
    """Standard port indexing: real ports first in declaration order, then
    virtual — the contract the engines above this module honor."""
    real = [n for n, p in net.ports.items() if isinstance(p, PortOnWire)]
    virt = [n for n, p in net.ports.items() if isinstance(p, PortVirtual)]
    port_to_idx = {n: i for i, n in enumerate(real + virt)}
    return NetworkReducer(net, port_to_idx, len(real) + len(virt))


def nodal_reference(y_full, driven):
    """Bare nodal reduction: driven nodes pinned at their EMF, the rest
    floating (I_ext = 0), currents I = Y·V. Exact for pure admittance stamps."""
    n = y_full.shape[0]
    idx = sorted(driven)
    other = [i for i in range(n) if i not in driven]
    v = np.zeros(n, dtype=np.complex128)
    for k, e in driven.items():
        v[k] = e
    if other:
        rhs = -y_full[np.ix_(other, idx)] @ np.array([driven[k] for k in idx])
        v[other] = np.linalg.solve(y_full[np.ix_(other, other)], rhs)
    return v, y_full @ v


class FixedY:
    """A hand-rolled `AdmittanceData` — the whole contract is ``nports`` and
    ``y_at(f_hz)``. momwire has no Touchstone parser (it stayed in
    antennaknobs), so the protocol IS what a measured-data branch is written
    to, and this stand-in is the only implementation the suite needs."""

    def __init__(self, y):
        self._y = np.asarray(y, dtype=np.complex128)
        self.calls = []

    @property
    def nports(self):
        return self._y.shape[0]

    def y_at(self, f_hz):
        self.calls.append(f_hz)
        return self._y


# ---------------------------------------------------------------------------
# 1. The flat Network's validation
# ---------------------------------------------------------------------------
def test_port_dict_key_must_match_port_name():
    with pytest.raises(ValueError, match="doesn't match Port.name"):
        Network(
            ports={"f": PortOnWire("other")},
            sources=[Driven(port="f")],
        )


def test_branch_referencing_unknown_port_raises():
    with pytest.raises(ValueError, match="references unknown port 'nope'"):
        Network(
            ports={"f": PortOnWire("f")},
            branches=[TL(a="f", b="nope", z0=50.0, length=1.0)],
            sources=[Driven(port="f")],
        )


def test_source_on_unknown_port_raises():
    with pytest.raises(ValueError, match="references unknown port"):
        Network(ports={"f": PortOnWire("f")}, sources=[Driven(port="ghost")])


def test_network_without_sources_raises():
    with pytest.raises(ValueError, match="no driven sources"):
        Network(ports={"f": PortOnWire("f")})


@pytest.mark.parametrize(
    "branch, refs",
    [
        (TL(a="x", b="y", z0=50.0, length=1.0), ("x", "y")),
        (TwoPort(a="x", b="y", r=1.0), ("x", "y")),
        (Transformer(a="x", b="y", n=2.0), ("x", "y")),
        (Autotransformer(a="x", b="y", l_lower=1e-6, l_upper=4e-6), ("x", "y")),
        (Load(port="x", r=1.0), ("x",)),
        (Shunt(port="x", c=1e-12), ("x",)),
        (Admittance(ports=("x", "y"), y=((1, 0), (0, 1))), ("x", "y")),
        (
            BalancedLine(a1="x", a2="y", b1="z", b2="w", zdiff=450.0, length=1.0),
            ("x", "y", "z", "w"),
        ),
        (FloatingBalun(primary="p", a="x", b="y", n=1.0), ("p", "x", "y")),
    ],
    ids=lambda v: type(v).__name__ if hasattr(v, "__dataclass_fields__") else "",
)
def test_branch_port_refs_names_every_terminal(branch, refs):
    """Every terminal, and only the terminals. `FloatingBalun` also carries
    ``.a``/``.b``, so its case must precede the generic fallback — a missed
    reference is a port the validation silently stops checking."""
    assert _branch_port_refs(branch) == refs


def test_floating_gap_is_addressed_by_its_two_terminals():
    """A `PortOnWireFloating` exposes ``.p``/``.n`` and NOT its bare name."""
    with pytest.raises(ValueError, match="references unknown port 'g'"):
        Network(
            ports={"f": PortOnWire("f"), "g": PortOnWireFloating("g")},
            branches=[Shunt(port="g", r=50.0)],
            sources=[Driven(port="f")],
        )
    # …and the dotted terminals are accepted.
    net = Network(
        ports={"f": PortOnWire("f"), "g": PortOnWireFloating("g")},
        branches=[TwoPort(a="g.p", b="g.n", r=50.0)],
        sources=[Driven(port="f")],
    )
    assert len(net.branches) == 1


def test_bonding_the_floating_minus_terminal_recovers_the_ordinary_gap():
    """Oracle for the floating stamp's congruence: a floating gap whose "−"
    terminal is hard-shorted to the datum is EXACTLY an ordinary node-to-datum
    gap port, so the two networks must agree to the last digit."""
    y = synth_y(2, 17)
    floating = Network(
        ports={"f": PortOnWire("f"), "g": PortOnWireFloating("g")},
        branches=[TwoPort(a="g.p", b="g.n", r=120.0), Shunt(port="g.n", l=0.0)],
        sources=[Driven(port="f")],
    )
    plain = Network(
        ports={"f": PortOnWire("f"), "g": PortOnWire("g")},
        branches=[Shunt(port="g", r=120.0)],
        sources=[Driven(port="f")],
    )
    z_float = reducer(floating).driven_impedance(y, WL)[0]
    z_plain = reducer(plain).driven_impedance(y, WL)[0]
    assert z_float == pytest.approx(z_plain, rel=1e-12)


def test_driving_a_floating_gap_is_refused():
    with pytest.raises(ValueError, match="floating gap is an attachment point"):
        Network(
            ports={"g": PortOnWireFloating("g")},
            sources=[Driven(port="g")],
        )


def test_cm_open_balanced_line_terminals_must_have_a_return():
    """``zcomm=None`` stamps only the differential block, so a node reachable
    ONLY through such terminals is common-mode floating — caught here rather
    than as a singular matrix at solve time."""
    ports = {
        "f": PortOnWire("f"),
        "g": PortOnWire("g"),
        "t1": PortVirtual("t1"),
        "t2": PortVirtual("t2"),
    }
    kw = dict(a1="f", a2="g", b1="t1", b2="t2", zdiff=450.0, length=3.0)
    with pytest.raises(ValueError, match=r"\['t1', 't2'\]"):
        Network(ports=ports, branches=[BalancedLine(**kw)], sources=[Driven(port="f")])
    # A zcomm-carrying line IS a common-mode return; the same network passes.
    net = Network(
        ports=dict(ports),
        branches=[BalancedLine(**kw, zcomm=250.0)],
        sources=[Driven(port="f")],
    )
    assert net.branch_paths == [""]


def test_floating_balun_secondary_must_have_a_return():
    """The secondary pair (a, b) is CM-floating by construction; the primary
    is pinned by the constitutive row and never counts as floating."""
    ports = {
        "f": PortOnWire("f"),
        "rig": PortVirtual("rig"),
        "sa": PortVirtual("sa"),
        "sb": PortVirtual("sb"),
    }
    bal = FloatingBalun(primary="rig", a="sa", b="sb", n=1.0)
    with pytest.raises(ValueError, match=r"\['sa', 'sb'\]"):
        Network(ports=ports, branches=[bal], sources=[Driven(port="rig")])
    net = Network(
        ports=dict(ports),
        branches=[bal, Admittance(ports=("sa", "sb"), y=((0.01, 0.0), (0.0, 0.01)))],
        sources=[Driven(port="rig")],
    )
    assert len(net.branches) == 2


# ---------------------------------------------------------------------------
# 2. branch_paths: derived in antennaknobs, PASSABLE here
# ---------------------------------------------------------------------------
def _tuner_net(branch_paths):
    return Network(
        ports={
            "f": PortOnWire("f"),
            "in": PortVirtual("in"),
            "tuner1.m": PortVirtual("tuner1.m"),
        },
        branches=[
            TwoPort(a="in", b="tuner1.m", l=1.2e-6, r=0.8),
            Shunt(port="tuner1.m", r=900.0, c=60e-12),
            TL(a="tuner1.m", b="f", z0=50.0, length=2.0),
        ],
        sources=[Driven(port="in")],
        branch_paths=branch_paths,
    )


def test_empty_branch_paths_normalizes_to_one_blank_per_branch():
    assert _tuner_net([]).branch_paths == ["", "", ""]


def test_branch_paths_length_mismatch_raises():
    with pytest.raises(ValueError, match="2 entries for 3 branches"):
        _tuner_net(["tuner1.", "tuner1."])


def test_branch_paths_prefix_the_power_budget_labels():
    """The label strings are a FROZEN cross-repo contract (antennaknobs#956 —
    the app's schematic layer pattern-matches them), so this pins the exact
    spelling, prefix and instance-namespace stripping included."""
    _v, _eff, _p_in, budget = reducer(
        _tuner_net(["tuner1.", "tuner1.", ""])
    ).excited_state(synth_y(1, 4), WL)
    labels = [label for label, _w in budget]
    assert labels == [
        "tuner1: TwoPort in→m",
        "tuner1: Shunt m",
        "TL tuner1.m→f",
    ]
    # …and without the paths, no prefix and no stripping.
    _v, _eff, _p_in, budget = reducer(_tuner_net([])).excited_state(synth_y(1, 4), WL)
    assert [label for label, _w in budget] == [
        "TwoPort in→tuner1.m",
        "Shunt tuner1.m",
        "TL tuner1.m→f",
    ]


def test_branch_paths_do_not_change_the_answer():
    y = synth_y(1, 4)
    z_paths = reducer(_tuner_net(["tuner1.", "tuner1.", ""])).driven_impedance(y, WL)[0]
    z_bare = reducer(_tuner_net([])).driven_impedance(y, WL)[0]
    assert z_paths == pytest.approx(z_bare, rel=1e-12)


# ---------------------------------------------------------------------------
# 2b. Structured probe identity (antennaknobs#956): every budget entry also
# carries `.key` = (instance_path, branch_index, subprobe) — independent of
# the (still frozen) label spelling, and unique across a budget.
# ---------------------------------------------------------------------------


def _pickled(obj):
    return pickle.loads(pickle.dumps(obj))


def _nested_autotransformer_net():
    """A 2-level-nested composite (`outer.` / `outer.auto.`) whose middle
    branch is an `Autotransformer`, which emits TWO probes ("lower"/"upper")
    off one branch — the case the issue calls out as needing distinct
    `subprobe` tokens within a single branch index."""
    return Network(
        ports={
            "f": PortOnWire("f"),
            "in": PortVirtual("in"),
            "tap": PortVirtual("tap"),
            "top": PortVirtual("top"),
        },
        branches=[
            TwoPort(a="in", b="tap", r=1.0),
            Autotransformer(a="tap", b="top", l_lower=1e-6, l_upper=4e-6, ql=50.0),
            TL(a="top", b="f", z0=50.0, length=1.0),
        ],
        sources=[Driven(port="f")],
        branch_paths=["outer.", "outer.auto.", ""],
    )


def test_probe_keys_are_unique_in_a_nested_composite_with_an_autotransformer():
    """Gate 3: uniqueness across a budget, on the network shape the issue
    names — nested instance paths plus an autotransformer's two sub-probes
    sharing one branch."""
    _v, _eff, _p_in, budget = reducer(_nested_autotransformer_net()).excited_state(
        synth_y(1, 4), WL
    )
    keys = [entry.key for entry in budget]
    assert len(keys) == len(set(keys)), f"duplicate probe keys: {keys}"

    # instance_path is the RAW `branch_paths` entry, not the "{p[:-1]}: "
    # label-display form (the design explicitly calls this out).
    paths = [k[0] for k in keys]
    assert "outer." in paths and "outer.auto." in paths
    assert not any(p.endswith(": ") for p in paths if p)

    # The autotransformer's two probes: same (instance_path, branch_index),
    # distinct subprobe tokens.
    auto_keys = [k for k in keys if k[0] == "outer.auto."]
    assert len(auto_keys) == 2
    assert auto_keys[0][:2] == auto_keys[1][:2]
    assert {k[2] for k in auto_keys} == {"lower", "upper"}


def _shared_loads_net():
    """Two co-located `Load`s (a resistor and a pure inductor, both on port
    "f") sharing one termination branch, under a composite's instance path."""
    return Network(
        ports={
            "f": PortOnWire("f"),
            "in": PortVirtual("in"),
            "tuner1.m": PortVirtual("tuner1.m"),
        },
        branches=[
            TwoPort(a="in", b="tuner1.m", l=1.2e-6, r=0.8),
            Shunt(port="tuner1.m", r=900.0, c=60e-12),
            Load(port="f", r=5.0),
            Load(port="f", l=1e-7),
            TL(a="tuner1.m", b="f", z0=50.0, length=2.0),
        ],
        sources=[Driven(port="in")],
        branch_paths=["tuner1.", "tuner1.", "", "", ""],
    )


def test_probe_keys_are_unique_across_a_composite_with_shared_loads():
    """Gate 3 (continued): co-located `Load`s each get their OWN budget row
    and key, and instance-path branches at multiple nesting depths must still
    all resolve to distinct keys."""
    _v, _eff, _p_in, budget = reducer(_shared_loads_net()).excited_state(
        synth_y(1, 4), WL
    )
    keys = [entry.key for entry in budget]
    assert len(keys) == len(set(keys)), f"duplicate probe keys: {keys}"
    # ONE row per contributing Load, each keyed on its own branch — branch
    # indices 2 and 3 in `_shared_loads_net`'s list. Folding them into a
    # single row (as the first cut of #956 did) makes the second Load's
    # identity unrepresentable, which is the thing the key exists to fix.
    load_rows = [e for e in budget if e.key[2] == "load"]
    assert [e.key for e in load_rows] == [("", 2, "load"), ("", 3, "load")]


def test_co_located_loads_get_their_own_watts_not_a_shared_row():
    """The per-Load rows carry per-Load PHYSICS, not a split label on one
    number: series loads share a current, so each burns ½·Re(z_i)·|j|². The
    resistor burns everything and the pure inductor burns nothing — an
    attribution no folded "Load f+f" row could ever express — and the two
    shares add up to exactly what the whole chain dissipates."""
    net = _shared_loads_net()
    red = reducer(net)
    _v, _eff, _p_in, budget = red.excited_state(synth_y(1, 4), WL)
    r_row, l_row = [e for e in budget if e.key[2] == "load"]

    assert r_row[0] == "Load f" and l_row[0] == "Load f"
    assert r_row[1] > 0.0  # the 5 Ω resistor is the only lossy part
    assert l_row[1] == 0.0  # jωL is pure reactance: no watts, exactly

    # The shares partition the chain: the historical whole-branch
    # "termination" probe (still `branch_power`'s documented kind) reads the
    # same total the two rows sum to.
    system = red.apply_branches(synth_y(1, 4), WL)
    node = red.port_to_idx["f"]
    whole = system.branch_power(("Load f+f", "termination", node))
    assert r_row[1] + l_row[1] == pytest.approx(whole, rel=1e-9)


def test_co_located_loads_split_under_a_forced_current_source_too():
    """The per-Load rows are emitted ONCE, above the forced-current /
    voltage split, because both terminations probe the same Loads at the
    same shared branch current. A `DrivenCurrent` port is the other half of
    that block: same rows, same keys, and the shares still add up to the
    chain dissipation ½·Re(Σz)·|I|² at the forced amps."""
    net = Network(
        ports={"f": PortOnWire("f")},
        branches=[Load(port="f", r=7.0), Load(port="f", r=3.0, l=2e-7)],
        sources=[DrivenCurrent(port="f", current=0.4 + 0j)],
    )
    red = reducer(net)
    _v, _eff, _p_in, budget = red.excited_state(synth_y(1, 11), WL)
    load_rows = [e for e in budget if e.key[2] == "load"]
    assert [e.key for e in load_rows] == [("", 0, "load"), ("", 1, "load")]
    # Forced current: each share is exactly ½·R_i·|I|², independent of the
    # antenna Y — the property that makes a current source worth having.
    i2 = abs(0.4) ** 2
    assert load_rows[0][1] == pytest.approx(0.5 * 7.0 * i2, rel=1e-12)
    assert load_rows[1][1] == pytest.approx(0.5 * 3.0 * i2, rel=1e-12)


def test_a_trap_at_resonance_gets_a_keyed_row_worth_zero_watts():
    """A parallel-LC Load at ω₀ has Z = ∞ — the intended open circuit. The
    per-Load share is Re(z)·|j|², so the naive form would read inf·0 = nan
    and poison the whole budget; the row must read exactly 0 W, the watts an
    open burns, and still carry its key."""
    l = 1.0e-6
    c = 1.0 / (OMEGA**2 * l)  # resonate the tank exactly at FREQ_MHZ
    net = Network(
        ports={"f": PortOnWire("f"), "arm": PortOnWire("arm")},
        branches=[Load(port="arm", l=l, c=c, parallel=True)],
        sources=[Driven(port="f")],
    )
    _v, eff, _p_in, budget = reducer(net).excited_state(synth_y(2, 5), WL)
    (row,) = [e for e in budget if e.key[2] == "load"]
    assert row[1] == 0.0  # not nan, not inf
    assert row.key == ("", 0, "load")
    assert eff == pytest.approx(1.0)


def test_budget_key_is_independent_of_label_spelling():
    """The decisive gate (antennaknobs#956): the label strings the reducer
    emits are still frozen (byte-identical), but `.key` is not DERIVED from
    them — the probe carries it as its own channel, not as something parsed
    back out of its label text. Prove it with a local re-render — take the
    reducer's own probes (the real ones `apply_branches` built) and re-render
    an entirely different display string per probe from the same (kind,
    payload) data. If the key were secretly built from the label text,
    mangling the label would perturb it; it does not, and the watts values
    (kind/payload-derived) do not move either."""
    net = _nested_autotransformer_net()
    system = reducer(net).apply_branches(synth_y(1, 4), WL)

    def render(labeller):
        return [
            _BudgetEntry(labeller(p[0], i), system.branch_power(p), p.key)
            for i, p in enumerate(system.probes)
        ]

    original = render(lambda label, i: label)
    # A future momwire-side label wording change, simulated: nothing about
    # this touches kind/payload/key, only the display string.
    reworded = render(lambda label, i: f"renamed probe #{i}!!")

    orig_labels = [e[0] for e in original]
    new_labels = [e[0] for e in reworded]
    assert orig_labels != new_labels  # the label genuinely changed
    assert new_labels == [f"renamed probe #{i}!!" for i in range(len(new_labels))]

    assert [e.key for e in original] == [e.key for e in reworded]
    assert [e[1] for e in original] == [e[1] for e in reworded]  # watts untouched
    keys = [e.key for e in reworded]
    assert len(keys) == len(set(keys))  # still unique after the rewording


def test_budget_key_survives_a_label_change_made_at_the_SOURCE():
    """The same claim as above, but proved where it has to hold.

    The test above re-renders labels in the test body and hands the SAME key
    object to both renders, so it would still pass if `apply_branches` built
    its keys by parsing label text. This one changes the label the reducer
    itself emits: port names are what `_short(br.a)`/`_short(br.b)` splice
    into every label, so a topologically IDENTICAL network with renamed
    ports emits different label strings from the same branch list, in the
    same order, under the same `branch_paths`. The keys are structural —
    (instance_path, branch_index, subprobe) — so they must not move.
    """
    twin = Network(
        ports={
            "fx": PortOnWire("fx"),
            "inx": PortVirtual("inx"),
            "tapx": PortVirtual("tapx"),
            "topx": PortVirtual("topx"),
        },
        branches=[
            TwoPort(a="inx", b="tapx", r=1.0),
            Autotransformer(a="tapx", b="topx", l_lower=1e-6, l_upper=4e-6, ql=50.0),
            TL(a="topx", b="fx", z0=50.0, length=1.0),
        ],
        sources=[Driven(port="fx")],
        branch_paths=["outer.", "outer.auto.", ""],
    )
    _v, _e, _p, base = reducer(_nested_autotransformer_net()).excited_state(
        synth_y(1, 4), WL
    )
    _v2, _e2, _p2, renamed = reducer(twin).excited_state(synth_y(1, 4), WL)

    # The reducer really did emit different labels...
    assert [e[0] for e in base] != [e[0] for e in renamed]
    # ...and the structural identity did not move.
    assert [e.key for e in base] == [e.key for e in renamed]


def test_budget_entries_unpack_as_two_tuples_backward_compat():
    """Gate 4: `for label, watts in budget` — the exact idiom `cli.py` and
    `schematic.py` (antennaknobs) and `_reducer.py` itself use — must keep
    working unchanged; `.key` rides along as an attribute, not a 3rd slot."""
    _v, _eff, _p_in, budget = reducer(
        _tuner_net(["tuner1.", "tuner1.", ""])
    ).excited_state(synth_y(1, 4), WL)
    assert budget  # nonempty, or the loop below proves nothing
    total = 0.0
    unpacked = 0
    for label, watts in budget:  # must not raise ValueError: too many values
        assert isinstance(label, str)
        assert isinstance(watts, float)
        total += watts
        unpacked += 1
    assert unpacked == len(budget)
    assert total == pytest.approx(sum(entry[1] for entry in budget))
    for entry in budget:
        assert len(entry) == 2
        assert isinstance(entry.key, tuple) and len(entry.key) == 3


@pytest.mark.parametrize("clone", [_pickled, copy.copy, copy.deepcopy])
def test_budget_rows_and_probes_survive_being_copied(clone):
    """A tuple subclass with a multi-argument `__new__` is NOT copyable by
    default: pickle and both copies reconstruct through `__new__` using the
    arguments `__getnewargs__` reports, and tuple's inherited version reports
    ONE (the plain tuple), so the 3-/4-argument `__new__` here raises
    TypeError. That bites the moment a budget lands in anything that
    round-trips — a multiprocessing sweep, a result cache, `copy.deepcopy` of
    a solve result. Both classes spell their own `__getnewargs__`, which
    fixes the crash AND carries `.key` across the copy."""
    _v, _eff, _p_in, budget = reducer(_shared_loads_net()).excited_state(
        synth_y(1, 4), WL
    )
    system = reducer(_shared_loads_net()).apply_branches(synth_y(1, 4), WL)

    for row in budget:
        again = clone(row)
        assert type(again) is type(row)
        assert tuple(again) == tuple(row)
        assert again.key == row.key

    for probe in system.probes:
        again = clone(probe)
        assert type(again) is type(probe)
        assert again[0] == probe[0] and again[1] == probe[1]
        assert again.key == probe.key

    # The whole list at once — the shape a cache or a worker boundary
    # actually moves.
    assert [e.key for e in clone(budget)] == [e.key for e in budget]


# ---------------------------------------------------------------------------
# 3. `Cable` as the formal line spec
# ---------------------------------------------------------------------------
def test_from_cable_takes_the_spec_and_fills_every_coefficient():
    cable = Cable(z0=50.0, vf=0.80, k1=0.27, k2=0.0055)
    tl = TL.from_cable(cable, "rig", "feed", 30.48, transposed=True)
    assert (tl.a, tl.b, tl.length, tl.transposed) == ("rig", "feed", 30.48, True)
    assert (tl.z0, tl.vf, tl.k1, tl.k2) == (cable.z0, cable.vf, cable.k1, cable.k2)


def test_from_cable_refuses_a_catalog_NAME_and_says_where_the_catalog_is():
    """momwire ships no cable table; name → spec resolution is the caller's."""
    with pytest.raises(TypeError) as exc:
        TL.from_cable("RG-8X", "rig", "feed", 30.48)
    msg = str(exc.value)
    assert "'RG-8X'" in msg and "cable_from_catalog" in msg


def test_from_cable_line_solves_the_same_as_a_hand_built_one():
    cable = Cable(z0=75.0, vf=0.66, k1=0.18, k2=0.003)
    ports = {"f": PortOnWire("f"), "rig": PortVirtual("rig")}
    y = synth_y(1, 21)

    def z_of(branch):
        net = Network(ports=dict(ports), branches=[branch], sources=[Driven("rig")])
        return reducer(net).driven_impedance(y, WL)[0]

    hand = TL("rig", "f", z0=75.0, length=12.0, vf=0.66, k1=0.18, k2=0.003)
    assert z_of(TL.from_cable(cable, "rig", "f", 12.0)) == pytest.approx(
        z_of(hand), rel=1e-12
    )


# ---------------------------------------------------------------------------
# 4. Per-branch stamp oracles
# ---------------------------------------------------------------------------
def test_admittance_1port_is_a_fixed_shunt_to_common():
    y = synth_y(1, 3)
    ysh = 0.003 - 0.002j
    net = Network(
        ports={"a": PortOnWire("a")},
        branches=[Admittance(ports=("a",), y=((ysh,),))],
        sources=[Driven(port="a")],
    )
    z = reducer(net).driven_impedance(y, WL)[0]
    assert z == pytest.approx(1.0 / (y[0, 0] + ysh), rel=1e-12)


def test_admittance_2port_matches_the_augmented_nodal_solve():
    y = synth_y(2, 11)
    yadm = np.array([[0.02 + 0.01j, -0.01 + 0.002j], [-0.01 + 0.002j, 0.015 - 0.004j]])
    net = Network(
        ports={"a": PortOnWire("a"), "b": PortOnWire("b")},
        branches=[Admittance(ports=("a", "b"), y=tuple(map(tuple, yadm)))],
        sources=[Driven(port="a")],
    )
    z = reducer(net).driven_impedance(y, WL)[0]
    v, i = nodal_reference(y + yadm, {0: 1 + 0j})
    assert z == pytest.approx(v[0] / i[0], rel=1e-9)


def test_admittance_is_stamped_verbatim_at_every_frequency():
    """No ω scaling — that is the whole difference from `Shunt`."""
    y = synth_y(1, 7)
    ysh = 1j * 0.004
    net = Network(
        ports={"a": PortOnWire("a")},
        branches=[Admittance(ports=("a",), y=((ysh,),))],
        sources=[Driven(port="a")],
    )
    red = reducer(net)
    assert red.driven_impedance(y, C_LIGHT / 10e6)[0] == pytest.approx(
        red.driven_impedance(y, C_LIGHT / 30e6)[0], rel=1e-12
    )


def test_admittance_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="must be 2×2"):
        Admittance(ports=("a", "b"), y=((0.01,),))


def test_touchstone_load_stamps_the_protocol_admittance_at_the_solve_frequency():
    y = synth_y(1, 3)
    data = FixedY([[0.01 + 0j]])
    net = Network(
        ports={"a": PortOnWire("a")},
        branches=[TouchstoneLoad("a", data)],
        sources=[Driven(port="a")],
    )
    z = reducer(net).driven_impedance(y, WL)[0]
    assert z == pytest.approx(1.0 / (y[0, 0] + 0.01), rel=1e-9)
    assert data.calls and all(f == pytest.approx(FREQ_MHZ * 1e6) for f in data.calls)


def test_touchstone_twoport_matches_the_augmented_nodal_solve():
    y = synth_y(2, 11)
    yblock = np.array([[0.012 + 0.004j, -0.009 + 0.001j], [-0.009 + 0.001j, 0.02 + 0j]])
    net = Network(
        ports={"a": PortOnWire("a"), "b": PortOnWire("b")},
        branches=[TouchstoneTwoPort("a", "b", FixedY(yblock))],
        sources=[Driven(port="a")],
    )
    z = reducer(net).driven_impedance(y, WL)[0]
    v, i = nodal_reference(y + yblock, {0: 1 + 0j})
    assert z == pytest.approx(v[0] / i[0], rel=1e-9)


def test_measured_branches_validate_the_protocol_port_count():
    with pytest.raises(ValueError, match="1-port"):
        TouchstoneLoad("a", FixedY(np.eye(2) * 0.01))
    with pytest.raises(ValueError, match="2-port"):
        TouchstoneTwoPort("a", "b", FixedY([[0.01 + 0j]]))


def _xfmr_zin(branch, z_l, wl=WL):
    net = Network(
        ports={"ant": PortOnWire("ant"), "rig": PortVirtual("rig")},
        branches=[branch],
        sources=[Driven("rig", 1 + 0j)],
    )
    red = NetworkReducer(net, {"ant": 0, "rig": 1}, 2)
    (z,) = red.driven_impedance(np.array([[1.0 / z_l]], dtype=np.complex128), wl)
    return z


@pytest.mark.parametrize("n", [2.0, 1.0, 0.5, 3.5])
def test_transformer_is_exactly_an_n_squared_impedance_transform(n):
    z_l = 300.0 - 40.0j
    assert _xfmr_zin(Transformer(a="rig", b="ant", n=n), z_l) == pytest.approx(
        n * n * z_l, rel=1e-12
    )


def test_transformer_winding_resistance_refers_to_side_a():
    z_l, n, r = 300.0 + 0j, 0.5, 1.2
    assert _xfmr_zin(Transformer(a="rig", b="ant", n=n, r=r), z_l) == pytest.approx(
        n * n * z_l + r, rel=1e-12
    )


def test_transformer_magnetizing_branch_parallels_the_ideal_ratio():
    z_l, n, lmag = 300.0 + 0j, 0.5, 10e-6
    z_ideal, z_mag = n * n * z_l, 1j * OMEGA * lmag
    assert _xfmr_zin(
        Transformer(a="rig", b="ant", n=n, lmag=lmag), z_l
    ) == pytest.approx(z_ideal * z_mag / (z_ideal + z_mag), rel=1e-12)


def test_zero_turns_ratio_is_rejected_at_stamp_time():
    with pytest.raises(ValueError, match="turns ratio n = 0"):
        _xfmr_zin(Transformer(a="rig", b="ant", n=0.0), 50.0)


def test_series_shunt_is_the_series_rlc_admittance_at_the_node():
    y = synth_y(2, 8)
    r, c = 12.0, 47e-12
    net = Network(
        ports={"f": PortOnWire("f"), "s": PortOnWire("s")},
        branches=[Shunt(port="s", r=r, c=c)],
        sources=[Driven(port="f")],
    )
    y_sh = 1.0 / (r + 1.0 / (1j * OMEGA * c))
    v, i = nodal_reference(y + np.diag([0.0, y_sh]).astype(np.complex128), {0: 1 + 0j})
    assert reducer(net).driven_impedance(y, WL)[0] == pytest.approx(
        v[0] / i[0], rel=1e-9
    )


def test_parallel_shunt_is_the_tank_admittance_at_the_node():
    y = synth_y(1, 7)
    r, l, c = 1000.0, 1e-6, 30e-12
    net = Network(
        ports={"f": PortOnWire("f")},
        branches=[Shunt(port="f", r=r, l=l, c=c, parallel=True)],
        sources=[Driven(port="f")],
    )
    y_tank = 1.0 / r + 1.0 / (1j * OMEGA * l) + 1j * OMEGA * c
    assert reducer(net).driven_impedance(y, WL)[0] == pytest.approx(
        1.0 / (y[0, 0] + y_tank), rel=1e-9
    )


@pytest.mark.parametrize("r_load", [0.0, 40.0, 250.0])
def test_driven_current_forces_its_amps_whatever_the_series_load(r_load):
    """A `DrivenCurrent` port carries exactly the forced current — series
    `Load` branches drop voltage inside the source loop without altering it
    (NEC's LD-in-driven-segment physics) — while the reported impedance
    still includes the load chain."""
    y = synth_y(1, 13)
    amps = 0.25 - 0.1j
    branches = [Load(port="f", r=r_load)] if r_load else []
    net = Network(
        ports={"f": PortOnWire("f")},
        branches=branches,
        sources=[DrivenCurrent(port="f", current=amps)],
    )
    red = reducer(net)
    system = red.apply_branches(y, WL, z_ref=50.0)
    _v, j = system.solve()
    col, _val, kind, _z_chain = system.terminations[0]
    assert kind == "i"
    assert j[col] == pytest.approx(amps, rel=1e-12)
    assert red.impedance_from_y(system)[0] == pytest.approx(
        1.0 / y[0, 0] + r_load, rel=1e-9
    )
