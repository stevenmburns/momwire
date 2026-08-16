"""The ``nec2`` dialect front-end.

Normative spec: ``site/src/content/docs/reference/deck-grammar-nec2.md``
("The nec2 deck dialect").  Every refusal message in this module is quoted
from that page's ``#refusals`` tables; the tests cite its anchors, so the page
cannot drift from the code without one going red.

This is where NEC's vocabulary lives and stops: the framing (comments,
geometry, the ``GE`` terminator), the execute-card state machine, the
environment (``GN``/``GD``), excitation and frequency (``EX``/``FR``),
loading and insulation (``LD``/``IS``) and the extended kernel (``EK``) are
all implemented here.  Nothing downstream of this module speaks NEC: a model
is handed to :func:`~momwire.deck.build_solver`, which knows about wires,
arclengths and ports and has never heard of a tag.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ._cards import Card, DeckError, parse_card
from ._nec2_geometry import Nec2Structure, build_geometry
from .model import (
    DeckModel,
    DeckWire,
    Environment,
    ExecuteGroup,
    FarFieldRequest,
    LoadSpec,
    NearFieldRequest,
    PrintControl,
    SecondMedium,
    WireMaterial,
)

__all__ = ["parse_nec2"]


# The geometry section.  GA/GH/GX/GR are NEC geometry too, and are refused by
# name below: the corpus never uses them and an untested geometry path is
# worse than an honest no.
_GEOMETRY_CARDS = frozenset({"GW", "GM", "GS", "GE"})

# Cards that RUN the pending group.  RP/NE/NH are not merely report requests:
# the engine executes on reading them and prints their table afterwards.
_EXECUTE_CARDS = frozenset({"XQ", "RP", "NE", "NH"})

# Cards that make the next execute card a real run rather than a no-op — the
# cards that move the operator or the drive (spec ``#arming``).  GD, MP and PT
# are deliberately absent: GD moves nothing outside the far field's cliff
# modes, MP is advisory, and PT changes what a run prints, not what it
# computes.  GN's membership is oracle-verified.
_ARMING_CARDS = frozenset({"EX", "FR", "LD", "GN", "EK"})

# The arming cards that move the OPERATOR — the matrix itself, not the drive
# it is solved against.  One of these between two execute cards refills
# without a fresh frequency list, which is `ExecuteGroup.refilled_partial`
# (spec ``#arming``).
#
# Measured on the oracle (probes ``XQ / <card> / XQ``, 2026-08-15): a ``GN``
# reprints the LOADING / ENVIRONMENT / MATRIX TIMING preamble with no
# FREQUENCY block and the impedance moves with the new half-space; an ``EK``
# prints the same shape; an ``EX`` prints no preamble at all, because moving
# the drive does not move the matrix.  ``LD`` is deliberately ABSENT from
# this set and present in `_ARMING_CARDS`: an ``LD`` between two execute
# cards makes the LOADING TABLE itself per group, which is a second change
# with its own fixture, and the flag alone would announce a table that had
# not moved.
#
# The test is the CARD, not the value it carries: NEC rebuilds because a
# ground or kernel card arrived, so ``EK`` at the kernel already in force
# refills exactly as a change does.
_OPERATOR_CARDS = frozenset({"GN", "EK"})

# NX ends the deck; EN ends it and additionally ends the run.
_TERMINATORS = frozenset({"NX", "EN"})

# The nec2 dialect's own default frequency, MHz — oracle-verified (probe
# ``no_fr_probe.nec``, momwire#359 unit C): nec2c's FMHZ starts at CVEL
# (299.8, the speed of light in its own units — a 1 m wavelength), and a deck
# that never sends an FR card runs at that frequency.  The spec is silent on
# this because every corpus deck sends one; the probe is what fills the gap.
_DEFAULT_FREQUENCY_MHZ = 299.8

# LD types 0, 1 and 4 expand into one port per segment; the range width
# beyond which that expansion is refused rather than silently truncated.
_LD_EXPAND_MAX = 8

_REFUSED_BY_NAME = MappingProxyType(
    {
        "TL": (
            "TL (transmission line) is not part of this engine's nec2 dialect, "
            "which is antenna-only; antennaknobs imports decks with networks"
        ),
        "NT": (
            "NT (two-port network) is not part of this engine's nec2 dialect, "
            "which is antenna-only; antennaknobs imports decks with networks"
        ),
        "GA": (
            "GA (wire arc) is not part of this engine's nec2 dialect, whose "
            "geometry is GW with GM / GS transforms"
        ),
        "GH": (
            "GH (helix) is not part of this engine's nec2 dialect, whose "
            "geometry is GW with GM / GS transforms"
        ),
        "GX": (
            "GX (structure reflection) is not part of this engine's nec2 "
            "dialect, whose geometry is GW with GM / GS transforms"
        ),
        "GR": (
            "GR (cylindrical structure rotation) is not part of this engine's "
            "nec2 dialect, whose geometry is GW with GM / GS transforms"
        ),
        "GC": (
            "GC (tapered wire continuation) is not part of this engine's nec2 dialect"
        ),
        "GF": (
            "GF (numerical Green's function) is not part of this engine's nec2 dialect"
        ),
        "SY": "SY (4nec2 symbolic variables) is not part of this engine's nec2 dialect",
        "SP": "SP (surface patch) is not supported by this engine yet",
        "SM": "SM (multiple-patch surface) is not supported by this engine yet",
        "KH": "KH (interaction approximation limit) is not supported by this engine",
        "PQ": "PQ (charge print control) is not supported by this engine",
        "CP": "CP (coupling request) is not supported by this engine",
        "PL": "PL (plot request) is not supported by this engine",
        "WG": "WG (NGF write request) is not supported by this engine",
        "ZO": "ZO (impedance normalisation) is not supported by this engine",
    }
)

# Far-field modes this engine computes.  1 is the surface wave (nothing here
# computes one); 4-6 fold a radial-wire screen into the reflection
# coefficient, and momwire has no screen model at all — running the deck as
# bare ground would be a wrong answer rather than a refusal.
_RP_MODES = frozenset({0, 2, 3})


def _directive(text: str, keyword: str) -> int | None:
    """``QQ n`` / ``FF n`` out of a comment body."""
    parts = text.split()
    for i, token in enumerate(parts):
        if token.upper() == keyword and i + 1 < len(parts):
            try:
                return int(float(parts[i + 1]))
            except ValueError:
                return None
    return None


@dataclass
class _PendingGroup:
    """One execute card's state, captured before the deck's feed union set is
    known.

    ``sources`` is in NEC address terms (``tag``, ``seg``, ``volts``) rather
    than resolved model terms, because the union port set — and therefore
    every group's ``ExecuteGroup.voltages`` vector — cannot be built until
    every group in the deck has been read (spec
    ``#one-geometry-one-port-set``).  :meth:`_Nec2Parser.model` is the second
    pass that turns these into real :class:`ExecuteGroup` objects.
    """

    sources: tuple[tuple[int, int, complex], ...]
    frequencies: tuple[float, ...]
    request: FarFieldRequest | NearFieldRequest | None
    print_control: PrintControl | None
    extended_kernel: bool
    multiprocessing: tuple[int, int] | None
    environment: Environment
    refilled: bool
    refilled_partial: bool


class _Nec2Parser:
    """The deck's state machine.

    Cards accumulate state; execute cards run the structure with whatever is
    in force and produce a group's worth of output (spec ``#execution``).
    """

    def __init__(self) -> None:
        self.comments: list[str] = []
        self.quiet = False
        self.reduced_field: int | None = None

        self._geometry_cards: list[Card] = []
        self._structure: Nec2Structure | None = None

        self.groups: list[_PendingGroup | None] = []

        # Execution state.  `armed` starts True because the first execute
        # card of a deck always runs.
        self._armed = True
        self._executed = 0
        self._fresh_fr = False
        self._print_control: PrintControl | None = None
        self._multiprocessing: tuple[int, int] | None = None
        self._saw_ex = False
        # An operator card since the last execute card (see
        # `_OPERATOR_CARDS`): the operator rebuilds, the frequency list does
        # not.  Never set before the FIRST execute card, which refills whole.
        self._operator_dirty = False

        # Excitation retention (spec ``#excitation-retention``): the first EX
        # after an execution REPLACES the list, every further one before the
        # next execution ADDS to it, and a source list is cleared lazily (at
        # the next EX, not at the execute card) so a re-run with no new EX
        # re-drives the previous set unchanged.
        self._sources: list[tuple[int, int, complex]] = []
        self._sources_stale = False

        # FR (spec ``#fr--frequency``): every FR is read, and a later one
        # replaces the list entirely; the default is nec2c's own (see
        # `_DEFAULT_FREQUENCY_MHZ`).
        self._frequencies: tuple[float, ...] = (_DEFAULT_FREQUENCY_MHZ,)

        # GN / GD (spec ``#gn--ground-parameters`` / ``#gd--additional-
        # ground-medium``).  None is free space; the ground plane is always
        # at z = 0 (no card moves it).
        self._ground: None | str | tuple[str, float, float] = None
        self._second_medium: SecondMedium | None = None

        # EK (spec ``#ek--extended-thin-wire-kernel``): I1 == -1 is the only
        # thing that turns it off.  The card's effect on the refill lives in
        # `_operator_dirty`, which every operator card sets.
        self._extended_kernel = False

        # LD / IS (spec ``#ld--loading`` / ``#is--insulated-sheath``).
        # `_loads` are the per-segment port loads (types 0, 1, 4); `_loaded`
        # is their (model wire, element) dedup set, so a second load on one
        # segment refuses rather than merges.  `LD -1` clears both, and also
        # the LD-family conductivity state (type 5 is read under the LD
        # mnemonic too) — but never `_wire_insulation`, which is IS's own
        # card and a wire property, not a load (spec: "and *only* loads").
        self._loads: list[tuple[int, float, LoadSpec]] = []
        self._loaded: set[tuple[int, int]] = set()
        self._global_conductivity: float | None = None
        self._wire_conductivity: dict[int, float] = {}
        self._wire_insulation: dict[int, tuple[float, float]] = {}

    # -- geometry ----------------------------------------------------------

    @property
    def structure(self) -> Nec2Structure:
        """The deck's geometry, built on first use and after every change.

        Built lazily rather than at ``GE`` so a geometry card's refusal fires
        in card order for a deck whose geometry section is well formed and at
        the first card that needs coordinates for one that is not.
        """
        if self._structure is None:
            self._structure = build_geometry(self._geometry_cards)
        return self._structure

    def _geometry(self, card: Card) -> None:
        self._geometry_cards.append(card)
        self._structure = None
        if card.mnemonic == "GE":
            # GE terminates the section, so build now: a deck that never
            # reaches a data card still gets its geometry refusals.
            _ = self.structure

    # -- GN / GD (spec ``#gn--ground-parameters``, ``#gd--additional-
    # ground-medium``) ------------------------------------------------------

    def _gn(self, card: Card) -> None:
        code = card.i(0)
        nradl = card.i(1)
        # The radial-screen refusal is oracle-verified as scoped to the two
        # reflection-coefficient ground types (probe `gn_pec_radial_probe*
        # .nec`, unit C): nec2c's PEC branch never looks at NRADL at all, so
        # `GN 1 4 ...` runs as plain perfect ground rather than refusing —
        # the screen folds into a *reflection coefficient*, which PEC (an
        # exact image) and free space (no ground) do not have.
        if code in (0, 2) and nradl != 0:
            raise DeckError(
                f"GN {code} with a {nradl}-wire radial ground screen is not "
                f"supported by this engine"
            )
        if code == -1:
            self._ground = None
        elif code == 1:
            self._ground = "pec"
        elif code == 0:
            self._ground = ("finite-fast", card.f(4), card.f(5))
        elif code == 2:
            self._ground = ("finite", card.f(4), card.f(5))
        else:
            raise DeckError(f"GN type {code} is not supported by this engine")
        if nradl == 0:
            # F3-F6 carry a whole second medium in the same four /FPAT/ slots
            # a GD card writes; a GN that reaches here always rewrites them,
            # so a bare `GN 1` clears an earlier cliff.
            self._second_medium = SecondMedium(
                card.f(6), card.f(7), card.f(8), card.f(9)
            )

    def _gd(self, card: Card) -> None:
        self._second_medium = SecondMedium(card.f(4), card.f(5), card.f(6), card.f(7))

    # -- FR / EX (spec ``#fr--frequency``, ``#ex--voltage-source``) --------

    def _fr(self, card: Card) -> None:
        sweep_type = card.i(0)
        n = max(card.i(1), 1)
        start, step = card.f(4), card.f(5)
        if sweep_type == 0:
            self._frequencies = tuple(start + i * step for i in range(n))
        else:
            self._frequencies = tuple(
                start * step**i if step > 0 else start for i in range(n)
            )
        self._fresh_fr = True

    def _ex(self, card: Card) -> None:
        ex_type = card.i(0)
        if ex_type != 0:
            raise DeckError(
                f"EX type {ex_type} is not a voltage source; this engine "
                f"drives EX 0 only"
            )
        self._saw_ex = True
        if self._sources_stale:
            self._sources = []
            self._sources_stale = False
        tag, seg = card.i(1), card.i(2)
        self._sources.append((tag, seg, complex(card.f(4), card.f(5))))

    # -- EK (spec ``#ek--extended-thin-wire-kernel``) -----------------------

    def _ek(self, card: Card) -> None:
        # The test is I1 == -1 and nothing else: EK, EK 0, EK 1, EK 2 and
        # EK -2 all turn the extended kernel on.
        self._extended_kernel = card.i(0) != -1

    # -- LD (spec ``#ld--loading``) ------------------------------------------

    def _load_spec(self, ldtyp: int, card: Card) -> LoadSpec | None:
        """The card's ``LoadSpec``, or ``None`` for a zero-valued no-op."""
        if ldtyp in (0, 1):
            r, l, c = card.f(4), card.f(5), card.f(6)
            if r == 0.0 and l == 0.0 and c == 0.0:
                return None
            return LoadSpec("series" if ldtyp == 0 else "parallel", r=r, l=l, c=c)
        r, x = card.f(4), card.f(5)
        if r == 0.0 and x == 0.0:
            return None
        return LoadSpec("fixed", r=r, x=x)

    def _ld(self, card: Card) -> None:
        ldtyp = card.i(0)
        if ldtyp == -1:
            # Nullify every load read so far — and only loads: type 5's
            # conductivity is read under this same mnemonic, so it clears
            # too, but IS's insulation (a different card) never does.
            self._loads = []
            self._loaded = set()
            self._global_conductivity = None
            self._wire_conductivity = {}
            return
        if ldtyp not in (0, 1, 4, 5):
            # 2/3 (per-metre RLC) and 6/7 (4nec2 extensions) refuse by name,
            # same as any type this engine does not recognise.
            raise DeckError(f"LD type {ldtyp} is not supported by this engine")

        tag, first, last = card.i(1), card.i(2), card.i(3)

        if ldtyp == 5:
            self._ld5(tag, first, last, card.f(4))
            return

        pairs = self.structure.ld_segment_range(tag, first, last)
        if len(pairs) > _LD_EXPAND_MAX:
            raise DeckError(
                f"LD over {len(pairs)} segments is not supported by this "
                f"engine — at most {_LD_EXPAND_MAX} segments expand into "
                f"per-segment loads"
            )
        spec = self._load_spec(ldtyp, card)
        if spec is None:
            return  # a zero-valued load is a no-op, dropped
        for wire, seg in pairs:
            piece_index, elem = self.structure.element_of(wire, seg)
            key = (piece_index, elem)
            if key in self._loaded:
                raise DeckError(
                    "LD on a segment that already carries a load is not "
                    "supported by this engine — a second load on one "
                    "segment is not merged"
                )
            self._loaded.add(key)
            _, arclength = self.structure.resolve_of(wire, seg)
            self._loads.append((piece_index, arclength, spec))

    def _ld5(self, tag: int, first: int, last: int, sigma: float) -> None:
        """``LD 5`` — whole-structure or per-wire conductivity, not a lumped
        element (spec: "a material property")."""
        if tag == 0 and first == 0:
            self._global_conductivity = sigma
            return
        pairs = self.structure.ld_segment_range(tag, first, last)
        by_wire: dict[int, set[int]] = {}
        for wire, seg in pairs:
            by_wire.setdefault(wire, set()).add(seg)
        for wire, segs in by_wire.items():
            if segs != set(range(1, self.structure.wires[wire].n_seg + 1)):
                raise DeckError(
                    "LD 5 conductivity on a partial-wire segment range is "
                    "not supported by this engine — per-wire conductivity "
                    "covers whole wires only"
                )
        for wire in by_wire:
            self._wire_conductivity[wire] = sigma

    # -- IS (spec ``#is--insulated-sheath``) ---------------------------------

    def _is(self, card: Card) -> None:
        if self._executed:
            raise DeckError(
                "IS after an execute request is not supported by this "
                "engine: wire insulation is part of the structure, so it "
                "cannot change between runs"
            )
        tag, first, last = card.i(1), card.i(2), card.i(3)
        eps_r, sigma, radius = card.f(4), card.f(5), card.f(6)
        if sigma != 0.0:
            raise DeckError(
                "IS with a conductive sheath (F2 != 0) is not modelled by "
                "this engine — the insulation jacket is a lossless "
                "dielectric; set the sheath conductivity to 0"
            )
        if radius <= 0.0 or eps_r <= 1.0:
            return  # electrically a vacuum: a no-op
        pairs = self.structure.ld_segment_range(tag, first, last)
        by_wire: dict[int, set[int]] = {}
        for wire, seg in pairs:
            by_wire.setdefault(wire, set()).add(seg)
        for wire, segs in by_wire.items():
            if segs != set(range(1, self.structure.wires[wire].n_seg + 1)):
                raise DeckError(
                    "IS: insulation on a partial-wire segment range — "
                    "per-wire specs cover whole wires only"
                )
        for wire in by_wire:
            if radius <= self.structure.wires[wire].radius:
                raise DeckError(
                    "IS: insulation whose outer radius does not exceed the "
                    "wire's conductor radius"
                )
        for wire in by_wire:
            self._wire_insulation[wire] = (radius, eps_r)

    # -- requests ----------------------------------------------------------

    def _far_field(self, card: Card) -> FarFieldRequest:
        mode = card.i(0)
        if mode not in _RP_MODES:
            raise DeckError(
                f"RP mode {mode} is not supported by this engine "
                f"(modes {', '.join(str(m) for m in sorted(_RP_MODES))} only)"
            )
        return FarFieldRequest(
            mode=mode,
            n_theta=max(card.i(1), 1),
            n_phi=max(card.i(2), 1),
            theta0_deg=card.f(4),
            phi0_deg=card.f(5),
            d_theta_deg=card.f(6),
            d_phi_deg=card.f(7),
            range_m=card.f(8),
        )

    def _near_field(self, card: Card) -> NearFieldRequest:
        if card.i(0) != 0:
            raise DeckError(
                f"{card.mnemonic} coordinate system {card.i(0)} (spherical) is "
                f"not supported by this engine; rectangular (0) only"
            )
        if isinstance(self._ground, tuple):
            # A finite ground ("finite-fast" or "finite"): the near field of
            # a lossy half-space is not an image, and a reflection
            # coefficient is a far-field construction. PEC and free space
            # are fine — PEC's near field IS an image.
            raise DeckError(
                f"{card.mnemonic} over a finite ground is not supported by "
                f"this engine (the near field of a Sommerfeld half-space is "
                f"not an image)"
            )
        return NearFieldRequest(
            magnetic=card.mnemonic == "NH",
            counts=(max(card.i(1), 1), max(card.i(2), 1), max(card.i(3), 1)),
            origin=(card.f(4), card.f(5), card.f(6)),
            step=(card.f(7), card.f(8), card.f(9)),
        )

    def _print_control_card(self, card: Card) -> PrintControl:
        """``PT`` — a persistent toggle on the element-current table.

        ``-1`` suppresses it, ``-2`` restores it, ``0`` with a real range
        prints only those segments, and every other flag is the ordinary full
        report.  An all-zero range means "no restriction" rather than "no
        rows".
        """
        flag = card.i(0)
        if flag == -1:
            return PrintControl(suppressed=True)
        tag, first, last = card.i(1), card.i(2), card.i(3)
        if flag == 0 and (first or last):
            return PrintControl(
                elements=self.structure.wire_of_segment_range(tag, first, last)
            )
        return PrintControl()

    def _multiprocessing_card(self, card: Card) -> tuple[int, int]:
        """``MP`` — advisory, and read strictly: NEC's integer-field reader
        refuses a fractional field, and so does this one."""
        for k in (0, 1):
            if card.f(k) != float(card.i(k)):
                raise DeckError(
                    f"MP field {k + 1} must be an integer, not {card.f(k)!r}"
                )
        return card.i(0), card.i(1)

    # -- execution ---------------------------------------------------------

    def _execute(self, card: Card) -> None:
        if not self._armed and self._executed:
            # An execute card with nothing new since the previous one
            # produces no output at all.
            self.groups.append(None)
            return
        request: FarFieldRequest | NearFieldRequest | None = None
        if card.mnemonic == "RP":
            request = self._far_field(card)
        elif card.mnemonic in ("NE", "NH"):
            request = self._near_field(card)
        refilled = self._fresh_fr or not self._executed
        self.groups.append(
            _PendingGroup(
                sources=tuple(self._sources),
                frequencies=self._frequencies
                if self._fresh_fr
                else self._frequencies[-1:],
                request=request,
                print_control=self._print_control,
                extended_kernel=self._extended_kernel,
                multiprocessing=self._multiprocessing,
                environment=Environment(
                    ground=self._ground,
                    # No card moves the plane; it is always z = 0 (spec
                    # ``#gn--ground-parameters``).
                    ground_z=0.0,
                    second_medium=self._second_medium,
                ),
                refilled=refilled,
                refilled_partial=self._operator_dirty and not refilled,
            )
        )
        self._executed += 1
        self._fresh_fr = False
        self._operator_dirty = False
        self._armed = False
        self._sources_stale = True

    # -- the loop ----------------------------------------------------------

    def feed(self, text: str) -> None:
        for line in text.splitlines():
            card = parse_card(line)
            if card is None:
                continue
            if card.mnemonic in _TERMINATORS:
                # Everything after the terminator is a new deck.
                break
            self.card(card)
        # Always close the geometry section, so a deck that never sent a GE
        # (or a data card) still gets its geometry refusals.
        _ = self.structure

    def card(self, card: Card) -> None:
        if card.mnemonic in ("CM", "CE"):
            text = card.text
            self.comments.append(text)
            if (qq := _directive(text, "QQ")) and qq > 0:
                self.quiet = True
            if (ff := _directive(text, "FF")) is not None:
                self.reduced_field = ff
            return
        if (message := _REFUSED_BY_NAME.get(card.mnemonic)) is not None:
            raise DeckError(message)
        if card.mnemonic in _GEOMETRY_CARDS:
            self._geometry(card)
            return
        if card.mnemonic in _ARMING_CARDS:
            self._armed = True
            if card.mnemonic in _OPERATOR_CARDS and self._executed:
                self._operator_dirty = True
        if card.mnemonic == "GN":
            self._gn(card)
            return
        if card.mnemonic == "GD":
            self._gd(card)
            return
        if card.mnemonic == "FR":
            self._fr(card)
            return
        if card.mnemonic == "EX":
            self._ex(card)
            return
        if card.mnemonic == "LD":
            self._ld(card)
            return
        if card.mnemonic == "IS":
            self._is(card)
            return
        if card.mnemonic == "EK":
            self._ek(card)
            return
        if card.mnemonic == "PT":
            self._print_control = self._print_control_card(card)
            return
        if card.mnemonic == "MP":
            self._multiprocessing = self._multiprocessing_card(card)
            return
        if card.mnemonic in _EXECUTE_CARDS:
            self._execute(card)
            return
        raise DeckError(f"unrecognised NEC card {card.mnemonic!r}")

    # -- the model ---------------------------------------------------------

    def _material_for(self, wire: int) -> WireMaterial | None:
        conductivity = self._wire_conductivity.get(wire, self._global_conductivity)
        insulation = self._wire_insulation.get(wire)
        if conductivity is None and insulation is None:
            return None
        if insulation is not None:
            radius, eps_r = insulation
            return WireMaterial(
                conductivity=conductivity,
                insulation_radius=radius,
                insulation_eps_r=eps_r,
            )
        return WireMaterial(conductivity=conductivity)

    def _feeds_and_groups(
        self, structure: Nec2Structure
    ) -> tuple[tuple[tuple[int, float, complex], ...], tuple[ExecuteGroup | None, ...]]:
        """The union feed set and the final :class:`ExecuteGroup` tuple.

        Every EX segment across every group becomes a port in one union set,
        in discovery order (spec ``#one-geometry-one-port-set``); a group's
        own drive is then a voltage vector over that set, zero at a port it
        did not drive.  This is the second pass over `self.groups`'
        `_PendingGroup` records that resolves NEC addresses into the model's
        own (wire, arclength) terms once every group has been read.
        """
        order: list[tuple[int, int]] = []
        position: dict[tuple[int, int], int] = {}
        resolved: list[tuple[int, float]] = []
        discovery_volts: list[complex] = []
        for pending in self.groups:
            if pending is None:
                continue
            for tag, seg, volts in pending.sources:
                key = (tag, seg)
                if key not in position:
                    position[key] = len(order)
                    order.append(key)
                    resolved.append(structure.resolve(tag, seg))
                    discovery_volts.append(volts)

        feeds = tuple(
            (wire, arclength, discovery_volts[i])
            for i, (wire, arclength) in enumerate(resolved)
        )

        groups: list[ExecuteGroup | None] = []
        for pending in self.groups:
            if pending is None:
                groups.append(None)
                continue
            voltages = [0j] * len(order)
            for tag, seg, volts in pending.sources:
                voltages[position[(tag, seg)]] = volts
            groups.append(
                ExecuteGroup(
                    frequencies=pending.frequencies,
                    voltages=tuple(voltages),
                    request=pending.request,
                    print_control=pending.print_control,
                    extended_kernel=pending.extended_kernel,
                    multiprocessing=pending.multiprocessing,
                    refilled=pending.refilled,
                    refilled_partial=pending.refilled_partial,
                    environment=pending.environment,
                )
            )
        return feeds, tuple(groups)

    def model(self) -> DeckModel:
        if not self._saw_ex:
            raise DeckError("deck has no EX card — nothing drives the structure")
        structure = self.structure
        wires = tuple(
            DeckWire(
                vertices=(piece.p1, piece.p2),
                radius=structure.wires[piece.wire].radius,
                edge_elements=(piece.n_seg,),
                material=self._material_for(piece.wire),
            )
            for piece in structure.pieces
        )
        feeds, groups = self._feeds_and_groups(structure)
        return DeckModel(
            wires=wires,
            feeds=feeds,
            loads=tuple(self._loads),
            ground=self._ground,
            ground_z=0.0,
            second_medium=self._second_medium,
            ground_plane_flag=structure.ground_plane_flag,
            groups=groups,
            comments=tuple(self.comments),
            quiet=self.quiet,
            reduced_field=self.reduced_field,
            deferred=(),
        )


def parse_nec2(text: str) -> DeckModel:
    """Parse one ``nec2`` deck body into a :class:`DeckModel`."""
    parser = _Nec2Parser()
    parser.feed(text)
    return parser.model()
