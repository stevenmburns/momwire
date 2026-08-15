"""The ``nec2`` dialect front-end.

Normative spec: ``site/src/content/docs/reference/deck-grammar-nec2.md``
("The nec2 deck dialect").  Every refusal message in this module is quoted
from that page's ``#refusals`` tables; the tests cite its anchors, so the page
cannot drift from the code without one going red.

This is where NEC's vocabulary lives and stops.  The framing (comments,
geometry, the ``GE`` terminator, the execute-card state machine) and the
geometry semantics are implemented here; the card handlers marked
``_defer`` are the seam the remaining units of momwire#359 fill — they route
their card to an explicit, recorded no-op rather than to a refusal, because a
card the reference engine accepts must never become a fabricated refusal.
"""

from __future__ import annotations

from types import MappingProxyType

from ._cards import Card, DeckError, parse_card
from ._nec2_geometry import Nec2Structure, build_geometry
from .model import (
    DeckModel,
    DeckWire,
    ExecuteGroup,
    FarFieldRequest,
    NearFieldRequest,
    PrintControl,
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

# NX ends the deck; EN ends it and additionally ends the run.
_TERMINATORS = frozenset({"NX", "EN"})

# Cards whose *semantics* belong to a later unit of momwire#359.  They are
# framed, armed and recorded here so the execution state machine is complete;
# what they mean is filled in by units C (excitation, frequency, ground,
# loading, insulation, kernel) and D (build_solver).
_DEFERRED_TO_LATER_UNITS = frozenset({"GN", "GD", "FR", "EX", "LD", "IS", "EK"})

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

        self.groups: list[ExecuteGroup | None] = []
        self.deferred: list[str] = []

        # Execution state.  `armed` starts True because the first execute
        # card of a deck always runs.
        self._armed = True
        self._executed = 0
        self._fresh_fr = False
        self._print_control: PrintControl | None = None
        self._multiprocessing: tuple[int, int] | None = None
        self._saw_ex = False

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

    # -- cards whose semantics a later unit owns ---------------------------

    def _defer(self, card: Card) -> None:
        if card.mnemonic not in self.deferred:
            self.deferred.append(card.mnemonic)
        if card.mnemonic == "EX":
            # Presence, not semantics: the structural refusal below needs to
            # know whether anything drives the structure at all.
            self._saw_ex = True
        if card.mnemonic == "FR":
            # Also structural: an execute card that follows a fresh FR runs
            # the whole frequency list and rebuilds the operator, and an
            # execute card with no new FR re-runs at the last frequency only
            # (spec ``#frequency-groups``).  Which frequencies those are is
            # unit C's; that there was a new FR is the group's shape.
            self._fresh_fr = True

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
        # The remaining refusal — NE/NH over a FINITE ground, whose near field
        # is not an image — needs the GN card's meaning, so it lands with the
        # ground in unit C.
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
        self.groups.append(
            ExecuteGroup(
                request=request,
                print_control=self._print_control,
                multiprocessing=self._multiprocessing,
                refilled=self._fresh_fr or not self._executed,
            )
        )
        self._executed += 1
        self._fresh_fr = False
        self._armed = False

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
        if card.mnemonic in _DEFERRED_TO_LATER_UNITS:
            self._defer(card)
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

    def model(self) -> DeckModel:
        if not self._saw_ex:
            raise DeckError("deck has no EX card — nothing drives the structure")
        structure = self.structure
        wires = tuple(
            DeckWire(
                vertices=(piece.p1, piece.p2),
                radius=structure.wires[piece.wire].radius,
                edge_elements=(piece.n_seg,),
            )
            for piece in structure.pieces
        )
        return DeckModel(
            wires=wires,
            ground_plane_flag=structure.ground_plane_flag,
            groups=tuple(self.groups),
            comments=tuple(self.comments),
            quiet=self.quiet,
            reduced_field=self.reduced_field,
            deferred=tuple(self.deferred),
        )


def parse_nec2(text: str) -> DeckModel:
    """Parse one ``nec2`` deck body into a :class:`DeckModel`."""
    parser = _Nec2Parser()
    parser.feed(text)
    return parser.model()
