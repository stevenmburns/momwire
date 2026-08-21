"""The ``nec5`` dialect front-end — the deck language EZNEC Pro+ v7 emits.

Sibling of :mod:`momwire.deck._nec2`, and deliberately not a variant of it:
NEC-5's connection cards address **nodes** (segment boundaries) through a
**favored wire**, which is a different animal from NEC-2's segment
addressing, so this dialect parses into its own model types
(:class:`Nec5Deck` and friends) rather than into :class:`DeckModel`.

Normative sources, both in the antennaknobs repo, both black-box:

``docs/status/2026-08-16-eznec-nec5-dialect-capture.md``
    the interface study (momwire#390/#414) — invocation, the card
    vocabulary, the signed-node encoding, the W7EL oracle triple.  Sections
    cited by name in the comments below: "Signed segment addressing",
    "The oracle triple", "Card vocabulary", "Dialect notes".
``docs/status/2026-08-20-eznec-nec5-scored-matrix.md``
    the scored matrix (momwire#456 ws3) — "Probe family 1" is where every
    ground semantic in this module was *measured*, against the licensed
    oracle, on 2026-08-20.

Courtesy stance, the same one the SimNEC and NEC-5 studies keep: every
dialect fact here comes from captured console I/O, captured printouts and
the published Users Manual.  Nothing in this module was learned from, or
describes, the engine's internals.

Scope is exactly the **fifteen observed mnemonics** — ``GW``, ``GE``,
``GN``, ``GD``, ``EX``, ``LD``, ``TL``, ``NT``, ``FR``, ``PQ``, ``RP``,
``XQ``, ``NE``, ``NH``, ``EN`` — plus the ``CM``/``CE`` comment framing.
(``NH`` is the fifteenth: capture sitting 4 falsified the study-era premise
that EZNEC never emits it — momwire#513, capture 0111.)  Everything
else refuses BY NAME.  This module reads a deck and records what it said; it
solves nothing and renders nothing (those are the seam's other units).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from ._cards import Card, DeckError, parse_card

__all__ = [
    "parse_nec5",
    "Nec5Deck",
    "Nec5Node",
    "Nec5Wire",
    "Nec5FreeSpace",
    "Nec5PerfectGround",
    "Nec5SommerfeldGround",
    "Nec5MininecGround",
    "Nec5Ground",
    "Nec5Source",
    "Nec5Load",
    "Nec5TransmissionLine",
    "Nec5Network",
    "Nec5FarFieldRequest",
    "Nec5NearFieldRequest",
    "Nec5ExecuteRequest",
    "Nec5Request",
]


Point = tuple[float, float, float]


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Nec5Node:
    """One connection endpoint: a **favored wire** and a **node index**.

    The dialect writes an endpoint as a field pair ``(tag, n)`` where ``n``
    is a node, not a segment: ``n = -1`` is node 0 of the wire (its end-1
    boundary) and ``n = +k`` is the far boundary of segment ``k`` (capture
    study, "Signed segment addressing is not limited to ``TL``" and
    "Dialect notes").  :attr:`node` holds the decoded index; :attr:`written`
    gives back the spelling the deck used, which the printout echoes.

    **There is no canonicalization here and there must never be one.**  The
    tag is the favored wire and it carries physics: NEC-5 attaches an
    inserted source or load not *at* a junction but on the wire next to it,
    named by the tag.  W7EL's ``Network Connection Test`` is the committed
    gate (capture study, "The oracle triple"): configs A (``NT 3,-1``) and B
    (``NT 2,3``) name the *same geometric point* through different tags and
    answer 114.47 + j21.096 alike, while config C — one address of each —
    answers 195.34 - j57.458.  A front-end that folds ``2,3`` and ``3,-1``
    into one node returns A's answer for C and is off by 70 %.

    So :class:`Nec5Node` compares by ``(tag, node)`` and by nothing else.
    Two addresses that describe one point through different tags are
    unequal, which is the whole point.
    """

    tag: int
    node: int

    @property
    def written(self) -> int:
        """The field as the deck spells it: ``-1`` for node 0, else ``+k``."""
        return -1 if self.node == 0 else self.node


@dataclass(frozen=True)
class Nec5Wire:
    """One ``GW``: a tag, a segment count, two endpoints and a radius."""

    tag: int
    segment_count: int
    end1: Point
    end2: Point
    radius: float


@dataclass(frozen=True)
class Nec5FreeSpace:
    """``GN -1`` — no ground at all.  Also what ``GD -1`` leaves behind."""


@dataclass(frozen=True)
class Nec5PerfectGround:
    """``GN 1`` — perfect ground, written bare (one field) by EZNEC."""


@dataclass(frozen=True)
class Nec5SommerfeldGround:
    """``GN 0`` / ``GN 2`` — finite ground, Sommerfeld solution.

    :attr:`spelling` records WHICH mnemonic spelling the deck used.  Probe
    family 1 measured ``GN 2`` ≡ ``GN 0`` to every printed digit (it is the
    NEC-4-compatible spelling and EZNEC never emits it), so the two are one
    ground — but a byte-gated printout echoes the deck's own card, so the
    spelling is recorded rather than normalized away.
    """

    eps_r: float
    sigma: float
    mu_r: complex = 1 + 0j
    spelling: int = 0


@dataclass(frozen=True)
class Nec5MininecGround:
    """A bare ``GD`` — EZNEC's "Real / MININEC type" ground.

    NOT NEC-2's ``GD``.  NEC-2's is a second radial-wire/cliff medium whose
    fields are distances and heights; this dialect's ``GD`` carries the same
    media payload as ``GN 0`` on a different mnemonic and asks for a
    different treatment.  No code is shared with the nec2 dialect's ``GD``
    handling, on purpose.

    Measured semantics (scored matrix, "Probe family 1", 2026-08-20): the
    mode is PEC currents plus a second-medium far field under an ordinary
    ``RP 0`` — on contacting and elevated geometry alike; ``Vert1`` answers
    35.571 - j1.4223 under bare ``GD``, identical to ``GN 1`` to every
    digit, while ``GN 0`` on the same geometry answers 47.789 - j0.78525.
    The two print the SAME ``FINITE GROUND.  SOMMERFELD SOLUTION`` banner,
    so the banner must never be read as a mode signal.

    :attr:`ix` is the first integer field.  ``-1`` cancels the ground (the
    parser yields :class:`Nec5FreeSpace` and this record is never built);
    ``0`` is normal, and other values were measured to behave as ``0``
    (``GD 0,7,8,9,…`` ≡ ``GD 0,0,0,0,…``), so the raw value is recorded
    rather than normalized.  :attr:`sigma_sets_im_epsc` flags the measured
    convention that a NEGATIVE ``sigma`` sets Im(εc) directly instead of
    being a conductivity.
    """

    ix: int
    eps_r: float
    sigma: float
    mu_r: complex = 1 + 0j
    sigma_sets_im_epsc: bool = False


Nec5Ground = (
    Nec5FreeSpace | Nec5PerfectGround | Nec5SommerfeldGround | Nec5MininecGround
)


@dataclass(frozen=True)
class Nec5Source:
    """One ``EX``: kind 0 (voltage) or 4 (current), at a node, complex drive.

    Corpus: ``EX 4`` in 47 of 49 captures and ``EX 0`` in 2 — the model's
    source-type setting picks one.  Magnitude is always √2 (a 1 W
    normalization, |I|²/2 = 1) and the phase rides in the complex pair, so a
    phased array is several ``EX`` cards and no network at all.  Multiple
    cards parse here; what a solver does with them is a later unit's
    business.  :attr:`option` is the fourth field, observed 0 throughout.
    """

    kind: int
    at: Nec5Node
    drive: complex
    option: int = 0


@dataclass(frozen=True)
class Nec5Load:
    """One ``LD 4``: a fixed impedance at a node address.

    ``LD 4`` is 67 of 67 loads in the corpus — EZNEC reduces whatever the
    user entered to an impedance at the frequency before writing the deck,
    so no ``LD 0/1/2`` RLC and no ``LD 5`` conductivity has ever been
    emitted.  The card is ``LD 4,tag,node,0,R,X``; :attr:`node_to` is that
    fourth field, observed 0 in every capture (loads are single-point, never
    ranges).  The ``1.E+10`` idiom — pinning a virtual wire's node open — is
    an ordinary load mechanically.
    """

    at: Nec5Node
    impedance: complex
    node_to: int = 0


@dataclass(frozen=True)
class Nec5TransmissionLine:
    """One ``TL``: two node endpoints, a line, and four shunt fields.

    :attr:`z0` keeps its SIGN: a negative characteristic impedance is NEC's
    crossed-line convention (the log-periodic's phase-reversal feeder emits
    ``TL 1,9,2,9,-490.0875,…``).  :attr:`length_m` is metres — EZNEC
    specifies line lengths in degrees in its own UI and resolves them to a
    physical length at each frequency before writing the deck.
    :attr:`shunt` is fields 3-6 as written: a shunt admittance per end, and
    the ``1.E+10`` spelling of a shorted stub.
    """

    end_a: Nec5Node
    end_b: Nec5Node
    z0: float
    length_m: float
    shunt: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def crossed(self) -> bool:
        """NEC's phase-reversal flag: the SIGN of ``Z0``, not its magnitude."""
        return self.z0 < 0.0


@dataclass(frozen=True)
class Nec5Network:
    """One ``NT``: two node endpoints and a reciprocal admittance matrix.

    The six reals are three complex Y entries, in card order Y11, Y12, Y22.
    Both ends frequently land on the virtual wire, making the card a pure
    circuit element; junction loads arrive this way too (W7EL's two parallel
    loads at a junction are ``NT 3,-1,4,1,.01,…`` and ``NT 3,-1,4,2,.005,…``,
    not ``LD`` cards at all), so an ordinary loaded model can still carry an
    ``NT``.
    """

    end_a: Nec5Node
    end_b: Nec5Node
    y11: complex
    y12: complex
    y22: complex


@dataclass(frozen=True)
class Nec5FarFieldRequest:
    """``RP 0`` — the pattern request, in the two ``XNDA`` forms observed.

    ``XNDA`` 1000 is the 2-D slice EZNEC's FF Plot asks for and 1001 the
    3-D form; the two have distinct printout formats, which is the
    renderer's business, not this module's.
    """

    n_theta: int
    n_phi: int
    xnda: int
    theta0_deg: float
    phi0_deg: float
    d_theta_deg: float
    d_phi_deg: float
    range_m: float = 0.0
    mode: int = 0


@dataclass(frozen=True)
class Nec5NearFieldRequest:
    """``NE`` / ``NH`` — near electric or magnetic field, rectangular.

    Both cards are EZNEC's (momwire#513): ``NE`` arrived first (one capture
    at the defaults, ``NE 0,1,1,1,0.,0.,0.,0.,0.,0.``), and capture sitting 4
    falsified the premise that ``NH`` is never emitted — it costs one radio
    button in the Near Field Analysis dialog (capture 0111, antennaknobs
    PR #970), and its field layout is ``NE``'s.  All ten fields are
    recorded; :attr:`magnetic` is which card wrote them.
    """

    coordinates: int
    counts: tuple[int, int, int]
    origin: Point
    step: Point
    magnetic: bool = False


@dataclass(frozen=True)
class Nec5ExecuteRequest:
    """``XQ`` — execute, no pattern requested.  ``XQ 0`` in every capture."""

    flag: int = 0


Nec5Request = Nec5FarFieldRequest | Nec5NearFieldRequest | Nec5ExecuteRequest


@dataclass(frozen=True)
class Nec5Deck:
    """One EZNEC-emitted NEC-5 deck, in the dialect's own terms.

    Nothing here is converted into solver objects: a wire is still a tag and
    a segment count, a connection is still a favored wire and a node.  The
    units downstream of this one do the translation, and they need the deck
    as written to do it — the printout renderer echoes :attr:`comments` and
    the deck's cards verbatim, and the physics units need the favored wire
    intact.
    """

    comments: tuple[str, ...] = ()
    ce_text: str = ""
    # The deck exactly as it arrived.  The printout renderer echoes card
    # IMAGES — the fields as written, not as parsed — so it needs the text
    # this model was read from; keeping it here is what lets a served run be
    # `(deck, data)` rather than `(deck, text, data)`.
    source_text: str = ""
    wires: tuple[Nec5Wire, ...] = ()
    ge_flag: int = 0
    ge_second: int = -1
    ground: Nec5Ground = Nec5FreeSpace()
    sources: tuple[Nec5Source, ...] = ()
    loads: tuple[Nec5Load, ...] = ()
    transmission_lines: tuple[Nec5TransmissionLine, ...] = ()
    networks: tuple[Nec5Network, ...] = ()
    frequency_mhz: float | None = None
    requests: tuple[Nec5Request, ...] = ()
    pq: int | None = None

    def wire(self, tag: int) -> Nec5Wire | None:
        """The wire a tag names, or ``None``."""
        for w in self.wires:
            if w.tag == tag:
                return w
        return None


# --------------------------------------------------------------------------
# The vocabulary
# --------------------------------------------------------------------------

# The fifteen mnemonics observed across the capture corpus — fourteen from
# the original 49-capture study plus NH, which capture sitting 4 measured off
# one radio button (momwire#513, capture 0111) — plus the CM/CE comment
# framing.  Anything outside this
# set refuses; the table below gives the ones a reader might plausibly send
# a reason of their own, and everything else falls through to the generic
# by-name refusal in `_Nec5Parser.card`.
_VOCABULARY = frozenset(
    {
        "CM",
        "CE",
        "GW",
        "GE",
        "GN",
        "GD",
        "EX",
        "LD",
        "TL",
        "NT",
        "FR",
        "PQ",
        "RP",
        "XQ",
        "NE",
        "NH",
        "EN",
    }
)

_REFUSED_BY_NAME = MappingProxyType(
    {
        # NEC geometry EZNEC never writes: it resolves every transform in its
        # own UI and emits explicit GW cards, so a deck carrying one of these
        # is not an EZNEC deck.  Ignoring them would mangle the geometry
        # silently, which is the one outcome worse than a refusal.
        "GM": "GM (geometry move) is not part of this engine's nec5 dialect, whose "
        "geometry is GW alone — EZNEC resolves its transforms before writing a deck",
        "GX": "GX (symmetry reflection) is not part of this engine's nec5 dialect, "
        "whose geometry is GW alone",
        "GR": "GR (cylindrical symmetry) is not part of this engine's nec5 dialect, "
        "whose geometry is GW alone",
        "GS": "GS (structure scale) is not part of this engine's nec5 dialect, whose "
        "geometry is GW alone",
        "GA": "GA (wire arc) is not part of this engine's nec5 dialect, whose "
        "geometry is GW alone",
        "GH": "GH (helix) is not part of this engine's nec5 dialect, whose geometry "
        "is GW alone",
        "GC": "GC (tapered wire continuation) is not part of this engine's nec5 "
        "dialect",
        "GF": "GF (numerical Green's function) is not part of this engine's nec5 "
        "dialect",
        "SP": "SP (surface patch) is not part of this engine's nec5 dialect, which "
        "models wires only",
        "SM": "SM (multiple-patch surface) is not part of this engine's nec5 dialect, "
        "which models wires only",
        "SC": "SC (surface patch continuation) is not part of this engine's nec5 "
        "dialect, which models wires only",
        "CP": "CP (coupling request) is not part of this engine's nec5 dialect",
        "PL": "PL (plot request) is not part of this engine's nec5 dialect",
        "WG": "WG (NGF write request) is not part of this engine's nec5 dialect",
        "ZO": "ZO (impedance normalisation) is not part of this engine's nec5 dialect",
        "KH": "KH (interaction approximation limit) is not part of this engine's nec5 "
        "dialect",
        # Cards the nec2 dialect serves and this one does not.
        "EK": "EK (extended thin-wire kernel) is not part of this engine's nec5 "
        "dialect — EZNEC emits no kernel card",
        "IS": "IS (insulated sheath) is not part of this engine's nec5 dialect",
        "PT": "PT (element-current print control) is not part of this engine's nec5 "
        "dialect; PQ is the only print-control card EZNEC emits",
        "MP": "MP (multiprocessing hint) is not part of this engine's nec5 dialect",
        "SY": "SY (4nec2 symbolic variables) is not part of this engine's nec5 dialect",
        "NX": "NX (next structure) is not part of this engine's nec5 dialect; EN is "
        "the terminator EZNEC writes",
    }
)

# The number of fields each card must carry.  EZNEC writes every field of
# every card it emits, so this dialect does no blank-field defaulting: a
# short card is a deck this front-end did not come from and refuses rather
# than zero-fills (the one exception is GD/GN 0's trailing complex mu, which
# the measured grammar lets default to 1).  GN is absent because its length
# depends on its first field — bare for -1 and 1, media-carrying for 0 and 2
# — and is checked in `_gn`.
_MIN_FIELDS = MappingProxyType(
    {
        "GW": 9,
        "GE": 2,
        "GD": 6,
        "EX": 6,
        "LD": 6,
        "TL": 10,
        "NT": 10,
        "FR": 5,
        "PQ": 1,
        "RP": 8,
        "NE": 10,
        "NH": 10,
    }
)

# EX kinds the dialect emits: 4 (elementary current source) 47/49 captures,
# 0 (voltage source) 2/49.  Every other NEC excitation — incident plane
# waves, EX 5's current-slope discontinuity — is off-vocabulary here.
_EX_KINDS = frozenset({0, 4})

# The two XNDA forms observed: 1000 (2-D slice) and 1001 (3-D).  The value is
# a packed set of output-format flags, and an unobserved packing would be a
# printout this seam has never been shown how to write.
_RP_XNDA = frozenset({1000, 1001})


def _complex(card: Card, k: int) -> complex:
    return complex(card.f(k), card.f(k + 1))


# --------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------


class _Nec5Parser:
    """A straight-line reader: cards accumulate, the deck comes out at the end.

    No execute-card state machine, unlike :class:`~momwire.deck._nec2.
    _Nec2Parser`.  EZNEC writes one request card per deck and launches the
    engine once per frequency point (capture study, "Sweep protocol"), so
    there is exactly one operating point in a deck and nothing for a group
    to separate.  Requests are collected in a tuple anyway, because the
    grammar allows more than one and silently dropping the second would be
    the wrong kind of quiet.
    """

    def __init__(self) -> None:
        self.comments: list[str] = []
        self.ce_text: str | None = None
        self.wires: list[Nec5Wire] = []
        self._by_tag: dict[int, Nec5Wire] = {}
        self.ge_flag: int | None = None
        self.ge_second: int = -1
        # Free space until a ground card says otherwise.  Every one of the 49
        # captures carries exactly one GN or GD, so this default is armor
        # rather than a hot path.
        self.ground: Nec5Ground = Nec5FreeSpace()
        self.sources: list[Nec5Source] = []
        self.loads: list[Nec5Load] = []
        self.lines: list[Nec5TransmissionLine] = []
        self.networks: list[Nec5Network] = []
        self.frequency_mhz: float | None = None
        self.requests: list[Nec5Request] = []
        self.pq: int | None = None
        self._saw_en = False

    # -- addressing --------------------------------------------------------

    def _address(self, card: Card, k: int) -> Nec5Node:
        """Fields ``k``/``k+1`` of ``card`` as a :class:`Nec5Node`.

        The one piece of code every connection card shares.  ``EX``, ``LD``,
        ``TL`` and ``NT`` all take this address (capture study, "Signed node
        addressing spans four cards"), and all four go through here so the
        favored wire survives identically on each.
        """
        tag = card.i(k)
        written = card.i(k + 1)
        if written == -1:
            node = 0
        elif written == 0:
            # Node 0 needs the -1 spelling precisely because 0 is not
            # available: a zero segment field means "all" elsewhere in NEC.
            raise DeckError(
                f"{card.mnemonic} addresses node 0 of tag {tag} as a literal 0; "
                f"this dialect spells node 0 (the wire's end-1 boundary) as -1, "
                f"and a zero connection field is not a node"
            )
        elif written < 0:
            # Every negative value in all 49 captures is exactly -1, as the
            # node-0 encoding predicts.  No other negative has ever been
            # observed, so its meaning is unknown and this engine will not
            # guess at one.
            raise DeckError(
                f"{card.mnemonic} addresses node {written} of tag {tag}: -1 is the "
                f"only negative node spelling in this dialect (it means node 0, the "
                f"wire's end-1 boundary) and no other negative value appears in any "
                f"captured deck"
            )
        else:
            node = written
        wire = self._by_tag.get(tag)
        if wire is None:
            raise DeckError(
                f"{card.mnemonic} names tag {tag}, which no GW card in this deck "
                f"declares"
            )
        if node > wire.segment_count:
            raise DeckError(
                f"{card.mnemonic} addresses node {node} of tag {tag}, whose "
                f"{wire.segment_count} segments give it nodes 0 to "
                f"{wire.segment_count}; there is no node past the end of a wire"
            )
        return Nec5Node(tag=tag, node=node)

    # -- geometry ----------------------------------------------------------

    def _gw(self, card: Card) -> None:
        tag = card.i(0)
        if tag in self._by_tag:
            # EZNEC numbers its wires 1..N, one GW each.  NEC-2 lets several
            # GW cards share a tag; a node address into such a group has no
            # single wire to count segments along, so this dialect refuses
            # the ambiguity rather than picking a wire.
            raise DeckError(
                f"GW declares tag {tag} a second time; this engine's nec5 dialect "
                f"gives each wire its own tag, because a node address names one "
                f"wire's segment boundary and a tag group has no single wire to "
                f"count along"
            )
        segments = card.i(1)
        if segments < 1:
            raise DeckError(
                f"GW tag {tag} asks for {segments} segments; a wire needs at least 1"
            )
        wire = Nec5Wire(
            tag=tag,
            segment_count=segments,
            end1=(card.f(2), card.f(3), card.f(4)),
            end2=(card.f(5), card.f(6), card.f(7)),
            radius=card.f(8),
        )
        self.wires.append(wire)
        self._by_tag[tag] = wire

    def _ge(self, card: Card) -> None:
        """``GE <ground-flag>,<second>``.

        NEC-2's ``GE`` takes one field; this dialect's takes two, and the
        second is ``-1`` in all 49 captures (capture study, "``GE`` takes a
        second parameter").  Its meaning is unobserved, so it is recorded and
        not interpreted.
        """
        self.ge_flag = card.i(0)
        self.ge_second = card.i(1)

    # -- ground ------------------------------------------------------------
    #
    # LAST GROUND CARD WINS.  Measured, scored matrix "Probe family 1" item 5:
    # `GN 1` then `GD` -> MININEC; `GD` then `GN 1` -> perfect; `GD` then
    # `GN 0` -> Sommerfeld; `GN 0` then `GD` -> MININEC; two `GD`s -> the
    # second.  There is NO NEC-2-style merge of GD into a GN environment in
    # this dialect, so neither handler below reads the ground already in
    # force.  EZNEC emits exactly one ground card per deck, so this is
    # refusal-grammar armor rather than a hot path.

    def _mu(self, card: Card) -> complex:
        """A ground card's trailing complex relative permeability.

        Per the Users Manual and measured in probe family 1: the trailing
        pair is μr, it moves the far field only (Z is pinned throughout),
        and omitted or zero behaves as 1.  EZNEC always writes ``1.,0.``.
        A non-unity μr refuses loudly — momwire has no magnetic ground, and
        running one as if it were free of magnetization would be a wrong
        answer rather than a refusal.
        """
        mu = _complex(card, 6)
        if mu == 0j:
            mu = 1 + 0j
        if mu != 1 + 0j:
            raise DeckError(
                f"{card.mnemonic} asks for a ground of relative permeability "
                f"{mu.real:g}{mu.imag:+g}j; this engine has no magnetic ground and "
                f"serves mu_r = 1 only (EZNEC writes 1.,0. on every ground card it "
                f"emits)"
            )
        return mu

    def _gn(self, card: Card) -> None:
        code = card.i(0)
        if code == -1:
            self.ground = Nec5FreeSpace()
            return
        if code == 1:
            self.ground = Nec5PerfectGround()
            return
        if code in (0, 2):
            # GN 2 is the NEC-4-compatible spelling; probe family 1 measured
            # it identical to GN 0 to every printed digit.  EZNEC emits only
            # GN 0, but the spelling is recorded so a byte-gated printout can
            # echo the card the deck actually wrote.
            if len(card.values) < 6:
                raise DeckError(
                    f"GN {code} carries {len(card.values)} fields; a finite ground "
                    f"needs at least 6 (the media payload's epsilon and sigma sit in "
                    f"fields 5 and 6)"
                )
            self.ground = Nec5SommerfeldGround(
                eps_r=card.f(4),
                sigma=card.f(5),
                mu_r=self._mu(card),
                spelling=code,
            )
            return
        raise DeckError(
            f"GN type {code} is not part of this engine's nec5 dialect, which serves "
            f"GN -1 (free space), GN 1 (perfect), and GN 0 / GN 2 (finite ground, "
            f"Sommerfeld solution)"
        )

    def _gd(self, card: Card) -> None:
        """A bare ``GD`` — the MININEC-type ground.

        Field layout is this dialect's, NOT NEC-2's (whose ``GD`` is a cliff:
        a second radial medium with distances and heights).  Here I1 is a
        cancel flag, I2-I4 are ignored, F1/F2 are ε and σ and F3/F4 are the
        complex μr.  No code is shared with
        :meth:`~momwire.deck._nec2._Nec2Parser._gd`, deliberately.
        """
        ix = card.i(0)
        if ix == -1:
            # Measured: `GD -1` cancels the GD ground and the deck answers as
            # free space.  Its media fields are then moot, so they are not
            # validated — including the mu refusal, which would be a false
            # refusal on a ground that is not there.
            self.ground = Nec5FreeSpace()
            return
        sigma = card.f(5)
        self.ground = Nec5MininecGround(
            ix=ix,
            eps_r=card.f(4),
            sigma=sigma,
            mu_r=self._mu(card),
            # Measured: a negative sigma sets Im(epsilon_c) directly rather
            # than being a conductivity (equivalent at the matched value).
            sigma_sets_im_epsc=sigma < 0.0,
        )

    # -- excitation, loading, networks -------------------------------------

    def _ex(self, card: Card) -> None:
        kind = card.i(0)
        if kind not in _EX_KINDS:
            raise DeckError(
                f"EX type {kind} is not part of this engine's nec5 dialect, which "
                f"serves EX 4 (elementary current source) and EX 0 (voltage source) "
                f"— the two EZNEC's source-type setting picks between"
            )
        self.sources.append(
            Nec5Source(
                kind=kind,
                at=self._address(card, 1),
                drive=_complex(card, 4),
                option=card.i(3),
            )
        )

    def _ld(self, card: Card) -> None:
        kind = card.i(0)
        if kind != 4:
            raise DeckError(
                f"LD type {kind} is not part of this engine's nec5 dialect, whose "
                f"loading is LD 4 (fixed impedance) alone — 67 of 67 loads across "
                f"the captured corpus, because EZNEC reduces a load to an impedance "
                f"at the frequency before it writes the deck"
            )
        node_to = card.i(3)
        if node_to != 0:
            # Observed 0 on every one of the 67 captured LD cards, which is
            # what makes this dialect's loads single-point rather than
            # ranges.  A nonzero field has never been seen, so its meaning
            # here is unobserved.
            raise DeckError(
                f"LD 4 carries {node_to} in its fourth field, which is 0 on every "
                f"captured card; this dialect's loads are single-point and the "
                f"field's meaning at any other value is unobserved"
            )
        self.loads.append(
            Nec5Load(
                at=self._address(card, 1),
                impedance=_complex(card, 4),
                node_to=node_to,
            )
        )

    def _tl(self, card: Card) -> None:
        end_a = self._address(card, 0)
        end_b = self._address(card, 2)
        z0 = card.f(4)
        if z0 == 0.0:
            raise DeckError(
                "TL with a zero characteristic impedance is not a transmission line; "
                "Z0 must be nonzero, and its SIGN — not its magnitude — is what "
                "selects a crossed line"
            )
        self.lines.append(
            Nec5TransmissionLine(
                end_a=end_a,
                end_b=end_b,
                z0=z0,
                length_m=card.f(5),
                shunt=(card.f(6), card.f(7), card.f(8), card.f(9)),
            )
        )

    def _nt(self, card: Card) -> None:
        self.networks.append(
            Nec5Network(
                end_a=self._address(card, 0),
                end_b=self._address(card, 2),
                y11=_complex(card, 4),
                y12=_complex(card, 6),
                y22=_complex(card, 8),
            )
        )

    # -- requests ----------------------------------------------------------

    def _fr(self, card: Card) -> None:
        """``FR 0,1,0,0,<MHz>`` — one frequency, always.

        EZNEC never uses NEC's multi-frequency stepping: a sweep is one
        process launch per point, each with a freshly regenerated deck
        (capture study, "Sweep protocol").  A stepping ``FR`` would ask this
        seam for something its protocol has no way to return.
        """
        n = card.i(1)
        if n > 1:
            raise DeckError(
                f"FR asks for {n} frequency points; this seam is one process per "
                f"point (EZNEC regenerates the whole deck per frequency and never "
                f"emits a stepping FR), so a multi-point FR is not served"
            )
        self.frequency_mhz = card.f(4)

    def _rp(self, card: Card) -> None:
        mode = card.i(0)
        if mode != 0:
            raise DeckError(
                f"RP mode {mode} is not part of this engine's nec5 dialect, which "
                f"serves RP 0 (the normal far-field request) alone"
            )
        xnda = card.i(3)
        if xnda not in _RP_XNDA:
            raise DeckError(
                f"RP 0 with XNDA {xnda} is not part of this engine's nec5 dialect, "
                f"which serves the two forms EZNEC emits: 1000 (2-D slice) and 1001 "
                f"(3-D)"
            )
        self.requests.append(
            Nec5FarFieldRequest(
                mode=mode,
                n_theta=max(card.i(1), 1),
                n_phi=max(card.i(2), 1),
                xnda=xnda,
                theta0_deg=card.f(4),
                phi0_deg=card.f(5),
                d_theta_deg=card.f(6),
                d_phi_deg=card.f(7),
                range_m=card.f(8),
            )
        )

    def _ne(self, card: Card) -> None:
        """``NE`` and ``NH`` — one field layout, one handler, one flag.

        Capture 0111 (momwire#513) pinned ``NH`` as ``NE``'s exact twin: the
        Near Field Analysis dialog's E/H radio button picks the mnemonic and
        nothing else about the card moves.
        """
        coordinates = card.i(0)
        if coordinates != 0:
            raise DeckError(
                f"{card.mnemonic} coordinate system {coordinates} (spherical) is "
                f"not part of this engine's nec5 dialect; rectangular (0) is the "
                f"form EZNEC emits"
            )
        self.requests.append(
            Nec5NearFieldRequest(
                coordinates=coordinates,
                counts=(max(card.i(1), 1), max(card.i(2), 1), max(card.i(3), 1)),
                origin=(card.f(4), card.f(5), card.f(6)),
                step=(card.f(7), card.f(8), card.f(9)),
                magnetic=card.mnemonic == "NH",
            )
        )

    def _pq(self, card: Card) -> None:
        """``PQ`` — accepted and recorded; serving it is U4's business.

        Every captured printout carries a ``Wire Charge Densities`` block
        under ``PQ 0`` (all ten byte-gate fixtures; the header is mixed-case,
        which briefly hid it from the scored matrix's first pass), so unlike
        the nec2 dialect — which refuses charge requests — this seam owes the
        block: charge density is a readout from the solved current,
        q = -(1/jω) dI/ds.  The parser records the value; the physics unit
        answers it.
        """
        self.pq = card.i(0)

    # -- the loop ----------------------------------------------------------

    def feed(self, text: str) -> None:
        for line in text.splitlines():
            card = parse_card(line)
            if card is None:
                continue
            if card.mnemonic == "EN":
                # EN terminates; anything after it is not this deck.
                self._saw_en = True
                break
            self.card(card)
        if not self._saw_en:
            # A divergence from the nec2 front-end, on purpose: `_nec2.feed`
            # closes a deck at EOF whether or not a terminator arrived,
            # because NEC's own reader does.  This seam serves EZNEC, which
            # writes EN on all 49 captures, and a deck that stops early is a
            # truncated file rather than a short deck — the failure mode the
            # fault-injection table showed EZNEC cannot diagnose for itself.
            raise DeckError(
                "deck has no EN card; every deck EZNEC writes terminates with one, "
                "so a deck that ends without EN is truncated"
            )

    def card(self, card: Card) -> None:
        if card.mnemonic == "CM":
            if self.ce_text is not None:
                raise DeckError(
                    "CM appears after the CE that closed the comment block; a deck's "
                    "comment cards are its first cards, and the printout echoes them "
                    "as one block"
                )
            self.comments.append(card.text)
            return
        if card.mnemonic == "CE":
            if self.ce_text is not None:
                raise DeckError("CE appears twice; one CE closes a comment block")
            # Bare in all 49 captures, but the card carries free text in NEC
            # and the renderer echoes whatever was written.
            self.ce_text = card.text
            return
        if self.ce_text is None:
            raise DeckError(
                f"{card.mnemonic} arrives before any CE card; a deck's comment block "
                f"must be closed by CE before its first data card"
            )
        if (message := _REFUSED_BY_NAME.get(card.mnemonic)) is not None:
            raise DeckError(message)
        if card.mnemonic not in _VOCABULARY:
            raise DeckError(
                f"{card.mnemonic} is not a card in this engine's nec5 dialect, whose "
                f"vocabulary is the fifteen mnemonics EZNEC emits: GW, GE, GN, GD, "
                f"EX, LD, TL, NT, FR, PQ, RP, XQ, NE, NH, EN"
            )
        minimum = _MIN_FIELDS.get(card.mnemonic)
        if minimum is not None and (written := len(card.values)) < minimum:
            raise DeckError(
                f"{card.mnemonic} carries {written} "
                f"{'field' if written == 1 else 'fields'} and needs at least "
                f"{minimum}; EZNEC writes every field of every card it emits, so "
                f"this dialect does no blank-field defaulting"
            )
        handler = {
            "GW": self._gw,
            "GE": self._ge,
            "GN": self._gn,
            "GD": self._gd,
            "EX": self._ex,
            "LD": self._ld,
            "TL": self._tl,
            "NT": self._nt,
            "FR": self._fr,
            "RP": self._rp,
            "NE": self._ne,
            "NH": self._ne,
            "PQ": self._pq,
        }.get(card.mnemonic)
        if handler is not None:
            handler(card)
            return
        assert card.mnemonic == "XQ"  # the vocabulary's remainder
        self.requests.append(Nec5ExecuteRequest(flag=card.i(0)))

    # -- the deck ----------------------------------------------------------

    def deck(self, source_text: str = "") -> Nec5Deck:
        if self.ge_flag is None:
            raise DeckError(
                "deck has no GE card; GE closes the geometry section and carries the "
                "ground flag, and EZNEC writes one on every deck"
            )
        return Nec5Deck(
            comments=tuple(self.comments),
            ce_text=self.ce_text or "",
            source_text=source_text,
            wires=tuple(self.wires),
            ge_flag=self.ge_flag,
            ge_second=self.ge_second,
            ground=self.ground,
            sources=tuple(self.sources),
            loads=tuple(self.loads),
            transmission_lines=tuple(self.lines),
            networks=tuple(self.networks),
            frequency_mhz=self.frequency_mhz,
            requests=tuple(self.requests),
            pq=self.pq,
        )


def parse_nec5(text: str) -> Nec5Deck:
    """Parse one ``nec5`` deck body into a :class:`Nec5Deck`.

    Raises :class:`~momwire.deck.DeckError` — the same exception family every
    dialect front-end raises — naming the offending card, for anything
    outside the fourteen observed mnemonics or their observed field forms.
    """
    parser = _Nec5Parser()
    parser.feed(text)
    return parser.deck(text)
