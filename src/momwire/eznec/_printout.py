"""The NEC-5 printout, top down: the header, the refusal frame, the results.

Layout is the contract; the numbers (when later units add them) are momwire's.
Everything in this module was derived from CAPTURED INPUT/OUTPUT ONLY — the
decks EZNEC wrote and the printouts the Windows engine left behind, kept under
``tests/fixtures/eznec/`` — and from the two interface studies:

* antennaknobs ``docs/status/2026-08-16-eznec-nec5-dialect-capture.md``
  (invocation protocol, the error convention, the stamp echo)
* antennaknobs ``docs/status/2026-08-20-eznec-nec5-scored-matrix.md``
  (the scored ladder and its normalizations)

No NEC-5 source, algorithm, or internal structure is described or relied on
here, and none may be: the binary is user-licensed and export-controlled.
What is reproduced is the shape of bytes a user's own engine already printed.

Why the header alone is worth a unit
------------------------------------
EZNEC does not read the engine's exit status at all; it reads the printout,
and it decides whether the printout belongs to THIS run by looking for the
comment block echoed back at the top.  EZNEC stamps the launch time into the
deck as a ``CM`` card, so that echo is the only thing in the file that
distinguishes one run from another.  A printout without it is rejected as
"written earlier from another calculation" and no message of any kind — not
even a refusal — reaches the user (capture doc, "Error convention").

So the header is not decoration.  It is the envelope every later unit's
results, and every refusal this engine will ever emit, has to travel in.

What U3 adds, and what it refuses to know
-----------------------------------------
Everything from ``- - - STRUCTURE SPECIFICATION - - -`` to ``RUN TIME``.
:func:`render_printout` is a FORMATTER: it is handed a :class:`RunData` full
of numbers and lays them out in the captured columns.  It computes no
physics — not a wavelength from a frequency, not a magnitude from a real and
an imaginary part, not an efficiency from two powers.  Everything printed is
carried, because a formatter that recomputes one printed value from another
turns a rounding difference into a byte difference, and because the numbers
belong to U4/U5 and the columns belong here.

Every column width, every blank column and every float format below was
MEASURED off the ten captured printouts in ``tests/fixtures/eznec/printouts/``
and is cited by capture id where a reader might reasonably doubt it.  The
round-trip gate in ``tests/test_eznec_printout.py`` re-derives a
:class:`RunData` from each captured printout and re-renders it, so a width
that is wrong anywhere is a failing test rather than a comment.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..deck._cards import Card, parse_card
from ..deck._nec5 import Nec5Deck

__all__ = [
    "STUB_REFUSAL",
    "ENVIRONMENT_FINITE_GROUND",
    "ENVIRONMENT_FREE_SPACE",
    "ENVIRONMENT_PERFECT_GROUND",
    "ChargeRow",
    "GroundMedium",
    "LineRow",
    "LoadRow",
    "NetworkRow",
    "PatternBlock",
    "PatternRow",
    "PortRow",
    "PowerBudget",
    "RunData",
    "WireCurrentRow",
    "render_header",
    "render_printout",
    "render_refusal",
]

# U1's stub refusal.  Since U4 nothing in the captured corpus reaches it: the
# dialect front end refuses an unknown card by name, :mod:`._serve` refuses a
# deck above rung 1 by name, and everything else is answered.  It survives as
# the shell's last line of defence — a deck that fails somewhere none of those
# three foresaw still comes back with a sentence rather than an empty file.
STUB_REFUSAL = "NEC-5 DIALECT NOT YET SERVED BY THIS ENGINE"

# --------------------------------------------------------------------------
# the header, column by column
# --------------------------------------------------------------------------
#
# Measured from all ten byte-gate printouts in tests/fixtures/eznec/printouts/
# (capture ids 0010, 0012, 0013, 0014, 0016, 0017, 0019, 0035, 0043, 0044).
# The skeleton is identical in every one of them; only the echoed comment
# lines differ, and they differ exactly as their decks do.

_PROLOGUE: tuple[str, ...] = (
    # A bare "1" — a Fortran carriage-control form feed, printed literally by
    # every capture — then the engine's build tag.
    "1",
    " x13",
    "",
    "",
    "",
)

_BANNER_INDENT = " " * 32
_BANNER_RULE = "*" * 47
_BANNER: tuple[str, ...] = (
    _BANNER_INDENT + _BANNER_RULE,
    _BANNER_INDENT + "*" + " " * 45 + "*",
    _BANNER_INDENT + "*  NUMERICAL ELECTROMAGNETICS CODE (NEC-5)    *",
    _BANNER_INDENT + "*" + " " * 45 + "*",
    _BANNER_INDENT + _BANNER_RULE,
    "",
    "",
    "",
    "",
)

# The comment box is 80 columns wide and starts four columns left of the text
# it wraps — and the text OVERRUNS it on the right, to column 103.  That is
# not a transcription slip: the box rule and the comment lines are written by
# different formats, and the echo's field is wider than the box.  Reproduced
# as observed.
_BOX_RULE = " " * 22 + "*" * 58

# An echoed comment line is 25 blanks followed by the card's columns 3-80,
# blank-padded to that full 78-column field — so an empty ``CM`` echoes as 103
# spaces, trailing blanks and all.  Reading the text POSITIONALLY (columns
# 3-80) rather than by stripping the mnemonic and a space is the one place
# U1 chose a model over a measurement: every captured card is either bare
# ``CM``/``CE`` or ``CM `` + text, so the two readings agree on all 49 decks
# and differ only on a card with two spaces after the mnemonic, which no
# capture contains.  The positional reading is what a fixed-format engine
# does, so it is the safer guess; if a future capture ever disagrees, this
# is the line to change.
_ECHO_INDENT = " " * 25
_ECHO_TEXT_COLUMNS = slice(2, 80)
_ECHO_WIDTH = 78

# Five blank lines separate the comment box from the STRUCTURE SPECIFICATION
# heading, which is where the header ends and U3's renderer begins.
_HEADER_TAIL: tuple[str, ...] = ("", "", "", "", "")

# The refusal frame, verbatim from the fault-injection experiment (capture doc,
# "Refusals can speak"): one leading space, five asterisks, the words, and the
# reason.  That exact line reached the operator through EZNEC's own viewer.
_ERROR_PREFIX = " ***** NEC ERROR - "


def comment_cards(deck_text: str) -> list[str]:
    """The deck's leading comment block, as the text each card echoes.

    NEC's comment block is the run of ``CM`` cards at the top of the deck,
    terminated by ``CE`` — and ``CE`` echoes too, as its own (usually empty)
    line, which is why a five-``CM`` deck prints six comment lines.  Anything
    after ``CE``, and anything that is not a comment card, belongs to U2.
    """
    texts: list[str] = []
    for line in deck_text.splitlines():
        mnemonic = line[:2].upper()
        if mnemonic == "CM":
            texts.append(line[_ECHO_TEXT_COLUMNS])
        elif mnemonic == "CE":
            texts.append(line[_ECHO_TEXT_COLUMNS])
            break
        else:
            break
    return texts


def _comment_box(deck_text: str) -> list[str]:
    """The boxed comment echo — the stamp EZNEC checks a printout's age by."""
    return [
        _BOX_RULE,
        "",
        *(_ECHO_INDENT + text.ljust(_ECHO_WIDTH) for text in comment_cards(deck_text)),
        "",
        _BOX_RULE,
    ]


def render_header(deck_text: str | None) -> str:
    """The printout down to (not including) ``- - - STRUCTURE SPECIFICATION - - -``.

    ``deck_text`` of ``None`` is the one case with no comment box to print:
    the input file could not be read at all, so there are no ``CM`` cards to
    echo.  Such a printout cannot pass EZNEC's belongs-to-this-run check —
    nothing can, without the stamp — but it is still written, because the
    alternative (no file) sends EZNEC down the "check where you installed the
    NEC program" path and blames the user's configuration for a read error.

    Line endings are LF.  The captured printouts are the Windows engine's, so
    they arrived CRLF; the byte-gates normalize before comparing (see the
    fixture manifest's ``normalizations`` key), and this engine writes the
    platform-neutral form.
    """
    lines = [*_PROLOGUE, *_BANNER]
    if deck_text is not None:
        lines += _comment_box(deck_text)
    lines += _HEADER_TAIL
    return "\n".join(lines) + "\n"


def render_refusal(deck_text: str | None, reason: str) -> str:
    """A complete printout that refuses: the header, then one ``NEC ERROR`` line.

    The order is the contract.  With the echo intact and the results missing,
    EZNEC offers to show the operator the file, and whatever stands where the
    results would have been is what they read — so a refusal that names itself
    arrives as a sentence rather than as an unexplained failure.  Put the same
    line BEFORE the echo and the file is discarded as stale instead.
    """
    return render_header(deck_text) + "\n" + _ERROR_PREFIX + reason + "\n"


# ==========================================================================
# what a printout carries
# ==========================================================================
#
# One dataclass per table, and every one of them is dumb on purpose: floats,
# ints and tuples, no solver types, no derived quantities.  A row holds each
# number the row PRINTS — a current's real part, imaginary part, magnitude
# AND phase — because the engine printed four numbers and this module's job
# is to put four numbers back.


@dataclass(frozen=True)
class LoadRow:
    """One row of the ``STRUCTURE IMPEDANCE LOADING`` table.

    Captured in 0012/0014/0016/0017 (W7EL's ``Network Connection Test``,
    whose two ``LD 4,4,n,0,1.E+10,0.`` cards pin a virtual wire's nodes
    open).  Those four printouts are the only loaded captures, and the
    dialect serves ``LD 4`` alone, so the fixed-impedance shape is the only
    row this table can carry: the RESISTANCE / INDUCTANCE / CAPACITANCE /
    CONDUCTIVITY columns have never been anything but blank and are not
    modelled here at all.

    :attr:`reactance` is optional because the captured rows leave the
    IMAGINARY column BLANK where the card wrote ``0.``  Whether the engine
    blanks a zero or never prints that column for ``LD 4`` is unobserved, so
    the cell is carried as present-or-absent rather than guessed at.

    :attr:`node_from` and :attr:`node_thru` are the DECODED node, and the
    contrast with :class:`NetworkRow` is the point: the same ``-1`` address
    that prints as a negative segment there prints as ``1`` here (0025's
    ``LD 4,1,-1`` against its ``TL 5,3,1,-1``, in one printout).  The loading
    table carries no sign at all.
    """

    tag: int
    node_from: int
    node_thru: int
    resistance: float
    reactance: float | None = None
    kind: str = "FIXED IMPEDANCE"


@dataclass(frozen=True)
class PortRow:
    """One row of ``ANTENNA INPUT PARAMETERS`` — and of its network twin.

    The two tables share a format exactly (compare 0012's lines under
    ``STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS`` with the ones
    under ``ANTENNA INPUT PARAMETERS``), so they share a row type.

    :attr:`end_index` is the trailing digit after the ``TAG``/``SEG.`` pair.
    It tracks the DECK'S SIGN, 9 of 9 rows in the capture study's table
    ("Signed segment addressing", 2026-08-16): a positive node field prints
    1, the ``-1`` spelling of node 0 prints 2.  It is not a port number and
    not an end-one/end-two flag — ``DipTL1`` reports 2 on a card's end one —
    so it is carried as read rather than derived from anything here.
    """

    tag: int
    segment: int
    end_index: int
    voltage: complex
    current: complex
    impedance: complex
    admittance: complex
    power: float


@dataclass(frozen=True)
class NetworkRow:
    """One row of ``NETWORK DATA``: two endpoints and a reciprocal Y matrix.

    The endpoints print as SIGNED SEGMENT numbers (0012: ``3   -10``), which
    is the engine's own rendering of the deck's ``(favored tag, node)``
    address and not something this module reconstructs.  0016 writes the same
    physical connection as ``2     9`` — same admittance, different address —
    which is exactly the distinction the W7EL gate exists to protect.
    """

    tag_from: int
    segment_from: int
    tag_to: int
    segment_to: int
    y11: complex
    y12: complex
    y22: complex


@dataclass(frozen=True)
class LineRow:
    """One row of ``NETWORK DATA``'s OTHER form: a ``TL`` card.

    The same table heading covers both, and the two rows START identically —
    four address fields then six ``E14.4`` cells, which is why 0012's ``NT``
    row and 0027's ``TL`` row line up column for column.  What differs is the
    COLUMN HEADINGS above them (an admittance matrix against an impedance, a
    length and two end shunts) and a seventh field the ``NT`` form has no room
    for: :attr:`crossed`, printed as ``STRAIGHT`` or ``CROSSED`` (0028's
    phase-reversal feeder is the corpus's only ``CROSSED``).

    :attr:`length_m` is the RESOLVED length in metres, not the card's field: a
    ``TL`` whose length field is zero prints the straight-line distance between
    the two addressed NODES (0028's four crossed lines all write ``0.`` and
    print 9.9764E-01 / 7.9826E-01 / 6.3831E-01 / 5.1090E-01, which are those
    node-to-node distances to every printed digit).  Resolving it is physics
    and belongs upstream; this row carries the number that was printed.
    """

    tag_from: int
    segment_from: int
    tag_to: int
    segment_to: int
    z0: float
    length_m: float
    shunt_a: complex
    shunt_b: complex
    crossed: bool = False

    @property
    def line_type(self) -> str:
        return "CROSSED" if self.crossed else "STRAIGHT"


@dataclass(frozen=True)
class WireCurrentRow:
    """One segment's row in the mixed-case ``Wire Currents`` table."""

    element: int
    tag: int
    center: tuple[float, float, float]
    length: float
    real: float
    imag: float
    magnitude: float
    phase_deg: float


@dataclass(frozen=True)
class ChargeRow:
    """One segment's row in ``Wire Charge Densities`` — the ``PQ`` block."""

    element: int
    tag: int
    center: tuple[float, float, float]
    length: float
    magnitude: float
    phase_deg: float


@dataclass(frozen=True)
class PowerBudget:
    """The four-or-five line ``POWER BUDGET`` block.

    :attr:`network_loss` is the fifth line, and carrying networks is not
    enough to earn it: 0012/0014/0016/0017/0018 print it, and 0027/0028 —
    which carry ``TL`` cards and print a whole ``NETWORK DATA`` table — do
    not.  What separates them is the SIGN (see :mod:`._serve`, "the budget's
    own arithmetic"), so the line is optional rather than zero, and whoever
    computes it decides whether there is one.
    """

    input_power: float
    radiated_power: float
    wire_loss: float
    efficiency_percent: float
    network_loss: float | None = None


@dataclass(frozen=True)
class PatternRow:
    """One direction's row of ``RADIATION PATTERNS``.

    :attr:`sense` is a STRING, blank included.  The captures print ``LINEAR``
    / ``LEFT`` / ``RIGHT`` in that column and leave it EMPTY on rows whose
    total gain is the ``-999.99`` null (0013 line 186 against line 187), and
    the rule behind that blank belongs to whoever computes the field, not to
    whoever prints it.

    The nulls themselves are ordinary floats: ``-999.99`` is what the engine
    prints for a gain of zero and it formats like any other number.
    """

    theta_deg: float
    phi_deg: float
    vert_db: float
    hor_db: float
    total_db: float
    axial_ratio: float
    tilt_deg: float
    sense: str
    e_theta_magnitude: float
    e_theta_phase_deg: float
    e_phi_magnitude: float
    e_phi_phase_deg: float


@dataclass(frozen=True)
class PatternBlock:
    """One ``RP`` card's answer: its rows, and the trailer the 3-D form adds.

    ``XNDA`` 1001 (0013, 0035) closes with an average-power-gain line and a
    power-radiated line; ``XNDA`` 1000 (0010, 0044) closes with neither.  The
    trailer is carried as three optional numbers rather than as a flag, so
    the renderer never has to know which form asked.
    """

    rows: tuple[PatternRow, ...] = ()
    average_power_gain: float | None = None
    solid_angle_pi: float | None = None
    power_radiated_4pi: float | None = None


# The three environments this arc serves — three names for FOUR ground cards,
# because the finite banner is SHARED by ``GN 0`` and by the bare ``GD``
# MININEC ground (scored matrix, "Probe family 1").  That is the arc's
# standing warning that the banner is not a mode signal, and both rungs have
# now landed under it: rung 2 solves IN the medium, rung 3 only reflects off
# it, and their environment blocks are byte-identical all four lines (0045
# against 0047, gated in ``tests/test_eznec_serve.py``).  A renderer must
# reproduce the shared banner; nothing may ever read it back as a mode.
ENVIRONMENT_FREE_SPACE = "FREE SPACE"
ENVIRONMENT_PERFECT_GROUND = "PERFECT GROUND"
ENVIRONMENT_FINITE_GROUND = "FINITE GROUND.  SOMMERFELD SOLUTION"


@dataclass(frozen=True)
class GroundMedium:
    """The three medium cells a finite-ground environment block prints.

    Numbers only, εc included: the complex dielectric constant is PHYSICS —
    it needs the frequency, which this module does not have and does not want
    — so it is computed by whoever solves the deck and carried here, exactly
    as a current's magnitude is.  (:mod:`._serve` measures the engine's own
    folding constant off the printed cell; see its
    ``EPSC_CONDUCTIVITY_FACTOR``.)
    """

    eps_r: float
    sigma: float
    eps_c: complex


@dataclass(frozen=True)
class RunData:
    """Everything a full printout needs that the deck does not already say.

    U4 builds one of these from a solve; the round-trip gate builds one by
    reading a captured printout.  Both hand it to :func:`render_printout`
    unchanged — that symmetry is what makes the gate a gate.
    """

    # -- the geometry summary, as its four inconsistently spaced labels ----
    node_count: int = 0
    wire_element_count: int = 0
    patch_element_count: int = 0
    unknown_count: int = 0
    # -- the operating point -----------------------------------------------
    frequency_mhz: float = 0.0
    wavelength_m: float = 0.0
    environment: str = ENVIRONMENT_FREE_SPACE
    # The medium under a finite-ground environment, and ``None`` under the two
    # that have none.  It is what picks the environment block's LAYOUT: the
    # two bannerless environments print their name centred, a finite one
    # prints four left-aligned lines (see :func:`_environment`).
    ground: GroundMedium | None = None
    # -- the tables --------------------------------------------------------
    loads: tuple[LoadRow, ...] = ()
    # One table, two row types — an ``NT`` card's admittance matrix and a
    # ``TL`` card's line.  No captured deck writes both (the corpus's three
    # mixed decks all stand over a ``GD`` ground and refuse a rung earlier),
    # so the heading a mixed table would carry is unobserved and the renderer
    # takes it from the first row.
    networks: tuple[NetworkRow | LineRow, ...] = ()
    network_excitation: tuple[PortRow, ...] = ()
    sources: tuple[PortRow, ...] = ()
    currents: tuple[WireCurrentRow, ...] = ()
    charges: tuple[ChargeRow, ...] = ()
    power: PowerBudget | None = None
    patterns: tuple[PatternBlock, ...] = ()
    # -- the timings -------------------------------------------------------
    #
    # Real seconds on the captured runs; the byte-gates normalize these lines
    # away (manifest ``normalizations``) because a timing is a property of
    # the machine, not of the answer.  Default 0.0 so a caller that does not
    # care prints the same zeros the fastest captures did.
    fill_seconds: float = 0.0
    factor_seconds: float = 0.0
    run_seconds: float = 0.0


# ==========================================================================
# the printout, section by section
# ==========================================================================
#
# Fortran field widths throughout.  Python's ``E`` format matches Fortran's
# ``1PE`` editing (one digit before the point, two exponent digits at these
# magnitudes), which is what every captured column uses; plain Fortran ``E``
# editing would print ``0.10000E+01`` and does not appear anywhere in the
# corpus.

_STRUCTURE_HEADING = " " * 33 + "- - - STRUCTURE SPECIFICATION - - -"
_COORDINATE_NOTE: tuple[str, ...] = (
    " " * 37 + "COORDINATES MUST BE INPUT IN",
    " " * 37 + "METERS OR BE SCALED TO METERS",
    " " * 37 + "BEFORE STRUCTURE INPUT IS ENDED",
)
_MODEL_COMMANDS = " Model definition commands:"

# Printed between the GE echo and the counts when GE asks for a ground plane
# (0019/0043/0044, ``GE 1,-1``); absent under ``GE 0,-1`` (every other
# capture).  The second GE field is -1 in all 49 decks, so whether it has any
# say in this note is unobserved.
_GROUND_PLANE_NOTE: tuple[str, ...] = (
    "   GROUND PLANE SPECIFIED.",
    "",
    "   WHERE WIRE ENDS TOUCH GROUND, CURRENT WILL BE INTERPOLATED TO IMAGE IN"
    " GROUND PLANE.",
    "",
    "",
)

# The counts block.  The ragged spacing before the colons is the engine's,
# not a transcription slip: "nodes" has a space before its colon, "patch
# elements" does not, and "unknowns" has lost the "of" entirely.
_COUNT_LABELS: tuple[str, ...] = (
    " Number of nodes :",
    " Number of wire elements :",
    " Number of patch elements:",
    " Number unknowns:",
)

_CARD_ECHO_PREFIX = " ***** INPUT LINE"

_FREQUENCY_HEADING = " " * 33 + "- - - - - - FREQUENCY - - - - - -"
_ENVIRONMENT_HEADING = " " * 34 + "- - - ANTENNA ENVIRONMENT - - -"

# Both observed bannerless environment names sit centred on the same column
# (44 blanks before ``FREE SPACE``, 42 before ``PERFECT GROUND``).  Centring
# in a 98-column field reproduces both; a 99-column field would too, and no
# odd-length banner has been captured to tell them apart.
_ENVIRONMENT_WIDTH = 98

# The finite-ground block is NOT centred — all four of its lines start in
# column 41 (0021/0022/0047/0048, measured), and centring the 35-character
# banner in the 98-column field above would start it in column 32.  Two
# different formats, so two different constants.
_MEDIUM_INDENT = " " * 40

_LOADING_HEADING = " " * 31 + "- - - STRUCTURE IMPEDANCE LOADING - - -"
_NOT_LOADED = " " * 35 + "THIS STRUCTURE IS NOT LOADED"
_LOADING_COLUMNS: tuple[str, ...] = (
    "       LOCATION          RESISTANCE   INDUCTANCE  CAPACITANCE       "
    "IMPEDANCE (OHMS)     CONDUCTIVITY    TYPE",
    "    ITAG FROM THRU          OHMS        HENRYS       FARADS        "
    "REAL      IMAGINARY    MHOS/METER",
)

# ALLOCATE CM is the one result line with nothing in front of it — every
# other line in the printout is indented at least one column — and the only
# integer in it is the matrix's element count: 100 = 10², 169 = 13²,
# 784 = 28², 8100 = 90² on the ten captures, i.e. unknowns squared every
# time.  Rendered from the unknown count so the identity cannot drift.
_ALLOCATE_LABEL = "ALLOCATE CM:"

_TIMING_HEADING = " " * 32 + "- - - MATRIX TIMING - - -"
_NETWORK_HEADING = " " * 44 + "- - - NETWORK DATA - - -"
_NETWORK_COLUMNS: tuple[str, ...] = (
    "      - FROM -    - TO -                          -  -  ADMITTANCE "
    "MATRIX ELEMENTS (MHOS)  -  -",
    "      TAG   SEG.   TAG  SEG.             (ONE,ONE)                   "
    "(ONE,TWO)                   (TWO,TWO)",
    "      NO.   NO.   NO.   NO.        REAL          IMAG.         REAL   "
    "       IMAG.         REAL          IMAG.",
)
# The ``TL`` form of the same table (0027, 0028).  Note the first heading row
# spells ``SEG.`` with one space less than the ``NT`` form does — the two
# headers were written separately in the engine and this transcription keeps
# both as they came.
_LINE_COLUMNS: tuple[str, ...] = (
    "      - FROM -    - TO -           TRANSMISSION LINE               "
    "-  -  SHUNT ADMITTANCES (MHOS)  -  -              LINE",
    "      TAG  SEG.   TAG  SEG.      IMPEDANCE      LENGTH            "
    "- END ONE -                 - END TWO -            TYPE",
    "      NO.   NO.   NO.   NO.         OHMS        METERS         REAL   "
    "       IMAG.         REAL          IMAG.",
)
_NETWORK_EXCITATION_HEADING = (
    " " * 27 + "- - - STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS - - -"
)
_ANTENNA_INPUT_HEADING = " " * 42 + "- - - ANTENNA INPUT PARAMETERS - - -"
_PORT_COLUMNS: tuple[str, ...] = (
    "   TAG   SEG.    VOLTAGE (VOLTS)         CURRENT (AMPS)         "
    "IMPEDANCE (OHMS)        ADMITTANCE (MHOS)      POWER",
    "   NO.   NO.    REAL        IMAG.       REAL        IMAG.       REAL"
    "        IMAG.       REAL        IMAG.     (WATTS)",
)

# Mixed case, both of them, and differently indented notes underneath — the
# case is why the scored matrix's first pass missed the charge block.
_CURRENTS_HEADING = " " * 39 + "- - - Wire Currents - - -"
_CURRENTS_NOTE = " " * 29 + "Lengths normalized by wavelength (or 2.*pi/CABS(k))"
_CURRENTS_COLUMNS: tuple[str, ...] = (
    "  Elem.  Tag            Coord. of seg. center          Seg.            "
    "- - - Current (Amps) - - -",
    "   No.   No.     X            Y            Z           Length       "
    "Real         Imag.        Mag.         Phase",
)
_CHARGES_HEADING = " " * 29 + "- - - Wire Charge Densities - - -"
_CHARGES_NOTE = " " * 21 + "Lengths normalized by wavelength (or 2.*pi/CABS(k))"
_CHARGES_COLUMNS: tuple[str, ...] = (
    "  ELEM.  TAG          Coord. of Seg. Center           Seg.        "
    "Charge Density (C/m)",
    "   No.   No.     X            Y            Z          Length        "
    "Mag.         Phase",
)

_POWER_HEADING = " " * 40 + "- - - POWER BUDGET - - -"
_POWER_INDENT = " " * 43
_POWER_LABEL_WIDTH = 14

_PATTERN_HEADING = " " * 48 + "- - - RADIATION PATTERNS - - -"
_PATTERN_COLUMNS: tuple[str, ...] = (
    "  - - ANGLES - -          - POWER GAINS -        - - - POLARIZATION "
    "- - -    - - - E(THETA) - - -    - - - E(PHI) - - -",
    "  THETA     PHI        VERT.   HOR.    TOTAL      AXIAL     TILT   "
    "SENSE     MAGNITUDE    PHASE      MAGNITUDE    PHASE ",
    " DEGREES  DEGREES       DB      DB      DB        RATIO     DEG.       "
    "        VOLTS     DEGREES       VOLTS     DEGREES",
)
_SENSE_WIDTH = 6

_RUN_TIME_LABEL = " RUN TIME ="

# Blank runs between sections.  Three after almost everything; FIVE after a
# radiation pattern (0010 and 0044 both leave six blank lines before the EN
# echo, against four in the patternless 0019/0014, and the echo itself owns
# one of them).
_SECTION_GAP = 3
_PATTERN_GAP = 5
_STRUCTURE_GAP = 4


def _e(value: float, width: int, digits: int) -> str:
    """One ``1PEw.d`` cell.  Python's ``E`` format is that format, byte for byte."""
    return f"{value:{width}.{digits}E}"


def _blank(count: int) -> list[str]:
    return [""] * count


def _post_ge_cards(deck_text: str) -> list[Card]:
    """Every card after ``GE``, re-tokenized from the deck's own lines.

    Deliberately NOT read off :class:`~momwire.deck._nec5.Nec5Deck`: the
    model normalizes as it parses (a zero ``RP`` count becomes 1, a node
    address becomes a decoded index) and the echo has to reproduce what was
    READ, field for field.  The echo is a card image, so it comes from the
    card image.
    """
    cards: list[Card] = []
    seen_ge = False
    for line in deck_text.splitlines():
        card = parse_card(line)
        if card is None:
            continue
        if not seen_ge:
            seen_ge = card.mnemonic == "GE"
            continue
        cards.append(card)
        if card.mnemonic == "EN":
            break
    return cards


def _card_echo(index: int, card: Card) -> str:
    """``***** INPUT LINE  n  XX`` — four integer fields, then six reals.

    Widths are the captures': the line number is I3, the first integer field
    I4 and the other three I5, then 6E13.5.  Six, not ten: 0012's ``NT`` echo
    is 121 columns wide, which is exactly four integers and six reals, and
    every ten-field card in the corpus (``NT``, ``TL``, ``NE``) puts its
    remaining fields in those six reals.
    """
    ints = "".join(f"{card.i(k):{4 if k == 0 else 5}d}" for k in range(4))
    reals = "".join(_e(card.f(4 + k), 13, 5) for k in range(6))
    return f"{_CARD_ECHO_PREFIX}{index:3d}  {card.mnemonic}{ints}{reals}"


def _structure_specification(deck: Nec5Deck, data: RunData) -> list[str]:
    """The heading, the geometry echo, and the four counts.

    Geometry echoes come from the deck's own ``GW``/``GE`` cards and use
    their own widths — ``GW`` prints I5 then I6 then 7E13.5, ``GE`` prints
    two I5 fields — which is a different card echo from the one every
    post-``GE`` card gets, and is why they are rendered here rather than
    through :func:`_card_echo`.
    """
    lines = [_STRUCTURE_HEADING, "", *_COORDINATE_NOTE, "", _MODEL_COMMANDS, ""]
    for wire in deck.wires:
        coordinates = "".join(
            _e(v, 13, 5) for v in (*wire.end1, *wire.end2, wire.radius)
        )
        lines.append(f" GW{wire.tag:5d}{wire.segment_count:6d}{coordinates}")
    lines.append(f" GE{deck.ge_flag:5d}{deck.ge_second:5d}")
    lines.append("")
    if deck.ge_flag != 0:
        lines += _GROUND_PLANE_NOTE
    counts = (
        data.node_count,
        data.wire_element_count,
        data.patch_element_count,
        data.unknown_count,
    )
    lines += [
        f"{label}{count:6d}" for label, count in zip(_COUNT_LABELS, counts, strict=True)
    ]
    return lines


def _frequency(data: RunData) -> list[str]:
    return [
        _FREQUENCY_HEADING,
        "",
        " " * 36 + "FREQUENCY=" + _e(data.frequency_mhz, 11, 4) + " MHZ",
        " " * 36 + "WAVELENGTH=" + _e(data.wavelength_m, 11, 4) + " METERS",
    ]


def _environment(data: RunData) -> list[str]:
    """The heading, one blank, and the environment — in one of its two forms.

    ONE blank under the heading in both, which is the reading the captures
    force once the ``SOMMPD.NEX`` cache blocks are taken out of them.  A
    finite-ground capture appears to carry two (0047 lines 71-72), but each
    cache block prints its OWN leading blank and there are two blocks in a
    two-frequency run — where the count would otherwise have to be three.
    Dropping "a blank and the cache lines that follow it" leaves every
    captured environment block with exactly one blank under its heading, which
    is also what ``FREE SPACE`` and ``PERFECT GROUND`` print.  The seam emits
    no cache block at all: it never reads and never writes that file.

    The three medium cells are the captures' own formats, measured against the
    linux oracle across four media (ε 1.5/5/13/80/100, σ 1.234E-6 to 12.345):
    ``F7.3`` for the relative constant (``100.000`` fills the field and
    ``  5.000`` pads it), ``E10.3`` for the conductivity, and two glued
    ``E12.5`` cells for the complex constant — whose imaginary part carries no
    ``j`` and wears its sign against the mantissa.
    """
    if data.ground is None:
        return [
            _ENVIRONMENT_HEADING,
            "",
            data.environment.center(_ENVIRONMENT_WIDTH).rstrip(),
        ]
    medium = data.ground
    return [
        _ENVIRONMENT_HEADING,
        "",
        _MEDIUM_INDENT + data.environment,
        _MEDIUM_INDENT + f"RELATIVE DIELECTRIC CONST.={medium.eps_r:7.3f}",
        _MEDIUM_INDENT + f"CONDUCTIVITY={medium.sigma:10.3E} MHOS/METER",
        _MEDIUM_INDENT
        + "COMPLEX DIELECTRIC CONSTANT="
        + _e(medium.eps_c.real, 12, 5)
        + _e(medium.eps_c.imag, 12, 5),
    ]


def _loading(data: RunData) -> list[str]:
    """The loading table, or the one line that says there is none.

    ``ALLOCATE CM:`` follows immediately, with no blank line between — it is
    printed by whatever allocates the matrix rather than by this section, and
    lands here in all ten captures.
    """
    lines = [_LOADING_HEADING, ""]
    if not data.loads:
        lines.append(_NOT_LOADED)
    else:
        lines += ["", *_LOADING_COLUMNS, ""]
        lines += [_load_row(row) for row in data.loads]
    lines.append(f"{_ALLOCATE_LABEL}{data.unknown_count**2:15d}")
    return lines


def _load_row(row: LoadRow) -> str:
    """One loading row, laid out as a canvas because most of it is blank.

    Measured on 0012: the three integers end in columns 8/13/18, the
    impedance's REAL cell is an E11.4 ending in column 73, and the type text
    starts in column 102 and is blank-padded to 16 (``FIXED IMPEDANCE `` —
    the trailing blank is real).

    The IMAGINARY cell is INFERRED and no capture fills it: its width is
    REAL's, and its right edge is put two columns past its own header the way
    REAL's sits two columns past ``REAL``.  The first printout that carries a
    reactive load is the line that corrects it.
    """
    canvas = [" "] * 118

    def place(text: str, end: int) -> None:
        canvas[end - len(text) : end] = list(text)

    place(f"{row.tag:d}", 8)
    place(f"{row.node_from:d}", 13)
    place(f"{row.node_thru:d}", 18)
    place(_e(row.resistance, 11, 4), 73)
    if row.reactance is not None:
        place(_e(row.reactance, 11, 4), 88)
    place(row.kind.ljust(16), 118)
    return "".join(canvas)


def _timing(data: RunData) -> list[str]:
    return [
        _TIMING_HEADING,
        "",
        " " * 24
        + f"FILL={data.fill_seconds:9.3f}"
        + f" SEC.,  FACTOR={data.factor_seconds:9.3f} SEC.",
    ]


def _port_row(row: PortRow) -> str:
    """A row of either port table: I4, I6, I2, then nine E12.4 cells."""
    cells = (
        row.voltage.real,
        row.voltage.imag,
        row.current.real,
        row.current.imag,
        row.impedance.real,
        row.impedance.imag,
        row.admittance.real,
        row.admittance.imag,
        row.power,
    )
    return f"{row.tag:4d}{row.segment:6d}{row.end_index:2d}" + "".join(
        _e(value, 12, 4) for value in cells
    )


def _network_row(row: NetworkRow | LineRow) -> str:
    """One ``NETWORK DATA`` row, either form.

    I9 then three I6, then ONE blank column, then six E14.4 cells: the gap
    between the addresses and the payload is a column wider than the gaps
    inside it (0012, where ``13`` ends in column 27 and the first admittance
    in column 42, against 14 for every cell after it).  Both forms share every
    one of those widths — the ``TL`` form adds a three-space gap and its line
    type, and nothing else (0027 against 0012, column for column).
    """
    if isinstance(row, LineRow):
        cells = (
            row.z0,
            row.length_m,
            row.shunt_a.real,
            row.shunt_a.imag,
            row.shunt_b.real,
            row.shunt_b.imag,
        )
        tail = "   " + row.line_type
    else:
        cells = (
            row.y11.real,
            row.y11.imag,
            row.y12.real,
            row.y12.imag,
            row.y22.real,
            row.y22.imag,
        )
        tail = ""
    return (
        f"{row.tag_from:9d}{row.segment_from:6d}"
        f"{row.tag_to:6d}{row.segment_to:6d} "
        + "".join(_e(value, 14, 4) for value in cells)
        + tail
    )


def _network_data(data: RunData) -> list[str]:
    """The table, under whichever of its two column headers the rows want.

    The header is taken from the FIRST row because no captured deck mixes the
    two forms and so no capture says what a mixed table's heading looks like
    (:mod:`._serve` refuses a mixed deck rather than guess).
    """
    columns = (
        _LINE_COLUMNS if isinstance(data.networks[0], LineRow) else _NETWORK_COLUMNS
    )
    return [_NETWORK_HEADING, "", *columns, *(_network_row(r) for r in data.networks)]


def _currents(data: RunData) -> list[str]:
    lines = [_CURRENTS_HEADING, "", _CURRENTS_NOTE, "", *_CURRENTS_COLUMNS]
    for row in data.currents:
        cells = (*row.center, row.length, row.real, row.imag, row.magnitude)
        lines.append(
            f"{row.element:7d}{row.tag:5d}"
            + "".join(_e(value, 13, 5) for value in cells)
            + f"{row.phase_deg:10.4f}"
        )
    return lines


def _charges(data: RunData) -> list[str]:
    lines = [_CHARGES_HEADING, "", _CHARGES_NOTE, "", *_CHARGES_COLUMNS]
    for row in data.charges:
        cells = (*row.center, row.length, row.magnitude)
        lines.append(
            f"{row.element:7d}{row.tag:5d}"
            + "".join(_e(value, 13, 5) for value in cells)
            + f"{row.phase_deg:10.4f}"
        )
    return lines


def _power_budget(power: PowerBudget) -> list[str]:
    def watts(label: str, value: float) -> str:
        return (
            _POWER_INDENT
            + label.ljust(_POWER_LABEL_WIDTH)
            + "="
            + _e(value, 11, 4)
            + " WATTS"
        )

    lines = [
        _POWER_HEADING,
        "",
        watts("INPUT POWER", power.input_power),
        watts("RADIATED POWER", power.radiated_power),
        watts("WIRE LOSS", power.wire_loss),
    ]
    if power.network_loss is not None:
        lines.append(watts("NETWORK LOSS", power.network_loss))
    lines.append(
        _POWER_INDENT
        + "EFFICIENCY".ljust(_POWER_LABEL_WIDTH)
        + "="
        + f"{power.efficiency_percent:7.2f}"
        + " PERCENT"
    )
    return lines


def _pattern(block: PatternBlock) -> list[str]:
    """One ``RP`` answer.  The 3-D trailer prints only when it is carried."""
    lines = [_PATTERN_HEADING, "", *_PATTERN_COLUMNS]
    for row in block.rows:
        lines.append(
            f"{row.theta_deg:8.2f}{row.phi_deg:9.2f}"
            f"{row.vert_db:11.2f}{row.hor_db:8.2f}{row.total_db:8.2f}"
            f"{row.axial_ratio:11.5f}{row.tilt_deg:9.2f}"
            + "  "
            + row.sense.ljust(_SENSE_WIDTH)
            + _e(row.e_theta_magnitude, 15, 5)
            + f"{row.e_theta_phase_deg:9.2f}"
            + _e(row.e_phi_magnitude, 15, 5)
            + f"{row.e_phi_phase_deg:9.2f}"
        )
    if block.average_power_gain is not None:
        lines += [
            "",
            "",
            "   AVERAGE POWER GAIN="
            + _e(block.average_power_gain, 12, 5)
            + "       SOLID ANGLE USED IN AVERAGING=("
            + f"{block.solid_angle_pi:7.4f}"
            + ")*PI STERADIANS.",
            "",
            "   POWER RADIATED ASSUMING RADIATION INTO 4*PI STERADIANS ="
            + _e(block.power_radiated_4pi, 12, 5)
            + " WATTS",
        ]
    return lines


def render_printout(deck: Nec5Deck, data: RunData) -> str:
    """A complete NEC-5 printout: U1's header, then everything through ``RUN TIME``.

    ``deck`` supplies what the printout ECHOES (its comment block, its
    geometry cards, its post-``GE`` card images); ``data`` supplies every
    number the run produced.  Nothing is computed here.

    Section order and the blank lines between sections are the captures',
    and the last card echo is the odd one out: ``EN`` is echoed AFTER the
    results, with the line number it would have had at the top, because the
    engine reads it only once the run it terminates is over.
    """
    cards = _post_ge_cards(deck.source_text)
    terminator = cards[-1] if cards and cards[-1].mnemonic == "EN" else None
    body: list[str] = []
    body += _structure_specification(deck, data)
    body += _blank(_STRUCTURE_GAP)
    body += [
        _card_echo(index, card)
        for index, card in enumerate(cards, start=1)
        if card is not terminator
    ]
    body += _blank(_STRUCTURE_GAP)
    body += _frequency(data)
    body += _blank(_SECTION_GAP)
    body += _environment(data)
    body += _blank(_SECTION_GAP)
    body += _loading(data)
    body += _blank(_SECTION_GAP)
    body += _timing(data)
    body += _blank(_SECTION_GAP)
    if data.networks:
        body += _network_data(data)
        body += _blank(_SECTION_GAP)
    if data.network_excitation:
        body += [_NETWORK_EXCITATION_HEADING, "", *_PORT_COLUMNS]
        body += [_port_row(row) for row in data.network_excitation]
        body += _blank(_SECTION_GAP)
    if data.sources:
        body += [_ANTENNA_INPUT_HEADING, "", *_PORT_COLUMNS]
        body += [_port_row(row) for row in data.sources]
        body += _blank(_SECTION_GAP)
    if data.currents:
        body += _currents(data)
        body += _blank(_SECTION_GAP)
    if data.charges:
        body += _charges(data)
        body += _blank(_SECTION_GAP)
    if data.power is not None:
        body += _power_budget(data.power)
        body += _blank(_SECTION_GAP)
    for block in data.patterns:
        body += _pattern(block)
        body += _blank(_PATTERN_GAP)
    body.append("")
    if terminator is not None:
        body.append(_card_echo(len(cards), terminator))
    body += ["", f"{_RUN_TIME_LABEL}{data.run_seconds:10.3f}"]
    return render_header(deck.source_text) + "\n".join(body) + "\n"
