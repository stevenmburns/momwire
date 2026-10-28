momwire-nec5-razor2p-linux: momwire's razor-2p NEC-5 engine as one native
Linux program, with no Python in it (momwire#1512).

It answers a NEC-5 deck the way the Python command `momwire-nec5-razor-2p`
does, byte for byte, without installing Python or momwire:

    ./momwire-nec5-razor2p-linux DECK.nec PRINTOUT.out   (EZNEC's and SimNEC's form)
    ./momwire-nec5-razor2p-linux - -                     (SimNEC's streaming form)
    ./momwire-nec5-razor2p-linux -version

To use it from SimNEC, point the NEC-5 engine setting at
momwire-nec5-razor2p-linux in this folder. Keep the three programs together:
momwire-nec5-razor2p-linux picks momwire-nec5-razor2p-linux-avx2 on a
processor with AVX2 and FMA, momwire-nec5-razor2p-linux-sse2 on any other
64-bit x86 processor.

Needs: 64-bit x86 Linux with glibc 2.28 or newer (2018 onwards: RHEL/Alma 8,
Debian 10, Ubuntu 18.10 and later). Everything else is built in.

What it does not serve yet is refused in the printout, by name ("not served
by this build: ..."); the Python command serves those decks.

momwire-nec5-bs2-linux is the same kind of program for momwire's default
B-spline formulation (bs2): it answers a deck as the Python command
`momwire-nec5-bspline` does, byte for byte, and takes the same three forms
and the same picking (momwire-nec5-bs2-linux-avx2 / -sse2 beside it). It
serves free space, wire loading, networks, the extended kernel, wires above
a perfect or Sommerfeld ground, a structure wholly below a Sommerfeld
ground, and wires on both sides of it (joined through a crossing junction,
or a raised antenna over a separate buried counterpoise).

PREVIEW: momwire-nec42-sinusoidal-linux is the same kind of program for
EZNEC's and SimNEC's NEC-4.2 slot (momwire#1607). It answers a NEC-4.2
deck as the Python command `momwire-nec42-sinusoidal` does, byte for byte
apart from the two lines that report measured time (FILL= ... SEC. and
RUN TIME =), and takes the same three forms and the same picking
(momwire-nec42-sinusoidal-linux-avx2 / -sse2 beside it). Point the NEC-4.2
engine setting at it; `nec42` in the name is what makes SimNEC write it
NEC-4.2 decks. It is a preview: it serves free space, a perfect ground
(GN 1), a Sommerfeld ground (GN 2 / GN 3) with every wire above it,
buried wires (wholly below it, joined to the wires above through a
crossing junction, or beside them with no junction, as a raised antenna
over a separate buried counterpoise is), a GD far-field ground (the second
medium's cliff), wire loading, networks (TL / NT), helices and near
fields, and it still refuses by name, for now:
    - buried wires beside raised ones with no crossing junction when the
      wires carry more than one radius (momwire#1718 S12);
    - an insulated wire (an IS card, momwire#1607 B5b).
The Python command serves those decks.

PREVIEW: momwire-nec42-sinusoidal-galerkin-linux and
momwire-nec5-sinusoidal-galerkin-linux are the same kind of program for
momwire's Sinusoidal-Galerkin formulation (momwire#1718), the native twins
of the Python commands `momwire-nec42-sinusoidal-galerkin` (EZNEC's and
SimNEC's NEC-4.2 slot) and `momwire-nec5-sinusoidal-galerkin` (SimNEC's
NEC-5 engine). Each answers a deck as its Python command does, byte for
byte apart from the two lines that report measured time, and takes the
same three forms and the same picking (-avx2 / -sse2 beside it). They
serve free space, a perfect ground (GN 1), a Sommerfeld ground (GN 2 /
GN 3) with every wire above it or a wire end on it, a structure wholly
below a Sommerfeld ground, a wire through the ground at a crossing
junction, wire loading, networks (TL / NT), node gaps and the extended
kernel, and still refuse by name, for now:
    - a deck with wires on both sides of the ground and no crossing
      junction (momwire#1718 S12).
The Python commands serve those decks.
