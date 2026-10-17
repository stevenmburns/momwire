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
by this build: ..."); the Python command serves those decks. Only the
razor-2p formulation is in this program; the other bases stay Python.
