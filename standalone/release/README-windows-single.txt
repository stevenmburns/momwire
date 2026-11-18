momwire-nec5-windows-x64-single (PREVIEW): two of momwire's NEC-5 engines as
single-file native Windows programs, with no Python in them.

This is a preview, shipped beside momwire-nec5-windows-x64.zip, which stays
the default. Use that zip unless you want one file with nothing beside it.

The programs:

    momwire-nec5-razor-2p.exe    nec5-razor-2p (the Python command
                                 `momwire-nec5-razor-2p`)
    momwire-nec5-bspline.exe     nec5-bspline, momwire's default B-spline
                                 formulation (`momwire-nec5-bspline`)

Each file holds an avx2 and an sse2 build and picks the one your processor can
run, so there is one program per engine. Nothing else needs to sit beside it.
Each takes the forms the real engine takes:

    momwire-nec5-razor-2p.exe DECK.nec PRINTOUT.out   (EZNEC's and SimNEC's form)
    momwire-nec5-razor-2p.exe - -                     (SimNEC's streaming form)
    momwire-nec5-razor-2p.exe -version

The -version line is the Python command's with "-standalone", the build that
answered and "-ucrt64" added to the name, e.g.

    NEC5 momwire-nec5-razor-2p-standalone-avx2-ucrt64 [sb] 0.95.0

so SimNEC (and the "CM version" card it puts on each deck) shows that this
preview answered, and not the programs in momwire-nec5-windows-x64.zip.

WHAT IT NEEDS

  64-bit Windows 10 or 11: nothing to install.

  Windows 7 SP1 or 8.1, 64-bit: the Universal C Runtime, which Windows does
  not always have there. Install update KB2999226 or any Visual C++ 2015 or
  later redistributable (x64). This zip has no win7 folder; the win7 folder of
  momwire-nec5-windows-x64.zip carries that runtime with it.

HOW IT DIFFERS FROM momwire-nec5-windows-x64.zip

The programs here are built with a different compiler (GCC, MSYS2 UCRT64) and
linked to a different math library. They are tested on the same decks and
held to the same tolerance against the Python engine, but their speed differs
from deck to deck: some decks run faster here and some slower. If speed
matters on your deck, time both.

To use one from EZNEC or SimNEC, point the NEC-5 engine setting at the
program.

A deck an engine does not serve yet is refused in the printout by name
("not served by this build: ..."). The Python commands, or the Python drop-in
engine (momwire-windows.zip), serve it. On Windows the printouts match the
Python commands' to the printed digit, apart from round-off in values that are
zero in exact arithmetic.

THIRD-PARTY-NOTICES.txt lists the libraries linked into these programs and
their licences.
