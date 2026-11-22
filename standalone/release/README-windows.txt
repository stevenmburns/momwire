momwire-nec5-windows-x64: two of momwire's NEC-5 engines as native
Windows programs, with no Python in them (momwire#1558).

They are one of two supported ways to run momwire under EZNEC, SimNEC or
4nec2 on Windows; the other is the Python drop-in, momwire-windows.zip.
Both are built from each momwire release and tested on every change: these
programs run every test deck on Windows (avx2 and sse2) and their printouts
are checked against the Python engine's.

The programs:

    momwire-nec5-razor2p.exe     nec5-razor2p (the Python command
                                 `momwire-nec5-razor2p`)
    momwire-nec5-bs2.exe         nec5-bs2, momwire's default B-spline
                                 formulation (`momwire-nec5-bs2`)

Each program holds an avx2 and an sse2 build and picks, when it starts, the
one your processor can run: avx2 on a processor with AVX2 and FMA (Intel from
2013's Haswell on, except some Celeron and Pentium models; AMD from 2015's
Excavator on, every Ryzen), sse2 on any other 64-bit x86 processor. So there
is one program per engine and nothing to choose. Each takes the forms the
real engine takes:

    momwire-nec5-razor2p.exe DECK.nec PRINTOUT.out   (EZNEC's and SimNEC's form)
    momwire-nec5-razor2p.exe - -                     (SimNEC's streaming form)
    momwire-nec5-razor2p.exe -version

The -version line is the Python command's with "-standalone" and the build
that answered added to the name, e.g.

    NEC5 momwire-nec5-razor2p-standalone-avx2 [sb] 0.97.0

so SimNEC (and the "CM version" card it puts on each deck) shows which
build ran. Line 2 of each printout names it too.

WHICH FOLDER TO USE

  Windows 10 or 11, 64-bit: the programs at the top of this folder.

  Windows 7 SP1, 64-bit: the win7 folder, the same two programs beside
  Microsoft's Universal C Runtime for Windows 7. On Windows 10 and 11
  Windows uses its own copy, so the win7 programs run there too.

Keep each folder together. The programs load the DLLs beside them: SciPy's
OpenBLAS (libscipy_openblas.dll), the Microsoft Visual C++ runtime and LLVM's
OpenMP runtime. Nothing needs to be installed.

To use one from EZNEC or SimNEC, point the NEC-5 engine setting at the
program, in this folder.

NAMES CHANGED IN 0.97.0.  Up to 0.96.0 each engine came twice, one program
per processor build, and you chose between them:

  DEPRECATED momwire-nec5-razor2p-avx2.exe       -> momwire-nec5-razor2p.exe
  DEPRECATED momwire-nec5-razor2p-sse2.exe       -> momwire-nec5-razor2p.exe
  DEPRECATED momwire-nec5-bs2-avx2.exe           -> momwire-nec5-bs2.exe
  DEPRECATED momwire-nec5-bs2-sse2.exe           -> momwire-nec5-bs2.exe
  DEPRECATED win7\momwire-nec5-razor2p-sse2.exe  -> win7\momwire-nec5-razor2p.exe
  DEPRECATED win7\momwire-nec5-bs2-sse2.exe      -> win7\momwire-nec5-bs2.exe

The old names still ship, as copies of the new programs, so an engine path
pointed at one keeps working for now. A copy runs the build its name names,
as before: an -sse2 name runs sse2 even on an AVX2 processor, and an -avx2
name on a processor without AVX2 refuses each deck in the printout, saying
why. Line 2 of its printout ends in "(renamed from momwire-nec5-bs2-sse2)"
or the like. The old names will stop shipping in a later release: change
the engine path to the new name, which picks the faster build by itself.

A deck an engine does not serve yet is refused in the printout by name
("not served by this build: ..."). The Python commands, or the Python drop-in
engine (momwire-windows.zip), serve it. On Windows the printouts match the
Python commands' to the printed digit, apart from round-off in values that
are zero in exact arithmetic.
