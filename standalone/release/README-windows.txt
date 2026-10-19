momwire-nec5-windows-x64 (PREVIEW): two of momwire's NEC-5 engines as native
Windows programs, with no Python in them (momwire#1558).

This is a preview. The drop-in, momwire-windows.zip, is still the
supported way to run momwire under EZNEC, SimNEC or 4nec2 on
Windows.

The engines are razor-2p (what the Python command `momwire-nec5-razor-2p`
answers) and bs2, momwire's default B-spline formulation (what
`momwire-nec5-bspline` answers). Each takes the forms the real engine takes:

    momwire-nec5-razor2p-avx2.exe DECK.nec PRINTOUT.out   (EZNEC's and SimNEC's form)
    momwire-nec5-razor2p-avx2.exe - -                     (SimNEC's streaming form)
    momwire-nec5-razor2p-avx2.exe -version

WHICH PROGRAM TO USE

  Windows 10 or 11, 64-bit, on a processor with AVX2 (Intel from 2013's
  Haswell on, except some Celeron and Pentium models; AMD from 2015's
  Excavator on, every Ryzen):
      momwire-nec5-razor2p-avx2.exe    momwire-nec5-bs2-avx2.exe

  Windows 10 or 11, 64-bit, on any other 64-bit x86 processor:
      momwire-nec5-razor2p-sse2.exe    momwire-nec5-bs2-sse2.exe

  Windows 7 SP1, 64-bit, on any 64-bit x86 processor: the win7 folder,
      win7\momwire-nec5-razor2p-sse2.exe    win7\momwire-nec5-bs2-sse2.exe

If you are not sure, use the sse2 program: it runs on every 64-bit x86
processor, only more slowly. An avx2 program on a processor without AVX2 does
not crash; it refuses each deck in the printout, saying so.

Keep each folder together. The programs load the DLLs beside them: SciPy's
OpenBLAS (libscipy_openblas.dll), the Microsoft Visual C++ runtime and LLVM's
OpenMP runtime. Nothing needs to be installed. The win7 folder also carries
Microsoft's Universal C Runtime for Windows 7. On Windows 10 and 11 Windows
uses its own copy, so the win7 programs run there too.

To use one from EZNEC or SimNEC, point the NEC-5 engine setting at the
program you chose, in this folder.

A deck an engine does not serve yet is refused in the printout by name
("not served by this build: ..."). The Python commands, or the EZNEC drop-in,
serve it. On Windows the printouts match the Python commands' to the printed
digit, apart from round-off in values that are zero in exact arithmetic.
