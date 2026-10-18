momwire-nec5-macos-arm64: two of momwire's NEC-5 engines as native macOS
programs for Apple silicon, with no Python in them (momwire#1559). This is a
preview release asset.

    ./momwire-nec5-razor2p-macos DECK.nec PRINTOUT.out   (EZNEC's and SimNEC's form)
    ./momwire-nec5-razor2p-macos - -                     (SimNEC's streaming form)
    ./momwire-nec5-razor2p-macos -version
    ./momwire-nec5-bs2-macos DECK.nec PRINTOUT.out       (the same forms, bs2)

They answer a deck the way the Python commands `momwire-nec5-razor-2p` and
`momwire-nec5-bspline` do on the same Mac, and need nothing installed: the
math library is the system's Accelerate, and the OpenMP runtime is built in.
There are no libraries beside them.

They are not signed with an Apple Developer ID or notarised, only ad-hoc
signed. Fetched with curl and unpacked with tar, they carry no quarantine
flag and run as they are:

    curl -LO <URL of momwire-nec5-macos-arm64.tar.gz>
    tar -xzf momwire-nec5-macos-arm64.tar.gz
    ./momwire-nec5-macos-arm64/momwire-nec5-razor2p-macos -version

A copy downloaded through a web browser is quarantined, and macOS then
refuses to open the programs. Clear the flag from the whole folder once:

    xattr -dr com.apple.quarantine momwire-nec5-macos-arm64

A deck an engine does not serve yet is refused in the printout by name
("not served by this build: ..."); the Python command serves it.
