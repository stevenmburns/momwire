from deck import solver

for name in ("sym4", "hex6"):
    s, doc = solver(name, True)
    m = s._rotational_map
    g = s._build_geometry()
    print(
        name,
        "segs",
        g["n_segs"],
        "N",
        m.n_sectors,
        "axial",
        m.axial,
        "rev",
        sum(map(sum, m.reversed)),
        "ports",
        len(s.feeds),
    )
