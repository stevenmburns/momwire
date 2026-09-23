"""Is razor's buried_serve_refusal EXACT against its own fill? Sweep the
two-node separation out to where razor refuses, and compare the pre-flight
verdict with what the fill does (raises which sentence, or serves)."""

import json

from probe3_multi_node import rz, two

for m in (1, 2, 4):
    for sep in (12.0, 24.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0):
        d = two(m, sep)
        s = rz(d)
        pre = s.buried_serve_refusal()
        try:
            s._assemble_Z(s._build_geometry(), s.k)
            fill = None
        except (ValueError, NotImplementedError) as e:
            fill = str(e)
        print(
            json.dumps(
                dict(
                    m=m,
                    sep=sep,
                    pre=None if pre is None else pre[:70],
                    fill=None if fill is None else fill[:70],
                    same=(pre == fill),
                )
            ),
            flush=True,
        )
