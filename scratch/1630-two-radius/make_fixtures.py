"""Write tests/fixtures/sg_two_radius_1630/: the synthetic 1-radial hub decks
and their licensed printouts (NEC-5 x13 nec5cl, NEC-4.2 nec42cl-static),
black box. Run from the momwire repo root on the box with the references."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "scratch/1630-two-radius")
import decks
from ref_ladder import NEC5, NEC42

OUT = Path("tests/fixtures/sg_two_radius_1630")
OUT.mkdir(parents=True, exist_ok=True)
R = 2
JOBS = {
    # NEC-5: the radius-response differential's three configs
    "hub1_all1mm.nec5": (decks.nec5, 0.001, 0.001, NEC5),
    "hub1_v19mm.nec5": (decks.nec5, 0.019, 0.001, NEC5),
    "hub1_b025mm.nec5": (decks.nec5, 0.001, 0.00025, NEC5),
    # the SAME decks in the NEC-4.2 dialect (momwire reads these)
    "hub1_all1mm.nec42": (decks.nec42, 0.001, 0.001, NEC42),
    "hub1_v19mm.nec42": (decks.nec42, 0.019, 0.001, NEC42),
    "hub1_b025mm.nec42": (decks.nec42, 0.001, 0.00025, NEC42),
    # NEC-4.2's step discontinuity at the crossing node
    "hub1_v1p1mm.nec42": (decks.nec42, 0.0011, 0.001, NEC42),
}
for stem, (mk, a_v, a_b, exe) in JOBS.items():
    text = mk("hub", R, "mid", a_v=a_v, a_b=a_b, n_rad=1)
    (OUT / f"{stem}.nec").write_text(text)
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "in.nec").write_text(text)
        subprocess.run(
            [str(exe), "in.nec", "out.nec"], cwd=td, check=True, capture_output=True
        )
        shutil.copy(Path(td) / "out.nec", OUT / f"{stem}.out")
    print(stem, (OUT / f"{stem}.out").stat().st_size)
