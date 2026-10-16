"""spot.py's routes at the nodes where sweep.py's strict rule walked long
(its value is then a long walk, not a referee): mpmath decides.

    python spot2.py [nproc]      (writes spot2.jsonl)
"""

import json
import sys
from multiprocessing import Pool

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import spot  # noqa: E402

NODES = [
    ("salt1.8", 0.02, 0.005, "strict-long"),
    ("dan", 0.02, 0.01, "strict-long"),
    ("salt1.8", 0.02, 0.01, "strict-long"),
    ("dan", 0.02, 0.0125, "strict-long"),
    ("salt28", 0.02, 0.04174882894334361, "strict-long"),
    ("salt28", 0.02, 0.4219933440435089, "strict-long"),
]

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with Pool(n) as p, open(__file__.rsplit("/", 1)[0] + "/spot2.jsonl", "w") as fh:
        for row in p.imap_unordered(spot.one, NODES):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print({k: row[k] for k in row if k not in ("ref",)}, flush=True)
