"""The one strict-long sweep node spot2.py does not cover.

python spot3.py      (writes spot3.jsonl)
"""

import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import spot  # noqa: E402

if __name__ == "__main__":
    row = spot.one(("salt28", 0.02, 0.0125, "strict-long"))
    with open(__file__.rsplit("/", 1)[0] + "/spot3.jsonl", "w") as fh:
        fh.write(json.dumps(row) + "\n")
    print({k: row[k] for k in row if k != "ref"}, flush=True)
