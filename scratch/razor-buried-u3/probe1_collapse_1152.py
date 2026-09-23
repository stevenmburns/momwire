"""momwire#1152: eps~=1 collapse per block on near-plane detached decks."""

import functools
import json

from common import collapse, detached, detached_hub

for gap in (0.01, 0.05, 0.1, 0.3):
    for dx in (0.0, 0.5):
        r = collapse(functools.partial(detached, 1, gap=gap, dx=dx))
        print(json.dumps(dict(deck="u1", gap=gap, dx=dx, rel=r)), flush=True)
    if gap < 0.15:
        r = collapse(functools.partial(detached_hub, 1, gap=gap))
        print(json.dumps(dict(deck="hub", gap=gap, rel=r)), flush=True)
