"""Figures for Act IV chapter 13 — "Arrays know their own symmetry".

Run from the repo root:  .venv/bin/python site/figures/fig_ch13.py
(schematic block structure + block-count dedup quoted from array_block.py / the
array_block_solver_plan design measurements.)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from style import AMBER, CYAN, GREEN, INK, MUTED, RED, save

# color per dedup key (self, then one per |displacement|)
_PALETTE = [RED, CYAN, AMBER, GREEN, "#b48ead", "#88c0d0"]


def fig_blocks():
    """A P-element array's Z is a P×P grid of blocks. Identical elements and
    free-space translation invariance make blocks with the same (shape,
    displacement) identical — so each diagonal band is computed once."""
    P = 5
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for i in range(P):
        for j in range(P):
            key = abs(i - j)  # 0 = self-block; k = coupling at displacement k
            color = _PALETTE[key % len(_PALETTE)]
            ax.add_patch(Rectangle((j, P - 1 - i), 1, 1, facecolor=color,
                                   edgecolor=INK, lw=1.5, alpha=0.85))
            label = "self" if key == 0 else f"±{key}"
            ax.text(j + 0.5, P - 1 - i + 0.5, label, ha="center", va="center",
                    fontsize=9, color=INK)
    ax.set_xlim(0, P)
    ax.set_ylim(0, P)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(P) + 0.5)
    ax.set_xticklabels([f"e{j}" for j in range(P)])
    ax.set_yticks(np.arange(P) + 0.5)
    ax.set_yticklabels([f"e{i}" for i in range(P - 1, -1, -1)])
    ax.tick_params(length=0)
    ax.set_title("Array Z is block-Toeplitz:\nsame color = computed once", fontsize=12)
    save(fig, "ch13-blocks")


def fig_dedup():
    """The collapse: P(P−1) coupling blocks dedup to a handful of unique
    (shape, shape, displacement) keys (measured, array_block.py P3)."""
    names = ["uniform\nlinear (P=4)", "invvee\narray", "bowtie\narray 2×4"]
    pairs = [12, 12, 56]
    unique = [3, 5, 13]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.barh(y + 0.2, pairs, height=0.36, color=MUTED, label="all coupling pairs")
    ax.barh(y - 0.2, unique, height=0.36, color=GREEN, label="unique blocks to compute")
    for yi, (p, u) in enumerate(zip(pairs, unique)):
        ax.text(p + 1, yi + 0.2, str(p), va="center", fontsize=9, color=MUTED)
        ax.text(u + 1, yi - 0.2, str(u), va="center", fontsize=9, color=GREEN)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("number of coupling blocks")
    ax.set_xlim(0, 62)
    ax.set_title("Translation invariance collapses the work to a few unique blocks")
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "ch13-dedup")


if __name__ == "__main__":
    fig_blocks()
    fig_dedup()
