"""Shared matplotlib style for the primer figures.

Instrument-panel look, matching the site's "Blueprint" dark palette
(site/src/styles/custom.css) — dark stage, light ink, signal-cyan accent.
The dark background is deliberate: figures read as panels on both the
light and dark site themes.

Every figure script imports this first, draws, then calls `save(fig, name)`
which writes an SVG into site/src/assets/figures/ (the chapters reference
them relatively from there).
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The app/site palette (custom.css / the app's dark token block).
BG = "#181b22"  # --sl-color-gray-7, panel surface
STAGE = "#0f1115"  # page background
INK = "#c4ccd8"  # body text
MUTED = "#9aa3b2"
GRID = "#343c4a"
CYAN = "#76d0ff"  # the app's signal cyan — primary series
AMBER = "#ffb86b"  # secondary series
GREEN = "#7fdc9c"  # tertiary series
RED = "#ff7b72"  # "this is wrong" series

ASSETS = Path(__file__).resolve().parent.parent / "src" / "assets" / "figures"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "legend.frameon": False,
        "figure.autolayout": True,
        "svg.fonttype": "none",  # real text in the SVG, not paths
    }
)


def save(fig, name: str) -> None:
    """Write `fig` to site/src/assets/figures/<name>.svg and report it.

    Set FIG_PNG_DIR to also emit a PNG there — a faithful matplotlib
    raster for eyeballing (external SVG rasterizers are unreliable)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"{name}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    print(f"wrote {out.relative_to(ASSETS.parent.parent.parent)}")
    png_dir = os.environ.get("FIG_PNG_DIR")
    if png_dir:
        fig.savefig(Path(png_dir) / f"{name}.png", dpi=110, bbox_inches="tight")
