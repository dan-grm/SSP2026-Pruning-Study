"""Matplotlib style matching the Descartes Beamer template of the SSP talk.

Frame: 16:9 Beamer at 10 pt, i.e. 160 x 90 mm. A full-width figure is about
5.5 in wide and at most ~2.6 in tall under a frame title. Text is 9 pt
Source Sans 3 (bundled in post_submission/fonts/, SIL OFL), so figures match
the slide text when included at their natural size (``width=<figure width>``
or without scaling).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_DIR = Path(__file__).resolve().parent / "fonts"

# Template palette.
NAVY = "#002F67"        # Marineblau
CORNFLOWER = "#5498FF"  # Kornblumenblau
CORAL = "#ED6A5B"       # Korallenrot
SAND = "#F1EBDF"        # Sandbeige: fills and bands only, never a line
GREY = "#6B6B6B"        # fourth series / reference lines
INK = "#1A1A1A"
INK_MUTED = "#55534F"
GRID = "#DDD8CC"

FULL_WIDTH = 5.5   # inches
HALF_WIDTH = 2.7
HEIGHT = 1.9   # leaves room for 1-2 lines of slide text under the figure


def apply() -> None:
    family = "DejaVu Sans"
    if FONT_DIR.is_dir():
        for ttf in FONT_DIR.glob("*.ttf"):
            font_manager.fontManager.addfont(str(ttf))
        family = "Source Sans 3"
    else:
        print(f"[style] {FONT_DIR} missing, falling back to {family}")
    plt.rcParams.update({
        "font.family": family,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "text.color": INK,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.transparent": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, outdir: Path, name: str, png: bool = True) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf")
    if png:  # quick-look preview; the slides use the PDF
        (outdir / "png").mkdir(exist_ok=True)
        fig.savefig(outdir / "png" / f"{name}.png", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"[plot] wrote {outdir / name}.pdf")
