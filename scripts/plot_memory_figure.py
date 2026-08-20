#!/usr/bin/env python3
r"""Memory consumption vs. pruning ratio, sized for Softwaretechnik-Trends.

Single-column figure (~\columnwidth) showing the structured-pruning memory
measurements for both models, without fine-tuning, plus a least-squares linear
fit per model.

Both series share one y-axis on purpose. They are the same quantity in the same
unit and the same order of magnitude, and the finding the figure has to carry is
that the big model both starts higher *and* falls faster. A second y-axis would
rescale the two curves onto near-identical slopes and destroy exactly that.

Usage:
    uv run --frozen python scripts/plot_memory_figure.py
    python3 scripts/plot_memory_figure.py --csv <path> --outdir <dir>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO / "results" / "reproduced" / "structured_pruning_data_3.csv"
DEFAULT_OUTDIR = REPO / "results" / "figures"

RATIOS = list(range(0, 100, 10))

# Categorical slots 1 and 2 of the validated reference palette (light mode).
# Checked with the palette validator: all-pairs CVD dE 24.7, normal-vision 33.6,
# both well clear of the floors, and each >= 3:1 against a light surface.
COLOR_BIG = "#2a78d6"    # blue
COLOR_SMALL = "#eb6834"  # orange

# Ink colors -- text never wears the series color.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d8d4"

SERIES = [
    # key prefix, label, color, marker
    ("big", "Big model", COLOR_BIG, "o"),
    ("small", "Small model", COLOR_SMALL, "s"),
]


def load_memory(csv_path: Path) -> dict[str, list[float]]:
    """Return {'big': [...], 'small': [...]} of mean memory in MB, by ratio."""
    rows = {r["metric"]: r for r in csv.DictReader(csv_path.open())}
    out: dict[str, list[float]] = {}
    for size, *_ in SERIES:
        values = []
        for ratio in RATIOS:
            key = f"{size}_memory_mb_pruned_{ratio}"
            if key not in rows:
                raise SystemExit(f"[plot] missing row {key!r} in {csv_path}")
            values.append(float(rows[key]["avg"]))
        out[size] = values
    return out


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType, not Type 3 -- required by many venues
        "ps.fonttype": 42,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--name", default="memory_vs_pruning_ratio")
    parser.add_argument("--width", type=float, default=3.4, help="inches (~1 column)")
    parser.add_argument("--height", type=float, default=2.35, help="inches")
    args = parser.parse_args()

    data = load_memory(args.csv)
    style()

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    x = np.array(RATIOS, dtype=float)
    grid = np.linspace(0, 90, 200)

    for size, label, color, marker in SERIES:
        y = np.array(data[size], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fit = slope * grid + intercept
        resid = y - (slope * x + intercept)
        r2 = 1.0 - resid.var() / y.var()

        # Fit first, so the measured points sit on top of it.
        ax.plot(grid, fit, linestyle="--", color=color, linewidth=1.0,
                dashes=(4, 2.5), zorder=2, alpha=0.9)
        ax.plot(x, y, linestyle="none", marker=marker, color=color,
                markersize=3.8, markeredgecolor="white", markeredgewidth=0.5,
                zorder=3, label=f"{label} ({slope:+.2f} MB per 1%)")

        print(f"[plot] {size:5s}: slope {slope:+.4f} MB per 1% | "
              f"intercept {intercept:7.3f} | R^2 {r2:.5f} | "
              f"{y[0]:.2f} -> {y[-1]:.2f} MB ({100*(y[-1]-y[0])/y[0]:+.1f}%)")

    ax.set_xlabel("Pruning ratio [%]")
    ax.set_ylabel("Memory consumption [MB]")
    ax.set_xlim(-4, 94)
    ax.set_xticks(RATIOS)
    ax.set_ylim(0, 180)
    ax.set_yticks([0, 40, 80, 120, 160])

    # Recessive grid and axes; no chartjunk.
    ax.grid(True, which="major", color=GRID, linewidth=0.4, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)

    legend = ax.legend(loc="lower left", frameon=False, handlelength=1.6,
                       borderpad=0.2, labelspacing=0.35, handletextpad=0.5)
    for text in legend.get_texts():
        text.set_color(INK)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = args.outdir / f"{args.name}.{ext}"
        fig.savefig(path)
        print(f"[plot] wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
