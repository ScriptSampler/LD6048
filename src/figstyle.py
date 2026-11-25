"""One consistent figure theme: Okabe-Ito palette, legible at A4 column width,
300 dpi PNG + SVG, every axis labelled by the caller."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cycler

from .config import FIGURES, PALETTE

# A4 text width is about 6.3 in; a full-width figure at 6.3 in with 9-10 pt text
# is legible after scaling. Base font 10 pt so that nothing drops below 9 pt.
COL_W = 6.3
HALF_W = 3.1


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "figure.constrained_layout.use": True,
    })


def save_fig(fig, name: str) -> None:
    """Save PNG (300 dpi) and SVG under figures/."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, letter: str) -> None:
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left")
