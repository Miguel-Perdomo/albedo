"""Plotting style and helpers for ALBEDO figures.

Call ``use_style()`` once before plotting. Every visual choice is defined
in this module (``COLORS`` and the ``rc`` dictionary in ``use_style``), so
changing the look of all figures only requires editing this file.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

# Colour cycle: lines take these colours in order unless one is given.
COLORS = {
    "red": "darkred",
    "blue": (0.0, 0.0, 0.7),
    "gray": (0.5, 0.5, 0.5),
    "olive": (0.7, 0.7, 0.0),
    "purple": (0.7, 0.0, 0.7),
    "orange": (1.0, 0.7, 0.0),
    "teal": (0.0, 0.3, 0.3),
}

# Colour of each material in geometry figures; vacuum is drawn white.
MATERIAL_COLORS = {
    "Pb": COLORS["gray"],
    "Sn": COLORS["teal"],
    "Cu": COLORS["orange"],
    "CsI": COLORS["blue"],
    "NaI": COLORS["purple"],
}
SOURCE_COLOR = COLORS["red"]
FONT_FAMILY_HTML = "CMU Serif, Latin Modern Roman, Computer Modern, serif"


def use_style(font_size: float = 18, usetex: bool = True, family: str = "serif") -> None:
    """Apply the ALBEDO figure style to matplotlib.

    Parameters
    ----------
    font_size
        Size (pt) of all text: labels, ticks and legend.
    usetex
        Render text with LaTeX. If no LaTeX installation is found, a
        warning is issued and matplotlib's mathtext is used instead.
    family
        ``"serif"`` (Computer Modern) or ``"sans-serif"``.
    """
    if usetex and shutil.which("latex") is None:
        warnings.warn("LaTeX not found; falling back to mathtext.", stacklevel=2)
        usetex = False

    plt.style.use("default")

    rc = {
        # Text
        "text.usetex": usetex,
        "font.family": family,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "legend.title_fontsize": font_size,
        # Colours
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.prop_cycle": cycler(color=list(COLORS.values())),
        # Ticks: inward, on all four sides, with minor ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Grid on major and minor ticks
        "axes.grid": True,
        "axes.grid.which": "both",
        "grid.alpha": 0.4,
        "grid.linewidth": 0.6,
        # Lines and legend
        "lines.linewidth": 1.5,
        "legend.labelspacing": 0.2,
        # Output
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }

    if usetex:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        rc["mathtext.fontset"] = "cm" if family == "serif" else "dejavusans"
        rc["axes.formatter.use_mathtext"] = True

    mpl.rcParams.update(rc)


def cm(width: float, height: float) -> tuple[float, float]:
    """Convert a figure size from centimetres to inches."""
    return width / 2.54, height / 2.54


def figure(width: float = 18.0, height: float = 7.0, nrows: int = 1, ncols: int = 1, **kwargs):
    """Create a figure of the given size in cm. Returns ``(fig, axes)``."""
    return plt.subplots(nrows, ncols, figsize=cm(width, height), layout="constrained", **kwargs)


def plot_spectrum(ax, energy, counts, *, label=None, color=None, logy=False, **kwargs):
    """Draw a histogrammed spectrum as a step line centred on each bin."""
    ax.step(energy, counts, where="mid", label=label, color=color, **kwargs)
    ax.set_xlabel(r"$E\,(\mathrm{keV})$")
    ax.set_ylabel("Counts")
    if logy:
        ax.set_yscale("log")
    return ax


def save(fig, name: str, directory: str | Path = "figures", formats=("pdf", "png")) -> list[Path]:
    """Save ``fig`` as ``directory/name.<fmt>`` for every format given."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{name}.{fmt}" for fmt in formats]
    for path in paths:
        fig.savefig(path)
    return paths
