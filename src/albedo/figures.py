"""Standard figures: a simulated spectrum and a comparison with a measurement.

Both functions draw on matplotlib axes with the ALBEDO style; call
``plotting.use_style()`` first. Spectra are drawn as step histograms.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from .plotting import COLORS, MATERIAL_COLORS, cm

COMPONENT_LABELS = {"direct": "Direct", "liner_Cu": "Cu", "liner_Sn": "Sn", "body": "Pb"}
COMPONENT_COLORS = {
    "direct": COLORS["blue"],
    "liner_Cu": MATERIAL_COLORS["Cu"],
    "liner_Sn": MATERIAL_COLORS["Sn"],
    "body": MATERIAL_COLORS["Pb"],
}


def _finish(ax, log, e_max, ylabel):
    if log:
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.5)
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_xlim(0.0, e_max)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=3, fontsize="small")


def plot_simulation(result, bin_width: float = 2.0, smear: bool = True, components: bool = False,
                    log: bool = False, seed: int = 0, width: float = 18.0, height: float = 10.0):
    """Simulated spectrum, optionally split by origin. Returns ``(fig, ax)``."""
    fig, ax = plt.subplots(figsize=cm(width, height), layout="constrained")
    e_max = 1.15 * float(result.source.energy.max())
    energy, counts = result.spectrum(bin_width, e_max, smear=smear, seed=seed)
    ax.step(energy, counts, where="mid", color=COLORS["red"], label="ALBEDO")
    if components:
        for name, mask in result.components().items():
            _, part = result.spectrum(bin_width, e_max, mask=mask, smear=smear, seed=seed)
            ax.step(energy, part, where="mid", lw=0.9, color=COMPONENT_COLORS.get(name),
                    label=COMPONENT_LABELS.get(name, name))
    ax.set_xlabel(r"$E\,(\mathrm{keV})$")
    ax.set_title(f"{result.source.nuclide}, {result.setup.name}")
    _finish(ax, log, e_max, f"Counts / {bin_width:g} keV")
    return fig, ax


def plot_comparison(comparison, components: bool = False, log: bool = False,
                    residuals: bool = True, width: float = 18.0, height: float = 14.0):
    """Measured and simulated spectra as step histograms. Returns ``(fig, axes)``."""
    m = comparison.measured
    e = m.energy
    e_max = min(float(m.edges[-1]), 1.2 * float(e[m.net > 0].max()))

    fig = plt.figure(figsize=cm(width, height), layout="constrained")
    if residuals:
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0])
        rx = fig.add_subplot(gs[1], sharex=ax)
    else:
        ax = fig.add_subplot()
        rx = None

    ax.step(e, m.net, where="mid", color="black", label="Measured")
    ax.step(e, comparison.simulated, where="mid", color=COLORS["red"], label="ALBEDO")
    if components:
        for name, counts in comparison.components.items():
            ax.step(e, counts, where="mid", lw=0.8, color=COMPONENT_COLORS.get(name),
                    label=COMPONENT_LABELS.get(name, name))
    ax.set_title(f"{m.source}, {m.setup} ({comparison.normalization} normalization)")
    _finish(ax, log, e_max, "Counts / channel")

    if rx is None:
        ax.set_xlabel(r"$E\,(\mathrm{keV})$")
        return fig, ax
    plt.setp(ax.get_xticklabels(), visible=False)
    rx.step(e, comparison.residuals, where="mid", color="black", lw=0.7)
    rx.axhline(0.0, color="gray")
    rx.set_ylim(-8, 8)
    rx.set_xlabel(r"$E\,(\mathrm{keV})$")
    rx.set_ylabel(r"$(\mathrm{exp}-\mathrm{sim})/\sigma$")
    return fig, (ax, rx)
