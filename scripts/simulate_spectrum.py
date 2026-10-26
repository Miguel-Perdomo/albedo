"""Simulate a source in a setup and plot the spectrum split by origin.

The deposited energy is shown before any detector resolution, so the
structure produced by each part of the geometry stays visible.

Examples
--------
    python scripts/simulate_spectrum.py Cs-137
    python scripts/simulate_spectrum.py Ba-133 -n 50000000 --save
    python scripts/simulate_spectrum.py Cs-137 --setup csi_747E --bin 2
"""

import argparse
import time

import matplotlib.pyplot as plt

from albedo import plotting, simulate

LABELS = {"direct": "Direct", "liner_Cu": "Cu", "liner_Sn": "Sn", "body": "Pb"}
COLORS = {
    "direct": plotting.COLORS["blue"],
    "liner_Cu": plotting.MATERIAL_COLORS["Cu"],
    "liner_Sn": plotting.MATERIAL_COLORS["Sn"],
    "body": plotting.MATERIAL_COLORS["Pb"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="nuclide, e.g. Cs-137")
    parser.add_argument("--setup", default="csi_747E", help="setup name (default: csi_747E)")
    parser.add_argument("-n", type=float, default=1e7, help="number of histories (default: 1e7)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bin", type=float, default=4.0, help="bin width in keV (default: 4)")
    parser.add_argument("--save", action="store_true", help="save the figure in figures/")
    args = parser.parse_args()

    start = time.perf_counter()
    result = simulate(args.setup, args.source, int(args.n), seed=args.seed)
    elapsed = time.perf_counter() - start
    print(f"{int(args.n):.3e} histories in {elapsed:.1f} s; "
          f"{len(result.deposited)} detected (efficiency {result.efficiency:.3e})")
    for name, mask in result.components().items():
        print(f"  {name:12s} {mask.sum():9d}")

    plotting.use_style()
    fig, ax = plotting.figure(18, 10)
    e_max = float(result.source.energy.max()) * 1.1
    energy, counts = result.spectrum(args.bin, e_max)
    ax.step(energy, counts, where="mid", color="black", label="Total")
    for name, mask in result.components().items():
        energy, counts = result.spectrum(args.bin, e_max, mask=mask)
        ax.step(energy, counts, where="mid", label=LABELS.get(name, name), color=COLORS.get(name))
    ax.set_yscale("log")
    ax.set_xlim(0, e_max)
    ax.set_ylim(bottom=0.8)
    ax.set_xlabel(r"$E_\mathrm{dep}\,(\mathrm{keV})$")
    ax.set_ylabel(f"Counts / {args.bin:g} keV")
    ax.set_title(f"{result.source.nuclide}, {result.setup.name}")
    ax.legend(ncol=5, loc="upper center", fontsize="small")

    if args.save:
        plotting.save(fig, f"{result.setup.name}_{result.source.nuclide}_components")
    plt.show()


if __name__ == "__main__":
    main()
