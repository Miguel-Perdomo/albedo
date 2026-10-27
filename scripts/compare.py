"""Compare a processed measurement with an ALBEDO simulation.

The simulation is broadened with the detector resolution, binned on the
measured channels and scaled either absolutely (source activity from the
measurement file) or to the net counts in the normalization window.

Examples
--------
    python scripts/compare.py cs137_2025-08-23
    python scripts/compare.py cs137_2025-08-23 -n 1e8 --log --components --save
    python scripts/compare.py cs137_2025-08-23 --window
"""

import argparse

import matplotlib.pyplot as plt

from albedo import plotting, simulate
from albedo.experiment import ROOT, compare, process_measurement
from albedo.figures import plot_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("measurement", help="name of a file in configs/measurements")
    parser.add_argument("-n", type=float, default=3e7, help="histories (default: 3e7)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--setup", help="override the setup of the measurement")
    parser.add_argument("--window", action="store_true",
                        help="scale to the normalization window instead of the activity")
    parser.add_argument("--components", action="store_true", help="draw the origin components")
    parser.add_argument("--log", action="store_true", help="logarithmic vertical axis")
    parser.add_argument("--save", action="store_true", help="save figure and CSV")
    args = parser.parse_args()

    m = process_measurement(args.measurement)
    result = simulate(args.setup or m.setup, m.source, int(args.n), seed=args.seed)
    c = compare(m, result, "window" if args.window else "absolute")

    print(f"{c.normalization} normalization, scale = {c.scale:.4g}")
    e = m.energy
    for lo, hi in [(40, 150), (150, 300), (300, 470), (470, 560), (600, 720)]:
        sel = (e >= lo) & (e < hi)
        print(f"  {lo:5.0f}-{hi:5.0f} keV   measured/simulated = "
              f"{m.net[sel].sum() / c.simulated[sel].sum():.3f}")

    plotting.use_style(font_size=14)
    fig, _ = plot_comparison(c, components=args.components, log=args.log)
    if args.save:
        plotting.save(fig, f"compare_{m.name}")
        c.to_csv(ROOT / "results" / f"compare_{m.name}.csv")
    plt.show()


if __name__ == "__main__":
    main()
