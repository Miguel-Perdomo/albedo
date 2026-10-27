"""Calibrate a measured spectrum, subtract its background and save it in keV.

Writes experimental/processed/<measurement>.csv and shows the fits of the
calibration peaks so they can be checked.

Example
-------
    python scripts/process_measurement.py cs137_2025-08-23
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from albedo import plotting
from albedo.experiment import PROCESSED_DIR, _peak_model, process_measurement


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("measurement", help="name of a file in configs/measurements")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    m = process_measurement(args.measurement)
    out = m.to_csv(PROCESSED_DIR / f"{m.name}.csv")
    cal = m.calibration
    print(f"calibration: E = {cal.a0:.3f} + {cal.a1:.5f} * channel  (keV)")
    for p in m.peaks:
        fwhm = 2.3548 * p.sigma * cal.a1
        print(f"  {p.energy:8.3f} keV  centroid {p.centroid:8.2f} +- {p.centroid_err:.2f} ch"
              f"   FWHM {fwhm:6.1f} keV")
    print(f"live time {m.live_time:.0f} s, net counts {m.net.sum():.0f}, "
          f"background {m.background.sum():.0f}")
    print(f"written: {out}")

    if args.no_plot:
        return
    plotting.use_style(font_size=12)
    fig, axes = plotting.figure(8.0 * len(m.peaks), 7, ncols=len(m.peaks))
    channels = np.arange(1, len(m.counts) + 1)
    for ax, p in zip(np.atleast_1d(axes), m.peaks):
        lo, hi = p.window
        sel = (channels >= lo) & (channels <= hi)
        x = np.linspace(lo, hi, 400)
        ax.errorbar(channels[sel], m.counts[sel], yerr=np.sqrt(np.maximum(m.counts[sel], 1)),
                    fmt="o", ms=2, color="black", elinewidth=0.5)
        ax.plot(x, _peak_model(x, *p.params))
        ax.set_title(f"{p.energy:g} keV")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Counts")
    plt.show()


if __name__ == "__main__":
    main()
