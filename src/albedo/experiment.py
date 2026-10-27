"""Measured spectra: reading, energy calibration and background subtraction.

A measurement is described by a TOML file in ``configs/measurements``
naming the raw spectrum, the background spectrum (both in
``experimental/raw``), the source, the setup and the peaks used for the
energy calibration. ``process_measurement`` turns it into a
``ProcessedSpectrum``: energy bins in keV, measured counts, background
scaled to the same live time, and net counts with their uncertainties.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "experimental" / "raw"
PROCESSED_DIR = ROOT / "experimental" / "processed"
MEASUREMENTS_DIR = ROOT / "configs" / "measurements"


# --- raw spectra ------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class MeasuredSpectrum:
    """Counts per channel; channel numbers start at 1."""

    name: str
    counts: np.ndarray
    live_time: float  # s
    real_time: float  # s
    start_time: str

    @property
    def channels(self) -> np.ndarray:
        return np.arange(1, len(self.counts) + 1)

    @property
    def dead_time_fraction(self) -> float:
        return 1.0 - self.live_time / self.real_time

    @property
    def start(self) -> datetime:
        """Start of the acquisition, from the 'Start time' header line."""
        return datetime.strptime(self.start_time, "%Y-%m-%d, %H:%M:%S")


def read_mca(path: str | Path) -> MeasuredSpectrum:
    """Read a text spectrum exported by the MCA software.

    Header lines start with '#' and hold 'key: value' pairs; data lines
    hold channel, energy, counts and rate. The energy column uses the
    calibration stored by the acquisition software and is ignored.
    """
    path = Path(path)
    header: dict[str, str] = {}
    counts: list[int] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                key, sep, value = line[1:].partition(":")
                if sep:
                    header[key.strip()] = value.strip()
                continue
            fields = line.split()
            if len(fields) >= 3:
                counts.append(int(float(fields[2])))

    if not counts:
        raise ValueError(f"{path}: no channel data found")
    return MeasuredSpectrum(
        name=path.stem,
        counts=np.array(counts, dtype=float),
        live_time=float(header["Live time (s)"]),
        real_time=float(header["Real time (s)"]),
        start_time=header.get("Start time", ""),
    )


# --- energy calibration -----------------------------------------------------

@dataclass(frozen=True)
class PeakFit:
    energy: float        # known energy of the peak (keV)
    centroid: float      # fitted centroid (channel)
    centroid_err: float
    sigma: float         # fitted Gaussian width (channels)
    window: tuple[int, int]
    params: tuple        # all fitted parameters, for plotting


def _peak_model(x, amplitude, centroid, sigma, c0, c1):
    return amplitude * np.exp(-0.5 * ((x - centroid) / sigma) ** 2) + c0 + c1 * (x - centroid)


def fit_peak(spectrum: MeasuredSpectrum, energy: float, window) -> PeakFit:
    """Fit a Gaussian on a linear background inside a channel window."""
    lo, hi = window
    ch = spectrum.channels
    mask = (ch >= lo) & (ch <= hi)
    x, y = ch[mask].astype(float), spectrum.counts[mask]
    p0 = [y.max() - y.min(), x[np.argmax(y)], (hi - lo) / 8.0, y.min(), 0.0]
    params, cov = curve_fit(_peak_model, x, y, p0=p0, sigma=np.sqrt(np.maximum(y, 1.0)),
                            absolute_sigma=True, maxfev=20000)
    return PeakFit(energy, params[1], float(np.sqrt(cov[1, 1])), abs(params[2]),
                   (lo, hi), tuple(params))


@dataclass(frozen=True)
class Calibration:
    """Linear energy calibration E = a0 + a1 * channel (keV)."""

    a0: float
    a1: float

    def energy(self, channel):
        return self.a0 + self.a1 * np.asarray(channel, dtype=float)

    def channel(self, energy):
        return (np.asarray(energy, dtype=float) - self.a0) / self.a1


def calibrate(spectrum: MeasuredSpectrum, peaks) -> tuple[Calibration, list[PeakFit]]:
    """Linear calibration from peaks given as (energy, (low, high)) pairs."""
    fits = [fit_peak(spectrum, e, w) for e, w in peaks]
    if len(fits) < 2:
        raise ValueError("at least two calibration peaks are needed")
    centroids = np.array([f.centroid for f in fits])
    energies = np.array([f.energy for f in fits])
    a1, a0 = np.polyfit(centroids, energies, 1)
    return Calibration(float(a0), float(a1)), fits


# --- processed spectrum -----------------------------------------------------

@dataclass(frozen=True, eq=False)
class ProcessedSpectrum:
    """Calibrated, background-subtracted spectrum on the detector channels."""

    name: str
    source: str
    setup: str
    calibration: Calibration
    live_time: float
    edges: np.ndarray       # keV, channel boundaries
    counts: np.ndarray      # measured counts
    background: np.ndarray  # background counts scaled to the same live time
    net: np.ndarray
    net_err: np.ndarray
    peaks: tuple[PeakFit, ...] = ()
    normalization: tuple[float, float] | None = None
    activity: float | None = None      # Bq at the start of the acquisition
    activity_err: float | None = None
    decays: float | None = None        # source decays during the live time

    @property
    def energy(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def bin_width(self) -> np.ndarray:
        return np.diff(self.edges)

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "\n".join([
            f"measurement: {self.name}",
            f"source: {self.source}",
            f"setup: {self.setup}",
            f"live time (s): {self.live_time}",
            f"calibration: E = {self.calibration.a0:.5f} + {self.calibration.a1:.6f} * channel (keV)",
            "background scaled by the live-time ratio; net_err includes both terms",
            "e_low,e_high,energy,counts,background,net,net_err",
        ])
        data = np.column_stack([self.edges[:-1], self.edges[1:], self.energy,
                                self.counts, self.background, self.net, self.net_err])
        np.savetxt(path, data, delimiter=",", header=header,
                   fmt=["%.4f", "%.4f", "%.4f", "%d", "%.4f", "%.4f", "%.4f"])
        return path


def process_measurement(name: str | Path) -> ProcessedSpectrum:
    """Calibrate and subtract the background of a configured measurement."""
    path = Path(name)
    if not path.exists():
        path = MEASUREMENTS_DIR / f"{name}.toml"
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    spectrum = read_mca(RAW_DIR / cfg["spectrum"])
    peaks = [(p["energy"], tuple(p["window"])) for p in cfg["calibration"]]
    calibration, fits = calibrate(spectrum, peaks)

    activity = activity_err = decays = None
    if "activity" in cfg:
        activity, activity_err, decays = _activity_at(cfg["activity"], cfg["source"], spectrum)

    counts = spectrum.counts
    background = np.zeros_like(counts)
    variance = counts.copy()
    if "background" in cfg:
        bkg = read_mca(RAW_DIR / cfg["background"])
        if len(bkg.counts) != len(counts):
            raise ValueError("spectrum and background have different numbers of channels")
        ratio = spectrum.live_time / bkg.live_time
        background = ratio * bkg.counts
        variance = counts + ratio**2 * bkg.counts

    channel_edges = np.arange(len(counts) + 1) + 0.5
    norm = tuple(cfg["normalization"]) if "normalization" in cfg else None
    return ProcessedSpectrum(
        name=path.stem,
        source=cfg["source"],
        setup=cfg["setup"],
        calibration=calibration,
        live_time=spectrum.live_time,
        edges=calibration.energy(channel_edges),
        counts=counts,
        background=background,
        net=counts - background,
        net_err=np.sqrt(np.maximum(variance, 1.0)),
        peaks=tuple(fits),
        normalization=norm,
        activity=activity,
        activity_err=activity_err,
        decays=decays,
    )


def _activity_at(cfg: dict, nuclide: str, spectrum: MeasuredSpectrum):
    """Activity (Bq) at the start of the acquisition and decays in the live time.

    The reference activity is propagated with A(t) = A_ref exp(-lambda (t - t_ref)),
    which also works when the reference date is after the measurement. The
    number of decays uses the mean activity over the real time, scaled to the
    live time.
    """
    from .sources import load_source

    half_life = load_source(nuclide).half_life
    if half_life is None:
        raise ValueError(f"no half-life available for {nuclide}")
    lam = np.log(2.0) / half_life

    reference = datetime.fromisoformat(str(cfg["reference_date"]))
    dt = (spectrum.start - reference).total_seconds()
    factor = np.exp(-lam * dt)
    activity = 1e3 * cfg["value"] * factor  # kBq -> Bq
    activity_err = 1e3 * cfg.get("uncertainty", 0.0) * factor

    x = lam * spectrum.real_time
    mean_factor = -np.expm1(-x) / x if x > 0 else 1.0
    decays = activity * mean_factor * spectrum.live_time
    return float(activity), float(activity_err), float(decays)


# --- comparison with simulation ---------------------------------------------

@dataclass(frozen=True, eq=False)
class Comparison:
    measured: ProcessedSpectrum
    simulated: np.ndarray          # scaled to the measurement
    simulated_err: np.ndarray
    components: dict[str, np.ndarray]
    scale: float
    normalization: str = "window"

    @property
    def residuals(self) -> np.ndarray:
        """(measured - simulated) / combined uncertainty, per bin."""
        sigma = np.sqrt(self.measured.net_err**2 + self.simulated_err**2)
        return (self.measured.net - self.simulated) / sigma

    def chi2(self, e_min: float, e_max: float) -> tuple[float, int]:
        """Chi-square and number of bins between e_min and e_max (keV)."""
        e = self.measured.energy
        mask = (e >= e_min) & (e <= e_max)
        return float(np.sum(self.residuals[mask] ** 2)), int(mask.sum())

    def to_csv(self, path: str | Path) -> Path:
        """Measured and simulated counts on the measurement bins."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        m = self.measured
        names = list(self.components)
        header = "\n".join([
            f"measurement: {m.name}",
            f"source: {m.source}",
            f"normalization: {self.normalization} (scale {self.scale:.6g})",
            "e_low,e_high,energy,measured_net,measured_err,simulated,simulated_err,"
            + ",".join(f"sim_{n}" for n in names),
        ])
        columns = [m.edges[:-1], m.edges[1:], m.energy, m.net, m.net_err,
                   self.simulated, self.simulated_err] + [self.components[n] for n in names]
        np.savetxt(path, np.column_stack(columns), delimiter=",", header=header, fmt="%.6g")
        return path


def compare(measured: ProcessedSpectrum, result, normalization: str = "absolute",
            window=None, activity=None, seed: int = 0) -> Comparison:
    """Put a simulation on the energy bins of a measurement.

    The simulated deposits are broadened with the detector resolution and
    histogrammed on the measured bins, then scaled in one of two ways:

    * ``"absolute"``: number of source decays in the measurement (from the
      activity in the measurement file, or ``activity`` in Bq at the start
      of the acquisition) divided by the number of simulated decays;
    * ``"window"``: equal net counts inside ``window`` (keV), by default
      the normalization window of the measurement.
    """
    smeared = result.smeared(seed)
    edges = measured.edges
    raw, _ = np.histogram(smeared, bins=edges)
    raw_components = {name: np.histogram(smeared[mask], bins=edges)[0]
                      for name, mask in result.components().items()}

    if normalization == "absolute":
        if activity is not None:
            decays = activity * measured.live_time
        elif measured.decays is not None:
            decays = measured.decays
        else:
            raise ValueError("absolute normalization needs the source activity")
        scale = decays / result.decays
    elif normalization == "window":
        window = window or measured.normalization
        if window is None:
            raise ValueError("give a normalization window or the source activity")
        e = measured.energy
        mask = (e >= window[0]) & (e <= window[1])
        scale = measured.net[mask].sum() / raw[mask].sum()
    else:
        raise ValueError("normalization must be 'absolute' or 'window'")

    return Comparison(
        measured=measured,
        simulated=scale * raw,
        simulated_err=scale * np.sqrt(raw),
        components={k: scale * v for k, v in raw_components.items()},
        scale=float(scale),
        normalization=normalization,
    )
