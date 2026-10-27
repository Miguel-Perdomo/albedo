import numpy as np
import pytest

from albedo.experiment import (RAW_DIR, Calibration, MeasuredSpectrum, calibrate, compare,
                               fit_peak, process_measurement, read_mca)


def synthetic(peaks, n_channels=1024, background=5.0, seed=0):
    """Spectrum with Gaussian peaks (centroid, sigma, area) on a flat background."""
    ch = np.arange(1, n_channels + 1)
    mean = np.full(n_channels, background)
    for c, s, a in peaks:
        mean += a / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((ch - c) / s) ** 2)
    counts = np.random.default_rng(seed).poisson(mean).astype(float)
    return MeasuredSpectrum("synthetic", counts, 3600.0, 3650.0, "")


def test_read_mca():
    s = read_mca(RAW_DIR / "CsI1x5_ST6us_700V_1h_G120_Cs137-100mm_2.txt")
    assert len(s.counts) == 1024
    assert s.live_time == 3600.0
    assert s.real_time == pytest.approx(3675.680)
    assert s.counts.sum() == 64288  # "Total counts" in the header
    assert s.start_time.startswith("2025-08-23")


def test_fit_peak_recovers_centroid():
    s = synthetic([(500.0, 25.0, 50_000)])
    fit = fit_peak(s, 662.0, (420, 580))
    assert fit.centroid == pytest.approx(500.0, abs=3 * fit.centroid_err)
    assert fit.sigma == pytest.approx(25.0, rel=0.05)


def test_calibration_recovers_line():
    true = Calibration(a0=3.0, a1=1.07)
    peaks = [(32.94, true.channel(32.94)), (661.657, true.channel(661.657))]
    s = synthetic([(c, 5.0 if c < 100 else 28.0, 40_000) for _, c in peaks])
    cal, _ = calibrate(s, [(32.94, (15, 45)), (661.657, (560, 680))])
    assert cal.a1 == pytest.approx(true.a1, rel=5e-3)
    assert cal.a0 == pytest.approx(true.a0, abs=1.0)
    assert cal.channel(cal.energy(123.4)) == pytest.approx(123.4)


def test_process_measurement():
    m = process_measurement("cs137_2025-08-23")
    assert len(m.edges) == len(m.counts) + 1
    assert np.all(np.diff(m.edges) > 0)
    # Background scaled by the live-time ratio 3600 / 7200
    assert m.background.sum() == pytest.approx(0.5 * 3658)
    assert np.allclose(m.net, m.counts - m.background)
    # Calibration close to the one obtained by hand in the analysis notebook
    assert m.calibration.a1 == pytest.approx(1.07, abs=0.01)
    peak = m.peaks[-1]
    assert m.calibration.energy(peak.centroid) == pytest.approx(661.657, abs=0.5)


def test_csv_roundtrip(tmp_path):
    m = process_measurement("cs137_2025-08-23")
    path = m.to_csv(tmp_path / "out.csv")
    data = np.loadtxt(path, delimiter=",")
    assert data.shape == (len(m.counts), 7)
    assert np.allclose(data[:, 3], m.counts)
    assert np.allclose(data[:, 5], m.net, atol=1e-3)


def test_compare_normalization():
    from albedo import simulate
    m = process_measurement("cs137_2025-08-23")
    r = simulate(m.setup, m.source, 1_000_000, seed=1)
    c = compare(m, r, "window")
    lo, hi = m.normalization
    sel = (m.energy >= lo) & (m.energy <= hi)
    assert c.simulated[sel].sum() == pytest.approx(m.net[sel].sum())
    total = sum(c.components.values())
    assert np.allclose(total, c.simulated)
    chi2, n = c.chi2(lo, hi)
    assert n == sel.sum() and chi2 >= 0.0


def test_activity_is_decay_corrected():
    m = process_measurement("cs137_2025-08-23")
    # Reference 2026-07-21 is ~332 days after the measurement: the activity
    # at the measurement is larger by exp(lambda dt), about 2 %.
    assert m.activity == pytest.approx(22.1e3 * 1.0212, rel=2e-3)
    assert m.decays == pytest.approx(m.activity * m.live_time, rel=1e-4)


def test_absolute_normalization():
    from albedo import simulate
    m = process_measurement("cs137_2025-08-23")
    r = simulate(m.setup, m.source, 200_000, seed=1)
    c = compare(m, r, "absolute")
    assert c.scale == pytest.approx(m.decays / r.decays)


def test_comparison_csv(tmp_path):
    from albedo import simulate
    m = process_measurement("cs137_2025-08-23")
    c = compare(m, simulate(m.setup, m.source, 200_000, seed=1), "absolute")
    data = np.loadtxt(c.to_csv(tmp_path / "cmp.csv"), delimiter=",")
    assert data.shape[0] == len(m.counts)
    assert np.allclose(data[:, 5], c.simulated, rtol=1e-5)


def test_activity_is_decay_corrected():
    m = process_measurement("cs137_2025-08-23")
    # Reference 2026-07-21 is ~332 days after the measurement: the activity
    # at the measurement is larger by exp(lambda dt), about 2 %.
    assert m.activity == pytest.approx(22.1e3 * 1.0212, rel=2e-3)
    assert m.decays == pytest.approx(m.activity * m.live_time, rel=1e-4)


def test_absolute_normalization():
    from albedo import simulate
    m = process_measurement("cs137_2025-08-23")
    r = simulate(m.setup, m.source, 200_000, seed=1)
    c = compare(m, r, "absolute")
    assert c.scale == pytest.approx(m.decays / r.decays)


def test_comparison_csv(tmp_path):
    from albedo import simulate
    m = process_measurement("cs137_2025-08-23")
    c = compare(m, simulate(m.setup, m.source, 200_000, seed=1), "absolute")
    data = np.loadtxt(c.to_csv(tmp_path / "cmp.csv"), delimiter=",")
    assert data.shape[0] == len(m.counts)
    assert np.allclose(data[:, 5], c.simulated, rtol=1e-5)
