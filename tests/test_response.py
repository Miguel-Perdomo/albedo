import numpy as np
import pytest

from albedo import load_setup, simulate
from albedo.response import FWHM_TO_SIGMA, Resolution

CSI = Resolution(a=2.18, b=0.023)


def test_fwhm_formula():
    assert CSI.fwhm(661.657) == pytest.approx(2.18 * np.sqrt(661.657) + 0.023 * 661.657)
    assert CSI.relative(661.657) == pytest.approx(0.1078, abs=1e-3)
    assert CSI.fwhm(0.0) == 0.0
    assert CSI.fwhm(-5.0) == 0.0


def test_fwhm_to_sigma():
    assert 1.0 / FWHM_TO_SIGMA == pytest.approx(2.354820, rel=1e-6)


@pytest.mark.parametrize("energy", [32.0, 356.0, 661.657, 1836.07])
def test_smearing_is_gaussian_with_the_right_width(energy):
    n = 400_000
    out = CSI.smear(np.full(n, energy), seed=1)
    sigma = CSI.fwhm(energy) * FWHM_TO_SIGMA
    assert out.mean() == pytest.approx(energy, abs=5.0 * sigma / np.sqrt(n))
    assert out.std() == pytest.approx(sigma, rel=5.0 / np.sqrt(2.0 * n))
    # Fraction within +-FWHM/2 of the centre is 76.1 % for a Gaussian
    inside = np.abs(out - energy) < 0.5 * CSI.fwhm(energy)
    assert inside.mean() == pytest.approx(0.7610, abs=5e-3)


def test_smearing_is_reproducible():
    e = np.linspace(10.0, 1000.0, 1000)
    assert np.array_equal(CSI.smear(e, seed=3), CSI.smear(e, seed=3))
    assert not np.array_equal(CSI.smear(e, seed=3), CSI.smear(e, seed=4))


def test_setup_loads_detector_resolution():
    res = load_setup("csi_747E").resolution
    assert res == Resolution(2.18, 0.023, res.reference)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown resolution model"):
        Resolution.from_config({"model": "quadratic", "a": 1, "b": 1})


def test_smeared_spectrum_keeps_counts():
    r = simulate("csi_747E", "Cs-137", 300_000, seed=1)
    _, raw = r.spectrum(2.0, 2000.0)
    _, wide = r.spectrum(2.0, 2000.0, smear=True)
    # Smearing moves counts between bins but does not create or remove them
    # (a few may fall below 0 keV and leave the histogram).
    assert abs(raw.sum() - wide.sum()) <= 0.01 * raw.sum()
