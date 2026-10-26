import numpy as np
import pytest

from albedo import compton_energy
from albedo._core import ELECTRON_MASS


def test_forward_scattering_keeps_energy():
    assert compton_energy(661.657, 1.0) == pytest.approx(661.657)


def test_backscatter_cs137():
    e0 = 661.657
    expected = e0 / (1.0 + 2.0 * e0 / ELECTRON_MASS)
    assert compton_energy(e0, -1.0) == pytest.approx(expected)
    assert compton_energy(e0, -1.0) == pytest.approx(184.32, abs=0.01)


def test_compton_energy_is_vectorized():
    cos_theta = np.linspace(-1.0, 1.0, 11)
    out = compton_energy(661.657, cos_theta)
    assert out.shape == cos_theta.shape
    assert np.all(np.diff(out) > 0)
