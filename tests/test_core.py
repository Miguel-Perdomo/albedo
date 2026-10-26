import pytest

from albedo import compton_energy

ME = 510.99895


def test_forward_scattering_keeps_energy():
    assert compton_energy(661.657, 1.0) == pytest.approx(661.657)


def test_backscatter_cs137():
    e0 = 661.657
    expected = e0 / (1.0 + 2.0 * e0 / ME)
    assert compton_energy(e0, -1.0) == pytest.approx(expected)
    assert compton_energy(e0, -1.0) == pytest.approx(184.32, abs=0.01)
