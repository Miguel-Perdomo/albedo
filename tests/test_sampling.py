import numpy as np
import pytest

from albedo import _core
from albedo._core import ELECTRON_MASS


def klein_nishina_pdf(cos_theta, energy):
    """Unnormalised d(sigma)/d(cos theta) for a free electron at rest."""
    eps = 1.0 / (1.0 + (energy / ELECTRON_MASS) * (1.0 - cos_theta))
    return eps**2 * (eps + 1.0 / eps - (1.0 - cos_theta**2))


def expected_fractions(energy, edges, n_sub=200):
    """Probability of each cos(theta) bin, by midpoint integration."""
    probs = []
    for a, b in zip(edges[:-1], edges[1:]):
        x = a + (np.arange(n_sub) + 0.5) * (b - a) / n_sub
        probs.append(klein_nishina_pdf(x, energy).mean() * (b - a))
    probs = np.array(probs)
    return probs / probs.sum()


@pytest.mark.parametrize("energy", [30.0, 184.3, 661.657, 1836.06])
def test_klein_nishina_matches_cross_section(energy):
    n = 1_000_000
    cos_theta = _core.sample_klein_nishina(energy, n, seed=12345)
    assert cos_theta.min() >= -1.0 and cos_theta.max() <= 1.0

    edges = np.linspace(-1.0, 1.0, 51)
    observed, _ = np.histogram(cos_theta, bins=edges)
    expected = n * expected_fractions(energy, edges)

    chi2 = np.sum((observed - expected) ** 2 / expected)
    ndf = len(observed) - 1
    # chi2 has mean ndf and standard deviation sqrt(2 ndf)
    assert abs(chi2 - ndf) < 5.0 * np.sqrt(2.0 * ndf), f"chi2/ndf = {chi2:.1f}/{ndf}"


def test_klein_nishina_energy_range():
    energy = 661.657
    cos_theta = _core.sample_klein_nishina(energy, 100_000, seed=1)
    e_out = _core.compton_energy(energy, cos_theta)
    e_min = energy / (1.0 + 2.0 * energy / ELECTRON_MASS)
    assert e_out.min() >= e_min - 1e-9
    assert e_out.max() <= energy + 1e-9


def test_same_seed_same_sequence():
    a = _core.sample_klein_nishina(661.657, 1000, seed=7)
    b = _core.sample_klein_nishina(661.657, 1000, seed=7)
    c = _core.sample_klein_nishina(661.657, 1000, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_isotropic_directions():
    d = _core.isotropic_directions(200_000, seed=3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    # For an isotropic distribution <x> = 0 and <x^2> = 1/3 per component
    n = len(d)
    assert np.all(np.abs(d.mean(axis=0)) < 5.0 / np.sqrt(3.0 * n))
    assert np.allclose((d**2).mean(axis=0), 1.0 / 3.0, atol=5e-3)


@pytest.mark.parametrize(
    "direction",
    [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.3, -0.5, 0.812404)],
)
@pytest.mark.parametrize("cos_theta", [-1.0, -0.4, 0.0, 0.7, 1.0])
def test_rotation_preserves_angle(direction, cos_theta):
    d = np.array(direction) / np.linalg.norm(direction)
    for phi in np.linspace(0.0, 2.0 * np.pi, 13):
        out = np.array(_core.rotate(d, cos_theta, phi))
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-12)
        assert np.dot(out, d) == pytest.approx(cos_theta, abs=1e-12)


def test_rotation_average_over_phi():
    # Averaging over a uniform azimuth leaves only the component along d.
    d = np.array([0.3, -0.5, 0.812404])
    d /= np.linalg.norm(d)
    cos_theta = 0.25
    phis = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    mean = np.mean([_core.rotate(d, cos_theta, p) for p in phis], axis=0)
    assert np.allclose(mean, cos_theta * d, atol=1e-12)
