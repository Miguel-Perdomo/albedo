import numpy as np
import pytest

from albedo import load_materials, load_setup, simulate
from albedo.geometry import Setup, Volume
from albedo.sources import Source


def line(energy):
    """Monoenergetic source."""
    return Source("test", "test", np.array([energy]), np.array([100.0]), ("g",), ("test",))


def bare_detector(radius, length, distance):
    """Detector in vacuum with a point source on its axis, above the face."""
    detector = Volume("detector", "CsI", radius, -length, 0.0)
    world = Volume("world", None, 100.0, -100.0, 100.0)
    return Setup("bare", (detector, world), (0.0, 0.0, distance))


def interaction_probability(mu, radius, length, distance, n=200_000):
    """Probability that an isotropic photon interacts in the cylinder.

    P = 1/2 * integral of (1 - exp(-mu l(theta))) d(cos theta) over the
    rays that hit the front face; l is the chord through the crystal.
    """
    cos_max = distance / np.hypot(distance, radius)
    c = cos_max + (np.arange(n) + 0.5) * (1.0 - cos_max) / n
    tan = np.sqrt(1.0 - c**2) / c
    through_back = (distance + length) * tan <= radius
    chord = np.where(through_back, length / c, radius / np.sqrt(1.0 - c**2) - distance / c)
    return 0.5 * np.mean(1.0 - np.exp(-mu * chord)) * (1.0 - cos_max)


@pytest.mark.parametrize("energy", [100.0, 661.657, 1836.07])
def test_bare_detector_efficiency(energy):
    radius, length, distance = 0.635, 5.08, 2.0
    mu = load_materials()["CsI"].mu_total(energy)
    expected = interaction_probability(mu, radius, length, distance)

    n = 2_000_000
    result = simulate(bare_detector(radius, length, distance), line(energy), n, seed=5)
    detected = len(result.deposited)
    sigma = np.sqrt(n * expected)
    assert abs(detected - n * expected) < 5.0 * sigma
    assert np.all(result.origin == -1)  # nothing to scatter from


def test_energy_is_conserved():
    result = simulate("csi_747E", "Ba-133", 500_000, seed=2)
    assert len(result.deposited) > 0
    assert np.all(result.deposited > 0.0)
    assert np.all(result.deposited <= result.emitted * (1 + 1e-12))
    assert set(np.unique(result.emitted)) <= set(result.source.energy)


def test_full_energy_peak_in_bare_detector():
    e0 = 661.657
    result = simulate(bare_detector(0.635, 5.08, 2.0), line(e0), 500_000, seed=3)
    full = np.isclose(result.deposited, e0)
    assert 0.0 < full.mean() < 1.0
    # A single Compton scattering cannot deposit more than the Compton edge,
    # so events between the edge and the peak need several interactions;
    # nothing can exceed the photon energy.
    assert result.deposited.max() <= e0 * (1 + 1e-12)


def test_same_seed_same_result():
    a = simulate("csi_747E", "Cs-137", 200_000, seed=9)
    b = simulate("csi_747E", "Cs-137", 200_000, seed=9)
    c = simulate("csi_747E", "Cs-137", 200_000, seed=10)
    assert np.array_equal(a.deposited, b.deposited)
    assert np.array_equal(a.origin, b.origin)
    assert not np.array_equal(a.deposited, c.deposited)


def test_liner_increases_backscatter(tmp_path):
    # The graded Cu/Sn liner reflects more 662 keV photons back to the
    # detector than bare lead (lower Z, less photoabsorption of the
    # scattered ~184 keV photons).
    lined = load_setup("csi_747E")
    cavity, hole, *liner, body = lined.volumes[1:]
    pb_only = Setup("pb_only", (lined.volumes[0], cavity, hole, body), lined.source)

    def backscatter(setup):
        r = simulate(setup, "Cs-137", 3_000_000, seed=4)
        window = (r.deposited > 150.0) & (r.deposited < 250.0)
        return np.sum(window & (r.origin != -1))

    n_lined, n_pb = backscatter(lined), backscatter(pb_only)
    assert n_lined > 2.0 * n_pb, (n_lined, n_pb)
