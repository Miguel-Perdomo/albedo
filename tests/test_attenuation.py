import numpy as np
import pytest

from albedo import AttenuationTable, load_materials


@pytest.fixture(scope="module")
def materials():
    return load_materials()


def _test_energies(material):
    # Dense log grid plus points just below/above every tabulated energy,
    # so that every absorption edge is probed from both sides.
    grid = np.geomspace(0.5, 4000.0, 2000)
    near = np.concatenate([material.energy * (1 - 1e-6), material.energy * (1 + 1e-6)])
    return np.sort(np.concatenate([grid, near]))


@pytest.mark.parametrize("name", ["Cu", "Sn", "Pb", "CsI", "NaI"])
def test_cpp_matches_python(materials, name):
    m = materials[name]
    table = m.attenuation_table()
    e = _test_energies(m)
    np.testing.assert_allclose(table.mu_incoherent(e), m.mu_incoherent(e), rtol=1e-12)
    np.testing.assert_allclose(table.mu_photoelectric(e), m.mu_photoelectric(e), rtol=1e-12)
    np.testing.assert_allclose(table.mu_total(e), m.mu_total(e), rtol=1e-12)


def test_scalar_call_returns_float(materials):
    table = materials["Pb"].attenuation_table()
    assert isinstance(table.mu_total(661.657), float)


def test_table_limits(materials):
    table = materials["Cu"].attenuation_table()
    assert table.energy_min == pytest.approx(1.0)
    assert table.energy_max == pytest.approx(3000.0)


@pytest.mark.parametrize(
    "energy, inc, ph",
    [
        ([1.0], [1.0], [1.0]),                      # too short
        ([1.0, 2.0], [1.0], [1.0, 1.0]),            # length mismatch
        ([2.0, 1.0], [1.0, 1.0], [1.0, 1.0]),       # decreasing energy
        ([1.0, 2.0], [1.0, 0.0], [1.0, 1.0]),       # non-positive coefficient
    ],
)
def test_invalid_input_raises(energy, inc, ph):
    with pytest.raises(ValueError):
        AttenuationTable(energy, inc, ph)
