import numpy as np
import pytest

from albedo.materials import load_materials, read_xcom, DATA_DIR

EXPECTED = {"Cu", "Sn", "Pb", "CsI", "NaI"}


@pytest.fixture(scope="module")
def materials():
    return load_materials()


def test_all_materials_load(materials):
    assert EXPECTED <= set(materials)
    for m in materials.values():
        assert m.density > 0
        assert np.all(np.diff(m.energy) >= 0)
        assert m.energy[0] == pytest.approx(1.0)
        assert m.energy[-1] == pytest.approx(3000.0)


def test_edges_are_reduced_to_pairs():
    # NaI has five rows at 1.072 keV (Na K and I M1 coincide)
    table = read_xcom(DATA_DIR / "xs" / "NaI.txt")
    _, counts = np.unique(table["energy"], return_counts=True)
    assert counts.max() == 2
    group = table["photoelectric"][np.isclose(table["energy"], 1.072)]
    assert group[0] == pytest.approx(6.736e3)
    assert group[-1] == pytest.approx(7.924e3)


def test_grid_point_is_exact(materials):
    pb = materials["Pb"]
    assert pb.mu_photoelectric(100.0) == pytest.approx(5.237 * 11.35)
    assert pb.mu_incoherent(600.0) == pytest.approx(6.263e-2 * 11.35)


def test_pb_k_edge_is_a_step(materials):
    pb = materials["Pb"]
    below = pb.mu_photoelectric(87.99) / pb.density
    above = pb.mu_photoelectric(88.01) / pb.density
    assert below == pytest.approx(1.547, rel=1e-3)
    assert above == pytest.approx(7.321, rel=1e-3)


def test_no_ramp_below_pb_k_edge(materials):
    # Removing the below-edge row turns the step into a ramp between 80 and
    # 88 keV. Correct log-log interpolation between 2.012 (80 keV) and
    # 1.547 (88 keV) gives about 1.70 cm2/g at 85 keV.
    pb = materials["Pb"]
    assert pb.mu_photoelectric(85.0) / pb.density == pytest.approx(1.702, rel=5e-3)


def test_csi_k_edges(materials):
    csi = materials["CsI"]
    for edge in (33.17, 35.98):
        assert csi.mu_photoelectric(edge + 0.01) > 1.5 * csi.mu_photoelectric(edge - 0.01)


def test_total_is_sum(materials):
    sn = materials["Sn"]
    e = np.array([50.0, 184.3, 661.657])
    assert np.allclose(sn.mu_total(e), sn.mu_incoherent(e) + sn.mu_photoelectric(e))


def test_clamped_outside_table(materials):
    cu = materials["Cu"]
    assert cu.mu_total(0.5) == pytest.approx(cu.mu_total(1.0))
    assert cu.mu_total(5000.0) == pytest.approx(cu.mu_total(3000.0))


def test_linear_table_matches_density(materials):
    csi = materials["CsI"]
    energy, mu_inc, mu_ph = csi.linear_table()
    assert np.allclose(mu_inc, csi.incoherent * 4.510)
    assert np.allclose(mu_ph, csi.photoelectric * 4.510)
