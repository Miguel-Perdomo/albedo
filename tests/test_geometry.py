import numpy as np
import pytest

from albedo import load_materials
from albedo.geometry import load_setup

CU, SN = 0.16, 0.13


@pytest.fixture(scope="module")
def setup():
    return load_setup("csi_747E")


@pytest.fixture(scope="module")
def geo(setup):
    geometry, _ = setup.build(load_materials())
    return geometry


def names(setup, segments):
    return [setup.volumes[v].name for v, _ in segments]


def test_setup_dimensions(setup):
    det = setup.volumes[setup.volume_index("detector")]
    assert det.radius == pytest.approx(0.635)
    assert det.z_max - det.z_min == pytest.approx(5.08)
    assert setup.source[2] - det.z_max == pytest.approx(10.55)
    assert setup.material_names == ["CsI", "Cu", "Sn", "Pb"]


def test_locate_each_region(setup, geo):
    r0 = setup.volumes[setup.volume_index("cavity")].radius
    det = setup.volumes[setup.volume_index("detector")]
    cases = {
        (0.0, 0.0, 0.5 * (det.z_min + det.z_max)): "detector",
        (0.0, 0.0, 28.0): "cavity",
        (r0 + 0.5 * CU, 0.0, 15.0): "liner_Cu",
        (r0 + CU + 0.5 * SN, 0.0, 15.0): "liner_Sn",
        (r0 + 5.0, 0.0, 15.0): "body",
        (0.0, 0.0, -0.5 * CU): "liner_Cu",
    }
    for p, expected in cases.items():
        assert setup.volumes[geo.locate(p)].name == expected
    assert geo.locate((30.0, 0.0, 15.0)) == -1


def test_horizontal_ray_crosses_every_layer(setup, geo):
    r0 = setup.volumes[setup.volume_index("cavity")].radius
    outer = setup.volumes[setup.volume_index("body")].radius
    segs = geo.trace((0.0, 0.0, 25.0), (1.0, 0.0, 0.0))
    assert names(setup, segs) == ["cavity", "liner_Cu", "liner_Sn", "body"]
    lengths = [length for _, length in segs]
    assert lengths == pytest.approx([r0, CU, SN, outer - r0 - CU - SN], abs=1e-6)


def test_vertical_ray_down_from_source(setup, geo):
    det = setup.volumes[setup.volume_index("detector")]
    body = setup.volumes[setup.volume_index("body")]
    zs = setup.source[2]
    segs = geo.trace(setup.source, (0.0, 0.0, -1.0))
    assert names(setup, segs) == ["cavity", "detector", "cavity", "liner_Cu", "liner_Sn", "body"]
    lengths = [length for _, length in segs]
    expected = [zs - det.z_max, det.z_max - det.z_min, det.z_min, CU, SN, -CU - SN - body.z_min]
    assert lengths == pytest.approx(expected, abs=1e-6)


def test_chord_through_detector(setup, geo):
    # Horizontal ray at impact parameter b crosses a chord 2 sqrt(R^2 - b^2)
    det = setup.volumes[setup.volume_index("detector")]
    z = 0.5 * (det.z_min + det.z_max)
    for b in (0.0, 0.3, 0.6):
        segs = geo.trace((-5.0, b, z), (1.0, 0.0, 0.0))
        chord = [length for v, length in segs if setup.volumes[v].name == "detector"]
        assert chord == pytest.approx([2.0 * np.sqrt(det.radius**2 - b**2)], abs=1e-6)


def test_isotropic_paths_end_on_outer_surface(setup, geo):
    # Total path from the source to the world boundary must equal the
    # analytic distance to the outer cylinder of the shield.
    body = setup.volumes[setup.volume_index("body")]
    rng = np.random.default_rng(0)
    p = np.array(setup.source)
    for _ in range(500):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        total = sum(length for _, length in geo.trace(p, d))

        a = d[0] ** 2 + d[1] ** 2
        t_side = np.inf
        if a > 0:
            b = p[0] * d[0] + p[1] * d[1]
            c = p[0] ** 2 + p[1] ** 2 - body.radius**2
            t_side = (-b + np.sqrt(b * b - a * c)) / a
        t_cap = (body.z_max - p[2]) / d[2] if d[2] > 0 else (body.z_min - p[2]) / d[2]
        assert total == pytest.approx(min(t_side, t_cap), abs=1e-5)


def test_detector_outside_cavity_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        'shield = "canberra_747E"\ndetector = "nai_3x3in"\n'
        "detector_face_height = 2.0\nsource_distance = 10.0\n"
    )
    with pytest.raises(ValueError, match="does not fit"):
        load_setup(bad)
