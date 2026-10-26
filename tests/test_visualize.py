import matplotlib

matplotlib.use("Agg")

import pytest

from albedo import load_setup, plotting
from albedo.visualize import section, view3d


@pytest.fixture(scope="module")
def setup():
    return load_setup("csi_747E")


def test_section(setup, tmp_path):
    plotting.use_style(usetex=False)
    fig, ax = section(setup)
    assert len(ax.patches) >= len(setup.volumes)
    fig.savefig(tmp_path / "section.png")


def test_view3d(setup, tmp_path):
    pytest.importorskip("plotly")
    fig = view3d(setup)
    legend = {t.name for t in fig.data if t.showlegend is not False}
    assert set(setup.material_names) | {"Source"} <= legend
    fig.write_html(tmp_path / "geometry.html")
    assert (tmp_path / "geometry.html").stat().st_size > 0
