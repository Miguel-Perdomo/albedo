import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import numpy as np
import pytest

from albedo import plotting


def test_style_and_save(tmp_path):
    plotting.use_style(usetex=False)
    assert mpl.rcParams["xtick.direction"] == "in"
    assert mpl.rcParams["xtick.top"] and mpl.rcParams["ytick.right"]

    fig, ax = plotting.figure(18, 7)
    assert fig.get_size_inches() == pytest.approx((18 / 2.54, 7 / 2.54))

    energy = np.arange(0.5, 800.0, 1.0)
    plotting.plot_spectrum(ax, energy, np.exp(-energy / 200.0), label="test", logy=True)
    ax.legend()
    paths = plotting.save(fig, "spectrum", directory=tmp_path, formats=("png",))
    assert paths[0].exists()


def test_missing_latex_falls_back(monkeypatch):
    monkeypatch.setattr(plotting.shutil, "which", lambda _: None)
    with pytest.warns(UserWarning, match="LaTeX not found"):
        plotting.use_style(usetex=True)
    assert mpl.rcParams["text.usetex"] is False
