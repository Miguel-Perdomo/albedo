import numpy as np
import pytest

from albedo.sources import SOURCES_DIR, load_source, read_lara

NUCLIDES = ["Cs-137", "Ba-133", "Mn-54", "Y-88"]


@pytest.mark.parametrize("nuclide", NUCLIDES)
def test_cuts_are_applied(nuclide):
    src = load_source(nuclide)
    assert np.all(src.energy >= 10.0)
    assert np.all(src.intensity >= 0.01)
    assert np.all(np.diff(src.energy) > 0)
    assert np.isclose(src.probabilities.sum(), 1.0)


def test_read_raw_file():
    header, lines = read_lara(SOURCES_DIR / "Ba-133.lara.txt")
    assert header["Nuclide"] == ["Ba-133"]
    assert len(lines) == 14
    assert lines[0]["type"] == "XL"


def test_ba133_lines():
    src = load_source("Ba-133")
    assert len(src.energy) == 13  # 14 lines minus the Cs L X-rays
    k_xrays = src.intensity[[k.startswith("XK") for k in src.kind]].sum()
    assert k_xrays == pytest.approx(33.8 + 62.4 + 18.24 + 4.45)
    i356 = src.intensity[np.isclose(src.energy, 356.0129)]
    assert i356 == pytest.approx([62.05])


def test_cs137_lines():
    src = load_source("Cs-137")
    assert src.energy[-1] == pytest.approx(661.6553)
    assert src.intensity[-1] == pytest.approx(85.01)
    # Ba K X-rays: about 7 photons per 100 decays, not 15
    assert src.intensity[:-1].sum() == pytest.approx(1.99 + 3.66 + 1.078 + 0.272)


def test_mn54_main_line():
    src = load_source("Mn-54")
    gamma = src.intensity[np.isclose(src.energy, 834.848)]
    assert gamma == pytest.approx([99.9752])


def test_y88_lines():
    src = load_source("Y-88")
    for e, i in [(898.042, 93.7), (1836.070, 99.346), (2734.092, 0.608)]:
        assert src.intensity[np.isclose(src.energy, e)] == pytest.approx([i])
    assert not np.any(np.isclose(src.energy, 3218.426))  # below 0.01 %
    assert src.photons_per_decay > 2.0


def test_cuts_can_be_relaxed():
    assert len(load_source("Ba-133", min_energy=0.0).energy) == 14


def test_unknown_nuclide():
    with pytest.raises(FileNotFoundError, match="available"):
        load_source("Co-60x")


@pytest.mark.parametrize("nuclide", ["Ba-133", "Y-88"])
def test_sampler_follows_intensities(nuclide):
    src = load_source(nuclide)
    n = 500_000
    energies = src.line_sampler().sample(n, seed=11)
    counts = np.array([np.sum(energies == e) for e in src.energy])
    assert counts.sum() == n  # every draw is one of the lines
    expected = n * src.probabilities
    chi2 = np.sum((counts - expected) ** 2 / expected)
    ndf = len(counts) - 1
    assert abs(chi2 - ndf) < 5.0 * np.sqrt(2.0 * ndf)
