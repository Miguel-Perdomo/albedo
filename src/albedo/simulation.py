"""Run the photon transport and collect the detected events.

``simulate`` combines a setup (geometry), a source and the material data,
runs the C++ transport and returns a ``SimulationResult`` holding, for
every history that deposited energy in the detector:

* ``deposited``: total energy deposited (keV), before any resolution;
* ``emitted``: energy of the emitted photon (keV);
* ``origin``: index of the volume where the photon last interacted before
  reaching the detector, or -1 if it arrived without interacting. This
  splits the spectrum into its direct and scattered components.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Setup, load_setup
from .materials import load_materials
from .sources import Source, load_source

DIRECT = -1


@dataclass(frozen=True, eq=False)
class SimulationResult:
    setup: Setup
    source: Source
    n_histories: int
    seed: int
    deposited: np.ndarray
    emitted: np.ndarray
    origin: np.ndarray

    @property
    def efficiency(self) -> float:
        """Fraction of emitted photons that deposit energy in the detector."""
        return len(self.deposited) / self.n_histories

    @property
    def decays(self) -> float:
        """Number of source decays equivalent to the simulated histories."""
        return self.n_histories / self.source.photons_per_decay

    def origin_name(self, index: int) -> str:
        return "direct" if index == DIRECT else self.setup.volumes[index].name

    def components(self) -> dict[str, np.ndarray]:
        """Boolean masks of the events, keyed by origin name."""
        return {self.origin_name(i): self.origin == i for i in np.unique(self.origin)}


    def smeared(self, seed: int = 0, resolution=None) -> np.ndarray:
        """Deposited energies broadened with the detector resolution.

        Uses the resolution of the setup's detector unless another one is
        given. The same seed always gives the same broadening.
        """
        resolution = self.setup.resolution if resolution is None else resolution
        if resolution is None:
            raise ValueError("the detector of this setup has no [resolution] table")
        return resolution.smear(self.deposited, seed=seed)

    def spectrum(self, bin_width: float = 1.0, e_max: float | None = None, mask=None,
                 smear: bool = False, seed: int = 0):
        """Histogram of deposited energy. Returns ``(bin_centres, counts)``.

        With ``smear`` the detector resolution is applied before binning.
        """
        if e_max is None:
            e_max = float(self.source.energy.max()) * 1.1
        edges = np.arange(0.0, e_max + bin_width, bin_width)
        values = self.smeared(seed) if smear else self.deposited
        if mask is not None:
            values = values[mask]
        counts, _ = np.histogram(values, bins=edges)
        return 0.5 * (edges[:-1] + edges[1:]), counts


def simulate(setup: Setup | str, source: Source | str, n_histories: int, seed: int = 1,
             energy_cut: float = 1.0, materials=None) -> SimulationResult:
    """Simulate ``n_histories`` photons emitted by ``source`` in ``setup``."""
    from ._core import Transport

    if isinstance(setup, str):
        setup = load_setup(setup)
    if isinstance(source, str):
        source = load_source(source)
    if materials is None:
        materials = load_materials()

    geometry, tables = setup.build(materials)
    transport = Transport(geometry, tables, source.line_sampler(), setup.source,
                          setup.volume_index("detector"), energy_cut)
    out = transport.run(int(n_histories), int(seed))
    return SimulationResult(setup, source, int(n_histories), int(seed),
                            out["deposited"], out["emitted"], out["origin"])
