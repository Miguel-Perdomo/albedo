"""Detector response: energy resolution applied to the deposited energy.

The transport code returns the energy deposited in the crystal. A real
detector converts it into a pulse height with a statistical spread; here
each deposit is replaced by a Gaussian deviate centred on it, with

    FWHM(E) = a sqrt(E) + b E          (E in keV)

The a sqrt(E) term reflects the Poisson statistics of the light and
photoelectron production; the b E term collects contributions that scale
with the signal (crystal non-proportionality, light collection
non-uniformity). The parameters come from the [resolution] table of the
detector file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class Resolution:
    """Gaussian energy resolution, FWHM(E) = a sqrt(E) + b E."""

    a: float  # keV^(1/2)
    b: float  # dimensionless
    reference: str = ""

    def fwhm(self, energy):
        """Full width at half maximum (keV) at ``energy`` (keV)."""
        e = np.clip(np.asarray(energy, dtype=float), 0.0, None)
        return self.a * np.sqrt(e) + self.b * e

    def relative(self, energy):
        """Resolution FWHM/E."""
        e = np.asarray(energy, dtype=float)
        return self.fwhm(e) / e

    def smear(self, energy, seed: int = 0) -> np.ndarray:
        """Replace each energy by a Gaussian deviate of width FWHM(E)."""
        e = np.asarray(energy, dtype=float)
        rng = np.random.default_rng(seed)
        return rng.normal(e, self.fwhm(e) * FWHM_TO_SIGMA)

    @classmethod
    def from_config(cls, cfg: dict) -> "Resolution":
        model = cfg.get("model", "sqrt_linear")
        if model != "sqrt_linear":
            raise ValueError(f"unknown resolution model {model!r}")
        return cls(a=float(cfg["a"]), b=float(cfg["b"]), reference=cfg.get("reference", ""))
