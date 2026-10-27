"""Radioactive sources: photon emission lines from LNHB/DDEP data.

Each nuclide is read from its Nucleide-Lara file in ``data/sources``
(``<nuclide>.lara.txt``), downloaded unmodified from
http://www.lnhb.fr/nuclides/. Intensities are photons per 100 decays and
may add up to more than 100 %.

Two cuts are applied when a source is loaded:

* ``min_energy`` (default 10 keV): lower-energy lines (L X-rays, Cr K
  X-rays of Mn-54) are absorbed by the detector housing, which is not
  modelled, or fall below the acquisition threshold.
* ``min_intensity`` (default 0.01 %): negligible lines are dropped.

Each simulated photon is drawn from a single line; coincidences between
photons emitted in the same decay are not modelled.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .materials import DATA_DIR

SOURCES_DIR = DATA_DIR / "sources"

_SECONDS = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0, "a": 365.25 * 86400.0}


def _half_life_seconds(header: dict[str, list[str]]) -> float | None:
    """Half-life in seconds from the 'Half-life (unit)' header lines."""
    values = {}
    for key, fields in header.items():
        if key.startswith("Half-life (") and fields:
            unit = key[len("Half-life ("):-1]
            if unit in _SECONDS:
                values[unit] = float(fields[0]) * _SECONDS[unit]
    if not values:
        return None
    return values.get("s", next(iter(values.values())))


def read_lara(path: str | Path) -> tuple[dict[str, list[str]], list[dict]]:
    """Read a Nucleide-Lara text file.

    Returns the header, as ``{key: [values...]}``, and the photon emission
    lines as dictionaries with keys ``energy``, ``energy_unc``,
    ``intensity``, ``intensity_unc``, ``type`` and ``origin``.
    """
    header: dict[str, list[str]] = {}
    lines: list[dict] = []
    in_table = False

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            text = raw.strip()
            if not text:
                continue
            if text.startswith("Energy (keV)"):
                in_table = True
                continue
            if text.startswith("=") or text.startswith("-"):
                continue

            fields = [x.strip() for x in text.split(";")]
            if not in_table:
                header[fields[0]] = [x for x in fields[1:] if x]
                continue

            kind = fields[4]
            if not (kind.startswith("X") or kind.startswith("g")):
                continue  # not a photon (e.g. alpha lines)
            try:
                line = {
                    "energy": float(fields[0]),
                    "energy_unc": float(fields[1]) if fields[1] else 0.0,
                    "intensity": float(fields[2]),
                    "intensity_unc": float(fields[3]) if fields[3] else 0.0,
                    "type": kind,
                    "origin": fields[5],
                }
            except ValueError:
                warnings.warn(f"{path}: skipping unreadable line: {text}", stacklevel=2)
                continue
            lines.append(line)
    return header, lines


@dataclass(frozen=True, eq=False)
class Source:
    """Photon emission lines of a nuclide after the energy/intensity cuts."""

    nuclide: str
    reference: str
    energy: np.ndarray  # keV
    intensity: np.ndarray  # photons per 100 decays
    kind: tuple[str, ...]
    origin: tuple[str, ...]
    half_life: float | None = None  # s

    @property
    def photons_per_decay(self) -> float:
        """Mean number of simulated photons emitted per decay."""
        return float(self.intensity.sum()) / 100.0

    @property
    def probabilities(self) -> np.ndarray:
        """Probability that an emitted photon belongs to each line."""
        return self.intensity / self.intensity.sum()

    def line_sampler(self):
        """C++ sampler of photon energies used by the transport code."""
        from ._core import LineSampler

        return LineSampler(self.energy, self.intensity)

    def __str__(self) -> str:
        rows = [f"{self.nuclide}  ({self.reference})",
                f"{'E (keV)':>10s} {'I (%)':>9s}  type"]
        for e, i, k, o in zip(self.energy, self.intensity, self.kind, self.origin):
            rows.append(f"{e:10.4f} {i:9.4f}  {k} ({o})")
        rows.append(f"photons per decay: {self.photons_per_decay:.4f}")
        return "\n".join(rows)


def load_source(nuclide: str, min_energy: float = 10.0, min_intensity: float = 0.01,
                directory: str | Path | None = None) -> Source:
    """Load the emission lines of ``nuclide`` (e.g. ``"Cs-137"``)."""
    directory = SOURCES_DIR if directory is None else Path(directory)
    path = directory / f"{nuclide}.lara.txt"
    if not path.exists():
        available = sorted(p.name.removesuffix(".lara.txt") for p in directory.glob("*.lara.txt"))
        raise FileNotFoundError(f"no data for {nuclide!r}; available: {available}")

    header, lines = read_lara(path)
    kept = [x for x in lines if x["energy"] >= min_energy and x["intensity"] >= min_intensity]
    if not kept:
        raise ValueError(f"{nuclide}: no lines left after the energy/intensity cuts")

    return Source(
        nuclide=header.get("Nuclide", [nuclide])[0],
        reference=" ".join(header.get("Reference", ["unknown"])),
        energy=np.array([x["energy"] for x in kept]),
        intensity=np.array([x["intensity"] for x in kept]),
        kind=tuple(x["type"] for x in kept),
        origin=tuple(x["origin"] for x in kept),
        half_life=_half_life_seconds(header),
    )        
    
