"""Photon attenuation data for the materials used in the simulations.

Mass attenuation coefficients are read from NIST XCOM output files
(standard grid, cm2/g) and converted to linear attenuation coefficients
(cm^-1) using the densities listed in ``data/materials.toml``.

Only incoherent scattering and photoelectric absorption are exposed,
since these are the only interactions handled by the transport code.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

XCOM_COLUMNS = (
    "energy",
    "coherent",
    "incoherent",
    "photoelectric",
    "pair_nuclear",
    "pair_electron",
    "total_with_coherent",
    "total_without_coherent",
)


def read_xcom(path: str | Path) -> dict[str, np.ndarray]:
    """Read an XCOM output file.

    Energies are returned in keV and coefficients in cm2/g. At an
    absorption edge XCOM repeats the energy, giving the values just below
    and just above the edge; when two edges coincide (e.g. Na K and I M1
    in NaI) more than two rows may share the same energy. Within each
    group of equal energies only the first and last rows are kept, so
    every edge becomes exactly one below/above pair.
    """
    data = np.loadtxt(path, skiprows=3, ndmin=2)
    if data.shape[1] != len(XCOM_COLUMNS):
        raise ValueError(
            f"{path}: expected {len(XCOM_COLUMNS)} columns, found {data.shape[1]}"
        )

    energy = data[:, 0] * 1.0e3  # MeV -> keV
    if np.any(np.diff(energy) < 0):
        raise ValueError(f"{path}: energies are not in increasing order")

    same_as_previous = np.r_[False, energy[1:] == energy[:-1]]
    same_as_next = np.r_[energy[1:] == energy[:-1], False]
    keep = ~(same_as_previous & same_as_next)

    table = {name: data[keep, i] for i, name in enumerate(XCOM_COLUMNS)}
    table["energy"] = energy[keep]
    return table


def loglog_interp(x, xp: np.ndarray, fp: np.ndarray):
    """Linear interpolation in (ln x, ln f), clamped at the table limits.

    ``xp`` may contain repeated values at absorption edges. The interval
    is located with a left-sided search (the same rule as
    ``std::lower_bound`` in the C++ engine), so energies below an edge use
    the value below the edge and energies above it use the value above.
    """
    x = np.asarray(x, dtype=float)
    idx = np.clip(np.searchsorted(xp, x, side="left"), 1, len(xp) - 1)

    lx0, lx1 = np.log(xp[idx - 1]), np.log(xp[idx])
    ly0, ly1 = np.log(fp[idx - 1]), np.log(fp[idx])
    t = (np.log(x) - lx0) / (lx1 - lx0)
    y = np.exp(ly0 + t * (ly1 - ly0))

    y = np.where(x <= xp[0], fp[0], y)
    y = np.where(x >= xp[-1], fp[-1], y)
    return y.item() if y.ndim == 0 else y


@dataclass(frozen=True, eq=False)
class Material:
    """A homogeneous material with tabulated attenuation coefficients.

    ``energy`` is in keV; ``incoherent`` and ``photoelectric`` are mass
    attenuation coefficients in cm2/g; ``density`` is in g/cm3.
    """

    name: str
    description: str
    density: float
    energy: np.ndarray
    incoherent: np.ndarray
    photoelectric: np.ndarray

    def mu_incoherent(self, energy):
        """Linear incoherent scattering coefficient (cm^-1)."""
        return self.density * loglog_interp(energy, self.energy, self.incoherent)

    def mu_photoelectric(self, energy):
        """Linear photoelectric absorption coefficient (cm^-1)."""
        return self.density * loglog_interp(energy, self.energy, self.photoelectric)

    def mu_total(self, energy):
        """Sum of the incoherent and photoelectric coefficients (cm^-1)."""
        return self.mu_incoherent(energy) + self.mu_photoelectric(energy)

    def linear_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Energy grid (keV) and linear coefficients (cm^-1) for the C++ engine."""
        return (
            self.energy.copy(),
            self.density * self.incoherent,
            self.density * self.photoelectric,
        )


def load_materials(path: str | Path | None = None) -> dict[str, Material]:
    """Load every material defined in a materials TOML file."""
    path = Path(path) if path is not None else DATA_DIR / "materials.toml"
    with open(path, "rb") as f:
        config = tomllib.load(f)

    materials = {}
    for name, entry in config.items():
        table = read_xcom(path.parent / entry["xcom"])
        materials[name] = Material(
            name=name,
            description=entry.get("description", ""),
            density=float(entry["density"]),
            energy=table["energy"],
            incoherent=table["incoherent"],
            photoelectric=table["photoelectric"],
        )
    return materials
