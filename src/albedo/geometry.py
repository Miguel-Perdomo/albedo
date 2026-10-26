"""Geometry of the experimental setup: shield, detector and source.

A setup is described by three TOML files in ``configs/`` (shield,
detector and the setup combining them) and converted into an ordered list
of coaxial cylinders. A point belongs to the first cylinder in the list
that contains it, so solid cylinders nested inside each other produce
shells: the cavity is listed before the Cu liner, the Cu liner before the
Sn liner, and so on.

Coordinates: z is the shield axis, pointing up, and z = 0 is the floor of
the free cavity (inside the liner). All lengths are in cm.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class Volume:
    """Solid cylinder on the z axis. ``material`` is None for vacuum."""

    name: str
    material: str | None
    radius: float
    z_min: float
    z_max: float

    def contains(self, x: float, y: float, z: float) -> bool:
        return self.z_min <= z <= self.z_max and x * x + y * y <= self.radius**2


@dataclass(frozen=True)
class Setup:
    """Volumes in priority order plus the source position."""

    name: str
    volumes: tuple[Volume, ...]
    source: tuple[float, float, float]

    @property
    def material_names(self) -> list[str]:
        """Materials present, in order of first appearance."""
        names = []
        for v in self.volumes:
            if v.material is not None and v.material not in names:
                names.append(v.material)
        return names

    def volume_index(self, name: str) -> int:
        for i, v in enumerate(self.volumes):
            if v.name == name:
                return i
        raise KeyError(f"no volume named {name!r}")

    def build(self, materials):
        """Create the C++ geometry and the attenuation table of each material.

        Returns ``(geometry, tables)``, where ``tables[i]`` belongs to
        ``material_names[i]``, the index stored in each cylinder.
        """
        from ._core import VACUUM, Geometry

        index = {name: i for i, name in enumerate(self.material_names)}
        geometry = Geometry()
        for v in self.volumes:
            material = VACUUM if v.material is None else index[v.material]
            geometry.add_cylinder(v.radius, v.z_min, v.z_max, material)
        tables = [materials[name].attenuation_table() for name in self.material_names]
        return geometry, tables


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def shield_volumes(cfg: dict) -> list[Volume]:
    """Cavity, liner layers and body of a cylindrical shield."""
    r = cfg["cavity_diameter"] / 2.0
    h = cfg["cavity_height"]
    volumes = [Volume("cavity", None, r, 0.0, h)]

    t = 0.0
    for layer in cfg.get("liner", []):
        t += layer["thickness"]
        volumes.append(Volume(f"liner_{layer['material']}", layer["material"], r + t, -t, h + t))

    outer_r = cfg["outer_diameter"] / 2.0
    cap = (cfg["outer_height"] - h) / 2.0
    if outer_r <= r + t or cap <= t:
        raise ValueError("shield body is thinner than the liner")
    volumes.append(Volume("body", cfg["body"], outer_r, -cap, h + cap))
    return volumes


def detector_volume(cfg: dict, face_height: float) -> Volume:
    """Crystal on the axis with its front face at ``face_height``, facing up."""
    return Volume(
        "detector",
        cfg["material"],
        cfg["diameter"] / 2.0,
        face_height - cfg["length"],
        face_height,
    )


def load_setup(name: str | Path) -> Setup:
    """Load a setup by name (``configs/setups/<name>.toml``) or by path."""
    path = Path(name)
    if not path.exists():
        path = CONFIG_DIR / "setups" / f"{name}.toml"
    cfg = _read_toml(path)

    shield = shield_volumes(_read_toml(CONFIG_DIR / "shields" / f"{cfg['shield']}.toml"))
    face = cfg["detector_face_height"]
    detector = detector_volume(_read_toml(CONFIG_DIR / "detectors" / f"{cfg['detector']}.toml"), face)
    source_z = face + cfg["source_distance"]

    cavity = shield[0]
    if not (
        detector.radius <= cavity.radius
        and detector.z_min >= cavity.z_min
        and detector.z_max <= cavity.z_max
    ):
        raise ValueError("detector does not fit inside the cavity")
    if not (cfg["source_distance"] > 0.0 and source_z < cavity.z_max):
        raise ValueError("source must lie between the detector face and the lid")

    return Setup(name=path.stem, volumes=(detector, *shield), source=(0.0, 0.0, source_z))
