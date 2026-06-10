from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

BANDS = ("u", "g", "r", "i", "z")


@dataclass(frozen=True)
class SourceRecord:
    """Ground-truth or weak-label source entry at catalog grain."""

    source_id: str
    cutout_id: str
    ra: float
    dec: float
    label: str
    x: float | None = None
    y: float | None = None
    mag_u: float | None = None
    mag_g: float | None = None
    mag_r: float | None = None
    mag_i: float | None = None
    mag_z: float | None = None
    flux_u: float | None = None
    flux_g: float | None = None
    flux_r: float | None = None
    flux_i: float | None = None
    flux_z: float | None = None
    size: float | None = None
    ellipticity: float | None = None
    crowding: float | None = None
    snr: float | None = None
    seeing: float | None = None
    psf_fwhm: float | None = None
    nearest_neighbor_arcsec: float | None = None
    galactic_latitude: float | None = None
    region_id: str | None = None
    quality_flags: str | None = None
    label_quality: str | None = None
    label_weight: float | None = None
    raw_source_id: str | None = None
    metadata: Mapping[str, str | int | float] | None = None

    def magnitude(self, band: str) -> float | None:
        return getattr(self, f"mag_{band}")

    def flux(self, band: str) -> float | None:
        return getattr(self, f"flux_{band}")


@dataclass(frozen=True)
class PredictionRecord:
    """Predicted catalog source entry."""

    prediction_id: str
    cutout_id: str
    ra: float
    dec: float
    label: str
    score: float
    x: float | None = None
    y: float | None = None
    mag_u: float | None = None
    mag_g: float | None = None
    mag_r: float | None = None
    mag_i: float | None = None
    mag_z: float | None = None
    flux_u: float | None = None
    flux_g: float | None = None
    flux_r: float | None = None
    flux_i: float | None = None
    flux_z: float | None = None
    size: float | None = None
    ellipticity: float | None = None

    def magnitude(self, band: str) -> float | None:
        return getattr(self, f"mag_{band}")

    def flux(self, band: str) -> float | None:
        return getattr(self, f"flux_{band}")
