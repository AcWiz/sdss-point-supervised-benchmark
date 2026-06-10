from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .schema import SourceRecord


@dataclass(frozen=True)
class InjectionSpec:
    source_id: str
    x: float
    y: float
    fluxes: Mapping[str, float]
    kind: str
    radius: float = 1.5
    ellipticity: float = 0.0
    metadata: Mapping[str, str | int | float] = field(default_factory=dict)


def inject_sources(
    image: np.ndarray,
    specs: Sequence[InjectionSpec],
    bands: Sequence[str] = ("u", "g", "r", "i", "z"),
    psf_sigma: float = 1.3,
    cutout_id: str = "synthetic",
) -> tuple[np.ndarray, list[SourceRecord]]:
    """Inject simple PSF-star and Sersic-like galaxy profiles into a multiband image."""

    if image.ndim != 3:
        raise ValueError("image must have shape (bands, height, width)")
    if image.shape[0] != len(bands):
        raise ValueError("image band axis must match bands")

    output = image.astype(np.float32, copy=True)
    truth: list[SourceRecord] = []

    for spec in specs:
        for band_index, band in enumerate(bands):
            flux = float(spec.fluxes.get(band, 0.0))
            if flux == 0.0:
                continue
            sigma = psf_sigma if spec.kind == "psf_star" else max(psf_sigma, spec.radius)
            profile = _normalized_gaussian(output.shape[1:], spec.x, spec.y, sigma)
            output[band_index] += flux * profile
        truth.append(
            SourceRecord(
                source_id=spec.source_id,
                cutout_id=cutout_id,
                ra=spec.x,
                dec=spec.y,
                label="star" if spec.kind == "psf_star" else "galaxy",
                flux_u=spec.fluxes.get("u"),
                flux_g=spec.fluxes.get("g"),
                flux_r=spec.fluxes.get("r"),
                flux_i=spec.fluxes.get("i"),
                flux_z=spec.fluxes.get("z"),
                size=spec.radius,
                ellipticity=spec.ellipticity,
                metadata=spec.metadata,
            )
        )
    return output, truth


def _normalized_gaussian(shape: tuple[int, int], x: float, y: float, sigma: float) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float32)
    profile = np.exp(-0.5 * (((xx - x) / sigma) ** 2 + ((yy - y) / sigma) ** 2))
    total = float(profile.sum())
    if total <= 0.0:
        return profile
    return profile / total
