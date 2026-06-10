from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import BANDS


@dataclass(frozen=True)
class FieldImageStack:
    images: np.ndarray
    bands: tuple[str, ...]
    headers: dict[str, dict[str, int | float | str]]


def load_field_image_stack(
    frame_paths: Mapping[str, str | Path],
    bands: Sequence[str] = BANDS,
    dtype: np.dtype | type = np.float32,
) -> FieldImageStack:
    """Load native SDSS corrected-frame images into a band-first array."""

    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise ImportError("reading SDSS FITS frames requires astropy; install the 'fits' extra") from exc

    arrays: list[np.ndarray] = []
    headers: dict[str, dict[str, int | float | str]] = {}
    expected_shape: tuple[int, int] | None = None
    for band in bands:
        path = Path(frame_paths.get(band, ""))
        if not path:
            raise FileNotFoundError(f"missing frame path for band {band}")
        if not path.exists():
            raise FileNotFoundError(f"missing frame file for band {band}: {path}")
        with fits.open(path, memmap=False) as hdul:
            hdu = next((candidate for candidate in hdul if candidate.data is not None), None)
            if hdu is None:
                raise ValueError(f"FITS file has no image data for band {band}: {path}")
            data = np.asarray(hdu.data, dtype=dtype)
            if data.ndim != 2:
                raise ValueError(f"expected 2D image for band {band}, got shape {data.shape}: {path}")
            shape = (int(data.shape[0]), int(data.shape[1]))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError(f"band {band} shape {shape} does not match expected {expected_shape}")
            arrays.append(data)
            headers[band] = _header_summary(hdu.header, shape)
    return FieldImageStack(images=np.stack(arrays, axis=0), bands=tuple(bands), headers=headers)


def _header_summary(header: Mapping[str, object], shape: tuple[int, int]) -> dict[str, int | float | str]:
    keys = [
        "SIMPLE",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "RUN",
        "RERUN",
        "CAMCOL",
        "FIELD",
        "FILTER",
        "CRPIX1",
        "CRPIX2",
        "CRVAL1",
        "CRVAL2",
        "CTYPE1",
        "CTYPE2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
        "CDELT1",
        "CDELT2",
    ]
    summary: dict[str, int | float | str] = {"NAXIS1": shape[1], "NAXIS2": shape[0]}
    for key in keys:
        if key not in header:
            continue
        value = header[key]
        if isinstance(value, (str, int, float, bool)):
            summary[key] = value
    return summary
