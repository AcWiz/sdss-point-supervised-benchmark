from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .schema import PredictionRecord, SourceRecord

SOURCE_FIELDNAMES = [
    "source_id",
    "cutout_id",
    "ra",
    "dec",
    "label",
    "x",
    "y",
    "mag_u",
    "mag_g",
    "mag_r",
    "mag_i",
    "mag_z",
    "flux_u",
    "flux_g",
    "flux_r",
    "flux_i",
    "flux_z",
    "size",
    "ellipticity",
    "crowding",
    "snr",
    "seeing",
    "psf_fwhm",
    "nearest_neighbor_arcsec",
    "galactic_latitude",
    "region_id",
    "quality_flags",
    "label_quality",
    "label_weight",
    "raw_source_id",
]


PREDICTION_FIELDNAMES = [
    "prediction_id",
    "cutout_id",
    "ra",
    "dec",
    "label",
    "score",
    "x",
    "y",
    "mag_u",
    "mag_g",
    "mag_r",
    "mag_i",
    "mag_z",
    "flux_u",
    "flux_g",
    "flux_r",
    "flux_i",
    "flux_z",
    "size",
    "ellipticity",
]


def load_source_catalog(path: str | Path) -> list[SourceRecord]:
    """Load benchmark source labels from a CSV catalog."""

    records: list[SourceRecord] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_id", "cutout_id", "ra", "dec", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"catalog is missing required columns: {sorted(missing)}")
        for row in reader:
            records.append(
                SourceRecord(
                    source_id=row["source_id"],
                    cutout_id=row["cutout_id"],
                    ra=float(row["ra"]),
                    dec=float(row["dec"]),
                    label=row["label"],
                    x=_optional_float(row.get("x")),
                    y=_optional_float(row.get("y")),
                    mag_u=_optional_float(row.get("mag_u")),
                    mag_g=_optional_float(row.get("mag_g")),
                    mag_r=_optional_float(row.get("mag_r")),
                    mag_i=_optional_float(row.get("mag_i")),
                    mag_z=_optional_float(row.get("mag_z")),
                    flux_u=_optional_float(row.get("flux_u")),
                    flux_g=_optional_float(row.get("flux_g")),
                    flux_r=_optional_float(row.get("flux_r")),
                    flux_i=_optional_float(row.get("flux_i")),
                    flux_z=_optional_float(row.get("flux_z")),
                    size=_optional_float(row.get("size")),
                    ellipticity=_optional_float(row.get("ellipticity")),
                    crowding=_optional_float(row.get("crowding")),
                    snr=_optional_float(row.get("snr")),
                    seeing=_optional_float(row.get("seeing")),
                    psf_fwhm=_optional_float(row.get("psf_fwhm")),
                    nearest_neighbor_arcsec=_optional_float(row.get("nearest_neighbor_arcsec")),
                    galactic_latitude=_optional_float(row.get("galactic_latitude")),
                    region_id=_optional_str(row.get("region_id")),
                    quality_flags=_optional_str(row.get("quality_flags")),
                    label_quality=_optional_str(row.get("label_quality")),
                    label_weight=_optional_float(row.get("label_weight")),
                    raw_source_id=_optional_str(row.get("raw_source_id")),
                    metadata=_extra_metadata(row),
                )
            )
    return records


def load_prediction_catalog(path: str | Path) -> list[PredictionRecord]:
    """Load predicted source catalog entries from CSV."""

    records: list[PredictionRecord] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"prediction_id", "cutout_id", "ra", "dec", "label", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"prediction catalog is missing required columns: {sorted(missing)}")
        for row in reader:
            records.append(
                PredictionRecord(
                    prediction_id=row["prediction_id"],
                    cutout_id=row["cutout_id"],
                    ra=float(row["ra"]),
                    dec=float(row["dec"]),
                    label=row["label"],
                    score=float(row["score"]),
                    x=_optional_float(row.get("x")),
                    y=_optional_float(row.get("y")),
                    mag_u=_optional_float(row.get("mag_u")),
                    mag_g=_optional_float(row.get("mag_g")),
                    mag_r=_optional_float(row.get("mag_r")),
                    mag_i=_optional_float(row.get("mag_i")),
                    mag_z=_optional_float(row.get("mag_z")),
                    flux_u=_optional_float(row.get("flux_u")),
                    flux_g=_optional_float(row.get("flux_g")),
                    flux_r=_optional_float(row.get("flux_r")),
                    flux_i=_optional_float(row.get("flux_i")),
                    flux_z=_optional_float(row.get("flux_z")),
                    size=_optional_float(row.get("size")),
                    ellipticity=_optional_float(row.get("ellipticity")),
                )
            )
    return records


def write_source_catalog(records: list[SourceRecord], path: str | Path) -> None:
    """Write source labels using the benchmark CSV contract."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow({field: _record_value(record, field) for field in SOURCE_FIELDNAMES})


def write_prediction_catalog(records: list[PredictionRecord], path: str | Path) -> None:
    """Write predicted catalog entries using the benchmark CSV contract."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow({field: _record_value(record, field) for field in PREDICTION_FIELDNAMES})


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_str(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _extra_metadata(row: dict[str, str]) -> dict[str, Any]:
    known = set(SourceRecord.__dataclass_fields__)
    return {key: value for key, value in row.items() if key not in known and value != ""}


def _record_value(record: PredictionRecord, field: str) -> str | float:
    value = getattr(record, field)
    return "" if value is None else value
