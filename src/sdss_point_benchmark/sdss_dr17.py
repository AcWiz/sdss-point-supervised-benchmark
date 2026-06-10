from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .schema import BANDS, SourceRecord


@dataclass(frozen=True)
class SdssFieldManifestRecord:
    field_id: str
    run: int
    rerun: int
    camcol: int
    field: int
    status: str
    catalog_path: str
    n_objects: int
    frame_paths: Mapping[str, str]


def build_sdss_field_manifest(
    data_root: str | Path,
    bands: Sequence[str] = BANDS,
) -> list[SdssFieldManifestRecord]:
    """Build a field-level manifest from SDSS frame and catalog manifests."""

    root = Path(data_root)
    frames_path = root / "manifest_frames.csv"
    catalogs_path = root / "manifest_catalogs.csv"
    if not frames_path.exists():
        raise FileNotFoundError(f"missing frame manifest: {frames_path}")
    if not catalogs_path.exists():
        raise FileNotFoundError(f"missing catalog manifest: {catalogs_path}")

    frame_groups: dict[tuple[int, int, int, int], dict[str, str]] = {}
    with frames_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = _field_key(row)
            if row.get("status") == "exists" and row.get("path"):
                frame_groups.setdefault(key, {})[row["band"]] = row["path"]

    catalog_rows: dict[tuple[int, int, int, int], dict[str, str]] = {}
    with catalogs_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            catalog_rows[_field_key(row)] = row

    records: list[SdssFieldManifestRecord] = []
    all_keys = sorted(set(frame_groups) | set(catalog_rows))
    for run, rerun, camcol, field in all_keys:
        frames = frame_groups.get((run, rerun, camcol, field), {})
        catalog = catalog_rows.get((run, rerun, camcol, field), {})
        has_bands = all(band in frames for band in bands)
        has_catalog = catalog.get("status") == "downloaded" and bool(catalog.get("path"))
        status = "ready" if has_bands and has_catalog else "partial"
        records.append(
            SdssFieldManifestRecord(
                field_id=sdss_field_id(run, camcol, field),
                run=run,
                rerun=rerun,
                camcol=camcol,
                field=field,
                status=status,
                catalog_path=catalog.get("path", ""),
                n_objects=int(float(catalog.get("n_objects") or 0)),
                frame_paths={band: frames.get(band, "") for band in bands},
            )
        )
    return records


def write_field_manifest(records: Iterable[SdssFieldManifestRecord], output: str | Path) -> None:
    """Write field-level manifest CSV with stable columns for automation."""

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "field_id",
        "run",
        "rerun",
        "camcol",
        "field",
        "status",
        "catalog_path",
        "n_objects",
        *[f"frame_{band}" for band in BANDS],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "field_id": record.field_id,
                "run": record.run,
                "rerun": record.rerun,
                "camcol": record.camcol,
                "field": record.field,
                "status": record.status,
                "catalog_path": record.catalog_path,
                "n_objects": record.n_objects,
            }
            row.update({f"frame_{band}": record.frame_paths.get(band, "") for band in BANDS})
            writer.writerow(row)


def load_sdss_source_catalog(path: str | Path, clean_only: bool = False) -> list[SourceRecord]:
    """Load SDSS PhotoObj-style CSV rows into the benchmark SourceRecord schema."""

    records: list[SourceRecord] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = _label_from_type(row.get("type_name", ""))
            if label is None:
                continue
            if clean_only and row.get("clean") not in {"1", "True", "true"}:
                continue
            run = int(float(row["run"]))
            camcol = int(float(row["camcol"]))
            field = int(float(row["field"]))
            source_id = row.get("objID") or f"{run}-{camcol}-{field}-{row.get('obj', len(records))}"
            records.append(
                SourceRecord(
                    source_id=str(source_id),
                    cutout_id=sdss_field_id(run, camcol, field),
                    ra=float(row["ra"]),
                    dec=float(row["dec"]),
                    label=label,
                    x=_optional_float(row.get("colc_r")),
                    y=_optional_float(row.get("rowc_r")),
                    mag_u=_sdss_magnitude(row, label, "u"),
                    mag_g=_sdss_magnitude(row, label, "g"),
                    mag_r=_sdss_magnitude(row, label, "r"),
                    mag_i=_sdss_magnitude(row, label, "i"),
                    mag_z=_sdss_magnitude(row, label, "z"),
                    size=_optional_float(row.get("petroR50_r")),
                    ellipticity=_ellipticity(row.get("expAB_r")),
                    galactic_latitude=_optional_float(row.get("b")),
                    region_id=sdss_field_id(run, camcol, field),
                    metadata={
                        "run": run,
                        "rerun": int(float(row.get("rerun") or 0)),
                        "camcol": camcol,
                        "field": field,
                        "sdss_type_name": row.get("type_name", ""),
                        "clean": row.get("clean", ""),
                        "flags": row.get("flags", ""),
                    },
                )
            )
    return records


def sdss_field_id(run: int, camcol: int, field: int) -> str:
    return f"run{run:06d}_camcol{camcol}_field{field:04d}"


def _field_key(row: Mapping[str, str]) -> tuple[int, int, int, int]:
    return (
        int(float(row["run"])),
        int(float(row["rerun"])),
        int(float(row["camcol"])),
        int(float(row["field"])),
    )


def _label_from_type(type_name: str) -> str | None:
    normalized = type_name.strip().lower()
    if normalized == "star":
        return "star"
    if normalized == "galaxy":
        return "galaxy"
    return None


def _sdss_magnitude(row: Mapping[str, str], label: str, band: str) -> float | None:
    preferred = [f"psfMag_{band}", f"cModelMag_{band}", f"modelMag_{band}"] if label == "star" else [
        f"cModelMag_{band}",
        f"modelMag_{band}",
        f"psfMag_{band}",
    ]
    for key in preferred:
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return None


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if number <= -9990:
        return None
    return number


def _ellipticity(axis_ratio: str | None) -> float | None:
    value = _optional_float(axis_ratio)
    if value is None:
        return None
    return max(0.0, min(1.0, 1.0 - value))
