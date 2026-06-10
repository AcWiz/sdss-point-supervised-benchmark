from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .fits_io import FieldImageStack, load_field_image_stack
from .io import SOURCE_FIELDNAMES
from .quality import (
    LABEL_QUALITY_TO_INT,
    QUALITY_TO_WEIGHT,
    raw_flag_value,
    source_label_quality,
    source_label_weight,
    source_quality_flags,
)
from .schema import BANDS, SourceRecord
from .sdss_dr17 import SdssFieldManifestRecord, build_sdss_field_manifest, load_sdss_source_catalog

LABEL_TO_INT = {"star": 1, "galaxy": 2}


@dataclass(frozen=True)
class CutoutLabel:
    source_id: str
    x: float
    y: float
    ra: float
    dec: float
    label: str
    mag_u: float | None
    mag_g: float | None
    mag_r: float | None
    mag_i: float | None
    mag_z: float | None
    flags: int
    quality_flags: tuple[str, ...]
    label_quality: str
    label_weight: float


@dataclass(frozen=True)
class CutoutSample:
    field_id: str
    center_source_id: str
    center_ra: float
    center_dec: float
    center_x: float
    center_y: float
    center_label: str
    origin_x: int
    origin_y: int
    image: np.ndarray
    labels: tuple[CutoutLabel, ...]
    center_index: int
    center_quality_flags: tuple[str, ...]
    center_label_quality: str
    center_label_weight: float


DATASET_MANIFEST_FIELDNAMES = [
    "shard",
    "sample_index",
    "shard_sample_index",
    "cutout_id",
    "field_id",
    "center_source_id",
    "ra",
    "dec",
    "x",
    "y",
    "origin_x",
    "origin_y",
    "label",
    "quality_flags",
    "label_quality",
    "label_weight",
    "n_sources",
]


class _SourceWindowIndex:
    def __init__(self, sources: Sequence[SourceRecord], cell_size: int) -> None:
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self._cell_size = cell_size
        self._cells: dict[tuple[int, int], list[tuple[int, SourceRecord]]] = {}
        for order, source in enumerate(sources):
            if not _has_valid_xy(source):
                continue
            cell = self._cell_for(float(source.x), float(source.y))
            self._cells.setdefault(cell, []).append((order, source))

    def query(self, origin_x: int, origin_y: int, cutout_size: int) -> list[SourceRecord]:
        end_x = np.nextafter(float(origin_x + cutout_size), -math.inf)
        end_y = np.nextafter(float(origin_y + cutout_size), -math.inf)
        min_cell_x, min_cell_y = self._cell_for(float(origin_x), float(origin_y))
        max_cell_x, max_cell_y = self._cell_for(end_x, end_y)
        matches: list[tuple[int, SourceRecord]] = []
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                for order, source in self._cells.get((cell_x, cell_y), []):
                    x = float(source.x)
                    y = float(source.y)
                    if origin_x <= x < origin_x + cutout_size and origin_y <= y < origin_y + cutout_size:
                        matches.append((order, source))
        matches.sort(key=lambda item: item[0])
        return [source for _, source in matches]

    def _cell_for(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / self._cell_size), math.floor(y / self._cell_size))


def build_cutouts_for_field(
    field_id: str,
    images: np.ndarray,
    sources: Sequence[SourceRecord],
    cutout_size: int = 128,
    *,
    image_dtype: np.dtype | type | None = None,
) -> tuple[list[CutoutSample], dict[str, int]]:
    """Build center-source cutouts from one native field image stack."""

    if images.ndim != 3:
        raise ValueError(f"expected images with shape (bands, height, width), got {images.shape}")
    _, height, width = images.shape
    half_size = cutout_size // 2
    stats = {
        "candidate_sources": 0,
        "invalid_type": 0,
        "invalid_center": 0,
        "edge_unsafe": 0,
        "empty_labels": 0,
        "suspect_center_excluded": 0,
        "accepted": 0,
    }
    accepted: list[CutoutSample] = []
    valid_sources = [source for source in sources if source.label in LABEL_TO_INT]
    source_index = _SourceWindowIndex(valid_sources, cell_size=cutout_size)
    for center in sources:
        if center.label not in LABEL_TO_INT:
            stats["invalid_type"] += 1
            continue
        stats["candidate_sources"] += 1
        if not _has_valid_xy(center):
            stats["invalid_center"] += 1
            continue
        center_label_quality = source_label_quality(center.metadata)
        center_label_weight = source_label_weight(center.metadata)
        if center_label_quality == "suspect":
            stats["suspect_center_excluded"] += 1
            continue
        origin_x = int(math.floor(float(center.x))) - half_size
        origin_y = int(math.floor(float(center.y))) - half_size
        if origin_x < 0 or origin_y < 0 or origin_x + cutout_size > width or origin_y + cutout_size > height:
            stats["edge_unsafe"] += 1
            continue

        labels = _labels_in_window(source_index.query(origin_x, origin_y, cutout_size), origin_x, origin_y, cutout_size)
        center_index = next((index for index, label in enumerate(labels) if label.source_id == center.source_id), -1)
        if center_index < 0:
            stats["empty_labels"] += 1
            continue
        image = images[:, origin_y : origin_y + cutout_size, origin_x : origin_x + cutout_size]
        image = image.copy() if image_dtype is None else image.astype(image_dtype, copy=True)
        accepted.append(
            CutoutSample(
                field_id=field_id,
                center_source_id=center.source_id,
                center_ra=center.ra,
                center_dec=center.dec,
                center_x=float(center.x),
                center_y=float(center.y),
                center_label=center.label,
                origin_x=origin_x,
                origin_y=origin_y,
                image=image,
                labels=tuple(labels),
                center_index=center_index,
                center_quality_flags=tuple(source_quality_flags(center.metadata)),
                center_label_quality=center_label_quality,
                center_label_weight=center_label_weight,
            )
        )
        stats["accepted"] += 1
    return accepted, stats


def encode_cutout_batch(cutouts: Sequence[CutoutSample], dtype: np.dtype | type = np.float16) -> dict[str, np.ndarray]:
    """Encode cutout images and variable-length labels into NPZ-ready arrays."""

    if not cutouts:
        raise ValueError("cannot encode an empty cutout batch")
    source_offsets = [0]
    source_x: list[float] = []
    source_y: list[float] = []
    source_label: list[int] = []
    source_flags: list[int] = []
    source_quality: list[int] = []
    source_weight: list[float] = []
    source_mags: dict[str, list[float]] = {band: [] for band in BANDS}
    center_index: list[int] = []
    for cutout in cutouts:
        center_index.append(cutout.center_index)
        for label in cutout.labels:
            source_x.append(label.x)
            source_y.append(label.y)
            source_label.append(LABEL_TO_INT[label.label])
            source_flags.append(label.flags)
            source_quality.append(LABEL_QUALITY_TO_INT[label.label_quality])
            source_weight.append(label.label_weight)
            for band in BANDS:
                source_mags[band].append(_nan_if_none(getattr(label, f"mag_{band}")))
        source_offsets.append(len(source_x))

    encoded = {
        "images": np.stack([cutout.image.astype(dtype, copy=False) for cutout in cutouts], axis=0),
        "source_offsets": np.asarray(source_offsets, dtype=np.int64),
        "source_x": np.asarray(source_x, dtype=np.float32),
        "source_y": np.asarray(source_y, dtype=np.float32),
        "source_label": np.asarray(source_label, dtype=np.int16),
        "source_flags": np.asarray(source_flags, dtype=np.uint64),
        "source_quality": np.asarray(source_quality, dtype=np.int8),
        "source_weight": np.asarray(source_weight, dtype=np.float32),
        "center_index": np.asarray(center_index, dtype=np.int64),
    }
    for band in BANDS:
        encoded[f"source_mag_{band}"] = np.asarray(source_mags[band], dtype=np.float32)
    return encoded


def build_dataset(
    config_path: str | Path,
    output_dir: str | Path,
    limit_fields: int | None = 100,
    cutout_size: int = 128,
    shard_size: int = 1024,
    dtype: str = "float16",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build or dry-run a native-frame SDSS pilot cutout dataset."""

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_config = config.get("data", {})
    data_root = Path(data_config["root"])
    bands = tuple(data_config.get("bands") or BANDS)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_records = build_sdss_field_manifest(data_root, bands=bands)
    ready_records = [record for record in all_records if record.status == "ready"]
    selected_records = ready_records[:limit_fields] if limit_fields is not None else ready_records
    np_dtype = np.dtype(dtype)

    sample_count = 0
    qa = _empty_qa(dry_run=dry_run)
    field_metadata: list[dict[str, Any]] = []
    writer = None if dry_run else _StreamingDatasetWriter(output / "manifest.csv", output / "truth_catalog.csv", output / "shards", shard_size, np_dtype)
    try:
        for record in selected_records:
            field_stack = _load_stack_for_record(record, bands)
            sources = load_sdss_source_catalog(record.catalog_path, clean_only=False)
            cutouts, stats = build_cutouts_for_field(
                record.field_id,
                field_stack.images,
                sources,
                cutout_size=cutout_size,
                image_dtype=np_dtype,
            )
            qa["filter_counts"] = _merge_counts(qa["filter_counts"], stats)
            qa["type_distribution"] = _merge_counts(qa["type_distribution"], _type_distribution(sources))
            quality_distribution = _label_quality_distribution(sources)
            qa["label_quality_counts"] = _merge_counts(qa["label_quality_counts"], quality_distribution)
            qa["label_quality_per_field"][record.field_id] = quality_distribution
            qa["samples_per_field"][record.field_id] = len(cutouts)
            qa["image_value_quantiles"][record.field_id] = _image_quantiles(field_stack.images)
            field_metadata.append(
                {
                    "field_id": record.field_id,
                    "run": record.run,
                    "rerun": record.rerun,
                    "camcol": record.camcol,
                    "field": record.field,
                    "headers": field_stack.headers,
                }
            )
            sample_count += len(cutouts)
            if writer is not None:
                writer.write_many(cutouts)
            del cutouts, sources, field_stack
    finally:
        if writer is not None:
            writer.close()

    qa["field_count"] = len(selected_records)
    qa["sample_count"] = sample_count
    qa["estimated_output_bytes"] = _estimated_output_bytes(sample_count, len(bands), cutout_size, np_dtype)
    _write_json(output / "qa_report.json", qa)

    metadata = {
        "protocol": config.get("protocol", "sdss-point-supervised-v1"),
        "config_path": str(config_path),
        "config": config,
        "generated_at": _utc_now(),
        "data_root": str(data_root),
        "output_dir": str(output),
        "bands": list(bands),
        "cutout_size": cutout_size,
        "dtype": str(np_dtype),
        "limit_fields": limit_fields,
        "field_count": len(selected_records),
        "sample_count": sample_count,
        "shard_size": shard_size,
        "dry_run": dry_run,
        "native_frame_policy": "native SDSS corrected-frame pixel grids; no WCS reprojection",
        "psf": {"status": "unavailable", "reason": "psField files are not part of the v1 pilot input tree"},
        "label_quality_policy": _label_quality_policy_metadata(),
        "git": _git_metadata(),
        "environment": _environment_metadata(),
        "fields": field_metadata,
    }
    _write_json(output / "metadata.json", metadata)
    return {"metadata": metadata, "qa": qa}


class _StreamingDatasetWriter:
    def __init__(
        self,
        manifest_path: Path,
        truth_catalog_path: Path,
        shards_dir: Path,
        shard_size: int,
        dtype: np.dtype,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self._shards_dir = shards_dir
        self._shard_size = shard_size
        self._dtype = dtype
        self._buffer: list[CutoutSample] = []
        self._sample_count = 0
        self._shard_index = 0

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        truth_catalog_path.parent.mkdir(parents=True, exist_ok=True)
        shards_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_handle = manifest_path.open("w", newline="", encoding="utf-8")
        self._truth_handle = truth_catalog_path.open("w", newline="", encoding="utf-8")
        self._manifest_writer = csv.DictWriter(self._manifest_handle, fieldnames=DATASET_MANIFEST_FIELDNAMES)
        self._truth_writer = csv.DictWriter(self._truth_handle, fieldnames=SOURCE_FIELDNAMES)
        self._manifest_writer.writeheader()
        self._truth_writer.writeheader()
        self._closed = False

    def write_many(self, cutouts: Sequence[CutoutSample]) -> None:
        self._raise_if_closed()
        for cutout in cutouts:
            shard_index = self._sample_count // self._shard_size
            shard_sample_index = self._sample_count % self._shard_size
            self._manifest_writer.writerow(_manifest_row(cutout, shard_index, self._sample_count, shard_sample_index))
            for record in _truth_records_for_cutout(cutout):
                self._truth_writer.writerow(_source_record_row(record))
            self._buffer.append(cutout)
            self._sample_count += 1
            if len(self._buffer) >= self._shard_size:
                self._flush_shard()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._flush_shard()
        finally:
            self._manifest_handle.close()
            self._truth_handle.close()
            self._closed = True

    def _flush_shard(self) -> None:
        if not self._buffer:
            return
        shard_path = self._shards_dir / f"shard_{self._shard_index:06d}.npz"
        np.savez_compressed(shard_path, **encode_cutout_batch(self._buffer, dtype=self._dtype))
        self._buffer.clear()
        self._shard_index += 1

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise ValueError("cannot write to a closed dataset writer")


def _load_stack_for_record(record: SdssFieldManifestRecord, bands: Sequence[str]) -> FieldImageStack:
    return load_field_image_stack({band: record.frame_paths.get(band, "") for band in bands}, bands=bands)


def _labels_in_window(
    sources: Iterable[SourceRecord],
    origin_x: int,
    origin_y: int,
    cutout_size: int,
) -> list[CutoutLabel]:
    labels: list[CutoutLabel] = []
    for source in sources:
        if not _has_valid_xy(source):
            continue
        x = float(source.x)
        y = float(source.y)
        if not (origin_x <= x < origin_x + cutout_size and origin_y <= y < origin_y + cutout_size):
            continue
        labels.append(
            CutoutLabel(
                source_id=source.source_id,
                x=x - origin_x,
                y=y - origin_y,
                ra=source.ra,
                dec=source.dec,
                label=source.label,
                mag_u=source.mag_u,
                mag_g=source.mag_g,
                mag_r=source.mag_r,
                mag_i=source.mag_i,
                mag_z=source.mag_z,
                flags=raw_flag_value(source.metadata),
                quality_flags=tuple(source_quality_flags(source.metadata)),
                label_quality=source_label_quality(source.metadata),
                label_weight=source_label_weight(source.metadata),
            )
        )
    return labels


def _has_valid_xy(source: SourceRecord) -> bool:
    return (
        source.x is not None
        and source.y is not None
        and math.isfinite(float(source.x))
        and math.isfinite(float(source.y))
    )


def _write_manifest_and_shards(
    cutouts: Sequence[CutoutSample],
    manifest_path: Path,
    shards_dir: Path,
    shard_size: int,
    dtype: np.dtype,
) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_MANIFEST_FIELDNAMES)
        writer.writeheader()
        for sample_index, cutout in enumerate(cutouts):
            shard_index = sample_index // shard_size
            shard_sample_index = sample_index % shard_size
            writer.writerow(_manifest_row(cutout, shard_index, sample_index, shard_sample_index))

    for shard_index, start in enumerate(range(0, len(cutouts), shard_size)):
        shard_cutouts = cutouts[start : start + shard_size]
        if not shard_cutouts:
            continue
        np.savez_compressed(shards_dir / f"shard_{shard_index:06d}.npz", **encode_cutout_batch(shard_cutouts, dtype=dtype))


def _manifest_row(
    cutout: CutoutSample,
    shard_index: int,
    sample_index: int,
    shard_sample_index: int,
) -> dict[str, str | int | float]:
    return {
        "shard": f"shard_{shard_index:06d}.npz",
        "sample_index": sample_index,
        "shard_sample_index": shard_sample_index,
        "cutout_id": f"{cutout.field_id}__{cutout.center_source_id}",
        "field_id": cutout.field_id,
        "center_source_id": cutout.center_source_id,
        "ra": cutout.center_ra,
        "dec": cutout.center_dec,
        "x": cutout.center_x,
        "y": cutout.center_y,
        "origin_x": cutout.origin_x,
        "origin_y": cutout.origin_y,
        "label": cutout.center_label,
        "quality_flags": ";".join(cutout.center_quality_flags),
        "label_quality": cutout.center_label_quality,
        "label_weight": cutout.center_label_weight,
        "n_sources": len(cutout.labels),
    }


def _truth_records_for_cutouts(cutouts: Sequence[CutoutSample]) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for cutout in cutouts:
        records.extend(_truth_records_for_cutout(cutout))
    return records


def _truth_records_for_cutout(cutout: CutoutSample) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    cutout_id = f"{cutout.field_id}__{cutout.center_source_id}"
    for label in cutout.labels:
        records.append(
            SourceRecord(
                source_id=f"{cutout_id}__{label.source_id}",
                cutout_id=cutout_id,
                ra=label.ra,
                dec=label.dec,
                label=label.label,
                x=label.x,
                y=label.y,
                mag_u=label.mag_u,
                mag_g=label.mag_g,
                mag_r=label.mag_r,
                mag_i=label.mag_i,
                mag_z=label.mag_z,
                region_id=cutout.field_id,
                quality_flags=";".join(label.quality_flags),
                label_quality=label.label_quality,
                label_weight=label.label_weight,
                raw_source_id=label.source_id,
                metadata={"raw_source_id": label.source_id, "flags": label.flags},
            )
        )
    return records


def _source_record_row(record: SourceRecord) -> dict[str, str | float]:
    return {field: "" if getattr(record, field) is None else getattr(record, field) for field in SOURCE_FIELDNAMES}


def _empty_qa(dry_run: bool) -> dict[str, Any]:
    return {
        "dry_run": dry_run,
        "field_count": 0,
        "sample_count": 0,
        "estimated_output_bytes": 0,
        "filter_counts": {
            "candidate_sources": 0,
            "invalid_type": 0,
            "invalid_center": 0,
            "edge_unsafe": 0,
            "empty_labels": 0,
            "suspect_center_excluded": 0,
            "accepted": 0,
        },
        "type_distribution": {},
        "label_quality_counts": {},
        "label_quality_per_field": {},
        "samples_per_field": {},
        "image_value_quantiles": {},
        "missing_or_abnormal_files": [],
    }


def _merge_counts(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, int]:
    merged = dict(left)
    for key, value in right.items():
        merged[key] = int(merged.get(key, 0)) + int(value)
    return merged


def _type_distribution(sources: Sequence[SourceRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.label] = counts.get(source.label, 0) + 1
    return counts


def _label_quality_distribution(sources: Sequence[SourceRecord]) -> dict[str, int]:
    counts = {"clean": 0, "suspect": 0, "weak": 0}
    for source in sources:
        if source.label not in LABEL_TO_INT:
            continue
        quality = source_label_quality(source.metadata)
        counts[quality] += 1
    return {key: value for key, value in counts.items() if value}


def _label_quality_policy_metadata() -> dict[str, Any]:
    return {
        "name": "sdss-photoobj-conservative-v1",
        "quality_to_weight": dict(sorted(QUALITY_TO_WEIGHT.items())),
        "quality_to_int": dict(sorted(LABEL_QUALITY_TO_INT.items())),
        "suspect_center_policy": "exclude suspect center sources from cutout generation",
        "suspect_neighbor_policy": "retain suspect neighbor labels with source_weight=0.0 for auditability",
    }


def _image_quantiles(images: np.ndarray) -> dict[str, float]:
    finite = images[np.isfinite(images)]
    if finite.size == 0:
        return {"p01": math.nan, "p50": math.nan, "p99": math.nan}
    p01, p50, p99 = np.percentile(finite, [1.0, 50.0, 99.0])
    return {"p01": float(p01), "p50": float(p50), "p99": float(p99)}


def _estimated_output_bytes(sample_count: int, band_count: int, cutout_size: int, dtype: np.dtype) -> int:
    image_bytes = sample_count * band_count * cutout_size * cutout_size * dtype.itemsize
    label_bytes = sample_count * 6 * 64
    return int(image_bytes + label_bytes)


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_metadata() -> dict[str, str | None]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": run_git(["status", "--short"]),
    }


def _environment_metadata() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
