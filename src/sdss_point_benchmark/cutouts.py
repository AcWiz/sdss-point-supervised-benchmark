from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from .schema import SourceRecord


def write_cutout_manifest(
    records: Sequence[SourceRecord],
    output: str | Path,
    cutout_size: int = 128,
    limit: int | None = None,
) -> None:
    """Write a deterministic cutout worklist centered on catalog source points."""

    selected = list(records[:limit] if limit is not None else records)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cutout_id",
        "source_id",
        "parent_cutout_id",
        "ra",
        "dec",
        "x",
        "y",
        "label",
        "size_pixels",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in selected:
            writer.writerow(
                {
                    "cutout_id": f"{record.cutout_id}__{record.source_id}",
                    "source_id": record.source_id,
                    "parent_cutout_id": record.cutout_id,
                    "ra": record.ra,
                    "dec": record.dec,
                    "x": "" if record.x is None else record.x,
                    "y": "" if record.y is None else record.y,
                    "label": record.label,
                    "size_pixels": cutout_size,
                }
            )
