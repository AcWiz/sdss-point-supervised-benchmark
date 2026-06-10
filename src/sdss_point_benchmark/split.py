from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .schema import SourceRecord


@dataclass(frozen=True)
class RegionSplit:
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    train_regions: tuple[str, ...]
    val_regions: tuple[str, ...]
    test_regions: tuple[str, ...]


def assign_region_id(ra: float, dec: float, ra_bin_deg: float = 1.0, dec_bin_deg: float = 1.0) -> str:
    """Assign a deterministic sky-region bin and wrap RA into [0, 360)."""

    if ra_bin_deg <= 0 or dec_bin_deg <= 0:
        raise ValueError("ra_bin_deg and dec_bin_deg must be positive")
    wrapped_ra = ra % 360.0
    ra_index = int(wrapped_ra // ra_bin_deg)
    dec_index = int((dec + 90.0) // dec_bin_deg)
    return f"r{ra_index:03d}_d{dec_index:03d}"


def make_region_split(
    records: Sequence[SourceRecord],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    ra_bin_deg: float = 1.0,
    dec_bin_deg: float = 1.0,
    seed: int = 0,
    region_mode: str = "sky-bin",
) -> RegionSplit:
    """Split source IDs by sky region so no region appears in multiple splits."""

    if region_mode not in {"sky-bin", "catalog-region"}:
        raise ValueError("region_mode must be 'sky-bin' or 'catalog-region'")
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")
    if not records:
        return RegionSplit((), (), (), (), (), ())

    by_region: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        if region_mode == "catalog-region":
            region = record.region_id or assign_region_id(record.ra, record.dec, ra_bin_deg, dec_bin_deg)
        else:
            region = assign_region_id(record.ra, record.dec, ra_bin_deg, dec_bin_deg)
        by_region[region].append(record)

    regions = sorted(by_region)
    rng = random.Random(seed)
    rng.shuffle(regions)

    n_regions = len(regions)
    train_cut = round(n_regions * train_fraction)
    val_cut = train_cut + round(n_regions * val_fraction)

    if n_regions >= 3:
        train_cut = min(max(train_cut, 1), n_regions - 2)
        val_cut = min(max(val_cut, train_cut + 1), n_regions - 1)

    train_regions = tuple(sorted(regions[:train_cut]))
    val_regions = tuple(sorted(regions[train_cut:val_cut]))
    test_regions = tuple(sorted(regions[val_cut:]))

    return RegionSplit(
        train_ids=_ids_for_regions(by_region, train_regions),
        val_ids=_ids_for_regions(by_region, val_regions),
        test_ids=_ids_for_regions(by_region, test_regions),
        train_regions=train_regions,
        val_regions=val_regions,
        test_regions=test_regions,
    )


def _ids_for_regions(by_region: dict[str, list[SourceRecord]], regions: Iterable[str]) -> tuple[str, ...]:
    ids = [record.source_id for region in regions for record in by_region[region]]
    return tuple(sorted(ids))
