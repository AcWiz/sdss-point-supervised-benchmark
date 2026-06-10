from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt

from .schema import PredictionRecord, SourceRecord


@dataclass(frozen=True)
class CatalogMatch:
    truth_id: str
    prediction_id: str
    distance_arcsec: float


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[CatalogMatch, ...]
    unmatched_truth_ids: list[str]
    unmatched_prediction_ids: list[str]
    truth_by_id: dict[str, SourceRecord]
    prediction_by_id: dict[str, PredictionRecord]


def angular_distance_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle angular separation in arcseconds."""

    ra1_rad, dec1_rad, ra2_rad, dec2_rad = map(radians, (ra1, dec1, ra2, dec2))
    delta_ra = ra2_rad - ra1_rad
    delta_dec = dec2_rad - dec1_rad
    a = sin(delta_dec / 2.0) ** 2 + cos(dec1_rad) * cos(dec2_rad) * sin(delta_ra / 2.0) ** 2
    return 2.0 * asin(min(1.0, sqrt(a))) * 206264.80624709636


def match_catalogs(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    radius_arcsec: float,
) -> MatchResult:
    """Greedy one-to-one source matching by angular distance within each cutout."""

    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    return _match_catalogs_with_radius_fn(truth, predictions, lambda record: radius_arcsec)


def match_catalogs_seeing_aware(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    max_radius_arcsec: float = 1.0,
    psf_fraction: float = 0.5,
) -> MatchResult:
    """Match with per-source radius min(max_radius_arcsec, psf_fraction * PSF_FWHM)."""

    if max_radius_arcsec <= 0 or psf_fraction <= 0:
        raise ValueError("max_radius_arcsec and psf_fraction must be positive")

    def radius_for(record: SourceRecord) -> float:
        if record.psf_fwhm is None:
            return max_radius_arcsec
        return min(max_radius_arcsec, psf_fraction * record.psf_fwhm)

    return _match_catalogs_with_radius_fn(truth, predictions, radius_for)


def _match_catalogs_with_radius_fn(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    radius_for: Callable[[SourceRecord], float],
) -> MatchResult:
    candidates: list[tuple[float, str, str]] = []
    truth_by_id = {record.source_id: record for record in truth}
    prediction_by_id = {record.prediction_id: record for record in predictions}
    predictions_by_cutout: dict[str, list[PredictionRecord]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_cutout[prediction.cutout_id].append(prediction)

    for truth_record in truth:
        for prediction in predictions_by_cutout.get(truth_record.cutout_id, []):
            distance = angular_distance_arcsec(truth_record.ra, truth_record.dec, prediction.ra, prediction.dec)
            if distance <= radius_for(truth_record):
                candidates.append((distance, truth_record.source_id, prediction.prediction_id))

    matched_truth: set[str] = set()
    matched_predictions: set[str] = set()
    matches: list[CatalogMatch] = []
    for distance, truth_id, prediction_id in sorted(candidates):
        if truth_id in matched_truth or prediction_id in matched_predictions:
            continue
        matched_truth.add(truth_id)
        matched_predictions.add(prediction_id)
        matches.append(CatalogMatch(truth_id, prediction_id, distance))

    return MatchResult(
        matches=tuple(matches),
        unmatched_truth_ids=sorted(set(truth_by_id) - matched_truth),
        unmatched_prediction_ids=sorted(set(prediction_by_id) - matched_predictions),
        truth_by_id=truth_by_id,
        prediction_by_id=prediction_by_id,
    )
