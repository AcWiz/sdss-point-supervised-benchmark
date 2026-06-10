from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import sqrt
from statistics import mean, pstdev

from .matching import MatchResult, angular_distance_arcsec
from .schema import PredictionRecord, SourceRecord


def detection_metrics(matches: MatchResult) -> dict[str, float]:
    tp = len(matches.matches)
    fp = len(matches.unmatched_prediction_ids)
    fn = len(matches.unmatched_truth_ids)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"tp": float(tp), "fp": float(fp), "fn": float(fn), "precision": precision, "recall": recall, "f1": f1}


def detection_average_precision(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    radius_arcsec: float,
) -> dict[str, float]:
    """Compute detection AP by sweeping unique prediction score thresholds."""

    return detection_score_curve(
        truth,
        predictions,
        max_radius_arcsec=radius_arcsec,
    )["average_precision"]


def detection_score_curve(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    *,
    max_radius_arcsec: float,
    psf_fraction: float | None = None,
) -> dict[str, object]:
    """Compute thresholded detection metrics while reusing candidate distances."""

    thresholds = sorted({prediction.score for prediction in predictions}, reverse=True)
    if not thresholds:
        empty = {"ap": 0.0, "best_f1": 0.0, "n_thresholds": 0.0}
        return {
            "best_threshold": 0.0,
            "best_metrics": {"tp": 0.0, "fp": 0.0, "fn": float(len(truth)), "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "average_precision": empty,
            "thresholds": [],
        }

    candidate_edges = build_candidate_edges(
        truth,
        predictions,
        max_radius_arcsec=max_radius_arcsec,
        psf_fraction=psf_fraction,
    )
    points: list[tuple[float, float]] = []
    rows: list[dict[str, float]] = []
    best_threshold = 0.0
    best_metrics = {"tp": 0.0, "fp": 0.0, "fn": float(len(truth)), "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in thresholds:
        active_prediction_ids = {prediction.prediction_id for prediction in predictions if prediction.score >= threshold}
        metrics = detection_metrics_from_edges(
            candidate_edges,
            truth_count=len(truth),
            prediction_count=len(active_prediction_ids),
            active_prediction_ids=active_prediction_ids,
        )
        rows.append({"threshold": threshold, **metrics})
        points.append((metrics["recall"], metrics["precision"]))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = metrics

    return {
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "average_precision": average_precision_from_points(points, best_metrics["f1"], len(thresholds)),
        "thresholds": rows,
    }


def build_candidate_edges(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    *,
    max_radius_arcsec: float,
    psf_fraction: float | None = None,
) -> list[tuple[float, str, str]]:
    predictions_by_cutout: dict[str, list[PredictionRecord]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_cutout[prediction.cutout_id].append(prediction)

    edges: list[tuple[float, str, str]] = []
    for truth_record in truth:
        radius_arcsec = max_radius_arcsec
        if psf_fraction is not None and truth_record.psf_fwhm is not None:
            radius_arcsec = min(max_radius_arcsec, psf_fraction * truth_record.psf_fwhm)
        for prediction in predictions_by_cutout.get(truth_record.cutout_id, []):
            distance = angular_distance_arcsec(truth_record.ra, truth_record.dec, prediction.ra, prediction.dec)
            if distance <= radius_arcsec:
                edges.append((distance, truth_record.source_id, prediction.prediction_id))
    return sorted(edges)


def detection_metrics_from_edges(
    candidate_edges: Sequence[tuple[float, str, str]],
    *,
    truth_count: int,
    prediction_count: int,
    active_prediction_ids: set[str],
) -> dict[str, float]:
    matched_truth: set[str] = set()
    matched_predictions: set[str] = set()
    for _, truth_id, prediction_id in candidate_edges:
        if prediction_id not in active_prediction_ids:
            continue
        if truth_id in matched_truth or prediction_id in matched_predictions:
            continue
        matched_truth.add(truth_id)
        matched_predictions.add(prediction_id)

    tp = len(matched_predictions)
    fp = prediction_count - tp
    fn = truth_count - tp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"tp": float(tp), "fp": float(fp), "fn": float(fn), "precision": precision, "recall": recall, "f1": f1}


def average_precision_from_points(
    points: Sequence[tuple[float, float]],
    best_f1: float,
    n_thresholds: int,
) -> dict[str, float]:
    precision_by_recall: dict[float, float] = {}
    for recall, precision in points:
        precision_by_recall[recall] = max(precision_by_recall.get(recall, 0.0), precision)

    envelope = 0.0
    envelope_by_recall: dict[float, float] = {}
    for recall in sorted(precision_by_recall, reverse=True):
        envelope = max(envelope, precision_by_recall[recall])
        envelope_by_recall[recall] = envelope

    ap = 0.0
    previous_recall = 0.0
    for recall in sorted(envelope_by_recall):
        ap += max(0.0, recall - previous_recall) * envelope_by_recall[recall]
        previous_recall = max(previous_recall, recall)

    return {"ap": ap, "best_f1": best_f1, "n_thresholds": float(n_thresholds)}


def classification_metrics(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    matches: MatchResult,
) -> dict[str, float]:
    labels = sorted({record.label for record in truth} | {record.label for record in predictions})
    if not labels:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    confusion: dict[str, dict[str, int]] = {label: defaultdict(int) for label in labels}
    correct = 0
    for match in matches.matches:
        truth_label = matches.truth_by_id[match.truth_id].label
        prediction_label = matches.prediction_by_id[match.prediction_id].label
        confusion[truth_label][prediction_label] += 1
        if truth_label == prediction_label:
            correct += 1

    f1s = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1s.append(_safe_div(2 * precision * recall, precision + recall))

    return {
        "accuracy": _safe_div(correct, len(matches.matches)),
        "macro_f1": mean(f1s),
    }


def photometry_metrics(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    matches: MatchResult,
    band: str = "r",
    outlier_threshold_mag: float = 0.5,
) -> dict[str, float]:
    deltas = []
    for match in matches.matches:
        truth_mag = matches.truth_by_id[match.truth_id].magnitude(band)
        prediction_mag = matches.prediction_by_id[match.prediction_id].magnitude(band)
        if truth_mag is None or prediction_mag is None:
            continue
        deltas.append(prediction_mag - truth_mag)

    if not deltas:
        return {"n": 0.0, "bias_mag": 0.0, "scatter_mag": 0.0, "outlier_rate": 0.0}
    return {
        "n": float(len(deltas)),
        "bias_mag": mean(deltas),
        "scatter_mag": pstdev(deltas),
        "outlier_rate": mean([abs(delta) > outlier_threshold_mag for delta in deltas]),
    }


def astrometry_metrics(matches: MatchResult) -> dict[str, float]:
    distances = [match.distance_arcsec for match in matches.matches]
    if not distances:
        return {"n": 0.0, "centroid_mae_arcsec": 0.0, "centroid_rmse_arcsec": 0.0}
    return {
        "n": float(len(distances)),
        "centroid_mae_arcsec": mean(distances),
        "centroid_rmse_arcsec": sqrt(mean([distance**2 for distance in distances])),
    }


def binned_detection_metrics(
    records: Sequence[SourceRecord],
    matches: MatchResult,
    bins: Sequence[float],
    field: str = "mag_r",
) -> Mapping[str, dict[str, float]]:
    """Report detection completeness by a scalar truth field such as magnitude or SNR."""

    matched = {match.truth_id for match in matches.matches}
    out: dict[str, dict[str, float]] = {}
    for lo, hi in zip(bins, bins[1:], strict=False):
        members = [record for record in records if (value := getattr(record, field)) is not None and lo <= value < hi]
        found = [record for record in members if record.source_id in matched]
        out[f"[{lo},{hi})"] = {"n": float(len(members)), "recall": _safe_div(len(found), len(members))}
    return out


def stratified_detection_report(
    records: Sequence[SourceRecord],
    matches: MatchResult,
    bins_by_field: Mapping[str, Sequence[float]],
) -> dict[str, Mapping[str, dict[str, float]]]:
    """Return detection completeness reports for multiple truth fields."""

    return {
        field: binned_detection_metrics(records, matches, bins, field=field)
        for field, bins in bins_by_field.items()
    }


def deblending_metrics(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    matches: MatchResult,
    close_pair_arcsec: float = 2.0,
    band: str = "r",
) -> dict[str, float]:
    """Evaluate close-pair detection and matched-source flux attribution."""

    del predictions
    close_truth = [
        record
        for record in truth
        if record.nearest_neighbor_arcsec is not None and record.nearest_neighbor_arcsec < close_pair_arcsec
    ]
    matched_truth_ids = {match.truth_id for match in matches.matches}
    close_detected = [record for record in close_truth if record.source_id in matched_truth_ids]

    attribution_errors = []
    for match in matches.matches:
        truth_record = matches.truth_by_id[match.truth_id]
        if truth_record.source_id not in {record.source_id for record in close_truth}:
            continue
        truth_flux = truth_record.flux(band)
        prediction_flux = matches.prediction_by_id[match.prediction_id].flux(band)
        if truth_flux is None or prediction_flux is None or truth_flux == 0:
            continue
        attribution_errors.append(abs(prediction_flux - truth_flux) / abs(truth_flux))

    close_pair_recall = _safe_div(len(close_detected), len(close_truth))
    return {
        "close_pair_truth": float(len(close_truth)),
        "close_pair_detected": float(len(close_detected)),
        "close_pair_recall": close_pair_recall,
        "missed_companion_rate": 1.0 - close_pair_recall if close_truth else 0.0,
        "flux_attribution_mae": mean(attribution_errors) if attribution_errors else 0.0,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
