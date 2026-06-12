from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_prediction_catalog, load_source_catalog
from .matching import MatchResult, match_catalogs, match_catalogs_seeing_aware
from .metrics import (
    average_precision_from_points,
    build_candidate_edges,
    detection_metrics,
    detection_metrics_from_edges,
    detection_score_curve,
    select_score_thresholds,
)
from .pilot_loop import load_json, write_json
from .schema import PredictionRecord, SourceRecord

AUDIT_SCHEMA_VERSION = 1
DEFAULT_MAG_R_BINS = (14.0, 16.0, 18.0, 20.0, 21.0, 22.0, 23.0)
DEFAULT_NEAREST_NEIGHBOR_BINS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf)
DEFAULT_SOURCE_DENSITY_BINS = (0.0, 3.0, 6.0, 10.0, 20.0, 50.0, math.inf)
DEFAULT_SNR_BINS = (0.0, 5.0, 10.0, 20.0, 50.0, 100.0)
DEFAULT_SEEING_BINS = (0.0, 1.2, 1.6, 2.0, 3.0)


@dataclass(frozen=True)
class RunAuditInputs:
    run_dir: Path
    run_id: str
    summary: Mapping[str, Any]
    val_threshold_sweep: Mapping[str, Any]
    truth: list[SourceRecord]
    predictions: list[PredictionRecord]
    paths: dict[str, str]

    @property
    def selected_threshold(self) -> float:
        return float(self.val_threshold_sweep["best_threshold"])


def build_evidence_audit(
    *,
    baseline_run_dir: str | Path,
    target_run_dir: str | Path,
    baseline_label: str = "baseline",
    target_label: str = "target",
    seed: int = 42,
    bootstrap_iterations: int = 200,
    bootstrap_max_thresholds: int = 128,
) -> dict[str, Any]:
    """Re-audit two existing pilot-loop runs without retraining or test tuning."""

    baseline = load_run_audit_inputs(baseline_run_dir)
    target = load_run_audit_inputs(target_run_dir)
    truth = _select_shared_truth(baseline.truth, target.truth)
    matching_options = _matching_options(target.summary, baseline.summary)
    pixel_scale_arcsec = _pixel_scale_arcsec(target.summary, baseline.summary)
    derived_maps = {
        "nearest_neighbor_arcsec": derive_nearest_neighbor_arcsec(truth, pixel_scale_arcsec=pixel_scale_arcsec),
        "source_density_per_cutout": derive_source_density_per_cutout(truth),
    }
    baseline_metrics = evaluate_run_evidence(
        truth=truth,
        predictions=baseline.predictions,
        threshold=baseline.selected_threshold,
        matching_options=matching_options,
        derived_maps=derived_maps,
    )
    target_metrics = evaluate_run_evidence(
        truth=truth,
        predictions=target.predictions,
        threshold=target.selected_threshold,
        matching_options=matching_options,
        derived_maps=derived_maps,
    )
    bootstrap = bootstrap_paired_deltas(
        truth=truth,
        baseline_predictions=baseline.predictions,
        target_predictions=target.predictions,
        baseline_threshold=baseline.selected_threshold,
        target_threshold=target.selected_threshold,
        matching_options=matching_options,
        seed=seed,
        iterations=bootstrap_iterations,
        max_thresholds=bootstrap_max_thresholds,
    )
    strata_comparison = compare_strata(
        baseline_metrics["strata"],
        target_metrics["strata"],
        baseline_label=baseline_label,
        target_label=target_label,
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol": "sdss-point-supervised-v1",
        "audit": "existing_e50_candidate_evidence",
        "baseline_label": baseline_label,
        "target_label": target_label,
        "runs": {
            baseline_label: _run_summary(baseline),
            target_label: _run_summary(target),
        },
        "matching": matching_options,
        "threshold_policy": {
            "source": "validation",
            "uses_test_threshold_tuning": False,
            baseline_label: {
                "selected_threshold": baseline.selected_threshold,
                "source_file": baseline.paths["val_threshold_sweep"],
            },
            target_label: {
                "selected_threshold": target.selected_threshold,
                "source_file": target.paths["val_threshold_sweep"],
            },
        },
        "aggregate": {
            baseline_label: baseline_metrics["aggregate"],
            target_label: target_metrics["aggregate"],
            "delta": compare_aggregate(baseline_metrics["aggregate"], target_metrics["aggregate"]),
        },
        "strata": {
            baseline_label: baseline_metrics["strata"],
            target_label: target_metrics["strata"],
            "comparison": strata_comparison,
        },
        "bootstrap": bootstrap,
        "claim_summary": build_claim_summary(strata_comparison),
        "notes": [
            "Audit reuses existing truth_test.csv and test candidate predictions.",
            "Thresholds are selected from each run's validation sweep and are not tuned on test.",
            "PhotoObj labels remain weak supervision; this audit strengthens candidate evidence but is not final paper truth.",
        ],
    }


def write_evidence_audit(
    *,
    baseline_run_dir: str | Path,
    target_run_dir: str | Path,
    output_path: str | Path,
    markdown_output_path: str | Path | None = None,
    baseline_label: str = "baseline",
    target_label: str = "target",
    seed: int = 42,
    bootstrap_iterations: int = 200,
    bootstrap_max_thresholds: int = 128,
) -> dict[str, Any]:
    payload = build_evidence_audit(
        baseline_run_dir=baseline_run_dir,
        target_run_dir=target_run_dir,
        baseline_label=baseline_label,
        target_label=target_label,
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_max_thresholds=bootstrap_max_thresholds,
    )
    write_json(output_path, payload)
    if markdown_output_path is not None:
        output = Path(markdown_output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_evidence_audit_markdown(payload), encoding="utf-8")
    return payload


def load_run_audit_inputs(run_dir: str | Path) -> RunAuditInputs:
    root = Path(run_dir)
    pilot_dir = root / "pilot_loop"
    summary_path = pilot_dir / "summary.json"
    val_sweep_path = pilot_dir / "val_threshold_sweep.json"
    summary = load_json(summary_path)
    val_sweep = load_json(val_sweep_path)
    outputs = summary.get("outputs", {}) if isinstance(summary.get("outputs"), Mapping) else {}
    truth_path = Path(str(outputs.get("test_truth") or pilot_dir / "truth_test.csv"))
    predictions_path = Path(str(outputs.get("test_predictions") or pilot_dir / "predictions_test_candidates.csv"))
    report_path = root / "report.json"
    run_id = root.name
    if report_path.exists():
        try:
            report = load_json(report_path)
            run_id = str(report.get("run_id") or run_id)
        except Exception:
            run_id = root.name
    return RunAuditInputs(
        run_dir=root,
        run_id=run_id,
        summary=summary,
        val_threshold_sweep=val_sweep,
        truth=load_source_catalog(truth_path),
        predictions=load_prediction_catalog(predictions_path),
        paths={
            "summary": str(summary_path),
            "val_threshold_sweep": str(val_sweep_path),
            "test_truth": str(truth_path),
            "test_predictions": str(predictions_path),
        },
    )


def evaluate_run_evidence(
    *,
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    threshold: float,
    matching_options: Mapping[str, Any],
    derived_maps: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    scored_predictions = [prediction for prediction in predictions if prediction.score >= threshold]
    matches = match_catalogs_for_options(truth, scored_predictions, matching_options)
    detection = detection_metrics(matches)
    ap = detection_score_curve(
        truth,
        predictions,
        max_radius_arcsec=float(matching_options["radius_arcsec"]),
        psf_fraction=float(matching_options["psf_fraction"]) if matching_options.get("seeing_aware") else None,
    )["average_precision"]
    aggregate = {
        "precision": detection["precision"],
        "recall": detection["recall"],
        "f1": detection["f1"],
        "ap": ap["ap"],
        "tp": detection["tp"],
        "fp": detection["fp"],
        "fn": detection["fn"],
        "truth": float(len(truth)),
        "predictions": float(len(scored_predictions)),
        "candidate_predictions": float(len(predictions)),
    }
    return {
        "aggregate": aggregate,
        "strata": stratified_recall_audit(truth, matches, derived_maps=derived_maps),
    }


def stratified_recall_audit(
    truth: Sequence[SourceRecord],
    matches: MatchResult,
    *,
    derived_maps: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    matched_truth_ids = {match.truth_id for match in matches.matches}
    return {
        "mag_r": binned_recall_status(
            truth,
            matched_truth_ids,
            DEFAULT_MAG_R_BINS,
            value_fn=lambda record: record.mag_r,
        ),
        "label": categorical_recall_status(
            truth,
            matched_truth_ids,
            value_fn=lambda record: record.label,
        ),
        "label_quality": categorical_recall_status(
            truth,
            matched_truth_ids,
            value_fn=lambda record: record.label_quality,
        ),
        "photoobj_flags": photoobj_flag_recall_status(truth, matched_truth_ids),
        "nearest_neighbor_arcsec_catalog": binned_recall_status(
            truth,
            matched_truth_ids,
            DEFAULT_NEAREST_NEIGHBOR_BINS,
            value_fn=lambda record: record.nearest_neighbor_arcsec,
        ),
        "nearest_neighbor_arcsec_derived": binned_recall_status(
            truth,
            matched_truth_ids,
            DEFAULT_NEAREST_NEIGHBOR_BINS,
            value_fn=lambda record: derived_maps["nearest_neighbor_arcsec"].get(record.source_id),
        ),
        "source_density_per_cutout": binned_recall_status(
            truth,
            matched_truth_ids,
            DEFAULT_SOURCE_DENSITY_BINS,
            value_fn=lambda record: derived_maps["source_density_per_cutout"].get(record.source_id),
        ),
        "crowding_catalog": binned_recall_status(
            truth,
            matched_truth_ids,
            DEFAULT_SOURCE_DENSITY_BINS,
            value_fn=lambda record: record.crowding,
        ),
        "snr": binned_recall_status(
            truth,
            matched_truth_ids,
            DEFAULT_SNR_BINS,
            value_fn=lambda record: record.snr,
        ),
        "seeing": binned_recall_status(
            truth,
            matched_truth_ids,
            DEFAULT_SEEING_BINS,
            value_fn=lambda record: record.seeing,
        ),
    }


def derive_nearest_neighbor_arcsec(
    records: Sequence[SourceRecord],
    *,
    pixel_scale_arcsec: float,
) -> dict[str, float]:
    by_cutout: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        by_cutout[record.cutout_id].append(record)
    nearest: dict[str, float] = {}
    for cutout_records in by_cutout.values():
        positioned = [
            record
            for record in cutout_records
            if record.x is not None and record.y is not None and math.isfinite(record.x) and math.isfinite(record.y)
        ]
        if len(positioned) < 2:
            continue
        xy = np.asarray([(float(record.x), float(record.y)) for record in positioned], dtype=np.float64)
        for index, record in enumerate(positioned):
            deltas = xy - xy[index]
            distances = np.sqrt(np.sum(deltas * deltas, axis=1))
            distances[index] = np.inf
            best = float(np.min(distances))
            if math.isfinite(best):
                nearest[record.source_id] = best * pixel_scale_arcsec
    return nearest


def derive_source_density_per_cutout(records: Sequence[SourceRecord]) -> dict[str, float]:
    counts = Counter(record.cutout_id for record in records)
    return {record.source_id: float(counts[record.cutout_id]) for record in records}


def binned_recall_status(
    records: Sequence[SourceRecord],
    matched_truth_ids: set[str],
    bins: Sequence[float],
    *,
    value_fn: Callable[[SourceRecord], float | None],
) -> dict[str, Any]:
    values = [(record, value_fn(record)) for record in records]
    available = [(record, float(value)) for record, value in values if value is not None and math.isfinite(float(value))]
    if not available:
        return {
            "status": "unavailable",
            "reason": "no finite values in truth catalog",
            "total_truth": float(len(records)),
            "available_truth": 0.0,
            "bins": {},
        }
    out: dict[str, dict[str, float]] = {}
    for lo, hi in zip(bins, bins[1:], strict=False):
        members = [record for record, value in available if lo <= value < hi]
        found = [record for record in members if record.source_id in matched_truth_ids]
        out[format_bin(lo, hi)] = {
            "n": float(len(members)),
            "matched": float(len(found)),
            "recall": safe_div(len(found), len(members)),
        }
    return {
        "status": "available",
        "kind": "binned",
        "total_truth": float(len(records)),
        "available_truth": float(len(available)),
        "bins": out,
    }


def categorical_recall_status(
    records: Sequence[SourceRecord],
    matched_truth_ids: set[str],
    *,
    value_fn: Callable[[SourceRecord], str | None],
) -> dict[str, Any]:
    values = [(record, value_fn(record)) for record in records]
    available = [(record, str(value)) for record, value in values if value not in {None, ""}]
    if not available:
        return {
            "status": "unavailable",
            "reason": "no non-empty categories in truth catalog",
            "total_truth": float(len(records)),
            "available_truth": 0.0,
            "categories": {},
        }
    categories: dict[str, dict[str, float]] = {}
    counts = Counter(value for _, value in available)
    for value, _ in sorted(counts.items(), key=lambda row: (-row[1], row[0])):
        members = [record for record, category in available if category == value]
        found = [record for record in members if record.source_id in matched_truth_ids]
        categories[value] = {
            "n": float(len(members)),
            "matched": float(len(found)),
            "recall": safe_div(len(found), len(members)),
        }
    return {
        "status": "available",
        "kind": "categorical",
        "total_truth": float(len(records)),
        "available_truth": float(len(available)),
        "categories": categories,
    }


def photoobj_flag_recall_status(
    records: Sequence[SourceRecord],
    matched_truth_ids: set[str],
) -> dict[str, Any]:
    flag_members: dict[str, list[SourceRecord]] = defaultdict(list)
    no_flag: list[SourceRecord] = []
    for record in records:
        flags = parse_photoobj_flags(record.quality_flags)
        if not flags:
            no_flag.append(record)
        for flag in flags:
            flag_members[flag].append(record)
    if not flag_members and not no_flag:
        return {
            "status": "unavailable",
            "reason": "no PhotoObj quality flags in truth catalog",
            "total_truth": float(len(records)),
            "available_truth": 0.0,
            "categories": {},
        }
    categories: dict[str, dict[str, float]] = {}
    for flag, members in sorted(flag_members.items(), key=lambda row: (-len(row[1]), row[0])):
        found = [record for record in members if record.source_id in matched_truth_ids]
        categories[flag] = {
            "n": float(len(members)),
            "matched": float(len(found)),
            "recall": safe_div(len(found), len(members)),
        }
    if no_flag:
        found = [record for record in no_flag if record.source_id in matched_truth_ids]
        categories["NO_FLAGS"] = {
            "n": float(len(no_flag)),
            "matched": float(len(found)),
            "recall": safe_div(len(found), len(no_flag)),
        }
    return {
        "status": "available",
        "kind": "multi_label_categorical",
        "total_truth": float(len(records)),
        "available_truth": float(len(records) - len(no_flag)),
        "categories": categories,
    }


def parse_photoobj_flags(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    normalized = value.replace("|", ";").replace(",", ";")
    return tuple(flag.strip() for flag in normalized.split(";") if flag.strip())


def compare_aggregate(baseline: Mapping[str, float], target: Mapping[str, float]) -> dict[str, float | None]:
    return {
        "precision": delta(target.get("precision"), baseline.get("precision")),
        "recall": delta(target.get("recall"), baseline.get("recall")),
        "f1": delta(target.get("f1"), baseline.get("f1")),
        "ap": delta(target.get("ap"), baseline.get("ap")),
    }


def compare_strata(
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    baseline_label: str,
    target_label: str,
) -> dict[str, Any]:
    comparison = {}
    for field in sorted(set(baseline) | set(target)):
        base_item = baseline.get(field, {})
        target_item = target.get(field, {})
        if base_item.get("status") != "available" or target_item.get("status") != "available":
            comparison[field] = {
                "status": "unavailable",
                "reason": base_item.get("reason") or target_item.get("reason") or "stratum unavailable in one run",
            }
            continue
        key = "bins" if "bins" in base_item else "categories"
        rows = {}
        for name in sorted(set(base_item.get(key, {})) | set(target_item.get(key, {})), key=sort_stratum_key):
            base_row = base_item.get(key, {}).get(name, {})
            target_row = target_item.get(key, {}).get(name, {})
            rows[name] = {
                f"{baseline_label}_n": base_row.get("n", 0.0),
                f"{baseline_label}_recall": base_row.get("recall", 0.0),
                f"{target_label}_n": target_row.get("n", 0.0),
                f"{target_label}_recall": target_row.get("recall", 0.0),
                "delta_recall": delta(target_row.get("recall", 0.0), base_row.get("recall", 0.0)),
            }
        comparison[field] = {"status": "available", "kind": base_item.get("kind", ""), key: rows}
    return comparison


def build_claim_summary(strata_comparison: Mapping[str, Any]) -> dict[str, Any]:
    available = sorted(field for field, item in strata_comparison.items() if item.get("status") == "available")
    unavailable = {
        field: item.get("reason", "unavailable")
        for field, item in strata_comparison.items()
        if item.get("status") != "available"
    }
    return {
        "available_strata": available,
        "unavailable_strata": unavailable,
        "claimable_strata": [
            field
            for field in available
            if field not in {"nearest_neighbor_arcsec_catalog", "crowding_catalog", "seeing", "snr"}
        ],
    }


def bootstrap_paired_deltas(
    *,
    truth: Sequence[SourceRecord],
    baseline_predictions: Sequence[PredictionRecord],
    target_predictions: Sequence[PredictionRecord],
    baseline_threshold: float,
    target_threshold: float,
    matching_options: Mapping[str, Any],
    seed: int,
    iterations: int,
    max_thresholds: int,
) -> dict[str, Any]:
    if iterations <= 0:
        return {"status": "skipped", "reason": "bootstrap_iterations <= 0"}
    cutout_ids = sorted(
        {record.cutout_id for record in truth}
        | {prediction.cutout_id for prediction in baseline_predictions}
        | {prediction.cutout_id for prediction in target_predictions}
    )
    baseline_fixed = per_cutout_threshold_counts(
        truth,
        baseline_predictions,
        threshold=baseline_threshold,
        matching_options=matching_options,
        cutout_ids=cutout_ids,
    )
    target_fixed = per_cutout_threshold_counts(
        truth,
        target_predictions,
        threshold=target_threshold,
        matching_options=matching_options,
        cutout_ids=cutout_ids,
    )
    baseline_curve = per_cutout_curve_counts(
        truth,
        baseline_predictions,
        matching_options=matching_options,
        cutout_ids=cutout_ids,
        max_thresholds=max_thresholds,
    )
    target_curve = per_cutout_curve_counts(
        truth,
        target_predictions,
        matching_options=matching_options,
        cutout_ids=cutout_ids,
        max_thresholds=max_thresholds,
    )

    rng = np.random.default_rng(seed)
    deltas_by_metric: dict[str, list[float]] = {"f1": [], "recall": [], "ap": []}
    n_cutouts = len(cutout_ids)
    for _ in range(iterations):
        sampled = rng.integers(0, n_cutouts, size=n_cutouts)
        weights = np.bincount(sampled, minlength=n_cutouts).astype(np.float64)
        baseline_metrics = counts_to_detection_metrics(
            float(weights @ baseline_fixed["tp"]),
            float(weights @ baseline_fixed["fp"]),
            float(weights @ baseline_fixed["fn"]),
        )
        target_metrics = counts_to_detection_metrics(
            float(weights @ target_fixed["tp"]),
            float(weights @ target_fixed["fp"]),
            float(weights @ target_fixed["fn"]),
        )
        baseline_ap = curve_counts_to_ap(baseline_curve, weights)
        target_ap = curve_counts_to_ap(target_curve, weights)
        deltas_by_metric["f1"].append(target_metrics["f1"] - baseline_metrics["f1"])
        deltas_by_metric["recall"].append(target_metrics["recall"] - baseline_metrics["recall"])
        deltas_by_metric["ap"].append(target_ap - baseline_ap)

    return {
        "status": "available",
        "unit": "cutout",
        "seed": seed,
        "iterations": iterations,
        "max_thresholds_for_ap": max_thresholds,
        "n_cutouts": n_cutouts,
        "delta_ci": {
            metric: percentile_summary(values)
            for metric, values in deltas_by_metric.items()
        },
    }


def per_cutout_threshold_counts(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    *,
    threshold: float,
    matching_options: Mapping[str, Any],
    cutout_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    truth_by_cutout = group_truth_by_cutout(truth)
    predictions_by_cutout = group_predictions_by_cutout(predictions)
    tp = np.zeros(len(cutout_ids), dtype=np.float64)
    fp = np.zeros(len(cutout_ids), dtype=np.float64)
    fn = np.zeros(len(cutout_ids), dtype=np.float64)
    for index, cutout_id in enumerate(cutout_ids):
        cutout_truth = truth_by_cutout.get(cutout_id, [])
        cutout_predictions = [
            prediction
            for prediction in predictions_by_cutout.get(cutout_id, [])
            if prediction.score >= threshold
        ]
        metrics = detection_metrics(match_catalogs_for_options(cutout_truth, cutout_predictions, matching_options))
        tp[index] = metrics["tp"]
        fp[index] = metrics["fp"]
        fn[index] = metrics["fn"]
    return {"tp": tp, "fp": fp, "fn": fn}


def per_cutout_curve_counts(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    *,
    matching_options: Mapping[str, Any],
    cutout_ids: Sequence[str],
    max_thresholds: int,
) -> dict[str, Any]:
    thresholds = select_score_thresholds(
        sorted({prediction.score for prediction in predictions}, reverse=True),
        max_thresholds=max_thresholds,
    )
    tp = np.zeros((len(cutout_ids), len(thresholds)), dtype=np.float64)
    fp = np.zeros((len(cutout_ids), len(thresholds)), dtype=np.float64)
    fn = np.zeros((len(cutout_ids), len(thresholds)), dtype=np.float64)
    if not thresholds:
        return {"thresholds": [], "tp": tp, "fp": fp, "fn": fn}
    truth_by_cutout = group_truth_by_cutout(truth)
    predictions_by_cutout = group_predictions_by_cutout(predictions)
    for index, cutout_id in enumerate(cutout_ids):
        cutout_truth = truth_by_cutout.get(cutout_id, [])
        cutout_predictions = predictions_by_cutout.get(cutout_id, [])
        edges = build_candidate_edges(
            cutout_truth,
            cutout_predictions,
            max_radius_arcsec=float(matching_options["radius_arcsec"]),
            psf_fraction=float(matching_options["psf_fraction"]) if matching_options.get("seeing_aware") else None,
        )
        scores = {prediction.prediction_id: prediction.score for prediction in cutout_predictions}
        for threshold_index, threshold in enumerate(thresholds):
            active_ids = {prediction_id for prediction_id, score in scores.items() if score >= threshold}
            metrics = detection_metrics_from_edges(
                edges,
                truth_count=len(cutout_truth),
                prediction_count=len(active_ids),
                active_prediction_ids=active_ids,
            )
            tp[index, threshold_index] = metrics["tp"]
            fp[index, threshold_index] = metrics["fp"]
            fn[index, threshold_index] = metrics["fn"]
    return {"thresholds": thresholds, "tp": tp, "fp": fp, "fn": fn}


def curve_counts_to_ap(curve_counts: Mapping[str, Any], weights: np.ndarray) -> float:
    thresholds = curve_counts.get("thresholds", [])
    if not thresholds:
        return 0.0
    tp = weights @ curve_counts["tp"]
    fp = weights @ curve_counts["fp"]
    fn = weights @ curve_counts["fn"]
    points = []
    best_f1 = 0.0
    for tp_value, fp_value, fn_value in zip(tp, fp, fn, strict=True):
        metrics = counts_to_detection_metrics(float(tp_value), float(fp_value), float(fn_value))
        points.append((metrics["recall"], metrics["precision"]))
        best_f1 = max(best_f1, metrics["f1"])
    return average_precision_from_points(points, best_f1, len(thresholds))["ap"]


def counts_to_detection_metrics(tp: float, fp: float, fn: float) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    percentiles = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 50.0, 97.5])
    return {"low": float(percentiles[0]), "median": float(percentiles[1]), "high": float(percentiles[2])}


def match_catalogs_for_options(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    matching_options: Mapping[str, Any],
) -> MatchResult:
    if matching_options.get("seeing_aware"):
        return match_catalogs_seeing_aware(
            truth,
            predictions,
            max_radius_arcsec=float(matching_options["radius_arcsec"]),
            psf_fraction=float(matching_options["psf_fraction"]),
        )
    return match_catalogs(truth, predictions, radius_arcsec=float(matching_options["radius_arcsec"]))


def group_truth_by_cutout(records: Sequence[SourceRecord]) -> dict[str, list[SourceRecord]]:
    grouped: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.cutout_id].append(record)
    return grouped


def group_predictions_by_cutout(records: Sequence[PredictionRecord]) -> dict[str, list[PredictionRecord]]:
    grouped: dict[str, list[PredictionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.cutout_id].append(record)
    return grouped


def render_evidence_audit_markdown(payload: Mapping[str, Any]) -> str:
    baseline_label = str(payload.get("baseline_label", "baseline"))
    target_label = str(payload.get("target_label", "target"))
    aggregate = payload.get("aggregate", {})
    baseline = aggregate.get(baseline_label, {}) if isinstance(aggregate, Mapping) else {}
    target = aggregate.get(target_label, {}) if isinstance(aggregate, Mapping) else {}
    delta_row = aggregate.get("delta", {}) if isinstance(aggregate, Mapping) else {}
    bootstrap = payload.get("bootstrap", {}) if isinstance(payload.get("bootstrap"), Mapping) else {}
    ci = bootstrap.get("delta_ci", {}) if isinstance(bootstrap.get("delta_ci"), Mapping) else {}
    claim_summary = payload.get("claim_summary", {}) if isinstance(payload.get("claim_summary"), Mapping) else {}
    lines = [
        "# Evidence Audit",
        "",
        f"- Baseline: {payload.get('runs', {}).get(baseline_label, {}).get('run_id', '')}",
        f"- Target: {payload.get('runs', {}).get(target_label, {}).get('run_id', '')}",
        f"- Threshold source: {payload.get('threshold_policy', {}).get('source', '')}",
        f"- Bootstrap: {bootstrap.get('iterations', 0)} cutout resamples, seed {bootstrap.get('seed', '')}",
        "",
        "## Aggregate",
        "",
        "| model | precision | recall | f1 | ap |",
        "|---|---:|---:|---:|---:|",
        _aggregate_markdown_row(baseline_label, baseline),
        _aggregate_markdown_row(target_label, target),
        _aggregate_markdown_row("delta", delta_row),
        "",
        "## Bootstrap Delta CI",
        "",
        "| metric | low | median | high |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("f1", "ap", "recall"):
        row = ci.get(metric, {}) if isinstance(ci, Mapping) else {}
        lines.append(f"| {metric} | {row.get('low')} | {row.get('median')} | {row.get('high')} |")
    lines.extend(["", "## Strata Availability", ""])
    available = ", ".join(claim_summary.get("available_strata", [])) or "n/a"
    lines.append(f"- Available: {available}")
    unavailable = claim_summary.get("unavailable_strata", {})
    for field, reason in sorted(unavailable.items()):
        lines.append(f"- Unavailable {field}: {reason}")
    lines.append("")
    return "\n".join(lines)


def _aggregate_markdown_row(label: str, row: Mapping[str, Any]) -> str:
    return "| {label} | {precision} | {recall} | {f1} | {ap} |".format(
        label=label,
        precision=row.get("precision"),
        recall=row.get("recall"),
        f1=row.get("f1"),
        ap=row.get("ap"),
    )


def load_evidence_audits(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root)
    audits = []
    if not base.exists():
        return audits
    for path in sorted(base.glob("**/*audit.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not is_evidence_audit_payload(payload):
            continue
        payload["_audit_path"] = str(path)
        audits.append(payload)
    return audits


def is_evidence_audit_payload(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    required = {
        "aggregate",
        "threshold_policy",
        "claim_summary",
        "baseline_label",
        "target_label",
    }
    return required.issubset(payload)


def format_bin(lo: float, hi: float) -> str:
    hi_text = "inf" if math.isinf(hi) else str(float(hi))
    return f"[{float(lo)},{hi_text})"


def sort_stratum_key(value: str) -> tuple[int, float | str]:
    if value.startswith("["):
        try:
            left = value.strip("[]()").split(",", 1)[0]
            return (0, float(left))
        except ValueError:
            return (0, value)
    return (1, value)


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _select_shared_truth(
    baseline_truth: Sequence[SourceRecord],
    target_truth: Sequence[SourceRecord],
) -> list[SourceRecord]:
    baseline_ids = [record.source_id for record in baseline_truth]
    target_ids = [record.source_id for record in target_truth]
    if baseline_ids != target_ids:
        raise ValueError("baseline and target truth_test.csv files do not contain the same source_id order")
    return list(baseline_truth)


def _matching_options(*summaries: Mapping[str, Any]) -> dict[str, Any]:
    for summary in summaries:
        matching = summary.get("matching", {}) if isinstance(summary.get("matching"), Mapping) else {}
        if matching:
            return {
                "radius_arcsec": float(matching.get("radius_arcsec", 1.0)),
                "seeing_aware": bool(matching.get("seeing_aware", False)),
                "psf_fraction": float(matching.get("psf_fraction", 0.5)),
            }
    return {"radius_arcsec": 1.0, "seeing_aware": False, "psf_fraction": 0.5}


def _pixel_scale_arcsec(*summaries: Mapping[str, Any]) -> float:
    for summary in summaries:
        decode = summary.get("decode", {}) if isinstance(summary.get("decode"), Mapping) else {}
        if decode.get("pixel_scale_arcsec") is not None:
            return float(decode["pixel_scale_arcsec"])
    return 0.396


def _run_summary(inputs: RunAuditInputs) -> dict[str, Any]:
    return {
        "run_id": inputs.run_id,
        "run_dir": str(inputs.run_dir),
        "paths": dict(inputs.paths),
        "candidate_predictions": len(inputs.predictions),
        "truth": len(inputs.truth),
        "validation_best_threshold": inputs.selected_threshold,
    }
