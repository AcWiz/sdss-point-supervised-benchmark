from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .io import load_source_catalog, write_source_catalog
from .matching import match_catalogs, match_catalogs_seeing_aware
from .metrics import (
    astrometry_metrics,
    classification_metrics,
    deblending_metrics,
    detection_average_precision,
    detection_metrics,
    photometry_metrics,
    stratified_detection_report,
)
from .schema import PredictionRecord, SourceRecord


@dataclass(frozen=True)
class SplitTruthSelection:
    records: list[SourceRecord]
    cutout_ids: set[str]
    source_ids: set[str]
    all_truth_count: int
    dropped_truth_count: int
    all_quality_counts: dict[str, int]
    kept_quality_counts: dict[str, int]


def run_pilot_loop(
    *,
    config: str | Path,
    dataset: str | Path,
    split: str | Path,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    train_split_name: str = "train",
    val_split_name: str = "val",
    test_split_name: str = "test",
    epochs: int = 1,
    train_limit_samples: int | None = None,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    base_channels: int = 32,
    model_arch: str = "baseline",
    loader_mode: str = "sample",
    shard_cache_size: int = 0,
    num_workers: int = 0,
    pin_memory: bool | str = "auto",
    device: str = "cpu",
    seed: int = 42,
    candidate_threshold: float = 0.2,
    nms_radius: int = 2,
    max_detections_per_cutout: int | None = None,
    predict_limit: int | None = None,
    pixel_scale_arcsec: float = 0.396,
    radius_arcsec: float = 1.0,
    band: str = "r",
    close_pair_arcsec: float = 2.0,
    seeing_aware: bool = False,
    psf_fraction: float = 0.5,
    include_suspect_truth: bool = False,
) -> dict:
    """Run the auditable train/val-threshold/test pilot loop."""

    from .training import predict_dataset, train_model

    output_path = Path(output_dir)
    checkpoint_path = Path(checkpoint_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    train_report = train_model(
        config_path=config,
        dataset_dir=dataset,
        output_dir=checkpoint_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        base_channels=base_channels,
        model_arch=model_arch,
        loader_mode=loader_mode,
        shard_cache_size=shard_cache_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        device=device,
        seed=seed,
        split_path=split,
        split_name=train_split_name,
        limit_samples=train_limit_samples,
    )
    best_checkpoint = checkpoint_path / "best.pt"

    val_truth_path = output_path / "truth_val.csv"
    test_truth_path = output_path / "truth_test.csv"
    val_truth = write_split_truth_catalog(
        dataset_dir=dataset,
        split_path=split,
        split_name=val_split_name,
        output_path=val_truth_path,
        include_suspect=include_suspect_truth,
        limit_samples=predict_limit,
    )
    test_truth = write_split_truth_catalog(
        dataset_dir=dataset,
        split_path=split,
        split_name=test_split_name,
        output_path=test_truth_path,
        include_suspect=include_suspect_truth,
        limit_samples=predict_limit,
    )

    val_predictions_path = output_path / "predictions_val_candidates.csv"
    test_predictions_path = output_path / "predictions_test_candidates.csv"
    val_predictions = predict_dataset(
        checkpoint_path=best_checkpoint,
        dataset_dir=dataset,
        output_path=val_predictions_path,
        batch_size=batch_size,
        threshold=candidate_threshold,
        nms_radius=nms_radius,
        device=device,
        pixel_scale_arcsec=pixel_scale_arcsec,
        split_path=split,
        split_name=val_split_name,
        max_detections_per_cutout=max_detections_per_cutout,
        limit_samples=predict_limit,
        shard_cache_size=shard_cache_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_predictions = predict_dataset(
        checkpoint_path=best_checkpoint,
        dataset_dir=dataset,
        output_path=test_predictions_path,
        batch_size=batch_size,
        threshold=candidate_threshold,
        nms_radius=nms_radius,
        device=device,
        pixel_scale_arcsec=pixel_scale_arcsec,
        split_path=split,
        split_name=test_split_name,
        max_detections_per_cutout=max_detections_per_cutout,
        limit_samples=predict_limit,
        shard_cache_size=shard_cache_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_sweep = {
        "protocol": "sdss-point-supervised-v1",
        "truth_catalog": str(val_truth_path),
        "prediction_catalog": str(val_predictions_path),
        **sweep_thresholds_payload(
            val_truth.records,
            val_predictions,
            radius_arcsec=radius_arcsec,
            seeing_aware=seeing_aware,
            psf_fraction=psf_fraction,
            filter_truth_to_prediction_cutouts=False,
        ),
    }
    val_sweep_path = output_path / "val_threshold_sweep.json"
    write_json(val_sweep_path, val_sweep)

    best_threshold = float(val_sweep["best_threshold"])
    test_metrics = {
        "protocol": "sdss-point-supervised-v1",
        "truth_catalog": str(test_truth_path),
        "prediction_catalog": str(test_predictions_path),
        **evaluate_payload(
            test_truth.records,
            test_predictions,
            radius_arcsec=radius_arcsec,
            band=band,
            close_pair_arcsec=close_pair_arcsec,
            seeing_aware=seeing_aware,
            psf_fraction=psf_fraction,
            min_score=best_threshold,
            filter_truth_to_prediction_cutouts=False,
        ),
    }
    test_metrics_path = output_path / "test_metrics.json"
    write_json(test_metrics_path, test_metrics)

    train_split = split_sample_summary(dataset, split, train_split_name)
    summary = {
        "protocol": "sdss-point-supervised-v1",
        "status": "generated",
        "generated_at": utc_now(),
        "config": str(Path(config)),
        "dataset": str(Path(dataset)),
        "split": str(Path(split)),
        "splits": {
            "train": {**train_split, "name": train_split_name},
            "val": truth_selection_summary(val_truth, val_split_name),
            "test": truth_selection_summary(test_truth, test_split_name),
        },
        "training": {
            "checkpoint": str(best_checkpoint),
            "report": str(checkpoint_path / "training_report.json"),
            "best_train_loss": train_report.get("best_train_loss"),
            "epochs": epochs,
            "train_limit_samples": train_limit_samples,
            "batch_size": batch_size,
            "base_channels": base_channels,
            "model_arch": model_arch,
            "loader": train_report.get("loader", {}),
            "device": device,
            "seed": seed,
        },
        "decode": {
            "candidate_threshold": candidate_threshold,
            "nms_radius_pixels": nms_radius,
            "max_detections_per_cutout": max_detections_per_cutout,
            "predict_limit": predict_limit,
            "pixel_scale_arcsec": pixel_scale_arcsec,
        },
        "threshold_selection": {
            "source": "val",
            "best_threshold": best_threshold,
            "best_metrics": val_sweep["best_metrics"],
        },
        "matching": test_metrics["matching"],
        "test_detection": test_metrics["metrics"]["detection"],
        "outputs": {
            "checkpoint": str(best_checkpoint),
            "training_report": str(checkpoint_path / "training_report.json"),
            "val_truth": str(val_truth_path),
            "test_truth": str(test_truth_path),
            "val_predictions": str(val_predictions_path),
            "test_predictions": str(test_predictions_path),
            "val_threshold_sweep": str(val_sweep_path),
            "test_metrics": str(test_metrics_path),
            "summary": str(output_path / "summary.json"),
        },
        "truth_policy": {
            "include_suspect_truth": include_suspect_truth,
            "default": "exclude label_quality=suspect and label_weight<=0 unless --include-suspect-truth is set",
        },
    }
    write_json(output_path / "summary.json", summary)
    return summary


def load_pilot_loop_outputs(output_dir: str | Path) -> dict[str, dict]:
    output_path = Path(output_dir)
    return {
        "summary": load_json(output_path / "summary.json"),
        "val_threshold_sweep": load_json(output_path / "val_threshold_sweep.json"),
        "test_metrics": load_json(output_path / "test_metrics.json"),
    }


def evaluate_payload(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    *,
    radius_arcsec: float,
    band: str,
    close_pair_arcsec: float,
    seeing_aware: bool,
    psf_fraction: float,
    min_score: float | None,
    filter_truth_to_prediction_cutouts: bool,
) -> dict:
    scored_predictions = filter_predictions_by_score(predictions, min_score)
    matches = match_catalogs_for_options(
        truth,
        scored_predictions,
        radius_arcsec=radius_arcsec,
        seeing_aware=seeing_aware,
        psf_fraction=psf_fraction,
    )
    metrics = {
        "detection": detection_metrics(matches),
        "average_precision": detection_average_precision_for_options(
            truth,
            predictions,
            radius_arcsec=radius_arcsec,
            seeing_aware=seeing_aware,
            psf_fraction=psf_fraction,
        ),
        "astrometry": astrometry_metrics(matches),
        "classification": classification_metrics(truth, scored_predictions, matches),
        f"photometry_{band}": photometry_metrics(truth, scored_predictions, matches, band=band),
        "deblending": deblending_metrics(
            truth,
            scored_predictions,
            matches,
            close_pair_arcsec=close_pair_arcsec,
            band=band,
        ),
        "stratified_detection": stratified_detection_report(
            truth,
            matches,
            {
                "mag_r": [14.0, 16.0, 18.0, 20.0, 21.0, 22.0, 23.0],
                "snr": [0.0, 5.0, 10.0, 20.0, 50.0, 100.0],
                "seeing": [0.0, 1.2, 1.6, 2.0, 3.0],
                "nearest_neighbor_arcsec": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
            },
        ),
    }
    return {
        "matching": {
            "radius_arcsec": radius_arcsec,
            "seeing_aware": seeing_aware,
            "psf_fraction": psf_fraction,
            "min_score": min_score,
            "filter_truth_to_prediction_cutouts": filter_truth_to_prediction_cutouts,
        },
        "counts": {
            "truth": len(truth),
            "predictions": len(scored_predictions),
            "candidate_predictions": len(predictions),
            "matches": len(matches.matches),
            "unmatched_truth": len(matches.unmatched_truth_ids),
            "unmatched_predictions": len(matches.unmatched_prediction_ids),
        },
        "metrics": metrics,
    }


def sweep_thresholds_payload(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    *,
    radius_arcsec: float,
    seeing_aware: bool,
    psf_fraction: float,
    filter_truth_to_prediction_cutouts: bool,
) -> dict:
    thresholds = sorted({prediction.score for prediction in predictions}, reverse=True)
    rows = []
    best_threshold = 0.0
    best_metrics = {"tp": 0.0, "fp": 0.0, "fn": float(len(truth)), "precision": 0.0, "recall": 0.0, "f1": 0.0}
    points: list[tuple[float, float]] = []
    for threshold in thresholds:
        filtered = filter_predictions_by_score(predictions, threshold)
        matches = match_catalogs_for_options(
            truth,
            filtered,
            radius_arcsec=radius_arcsec,
            seeing_aware=seeing_aware,
            psf_fraction=psf_fraction,
        )
        metrics = detection_metrics(matches)
        rows.append({"threshold": threshold, **metrics})
        points.append((metrics["recall"], metrics["precision"]))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = metrics

    return {
        "matching": {
            "radius_arcsec": radius_arcsec,
            "seeing_aware": seeing_aware,
            "psf_fraction": psf_fraction,
            "filter_truth_to_prediction_cutouts": filter_truth_to_prediction_cutouts,
        },
        "counts": {
            "truth": len(truth),
            "candidate_predictions": len(predictions),
            "n_thresholds": len(thresholds),
        },
        "best_threshold": best_threshold,
        "best_metrics": best_metrics,
        "average_precision": average_precision_from_points(points, best_metrics["f1"], len(thresholds)),
        "thresholds": rows,
    }


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


def write_split_truth_catalog(
    *,
    dataset_dir: str | Path,
    split_path: str | Path,
    split_name: str,
    output_path: str | Path,
    include_suspect: bool,
    limit_samples: int | None = None,
) -> SplitTruthSelection:
    selection = select_split_truth_records(
        dataset_dir=dataset_dir,
        split_path=split_path,
        split_name=split_name,
        include_suspect=include_suspect,
        limit_samples=limit_samples,
    )
    write_source_catalog(selection.records, output_path)
    return selection


def select_split_truth_records(
    *,
    dataset_dir: str | Path,
    split_path: str | Path,
    split_name: str,
    include_suspect: bool = False,
    limit_samples: int | None = None,
) -> SplitTruthSelection:
    dataset = Path(dataset_dir)
    source_ids = load_split_ids(split_path, split_name)
    cutout_ids = cutout_ids_for_split(dataset / "manifest.csv", source_ids, limit_samples=limit_samples)
    all_records = [record for record in load_source_catalog(dataset / "truth_catalog.csv") if record.cutout_id in cutout_ids]
    if include_suspect:
        kept_records = list(all_records)
    else:
        kept_records = [record for record in all_records if is_primary_truth(record)]
    return SplitTruthSelection(
        records=kept_records,
        cutout_ids=cutout_ids,
        source_ids=source_ids,
        all_truth_count=len(all_records),
        dropped_truth_count=len(all_records) - len(kept_records),
        all_quality_counts=quality_counts(all_records),
        kept_quality_counts=quality_counts(kept_records),
    )


def split_sample_summary(dataset_dir: str | Path, split_path: str | Path, split_name: str) -> dict[str, int]:
    source_ids = load_split_ids(split_path, split_name)
    cutout_ids = cutout_ids_for_split(Path(dataset_dir) / "manifest.csv", source_ids)
    return {"source_ids": len(source_ids), "cutouts": len(cutout_ids)}


def truth_selection_summary(selection: SplitTruthSelection, split_name: str) -> dict:
    return {
        "name": split_name,
        "source_ids": len(selection.source_ids),
        "cutouts": len(selection.cutout_ids),
        "truth_all": selection.all_truth_count,
        "truth_kept": len(selection.records),
        "truth_dropped": selection.dropped_truth_count,
        "quality_all": selection.all_quality_counts,
        "quality_kept": selection.kept_quality_counts,
    }


def filter_predictions_by_score(predictions, min_score: float | None):
    if min_score is None:
        return predictions
    return [prediction for prediction in predictions if prediction.score >= min_score]


def detection_average_precision_for_options(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    *,
    radius_arcsec: float,
    seeing_aware: bool,
    psf_fraction: float,
) -> dict[str, float]:
    if not seeing_aware:
        return detection_average_precision(truth, predictions, radius_arcsec=radius_arcsec)
    thresholds = sorted({prediction.score for prediction in predictions}, reverse=True)
    if not thresholds:
        return {"ap": 0.0, "best_f1": 0.0, "n_thresholds": 0.0}

    points: list[tuple[float, float]] = []
    best_f1 = 0.0
    for threshold in thresholds:
        filtered = filter_predictions_by_score(predictions, threshold)
        metrics = detection_metrics(
            match_catalogs_seeing_aware(
                truth,
                filtered,
                max_radius_arcsec=radius_arcsec,
                psf_fraction=psf_fraction,
            )
        )
        points.append((metrics["recall"], metrics["precision"]))
        best_f1 = max(best_f1, metrics["f1"])

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

    return {"ap": ap, "best_f1": best_f1, "n_thresholds": float(len(thresholds))}


def match_catalogs_for_options(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    *,
    radius_arcsec: float,
    seeing_aware: bool,
    psf_fraction: float,
):
    if seeing_aware:
        return match_catalogs_seeing_aware(
            truth,
            predictions,
            max_radius_arcsec=radius_arcsec,
            psf_fraction=psf_fraction,
        )
    return match_catalogs(truth, predictions, radius_arcsec=radius_arcsec)


def cutout_ids_for_split(
    manifest_path: Path,
    source_ids: set[str],
    limit_samples: int | None = None,
) -> set[str]:
    if limit_samples is not None and limit_samples < 0:
        raise ValueError("limit_samples must be non-negative")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"center_source_id", "cutout_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"dataset manifest is missing required columns: {sorted(missing)}")
        cutout_ids: list[str] = []
        for row in reader:
            if row["center_source_id"] not in source_ids:
                continue
            cutout_ids.append(row["cutout_id"])
            if limit_samples is not None and len(cutout_ids) >= limit_samples:
                break
        return set(cutout_ids)


def load_split_ids(path: str | Path, split_name: str) -> set[str]:
    payload = load_json(path)
    splits = payload.get("splits", {})
    if split_name not in splits:
        raise ValueError(f"split {split_name!r} not found in {path}")
    return {str(source_id) for source_id in splits[split_name]}


def is_primary_truth(record: SourceRecord) -> bool:
    if (record.label_quality or "").lower() == "suspect":
        return False
    return record.label_weight is None or record.label_weight > 0.0


def quality_counts(records: Sequence[SourceRecord]) -> dict[str, int]:
    counts = Counter((record.label_quality or "unknown").lower() for record in records)
    return dict(sorted(counts.items()))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
