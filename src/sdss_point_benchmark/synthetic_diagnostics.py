from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

from .baseline import make_catalog_model
from .decode import decode_predictions
from .io import load_prediction_catalog, load_source_catalog, write_prediction_catalog, write_source_catalog
from .matching import angular_distance_arcsec, match_catalogs
from .metrics import detection_metrics, detection_score_curve, select_score_thresholds
from .pilot_loop import write_json
from .schema import BANDS, PredictionRecord, SourceRecord
from .synthetic import InjectionSpec, inject_sources
from .synthetic_validation import (
    local_pixel_to_radec,
    mag_to_flux,
    source_records_with_injection_metadata,
)
from .synthetic_validation import injection_specs_for_background
from .training import NpzCutoutDataset

SYNTHETIC_DIAGNOSTIC_SCHEMA_VERSION = 1
MAG_R_BINS = [(17.5, 19.0), (19.0, 20.5), (20.5, 21.5), (21.5, 22.5)]


def build_synthetic_faint_recovery_diagnostic(
    *,
    validation_dir: str | Path,
    output_dir: str | Path,
    threshold: float | None = None,
    match_radius_pixels: float | None = None,
    max_thresholds: int = 64,
) -> dict[str, Any]:
    validation_path = Path(validation_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    validation_report = load_validation_report(validation_path)
    resolved_threshold = resolve_threshold(validation_report, threshold)
    resolved_radius = resolve_match_radius(validation_report, match_radius_pixels)
    truth_path = validation_output_path(validation_path, validation_report, "truth_injected", "truth_injected.csv")
    prediction_path = validation_output_path(
        validation_path,
        validation_report,
        "predictions",
        "predictions_candidates.csv",
    )
    truth = load_source_catalog(truth_path)
    predictions = load_prediction_catalog(prediction_path)
    thresholded_predictions = [prediction for prediction in predictions if prediction.score >= resolved_threshold]
    fixed_matches = match_catalogs(truth, thresholded_predictions, radius_arcsec=resolved_radius)
    curve = detection_score_curve(
        truth,
        predictions,
        max_radius_arcsec=resolved_radius,
        max_thresholds=max_thresholds,
    )
    fixed_detection = detection_metrics(fixed_matches)
    false_negative_rows = false_negative_diagnostics(
        truth,
        predictions,
        fixed_matches.unmatched_truth_ids,
        radius_arcsec=resolved_radius,
    )
    by_mag = mag_bin_diagnostics(
        truth,
        predictions,
        radius_arcsec=resolved_radius,
        threshold=resolved_threshold,
        max_thresholds=max_thresholds,
        false_negative_rows=false_negative_rows,
    )
    findings, next_actions = diagnostic_findings(by_mag, resolved_threshold)
    false_negative_csv = output_path / "false_negatives.csv"
    write_false_negative_rows(false_negative_rows, false_negative_csv)

    payload = {
        "schema_version": SYNTHETIC_DIAGNOSTIC_SCHEMA_VERSION,
        "protocol": "sdss-point-supervised-v1",
        "run_id": output_path.name,
        "program_id": "synthetic_injection_mainline",
        "variant_id": output_path.name,
        "status": "executed",
        "diagnostic": "synthetic_faint_recovery",
        "validation": "synthetic_injection_mainline",
        "objective": "Diagnose faint-source recovery in the controlled synthetic-injection validation.",
        "hypothesis": "The faintest injected-source failures are caused by candidate decoding or score calibration limits.",
        "tags": [
            "paper_validation",
            "mainline_validation",
            "synthetic_injection",
            "faint_recovery_diagnostic",
            "fixed_split",
            "native_frame",
        ],
        "claims": ["synthetic_injection", "stratified_metrics"],
        "validation_run_dir": str(validation_path),
        "validation_report": str(validation_path / "report.json"),
        "threshold": resolved_threshold,
        "match_radius_pixels": resolved_radius,
        "counts": {
            "truth": len(truth),
            "candidate_predictions": len(predictions),
            "thresholded_predictions": len(thresholded_predictions),
            "false_negatives": len(false_negative_rows),
        },
        "metrics": {
            "test": {
                "counts": {
                    "truth": len(truth),
                    "candidate_predictions": len(thresholded_predictions),
                    "matches": int(fixed_detection.get("tp", 0.0)),
                    "unmatched_predictions": int(fixed_detection.get("fp", 0.0)),
                    "unmatched_truth": int(fixed_detection.get("fn", 0.0)),
                },
                "detection": fixed_detection,
                "average_precision": curve["average_precision"],
                "stratified_detection": {"injected_mag_r": by_mag},
            },
            "validation": {
                "type": "synthetic_faint_recovery",
                "source_validation_run_id": validation_report.get("run_id"),
                "threshold": resolved_threshold,
                "match_radius_pixels": resolved_radius,
            },
            "fixed_threshold_detection": fixed_detection,
            "score_curve": compact_score_curve(curve),
            "by_mag_r": by_mag,
        },
        "findings": findings,
        "next_actions": next_actions,
        "outputs": {
            "false_negatives": str(false_negative_csv),
            "report": str(output_path / "report.json"),
            "markdown": str(output_path / "report.md"),
        },
        "claim_gate": {
            "status": "engineering_check",
            "paper_ready": False,
            "paper_claim_allowed": False,
            "reasons": ["derived diagnostic over an existing synthetic-validation report; not independent evidence"],
        },
    }
    write_json(output_path / "report.json", payload)
    (output_path / "report.md").write_text(render_synthetic_diagnostic_markdown(payload), encoding="utf-8")
    return payload


def build_synthetic_heatmap_response_diagnostic(
    *,
    validation_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    search_radius_pixels: float = 8.0,
    low_floor: float = 0.05,
    shard_cache_size: int | None = None,
) -> dict[str, Any]:
    validation_path = Path(validation_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    validation_report = load_validation_report(validation_path)
    parameters = validation_report.get("parameters", {}) if isinstance(validation_report.get("parameters"), Mapping) else {}
    checkpoint_path = Path(str(validation_report.get("checkpoint", "")))
    dataset_dir = Path(str(validation_report.get("dataset", "")))
    split_path = validation_report.get("split")
    split_name = validation_report.get("split_name")
    seed = int(validation_report.get("seed", 42) or 42)
    num_backgrounds = int(parameters.get("num_backgrounds", 16) or 16)
    injections_per_background = int(parameters.get("injections_per_background", 4) or 4)
    threshold = float(parameters.get("threshold", 0.2) or 0.2)
    match_radius_pixels = float(parameters.get("match_radius_pixels", 2.0) or 2.0)
    psf_sigma = float(parameters.get("psf_sigma", 1.3) or 1.3)
    cache_size = int(shard_cache_size if shard_cache_size is not None else 2)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found in validation report: {checkpoint_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset not found in validation report: {dataset_dir}")

    rng = random.Random(seed)
    dataset = NpzCutoutDataset(
        dataset_dir,
        split_path=split_path,
        split_name=split_name,
        limit_samples=num_backgrounds,
        shard_cache_size=cache_size,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model", {})
    model = make_catalog_model(
        str(model_config.get("model_arch") or model_config.get("name") or "baseline"),
        in_channels=int(model_config.get("in_channels", 5)),
        num_classes=int(model_config.get("num_classes", 2)),
        base_channels=int(model_config.get("base_channels", 32)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            background = sample["image"].detach().cpu().numpy()
            cutout_id = f"synthetic_{index:04d}"
            specs = injection_specs_for_background(
                rng,
                cutout_id=cutout_id,
                count=injections_per_background,
                shape=background.shape[-2:],
            )
            injected_image, _truth = inject_sources(
                background,
                specs,
                bands=BANDS[: background.shape[0]],
                psf_sigma=psf_sigma,
                cutout_id=cutout_id,
            )
            image_tensor = torch.from_numpy(injected_image).unsqueeze(0).to(device_obj)
            outputs = model(image_tensor)
            scores = torch.sigmoid(outputs.center_heatmap.detach())[0, 0].cpu().numpy()
            for spec in specs:
                rows.append(
                    heatmap_response_row(
                        scores,
                        cutout_id=cutout_id,
                        source_id=spec.source_id,
                        x=float(spec.x),
                        y=float(spec.y),
                        mag_r=float(spec.metadata.get("mag_r")),
                        kind=spec.kind,
                        fixed_threshold=threshold,
                        low_floor=low_floor,
                        match_radius_pixels=match_radius_pixels,
                        search_radius_pixels=search_radius_pixels,
                    )
                )

    by_mag = heatmap_response_by_mag(rows)
    by_mag_and_label = heatmap_response_by_mag_and_label(rows)
    findings, next_actions = heatmap_response_findings(by_mag, low_floor=low_floor)
    response_csv = output_path / "heatmap_response.csv"
    write_heatmap_response_rows(rows, response_csv)
    payload = {
        "schema_version": SYNTHETIC_DIAGNOSTIC_SCHEMA_VERSION,
        "protocol": "sdss-point-supervised-v1",
        "run_id": output_path.name,
        "program_id": "synthetic_injection_mainline",
        "variant_id": output_path.name,
        "status": "executed",
        "diagnostic": "synthetic_heatmap_response",
        "validation": "synthetic_injection_mainline",
        "objective": "Measure raw center-heatmap response at controlled injected-source positions.",
        "hypothesis": "Faint synthetic misses are due to weak local center-heatmap response or off-radius heatmap peaks.",
        "tags": [
            "paper_validation",
            "mainline_validation",
            "synthetic_injection",
            "heatmap_response_diagnostic",
            "fixed_split",
            "native_frame",
        ],
        "claims": ["synthetic_injection", "stratified_metrics"],
        "validation_run_dir": str(validation_path),
        "validation_report": str(validation_path / "report.json"),
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_dir),
        "split": str(split_path) if split_path is not None else None,
        "split_name": split_name,
        "threshold": threshold,
        "low_floor": low_floor,
        "match_radius_pixels": match_radius_pixels,
        "search_radius_pixels": search_radius_pixels,
        "counts": {
            "backgrounds": len(dataset),
            "injected_truth": len(rows),
        },
        "metrics": {
            "validation": {
                "type": "synthetic_heatmap_response",
                "source_validation_run_id": validation_report.get("run_id"),
                "threshold": threshold,
                "low_floor": low_floor,
                "match_radius_pixels": match_radius_pixels,
                "search_radius_pixels": search_radius_pixels,
            },
            "by_mag_r": by_mag,
            "by_mag_r_and_label": by_mag_and_label,
            "overall": summarize_heatmap_response_rows(rows),
        },
        "findings": findings,
        "next_actions": next_actions,
        "outputs": {
            "heatmap_response": str(response_csv),
            "report": str(output_path / "report.json"),
            "markdown": str(output_path / "report.md"),
        },
        "claim_gate": {
            "status": "engineering_check",
            "paper_ready": False,
            "paper_claim_allowed": False,
            "reasons": ["raw model-response diagnostic over an existing synthetic-validation report"],
        },
    }
    write_json(output_path / "report.json", payload)
    (output_path / "report.md").write_text(render_synthetic_heatmap_diagnostic_markdown(payload), encoding="utf-8")
    return payload


def build_synthetic_morphology_diagnostic(
    *,
    validation_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    search_radius_pixels: float = 8.0,
    low_floor: float = 0.05,
    shard_cache_size: int | None = None,
) -> dict[str, Any]:
    validation_path = Path(validation_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    validation_report = load_validation_report(validation_path)
    parameters = validation_report.get("parameters", {}) if isinstance(validation_report.get("parameters"), Mapping) else {}
    checkpoint_path = Path(str(validation_report.get("checkpoint", "")))
    dataset_dir = Path(str(validation_report.get("dataset", "")))
    split_path = validation_report.get("split")
    split_name = validation_report.get("split_name")
    seed = int(validation_report.get("seed", 42) or 42)
    num_backgrounds = int(parameters.get("num_backgrounds", 16) or 16)
    threshold = float(parameters.get("threshold", 0.2) or 0.2)
    nms_radius = int(parameters.get("nms_radius", 2) or 2)
    match_radius_pixels = float(parameters.get("match_radius_pixels", 2.0) or 2.0)
    max_detections = parameters.get("max_detections_per_cutout", 32)
    max_detections_per_cutout = int(max_detections) if max_detections is not None else None
    psf_sigma = float(parameters.get("psf_sigma", 1.3) or 1.3)
    cache_size = int(shard_cache_size if shard_cache_size is not None else 2)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found in validation report: {checkpoint_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset not found in validation report: {dataset_dir}")

    rng = random.Random(seed)
    dataset = NpzCutoutDataset(
        dataset_dir,
        split_path=split_path,
        split_name=split_name,
        limit_samples=num_backgrounds,
        shard_cache_size=cache_size,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model", {})
    model = make_catalog_model(
        str(model_config.get("model_arch") or model_config.get("name") or "baseline"),
        in_channels=int(model_config.get("in_channels", 5)),
        num_classes=int(model_config.get("num_classes", 2)),
        base_channels=int(model_config.get("base_channels", 32)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    truth: list[SourceRecord] = []
    predictions: list[PredictionRecord] = []
    response_rows: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    per_cutout: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            background = sample["image"].detach().cpu().numpy()
            cutout_id = f"morphology_{index:04d}"
            specs = morphology_injection_specs_for_background(
                rng,
                cutout_id=cutout_id,
                shape=background.shape[-2:],
            )
            injected_image, cutout_truth = inject_sources(
                background,
                specs,
                bands=BANDS[: background.shape[0]],
                psf_sigma=psf_sigma,
                cutout_id=cutout_id,
            )
            cutout_truth = source_records_with_injection_metadata(cutout_truth, specs)
            truth.extend(cutout_truth)
            plans.extend([morphology_plan_row(spec) for spec in specs])

            image_tensor = torch.from_numpy(injected_image).unsqueeze(0).to(device_obj)
            outputs = model(image_tensor)
            cutout_predictions = decode_predictions(
                outputs,
                cutout_ids=[cutout_id],
                origin_radec=[(0.0, 0.0)],
                pixel_to_radec=lambda _batch_index, x, y: local_pixel_to_radec(x, y),
                pixel_scale_arcsec=1.0,
                threshold=threshold,
                nms_radius=nms_radius,
                max_detections_per_cutout=max_detections_per_cutout,
            )
            predictions.extend(cutout_predictions)
            scores = torch.sigmoid(outputs.center_heatmap.detach())[0, 0].cpu().numpy()
            for spec in specs:
                response_rows.append(
                    heatmap_response_row(
                        scores,
                        cutout_id=cutout_id,
                        source_id=spec.source_id,
                        x=float(spec.x),
                        y=float(spec.y),
                        mag_r=float(spec.metadata.get("mag_r")),
                        kind=spec.kind,
                        fixed_threshold=threshold,
                        low_floor=low_floor,
                        match_radius_pixels=match_radius_pixels,
                        search_radius_pixels=search_radius_pixels,
                    )
                    | {
                        "radius": float(spec.radius),
                        "ellipticity": float(spec.ellipticity),
                        "condition": morphology_condition(spec),
                    }
                )
            per_cutout.append(
                {
                    "cutout_id": cutout_id,
                    "background_cutout_id": str(sample["cutout_id"]),
                    "injected": len(cutout_truth),
                    "candidate_predictions": len(cutout_predictions),
                }
            )

    scored_predictions = [prediction for prediction in predictions if prediction.score >= threshold]
    matches = match_catalogs(truth, scored_predictions, radius_arcsec=match_radius_pixels)
    curve = detection_score_curve(truth, predictions, max_radius_arcsec=match_radius_pixels)
    detection = detection_metrics(matches)
    recall_by_condition = morphology_recall_by_condition(truth, matches)
    response_by_condition = morphology_response_by_condition(response_rows)
    findings, next_actions = morphology_diagnostic_findings(response_by_condition, recall_by_condition)

    truth_csv = output_path / "truth_injected.csv"
    prediction_csv = output_path / "predictions_candidates.csv"
    response_csv = output_path / "morphology_response.csv"
    write_source_catalog(truth, truth_csv)
    write_prediction_catalog(predictions, prediction_csv)
    write_morphology_response_rows(response_rows, response_csv)
    write_json(output_path / "injection_plan.json", {"seed": seed, "plans": plans})

    payload = {
        "schema_version": SYNTHETIC_DIAGNOSTIC_SCHEMA_VERSION,
        "protocol": "sdss-point-supervised-v1",
        "run_id": output_path.name,
        "program_id": "synthetic_injection_mainline",
        "variant_id": output_path.name,
        "status": "executed",
        "diagnostic": "synthetic_morphology_response",
        "validation": "synthetic_injection_mainline",
        "objective": "Sweep faint synthetic source morphology to isolate extended-source center-response failures.",
        "hypothesis": (
            "If faint galaxy recovery fails because the center-only checkpoint is insensitive to low surface brightness, "
            "larger-radius galaxy injections should have weaker match-radius heatmap response than star controls."
        ),
        "tags": [
            "paper_validation",
            "mainline_validation",
            "synthetic_injection",
            "morphology_response_diagnostic",
            "fixed_split",
            "native_frame",
        ],
        "claims": ["synthetic_injection", "stratified_metrics"],
        "validation_run_dir": str(validation_path),
        "validation_report": str(validation_path / "report.json"),
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_dir),
        "split": str(split_path) if split_path is not None else None,
        "split_name": split_name,
        "threshold": threshold,
        "low_floor": low_floor,
        "match_radius_pixels": match_radius_pixels,
        "search_radius_pixels": search_radius_pixels,
        "parameters": {
            "num_backgrounds": len(dataset),
            "injections_per_background": len(morphology_condition_grid()),
            "mag_r": [21.0, 22.0],
            "galaxy_radii": [1.3, 2.4, 3.6],
            "star_control_radius": psf_sigma,
            "nms_radius": nms_radius,
            "max_detections_per_cutout": max_detections_per_cutout,
            "psf_sigma": psf_sigma,
        },
        "counts": {
            "backgrounds": len(dataset),
            "injected_truth": len(truth),
            "candidate_predictions": len(predictions),
            "thresholded_predictions": len(scored_predictions),
        },
        "metrics": {
            "test": {
                "counts": {
                    "truth": len(truth),
                    "candidate_predictions": len(scored_predictions),
                    "matches": int(detection.get("tp", 0.0)),
                    "unmatched_predictions": int(detection.get("fp", 0.0)),
                    "unmatched_truth": int(detection.get("fn", 0.0)),
                },
                "detection": detection,
                "average_precision": curve["average_precision"],
                "stratified_detection": {"morphology_condition": recall_by_condition},
            },
            "validation": {
                "type": "synthetic_morphology_response",
                "source_validation_run_id": validation_report.get("run_id"),
                "threshold": threshold,
                "low_floor": low_floor,
                "match_radius_pixels": match_radius_pixels,
                "search_radius_pixels": search_radius_pixels,
            },
            "fixed_threshold_detection": detection,
            "average_precision": curve["average_precision"],
            "recall_by_condition": recall_by_condition,
            "response_by_condition": response_by_condition,
            "overall_response": summarize_heatmap_response_rows(response_rows),
        },
        "findings": findings,
        "next_actions": next_actions,
        "per_cutout": per_cutout,
        "outputs": {
            "truth_injected": str(truth_csv),
            "predictions": str(prediction_csv),
            "morphology_response": str(response_csv),
            "injection_plan": str(output_path / "injection_plan.json"),
            "report": str(output_path / "report.json"),
            "markdown": str(output_path / "report.md"),
        },
        "claim_gate": {
            "status": "engineering_check",
            "paper_ready": False,
            "paper_claim_allowed": False,
            "reasons": ["morphology sweep diagnostic over an existing mainline checkpoint; not independent evidence"],
        },
    }
    write_json(output_path / "report.json", payload)
    (output_path / "report.md").write_text(render_synthetic_morphology_diagnostic_markdown(payload), encoding="utf-8")
    return payload


def load_validation_report(validation_dir: Path) -> dict[str, Any]:
    report_path = validation_dir / "report.json"
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text(encoding="utf-8"))


def heatmap_response_row(
    scores: Any,
    *,
    cutout_id: str,
    source_id: str,
    x: float,
    y: float,
    mag_r: float,
    kind: str,
    fixed_threshold: float,
    low_floor: float,
    match_radius_pixels: float,
    search_radius_pixels: float,
) -> dict[str, Any]:
    height, width = scores.shape
    nearest_x = min(max(int(round(x)), 0), width - 1)
    nearest_y = min(max(int(round(y)), 0), height - 1)
    center_score = float(scores[nearest_y, nearest_x])
    local = best_local_score(
        scores,
        x=x,
        y=y,
        radius_pixels=search_radius_pixels,
    )
    best_distance = float(local["distance_pixels"])
    best_score = float(local["score"])
    return {
        "source_id": source_id,
        "cutout_id": cutout_id,
        "mag_r": mag_r,
        "mag_r_bin": mag_bin_for_mag(mag_r),
        "kind": kind,
        "label": "star" if kind == "psf_star" else "galaxy",
        "x": x,
        "y": y,
        "rounded_x": float(nearest_x),
        "rounded_y": float(nearest_y),
        "center_score": center_score,
        "best_score": best_score,
        "best_x": float(local["x"]),
        "best_y": float(local["y"]),
        "best_distance_pixels": best_distance,
        "center_score_ge_low_floor": center_score >= low_floor,
        "best_score_ge_low_floor": best_score >= low_floor,
        "best_score_ge_fixed_threshold": best_score >= fixed_threshold,
        "best_within_match_radius": best_distance <= match_radius_pixels,
        "best_within_match_radius_and_low_floor": best_distance <= match_radius_pixels and best_score >= low_floor,
        "best_within_match_radius_and_fixed_threshold": best_distance <= match_radius_pixels and best_score >= fixed_threshold,
    }


def morphology_condition_grid() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for mag_r in (21.0, 22.0):
        rows.append({"mag_r": mag_r, "kind": "psf_star", "radius": 1.3, "ellipticity": 0.0})
        for radius in (1.3, 2.4, 3.6):
            rows.append({"mag_r": mag_r, "kind": "sersic_galaxy", "radius": radius, "ellipticity": 0.25})
    return rows


def morphology_injection_specs_for_background(
    rng: random.Random,
    *,
    cutout_id: str,
    shape: tuple[int, int],
) -> list[InjectionSpec]:
    height, width = shape
    margin = 12.0
    color_scale = {"u": 0.45, "g": 0.75, "r": 1.0, "i": 0.9, "z": 0.7}
    specs: list[InjectionSpec] = []
    for index, row in enumerate(morphology_condition_grid()):
        mag_r = float(row["mag_r"])
        kind = str(row["kind"])
        radius = float(row["radius"])
        x = rng.uniform(margin, max(margin, width - margin - 1.0))
        y = rng.uniform(margin, max(margin, height - margin - 1.0))
        flux_r = mag_to_flux(mag_r)
        specs.append(
            InjectionSpec(
                source_id=f"{cutout_id}__morph{index:03d}",
                x=x,
                y=y,
                fluxes={band: flux_r * color_scale[band] for band in BANDS},
                kind=kind,
                radius=radius,
                ellipticity=float(row["ellipticity"]),
                metadata={
                    "mag_r": mag_r,
                    "radius": radius,
                    "condition": morphology_condition_values(mag_r=mag_r, kind=kind, radius=radius),
                },
            )
        )
    return specs


def morphology_condition(spec: InjectionSpec) -> str:
    mag_r = float(spec.metadata.get("mag_r", 0.0))
    return morphology_condition_values(mag_r=mag_r, kind=spec.kind, radius=float(spec.radius))


def morphology_condition_values(*, mag_r: float, kind: str, radius: float) -> str:
    label = "star" if kind == "psf_star" else "galaxy"
    return f"mag{mag_r:g}_{label}_r{radius:g}"


def morphology_plan_row(spec: InjectionSpec) -> dict[str, Any]:
    return {
        "source_id": spec.source_id,
        "x": spec.x,
        "y": spec.y,
        "kind": spec.kind,
        "label": "star" if spec.kind == "psf_star" else "galaxy",
        "radius": spec.radius,
        "ellipticity": spec.ellipticity,
        "mag_r": spec.metadata.get("mag_r"),
        "flux_r": spec.fluxes.get("r"),
        "condition": morphology_condition(spec),
    }


def morphology_recall_by_condition(truth: Sequence[SourceRecord], matches: Any) -> dict[str, dict[str, float | str]]:
    matched = {match.truth_id for match in matches.matches}
    rows: dict[str, dict[str, float | str]] = {}
    for record in truth:
        if record.mag_r is None or record.size is None:
            continue
        condition = morphology_condition_values(
            mag_r=float(record.mag_r),
            kind="psf_star" if record.label == "star" else "sersic_galaxy",
            radius=float(record.size),
        )
        row = rows.setdefault(
            condition,
            {
                "condition": condition,
                "mag_r": float(record.mag_r),
                "label": record.label,
                "radius": float(record.size),
                "n": 0.0,
                "detected": 0.0,
                "recall": 0.0,
            },
        )
        row["n"] = float(row["n"]) + 1.0
        if record.source_id in matched:
            row["detected"] = float(row["detected"]) + 1.0
    for row in rows.values():
        row["recall"] = safe_div(float(row["detected"]), float(row["n"]))
    return dict(sorted(rows.items(), key=lambda item: (float(item[1]["mag_r"]), str(item[1]["label"]), float(item[1]["radius"]))))


def morphology_response_by_condition(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    conditions = sorted(
        {str(row.get("condition")) for row in rows if row.get("condition")},
        key=morphology_condition_sort_key,
    )
    out: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        members = [row for row in rows if row.get("condition") == condition]
        if members:
            first = members[0]
            summary = summarize_heatmap_response_rows(members)
            summary.update(
                {
                    "condition": condition,
                    "mag_r": float(first.get("mag_r")),
                    "label": str(first.get("label")),
                    "kind": str(first.get("kind")),
                    "radius": float(first.get("radius")),
                }
            )
            out[condition] = summary
    return out


def morphology_condition_sort_key(condition: str) -> tuple[float, str, float]:
    parts = condition.split("_")
    mag = 0.0
    label = ""
    radius = 0.0
    for part in parts:
        if part.startswith("mag"):
            try:
                mag = float(part.removeprefix("mag"))
            except ValueError:
                mag = 0.0
        elif part.startswith("r"):
            try:
                radius = float(part.removeprefix("r"))
            except ValueError:
                radius = 0.0
        else:
            label = part
    return mag, label, radius


def morphology_diagnostic_findings(
    response_by_condition: Mapping[str, Mapping[str, Any]],
    recall_by_condition: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, str]]]:
    findings: list[str] = []
    next_actions: list[dict[str, str]] = []
    star22 = response_by_condition.get("mag22_star_r1.3", {})
    galaxy22_small = response_by_condition.get("mag22_galaxy_r1.3", {})
    galaxy22_mid = response_by_condition.get("mag22_galaxy_r2.4", {})
    galaxy22_large = response_by_condition.get("mag22_galaxy_r3.6", {})
    star_response = float(star22.get("best_within_match_radius_and_low_floor", 0.0) or 0.0)
    small_response = float(galaxy22_small.get("best_within_match_radius_and_low_floor", 0.0) or 0.0)
    mid_response = float(galaxy22_mid.get("best_within_match_radius_and_low_floor", 0.0) or 0.0)
    large_response = float(galaxy22_large.get("best_within_match_radius_and_low_floor", 0.0) or 0.0)
    star_recall = float(recall_by_condition.get("mag22_star_r1.3", {}).get("recall", 0.0) or 0.0)
    mid_recall = float(recall_by_condition.get("mag22_galaxy_r2.4", {}).get("recall", 0.0) or 0.0)
    large_recall = float(recall_by_condition.get("mag22_galaxy_r3.6", {}).get("recall", 0.0) or 0.0)
    findings.append(
        "mag22 response: "
        f"star_r1.3 low-floor={star_response}, galaxy_r1.3={small_response}, "
        f"galaxy_r2.4={mid_response}, galaxy_r3.6={large_response}."
    )
    findings.append(
        f"mag22 fixed-threshold recall: star_r1.3={star_recall}, galaxy_r2.4={mid_recall}, galaxy_r3.6={large_recall}."
    )
    if star_response - mid_response >= 0.3 and small_response >= mid_response:
        next_actions.append(
            {
                "action": "design_surface_brightness_or_extended_profile_rescue",
                "reason": "faint galaxy response worsens with extended radius while star controls remain stronger",
            }
        )
    elif star_response - small_response >= 0.3:
        next_actions.append(
            {
                "action": "audit_synthetic_profile_or_catalog_label_protocol",
                "reason": "even compact faint galaxy injections underperform star controls, suggesting protocol or label-shape mismatch",
            }
        )
    else:
        next_actions.append(
            {
                "action": "expand_synthetic_validation_backgrounds",
                "reason": "morphology response is not the dominant faint-source failure under this sweep",
            }
        )
    return findings, next_actions


def write_morphology_response_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "source_id",
        "cutout_id",
        "condition",
        "mag_r",
        "mag_r_bin",
        "kind",
        "label",
        "radius",
        "ellipticity",
        "x",
        "y",
        "rounded_x",
        "rounded_y",
        "center_score",
        "best_score",
        "best_x",
        "best_y",
        "best_distance_pixels",
        "center_score_ge_low_floor",
        "best_score_ge_low_floor",
        "best_score_ge_fixed_threshold",
        "best_within_match_radius",
        "best_within_match_radius_and_low_floor",
        "best_within_match_radius_and_fixed_threshold",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def render_synthetic_morphology_diagnostic_markdown(payload: Mapping[str, Any]) -> str:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), Mapping) else {}
    recall = metrics.get("recall_by_condition", {}) if isinstance(metrics.get("recall_by_condition"), Mapping) else {}
    response = (
        metrics.get("response_by_condition", {}) if isinstance(metrics.get("response_by_condition"), Mapping) else {}
    )
    detection = (
        metrics.get("fixed_threshold_detection", {})
        if isinstance(metrics.get("fixed_threshold_detection"), Mapping)
        else {}
    )
    ap = metrics.get("average_precision", {}) if isinstance(metrics.get("average_precision"), Mapping) else {}
    lines = [
        "# Synthetic Morphology-Response Diagnostic",
        "",
        f"- Validation run: {payload.get('validation_run_dir', '')}",
        f"- Threshold: {payload.get('threshold')}",
        f"- Low floor: {payload.get('low_floor')}",
        f"- Match radius pixels: {payload.get('match_radius_pixels')}",
        f"- Search radius pixels: {payload.get('search_radius_pixels')}",
        f"- Gate: {payload.get('claim_gate', {}).get('status', '') if isinstance(payload.get('claim_gate'), Mapping) else ''}",
        "",
        "## Fixed Threshold",
        "",
        f"- Precision: {detection.get('precision')}",
        f"- Recall: {detection.get('recall')}",
        f"- F1: {detection.get('f1')}",
        f"- AP: {ap.get('ap')}",
        "",
        "## Condition Summary",
        "",
    ]
    for condition, response_row in response.items():
        recall_row = recall.get(condition, {}) if isinstance(recall.get(condition), Mapping) else {}
        best = response_row.get("best_score", {}) if isinstance(response_row.get("best_score"), Mapping) else {}
        lines.append(
            f"- {condition}: n={response_row.get('n')} recall={recall_row.get('recall')} "
            f"best_median={best.get('median')} "
            f"within_radius_low_floor={response_row.get('best_within_match_radius_and_low_floor')}"
        )
    lines.extend(["", "## Findings", ""])
    for finding in payload.get("findings", []):
        lines.append(f"- {finding}")
    lines.extend(["", "## Next Actions", ""])
    for action in payload.get("next_actions", []):
        lines.append(f"- {action.get('action')}: {action.get('reason')}")
    lines.append("")
    return "\n".join(lines)


def best_local_score(scores: Any, *, x: float, y: float, radius_pixels: float) -> dict[str, float]:
    height, width = scores.shape
    radius = max(0, int(round(radius_pixels)))
    x0 = max(0, int(round(x)) - radius)
    x1 = min(width - 1, int(round(x)) + radius)
    y0 = max(0, int(round(y)) - radius)
    y1 = min(height - 1, int(round(y)) + radius)
    best = {"score": float("-inf"), "x": float(x0), "y": float(y0), "distance_pixels": float("inf")}
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            distance = ((float(xx) - x) ** 2 + (float(yy) - y) ** 2) ** 0.5
            if distance > radius_pixels:
                continue
            score = float(scores[yy, xx])
            if score > best["score"]:
                best = {"score": score, "x": float(xx), "y": float(yy), "distance_pixels": distance}
    return best


def heatmap_response_by_mag(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for lo, hi in MAG_R_BINS:
        key = mag_bin_label(lo, hi)
        members = [row for row in rows if row.get("mag_r_bin") == key]
        out[key] = summarize_heatmap_response_rows(members)
    return out


def heatmap_response_by_mag_and_label(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    labels = sorted({str(row.get("label") or row.get("kind") or "") for row in rows if row.get("label") or row.get("kind")})
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for lo, hi in MAG_R_BINS:
        key = mag_bin_label(lo, hi)
        out[key] = {}
        for label in labels:
            members = [
                row
                for row in rows
                if row.get("mag_r_bin") == key and str(row.get("label") or row.get("kind") or "") == label
            ]
            out[key][label] = summarize_heatmap_response_rows(members)
    return out


def summarize_heatmap_response_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "n": float(count),
        "center_score": summarize_values([float(row["center_score"]) for row in rows]),
        "best_score": summarize_values([float(row["best_score"]) for row in rows]),
        "best_distance_pixels": summarize_values([float(row["best_distance_pixels"]) for row in rows]),
        "center_score_ge_low_floor": safe_div(sum(bool(row.get("center_score_ge_low_floor")) for row in rows), count),
        "best_score_ge_low_floor": safe_div(sum(bool(row.get("best_score_ge_low_floor")) for row in rows), count),
        "best_score_ge_fixed_threshold": safe_div(
            sum(bool(row.get("best_score_ge_fixed_threshold")) for row in rows),
            count,
        ),
        "best_within_match_radius": safe_div(sum(bool(row.get("best_within_match_radius")) for row in rows), count),
        "best_within_match_radius_and_low_floor": safe_div(
            sum(bool(row.get("best_within_match_radius_and_low_floor")) for row in rows),
            count,
        ),
        "best_within_match_radius_and_fixed_threshold": safe_div(
            sum(bool(row.get("best_within_match_radius_and_fixed_threshold")) for row in rows),
            count,
        ),
    }


def heatmap_response_findings(
    by_mag: Mapping[str, Mapping[str, Any]],
    *,
    low_floor: float,
) -> tuple[list[str], list[dict[str, str]]]:
    findings: list[str] = []
    next_actions: list[dict[str, str]] = []
    faint_key = mag_bin_label(*MAG_R_BINS[-1])
    faint = by_mag.get(faint_key, {})
    within_low_floor = float(faint.get("best_within_match_radius_and_low_floor", 0.0) or 0.0)
    best_score = faint.get("best_score", {}) if isinstance(faint.get("best_score"), Mapping) else {}
    center_score = faint.get("center_score", {}) if isinstance(faint.get("center_score"), Mapping) else {}
    findings.append(
        f"Faintest injected bin {faint_key} has match-radius local response above {low_floor} for {within_low_floor} of sources."
    )
    if within_low_floor < 0.5:
        next_actions.append(
            {
                "action": "increase_faint_source_signal_or_loss_weight_diagnostic",
                "reason": (
                    "raw center heatmap response is weak near most faint injected sources; candidate threshold changes are insufficient"
                ),
            }
        )
    else:
        next_actions.append(
            {
                "action": "inspect_decode_nms_or_matching_radius",
                "reason": (
                    "raw heatmap response exists near faint sources, so missed detections may be caused by decoding or matching"
                ),
            }
        )
    findings.append(
        "Faintest bin score summary: "
        f"center median={center_score.get('median')}, best local median={best_score.get('median')}."
    )
    return findings, next_actions


def write_heatmap_response_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "source_id",
        "cutout_id",
        "mag_r",
        "mag_r_bin",
        "kind",
        "label",
        "x",
        "y",
        "rounded_x",
        "rounded_y",
        "center_score",
        "best_score",
        "best_x",
        "best_y",
        "best_distance_pixels",
        "center_score_ge_low_floor",
        "best_score_ge_low_floor",
        "best_score_ge_fixed_threshold",
        "best_within_match_radius",
        "best_within_match_radius_and_low_floor",
        "best_within_match_radius_and_fixed_threshold",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def render_synthetic_heatmap_diagnostic_markdown(payload: Mapping[str, Any]) -> str:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), Mapping) else {}
    by_mag = metrics.get("by_mag_r", {}) if isinstance(metrics.get("by_mag_r"), Mapping) else {}
    by_mag_label = metrics.get("by_mag_r_and_label", {}) if isinstance(metrics.get("by_mag_r_and_label"), Mapping) else {}
    lines = [
        "# Synthetic Heatmap-Response Diagnostic",
        "",
        f"- Validation run: {payload.get('validation_run_dir', '')}",
        f"- Threshold: {payload.get('threshold')}",
        f"- Low floor: {payload.get('low_floor')}",
        f"- Match radius pixels: {payload.get('match_radius_pixels')}",
        f"- Search radius pixels: {payload.get('search_radius_pixels')}",
        f"- Gate: {payload.get('claim_gate', {}).get('status', '') if isinstance(payload.get('claim_gate'), Mapping) else ''}",
        "",
        "## Response By Injected mag_r",
        "",
    ]
    for key, row in by_mag.items():
        best = row.get("best_score", {}) if isinstance(row, Mapping) else {}
        center = row.get("center_score", {}) if isinstance(row, Mapping) else {}
        lines.append(
            f"- {key}: n={row.get('n')} center_median={center.get('median')} "
            f"best_median={best.get('median')} "
            f"within_radius_low_floor={row.get('best_within_match_radius_and_low_floor')}"
        )
    if by_mag_label:
        lines.extend(["", "## Response By Injected mag_r And Label", ""])
        for key, label_rows in by_mag_label.items():
            if not isinstance(label_rows, Mapping):
                continue
            for label, row in label_rows.items():
                if not isinstance(row, Mapping):
                    continue
                best = row.get("best_score", {}) if isinstance(row.get("best_score"), Mapping) else {}
                lines.append(
                    f"- {key} {label}: n={row.get('n')} best_median={best.get('median')} "
                    f"within_radius_low_floor={row.get('best_within_match_radius_and_low_floor')}"
                )
    lines.extend(["", "## Findings", ""])
    for finding in payload.get("findings", []):
        lines.append(f"- {finding}")
    lines.extend(["", "## Next Actions", ""])
    for action in payload.get("next_actions", []):
        lines.append(f"- {action.get('action')}: {action.get('reason')}")
    lines.append("")
    return "\n".join(lines)


def resolve_threshold(report: Mapping[str, Any], threshold: float | None) -> float:
    if threshold is not None:
        return float(threshold)
    parameters = report.get("parameters", {}) if isinstance(report.get("parameters"), Mapping) else {}
    return float(parameters.get("threshold", 0.2))


def resolve_match_radius(report: Mapping[str, Any], match_radius_pixels: float | None) -> float:
    if match_radius_pixels is not None:
        return float(match_radius_pixels)
    parameters = report.get("parameters", {}) if isinstance(report.get("parameters"), Mapping) else {}
    return float(parameters.get("match_radius_pixels", 2.0))


def validation_output_path(
    validation_dir: Path,
    report: Mapping[str, Any],
    key: str,
    default_name: str,
) -> Path:
    outputs = report.get("outputs", {}) if isinstance(report.get("outputs"), Mapping) else {}
    value = outputs.get(key)
    if value:
        return Path(str(value))
    return validation_dir / default_name


def mag_bin_diagnostics(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    *,
    radius_arcsec: float,
    threshold: float,
    max_thresholds: int,
    false_negative_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    false_negatives_by_bin: dict[str, list[Mapping[str, Any]]] = {}
    for row in false_negative_rows:
        false_negatives_by_bin.setdefault(str(row.get("mag_r_bin", "")), []).append(row)

    thresholds = select_score_thresholds(
        sorted({prediction.score for prediction in predictions}, reverse=True),
        max_thresholds=max_thresholds,
    )
    rows: dict[str, dict[str, Any]] = {}
    for lo, hi in MAG_R_BINS:
        key = mag_bin_label(lo, hi)
        members = [record for record in truth if record.mag_r is not None and lo <= float(record.mag_r) < hi]
        fixed_predictions = [prediction for prediction in predictions if prediction.score >= threshold]
        fixed_matches = match_catalogs(members, fixed_predictions, radius_arcsec=radius_arcsec)
        matched_scores = [
            fixed_matches.prediction_by_id[match.prediction_id].score
            for match in fixed_matches.matches
            if match.prediction_id in fixed_matches.prediction_by_id
        ]
        bin_false_negatives = false_negatives_by_bin.get(key, [])
        rows[key] = {
            "n": float(len(members)),
            "fixed_threshold": {
                "threshold": threshold,
                "detected": float(len(fixed_matches.matches)),
                "recall": safe_div(len(fixed_matches.matches), len(members)),
                "matched_score": summarize_values(matched_scores),
                "false_negatives": float(len(bin_false_negatives)),
                "false_negative_nearest_candidate": summarize_false_negative_nearest_candidates(bin_false_negatives),
            },
            "threshold_sweep": recall_sweep(members, predictions, radius_arcsec=radius_arcsec, thresholds=thresholds),
        }
    return rows


def recall_sweep(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    *,
    radius_arcsec: float,
    thresholds: Sequence[float],
) -> list[dict[str, float]]:
    if not truth:
        return []
    rows = []
    for threshold in thresholds:
        active = [prediction for prediction in predictions if prediction.score >= threshold]
        matches = match_catalogs(truth, active, radius_arcsec=radius_arcsec)
        rows.append(
            {
                "threshold": float(threshold),
                "detected": float(len(matches.matches)),
                "recall": safe_div(len(matches.matches), len(truth)),
            }
        )
    return rows


def false_negative_diagnostics(
    truth: Sequence[SourceRecord],
    predictions: Sequence[PredictionRecord],
    unmatched_truth_ids: Sequence[str],
    *,
    radius_arcsec: float,
) -> list[dict[str, Any]]:
    truth_by_id = {record.source_id: record for record in truth}
    predictions_by_cutout: dict[str, list[PredictionRecord]] = {}
    for prediction in predictions:
        predictions_by_cutout.setdefault(prediction.cutout_id, []).append(prediction)

    rows = []
    for truth_id in unmatched_truth_ids:
        record = truth_by_id[truth_id]
        nearest = nearest_prediction(record, predictions_by_cutout.get(record.cutout_id, []))
        nearest_within_radius = nearest is not None and float(nearest["distance_arcsec"]) <= radius_arcsec
        rows.append(
            {
                "source_id": record.source_id,
                "cutout_id": record.cutout_id,
                "mag_r": record.mag_r,
                "mag_r_bin": mag_bin_for_record(record),
                "x": record.x,
                "y": record.y,
                "nearest_prediction_id": nearest.get("prediction_id") if nearest else None,
                "nearest_score": nearest.get("score") if nearest else None,
                "nearest_distance_arcsec": nearest.get("distance_arcsec") if nearest else None,
                "nearest_within_match_radius": nearest_within_radius,
            }
        )
    return rows


def nearest_prediction(record: SourceRecord, predictions: Sequence[PredictionRecord]) -> dict[str, Any] | None:
    nearest: dict[str, Any] | None = None
    for prediction in predictions:
        distance = angular_distance_arcsec(record.ra, record.dec, prediction.ra, prediction.dec)
        if nearest is None or distance < float(nearest["distance_arcsec"]):
            nearest = {
                "prediction_id": prediction.prediction_id,
                "score": prediction.score,
                "distance_arcsec": distance,
            }
    return nearest


def summarize_false_negative_nearest_candidates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    distances = [float(row["nearest_distance_arcsec"]) for row in rows if row.get("nearest_distance_arcsec") is not None]
    scores = [float(row["nearest_score"]) for row in rows if row.get("nearest_score") is not None]
    within = sum(1 for row in rows if bool(row.get("nearest_within_match_radius")))
    return {
        "count": float(len(rows)),
        "nearest_within_match_radius": float(within),
        "no_candidate_within_match_radius": float(len(rows) - within),
        "nearest_distance_arcsec": summarize_values(distances),
        "nearest_score": summarize_values(scores),
    }


def summarize_values(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0.0, "min": None, "median": None, "mean": None, "max": None}
    sorted_values = sorted(float(value) for value in values)
    return {
        "n": float(len(sorted_values)),
        "min": sorted_values[0],
        "median": median(sorted_values),
        "mean": mean(sorted_values),
        "max": sorted_values[-1],
    }


def compact_score_curve(curve: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = curve.get("thresholds", [])
    return {
        "best_threshold": curve.get("best_threshold"),
        "best_metrics": curve.get("best_metrics", {}),
        "average_precision": curve.get("average_precision", {}),
        "candidate_thresholds": curve.get("candidate_thresholds"),
        "threshold_rows": len(thresholds) if isinstance(thresholds, Sequence) else 0,
    }


def diagnostic_findings(by_mag: Mapping[str, Mapping[str, Any]], threshold: float) -> tuple[list[str], list[dict[str, str]]]:
    findings: list[str] = []
    next_actions: list[dict[str, str]] = []
    faint_key = mag_bin_label(*MAG_R_BINS[-1])
    faint = by_mag.get(faint_key, {})
    fixed = faint.get("fixed_threshold", {}) if isinstance(faint.get("fixed_threshold"), Mapping) else {}
    recall = float(fixed.get("recall", 0.0) or 0.0)
    fn = (
        fixed.get("false_negative_nearest_candidate", {})
        if isinstance(fixed.get("false_negative_nearest_candidate"), Mapping)
        else {}
    )
    no_candidate = float(fn.get("no_candidate_within_match_radius", 0.0) or 0.0)
    false_negative_count = float(fn.get("count", 0.0) or 0.0)
    if recall < 0.5:
        findings.append(f"Faintest injected bin {faint_key} has low fixed-threshold recall {recall}.")
    if false_negative_count and no_candidate / false_negative_count >= 0.5:
        findings.append(
            "Most faint false negatives have no decoded candidate within the match radius at the current candidate floor."
        )
        next_actions.append(
            {
                "action": "rerun_synthetic_low_candidate_floor_diagnostic",
                "reason": (
                    f"threshold sweeps above the current floor {threshold} cannot recover sources that were never decoded"
                ),
            }
        )
    else:
        next_actions.append(
            {
                "action": "inspect_faint_candidate_scores_and_conflicts",
                "reason": "some faint false negatives have nearby decoded candidates, so score calibration or matching conflicts may dominate",
            }
        )
    if not findings:
        findings.append("No faint-bin recovery failure is apparent under the current diagnostic thresholds.")
    return findings, next_actions


def write_false_negative_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "source_id",
        "cutout_id",
        "mag_r",
        "mag_r_bin",
        "x",
        "y",
        "nearest_prediction_id",
        "nearest_score",
        "nearest_distance_arcsec",
        "nearest_within_match_radius",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def render_synthetic_diagnostic_markdown(payload: Mapping[str, Any]) -> str:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), Mapping) else {}
    fixed = metrics.get("fixed_threshold_detection", {}) if isinstance(metrics.get("fixed_threshold_detection"), Mapping) else {}
    by_mag = metrics.get("by_mag_r", {}) if isinstance(metrics.get("by_mag_r"), Mapping) else {}
    lines = [
        "# Synthetic Faint-Recovery Diagnostic",
        "",
        f"- Validation run: {payload.get('validation_run_dir', '')}",
        f"- Threshold: {payload.get('threshold')}",
        f"- Match radius pixels: {payload.get('match_radius_pixels')}",
        f"- Gate: {payload.get('claim_gate', {}).get('status', '') if isinstance(payload.get('claim_gate'), Mapping) else ''}",
        "",
        "## Fixed Threshold",
        "",
        f"- Precision: {fixed.get('precision')}",
        f"- Recall: {fixed.get('recall')}",
        f"- F1: {fixed.get('f1')}",
        "",
        "## Recall By Injected mag_r",
        "",
    ]
    for key, row in by_mag.items():
        fixed_row = row.get("fixed_threshold", {}) if isinstance(row, Mapping) else {}
        nearest = (
            fixed_row.get("false_negative_nearest_candidate", {})
            if isinstance(fixed_row.get("false_negative_nearest_candidate"), Mapping)
            else {}
        )
        lines.append(
            f"- {key}: n={row.get('n')} recall={fixed_row.get('recall')} "
            f"false_negatives={fixed_row.get('false_negatives')} "
            f"no_candidate_within_radius={nearest.get('no_candidate_within_match_radius')}"
        )
    lines.extend(["", "## Findings", ""])
    for finding in payload.get("findings", []):
        lines.append(f"- {finding}")
    lines.extend(["", "## Next Actions", ""])
    for action in payload.get("next_actions", []):
        lines.append(f"- {action.get('action')}: {action.get('reason')}")
    lines.append("")
    return "\n".join(lines)


def mag_bin_for_record(record: SourceRecord) -> str:
    if record.mag_r is None:
        return "unknown"
    return mag_bin_for_mag(float(record.mag_r))


def mag_bin_for_mag(mag_r: float) -> str:
    for lo, hi in MAG_R_BINS:
        if lo <= mag_r < hi:
            return mag_bin_label(lo, hi)
    return "out_of_range"


def mag_bin_label(lo: float, hi: float) -> str:
    return f"[{lo},{hi})"


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
