from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baseline import make_catalog_model
from .decode import decode_predictions
from .io import write_prediction_catalog, write_source_catalog
from .matching import match_catalogs
from .metrics import detection_metrics, detection_score_curve
from .pilot_loop import write_json
from .schema import BANDS, PredictionRecord, SourceRecord
from .synthetic import InjectionSpec, inject_sources
from .training import NpzCutoutDataset

SYNTHETIC_VALIDATION_SCHEMA_VERSION = 1
SYNTHETIC_VALIDATION_PROGRAM_ID = "synthetic_injection_mainline"


def build_synthetic_injection_validation(
    *,
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    split_path: str | Path | None = None,
    split_name: str | None = None,
    device: str = "cpu",
    seed: int = 42,
    num_backgrounds: int = 16,
    injections_per_background: int = 4,
    threshold: float = 0.2,
    nms_radius: int = 2,
    match_radius_pixels: float = 2.0,
    max_detections_per_cutout: int | None = 16,
    shard_cache_size: int = 2,
    psf_sigma: float = 1.3,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    dataset = NpzCutoutDataset(
        dataset_dir,
        split_path=split_path,
        split_name=split_name,
        limit_samples=num_backgrounds,
        shard_cache_size=shard_cache_size,
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

    all_truth: list[SourceRecord] = []
    injected_truth: list[SourceRecord] = []
    predictions: list[PredictionRecord] = []
    per_cutout: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []

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
            injected_image, truth = inject_sources(
                background,
                specs,
                bands=BANDS[: background.shape[0]],
                psf_sigma=psf_sigma,
                cutout_id=cutout_id,
            )
            truth = source_records_with_injection_metadata(truth, specs)
            existing_truth = existing_truth_from_sample(sample, cutout_id=cutout_id)
            all_truth.extend(existing_truth)
            all_truth.extend(truth)
            injected_truth.extend(truth)
            plans.extend([injection_plan_row(spec) for spec in specs])

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
            per_cutout.append(
                {
                    "cutout_id": cutout_id,
                    "background_cutout_id": str(sample["cutout_id"]),
                    "injected": len(truth),
                    "existing_truth": len(existing_truth),
                    "candidate_predictions": len(cutout_predictions),
                }
            )

    scored_predictions = [prediction for prediction in predictions if prediction.score >= threshold]
    all_matches = match_catalogs(all_truth, scored_predictions, radius_arcsec=match_radius_pixels)
    injected_matches = match_catalogs(injected_truth, scored_predictions, radius_arcsec=match_radius_pixels)
    all_curve = detection_score_curve(all_truth, predictions, max_radius_arcsec=match_radius_pixels)
    injected_curve = detection_score_curve(injected_truth, predictions, max_radius_arcsec=match_radius_pixels)
    by_mag = injected_recall_by_mag(injected_truth, injected_matches)
    by_mag_and_label = injected_recall_by_mag_and_label(injected_truth, injected_matches)
    all_detection = detection_metrics(all_matches)
    injected_detection = detection_metrics(injected_matches)
    run_id = output.name
    variant_id = output.name
    source_run_id = Path(checkpoint_path).parent.name
    source_variant_id = infer_source_variant_id(source_run_id)
    run_options = {
        "epochs": checkpoint.get("epoch"),
        "train_limit_samples": None,
        "batch_size": None,
        "base_channels": model_config.get("base_channels"),
        "heatmap_sigma": checkpoint.get("heatmap_sigma"),
        "model_arch": model_config.get("model_arch") or model_config.get("name"),
        "loader_mode": "synthetic_injection",
        "predict_limit": num_backgrounds,
        "device": device,
        "loss_variant": infer_source_loss_variant(source_variant_id),
        "source_run_id": source_run_id,
        "source_variant_id": source_variant_id,
    }
    counts = {
        "backgrounds": len(dataset),
        "all_truth": len(all_truth),
        "injected_truth": len(injected_truth),
        "candidate_predictions": len(predictions),
        "thresholded_predictions": len(scored_predictions),
    }
    metrics = {
        "test": {
            "counts": {
                "truth": len(injected_truth),
                "candidate_predictions": len(scored_predictions),
                "matches": int(injected_detection.get("tp", 0.0)),
                "unmatched_predictions": int(injected_detection.get("fp", 0.0)),
                "unmatched_truth": int(injected_detection.get("fn", 0.0)),
            },
            "detection": injected_detection,
            "average_precision": injected_curve["average_precision"],
            "stratified_detection": {
                "injected_mag_r": by_mag,
                "injected_mag_r_and_label": by_mag_and_label,
            },
        },
        "validation": {
            "type": "synthetic_injection_mainline",
            "source_run_id": source_run_id,
            "source_variant_id": source_variant_id,
            "threshold": threshold,
            "match_radius_pixels": match_radius_pixels,
        },
        "all_truth_detection": all_detection,
        "injected_detection": injected_detection,
        "all_truth_average_precision": all_curve["average_precision"],
        "injected_average_precision": injected_curve["average_precision"],
        "injected_recall_by_mag_r": by_mag,
        "injected_recall_by_mag_r_and_label": by_mag_and_label,
    }
    claim_gate = synthetic_claim_gate(
        output_dir=output,
        num_backgrounds=len(dataset),
        injected_truth_count=len(injected_truth),
        thresholded_prediction_count=len(scored_predictions),
        injected_detection=injected_detection,
        injected_average_precision=injected_curve["average_precision"],
    )

    write_source_catalog(all_truth, output / "truth_all.csv")
    write_source_catalog(injected_truth, output / "truth_injected.csv")
    write_prediction_catalog(predictions, output / "predictions_candidates.csv")
    write_json(output / "injection_plan.json", {"seed": seed, "plans": plans})

    payload = {
        "schema_version": SYNTHETIC_VALIDATION_SCHEMA_VERSION,
        "protocol": "sdss-point-supervised-v1",
        "run_id": run_id,
        "program_id": SYNTHETIC_VALIDATION_PROGRAM_ID,
        "variant_id": variant_id,
        "status": "executed",
        "objective": "Validate the center-only e50 mainline checkpoint on controlled synthetic injections.",
        "hypothesis": (
            "A checkpoint that agrees with weak PhotoObj labels should also recover controlled injected sources "
            "across the fixed magnitude ladder."
        ),
        "tags": [
            "paper_validation",
            "mainline_validation",
            "synthetic_injection",
            "fixed_split",
            "native_frame",
            "center_only_e50",
        ],
        "claims": ["synthetic_injection", "architecture_improvement", "stratified_metrics"],
        "validation": "synthetic_injection_mainline",
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_dir),
        "split": str(split_path) if split_path is not None else None,
        "split_name": split_name,
        "seed": seed,
        "inputs": {
            "checkpoint": str(checkpoint_path),
            "dataset": str(dataset_dir),
            "split": str(split_path) if split_path is not None else None,
        },
        "run_options": run_options,
        "parameters": {
            "num_backgrounds": num_backgrounds,
            "injections_per_background": injections_per_background,
            "threshold": threshold,
            "nms_radius": nms_radius,
            "match_radius_pixels": match_radius_pixels,
            "max_detections_per_cutout": max_detections_per_cutout,
            "psf_sigma": psf_sigma,
        },
        "counts": counts,
        "metrics": metrics,
        "per_cutout": per_cutout,
        "outputs": {
            "truth_all": str(output / "truth_all.csv"),
            "truth_injected": str(output / "truth_injected.csv"),
            "predictions": str(output / "predictions_candidates.csv"),
            "injection_plan": str(output / "injection_plan.json"),
        },
        "claim_gate": claim_gate,
        "next_actions": synthetic_next_actions(by_mag),
        "notes": [
            "Pixel coordinates are used as a local angular coordinate system with one pixel equal to one matching unit.",
            "Existing weak labels in each background are included in all-truth metrics so background sources are not automatically counted as false positives.",
            "Injected-only metrics isolate recovery of controlled added sources.",
        ],
    }
    write_json(output / "report.json", payload)
    (output / "report.md").write_text(render_synthetic_validation_markdown(payload), encoding="utf-8")
    return payload


def infer_source_variant_id(source_run_id: str) -> str | None:
    marker = "_agent_loss_diagnosis_v1_"
    if marker in source_run_id:
        return source_run_id.split(marker, 1)[1]
    marker = "_pilot_"
    if marker in source_run_id:
        return source_run_id.split(marker, 1)[1]
    return None


def infer_source_loss_variant(source_variant_id: str | None) -> str:
    if not source_variant_id:
        return "unknown_checkpoint"
    if "center_only" in source_variant_id:
        return "center_only"
    if "no_psf" in source_variant_id:
        return "no_psf_reconstruction"
    if "full_psf" in source_variant_id:
        return "full_psf_point_supervised"
    return "unknown_checkpoint"


def synthetic_claim_gate(
    *,
    output_dir: Path,
    num_backgrounds: int,
    injected_truth_count: int,
    thresholded_prediction_count: int,
    injected_detection: Mapping[str, Any],
    injected_average_precision: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if injected_truth_count <= 0:
        reasons.append("synthetic injected truth count is zero")
    if thresholded_prediction_count <= 0:
        reasons.append("synthetic validation emitted no thresholded predictions")
    if reasons:
        return {
            "status": "blocked",
            "paper_ready": False,
            "paper_claim_allowed": False,
            "reasons": reasons,
        }

    precision = float(injected_detection.get("precision", 0.0) or 0.0)
    recall = float(injected_detection.get("recall", 0.0) or 0.0)
    ap = float(injected_average_precision.get("ap", 0.0) or 0.0)
    engineering_reasons = []
    output_text = str(output_dir).lower()
    if "smoke" in output_text:
        engineering_reasons.append("smoke synthetic validation output")
    if "diagnostic" in output_text:
        engineering_reasons.append("diagnostic synthetic validation output")
    if num_backgrounds < 8:
        engineering_reasons.append("fewer than 8 synthetic backgrounds")
    if engineering_reasons:
        return {
            "status": "engineering_check",
            "paper_ready": False,
            "paper_claim_allowed": False,
            "reasons": engineering_reasons,
            "observed_precision": precision,
            "observed_recall": recall,
            "observed_ap": ap,
        }
    if recall > 0.0 and ap > 0.0:
        return {
            "status": "candidate_evidence",
            "paper_ready": False,
            "paper_claim_allowed": False,
            "reasons": [
                "controlled synthetic truth is available",
                "validation uses a fixed checkpoint and fixed threshold",
            ],
            "observed_precision": precision,
            "observed_recall": recall,
            "observed_ap": ap,
        }
    return {
        "status": "blocked",
        "paper_ready": False,
        "paper_claim_allowed": False,
        "reasons": ["synthetic injected recall or AP is zero"],
        "observed_precision": precision,
        "observed_recall": recall,
        "observed_ap": ap,
    }


def synthetic_next_actions(by_mag: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    faint = faintest_mag_bin(by_mag)
    if faint and float(faint.get("recall", 0.0) or 0.0) < 0.5:
        return [
            {
                "action": "diagnose_faint_synthetic_recovery",
                "reason": f"faintest injected mag_r bin {faint.get('bin')} has recall {faint.get('recall')}",
            }
        ]
    return [
        {
            "action": "expand_synthetic_validation",
            "reason": "initial controlled-injection validation has nonzero recovery and should be scaled after protocol review",
        }
    ]


def faintest_mag_bin(by_mag: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    rows: list[tuple[float, str, Mapping[str, Any]]] = []
    for key, row in by_mag.items():
        hi = parse_mag_bin_upper(key)
        if hi is None:
            continue
        rows.append((hi, key, row))
    if not rows:
        return None
    _hi, key, row = sorted(rows, key=lambda item: item[0], reverse=True)[0]
    return {"bin": key, **dict(row)}


def parse_mag_bin_upper(label: str) -> float | None:
    try:
        return float(label.rsplit(",", 1)[1].rstrip(")"))
    except (IndexError, ValueError):
        return None


def injection_specs_for_background(
    rng: random.Random,
    *,
    cutout_id: str,
    count: int,
    shape: tuple[int, int],
) -> list[InjectionSpec]:
    height, width = shape
    margin = 12.0
    mags = [18.0, 20.0, 21.0, 22.0]
    kinds = ["psf_star", "sersic_galaxy"]
    specs: list[InjectionSpec] = []
    for index in range(count):
        x = rng.uniform(margin, max(margin, width - margin - 1.0))
        y = rng.uniform(margin, max(margin, height - margin - 1.0))
        if count >= len(mags) * len(kinds):
            mag_r = mags[(index // len(kinds)) % len(mags)]
            kind = kinds[index % len(kinds)]
        else:
            mag_r = mags[index % len(mags)]
            kind = kinds[index % len(kinds)]
        flux_r = mag_to_flux(mag_r)
        color_scale = {"u": 0.45, "g": 0.75, "r": 1.0, "i": 0.9, "z": 0.7}
        specs.append(
            InjectionSpec(
                source_id=f"{cutout_id}__inj{index:03d}",
                x=x,
                y=y,
                fluxes={band: flux_r * color_scale[band] for band in BANDS},
                kind=kind,
                radius=1.5 if kind == "psf_star" else 2.4,
                ellipticity=0.0 if kind == "psf_star" else 0.25,
                metadata={"mag_r": mag_r},
            )
        )
    return specs


def source_records_with_injection_metadata(records: Sequence[SourceRecord], specs: Sequence[InjectionSpec]) -> list[SourceRecord]:
    by_id = {spec.source_id: spec for spec in specs}
    enriched: list[SourceRecord] = []
    for record in records:
        spec = by_id.get(record.source_id)
        mag_r = float(spec.metadata["mag_r"]) if spec and "mag_r" in spec.metadata else record.mag_r
        enriched.append(
            SourceRecord(
                source_id=record.source_id,
                cutout_id=record.cutout_id,
                ra=float(record.ra) / 3600.0,
                dec=float(record.dec) / 3600.0,
                label=record.label,
                x=record.ra,
                y=record.dec,
                mag_r=mag_r,
                flux_u=record.flux_u,
                flux_g=record.flux_g,
                flux_r=record.flux_r,
                flux_i=record.flux_i,
                flux_z=record.flux_z,
                size=record.size,
                ellipticity=record.ellipticity,
                label_quality="synthetic",
                label_weight=1.0,
                metadata=record.metadata,
            )
        )
    return enriched


def existing_truth_from_sample(sample: Mapping[str, Any], *, cutout_id: str) -> list[SourceRecord]:
    target_heatmap = sample.get("target_heatmap")
    if target_heatmap is None:
        return []
    heatmap = target_heatmap.squeeze(0).detach().cpu().numpy()
    ys, xs = np.where(heatmap >= 0.95)
    records: list[SourceRecord] = []
    for index, (y, x) in enumerate(zip(ys.tolist(), xs.tolist(), strict=True)):
        records.append(
            SourceRecord(
                source_id=f"{cutout_id}__background{index:04d}",
                cutout_id=cutout_id,
                ra=float(x) / 3600.0,
                dec=float(y) / 3600.0,
                label="star",
                x=float(x),
                y=float(y),
                label_quality="weak_background",
                label_weight=0.5,
            )
        )
    return records


def injected_recall_by_mag(
    injected_truth: Sequence[SourceRecord],
    matches: Any,
) -> dict[str, dict[str, float]]:
    matched = {match.truth_id for match in matches.matches}
    bins = [(17.5, 19.0), (19.0, 20.5), (20.5, 21.5), (21.5, 22.5)]
    rows: dict[str, dict[str, float]] = {}
    for lo, hi in bins:
        members = [
            record
            for record in injected_truth
            if record.mag_r is not None and lo <= float(record.mag_r) < hi
        ]
        found = sum(1 for record in members if record.source_id in matched)
        key = f"[{lo},{hi})"
        rows[key] = {
            "n": float(len(members)),
            "detected": float(found),
            "recall": float(found / len(members)) if members else 0.0,
        }
    return rows


def injected_recall_by_mag_and_label(
    injected_truth: Sequence[SourceRecord],
    matches: Any,
) -> dict[str, dict[str, dict[str, float]]]:
    matched = {match.truth_id for match in matches.matches}
    bins = [(17.5, 19.0), (19.0, 20.5), (20.5, 21.5), (21.5, 22.5)]
    labels = sorted({record.label for record in injected_truth})
    rows: dict[str, dict[str, dict[str, float]]] = {}
    for lo, hi in bins:
        key = f"[{lo},{hi})"
        rows[key] = {}
        for label in labels:
            members = [
                record
                for record in injected_truth
                if record.label == label and record.mag_r is not None and lo <= float(record.mag_r) < hi
            ]
            found = sum(1 for record in members if record.source_id in matched)
            rows[key][label] = {
                "n": float(len(members)),
                "detected": float(found),
                "recall": float(found / len(members)) if members else 0.0,
            }
    return rows


def injection_plan_row(spec: InjectionSpec) -> dict[str, Any]:
    return {
        "source_id": spec.source_id,
        "x": spec.x,
        "y": spec.y,
        "kind": spec.kind,
        "radius": spec.radius,
        "ellipticity": spec.ellipticity,
        "mag_r": spec.metadata.get("mag_r"),
        "flux_r": spec.fluxes.get("r"),
    }


def mag_to_flux(magnitude: float, zeropoint: float = 22.5) -> float:
    if not math.isfinite(magnitude):
        return 0.0
    return float(10.0 ** (-0.4 * (magnitude - zeropoint)))


def local_pixel_to_radec(x: float, y: float) -> tuple[float, float]:
    return float(x) / 3600.0, float(y) / 3600.0


def render_synthetic_validation_markdown(payload: Mapping[str, Any]) -> str:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), Mapping) else {}
    injected = metrics.get("injected_detection", {}) if isinstance(metrics.get("injected_detection"), Mapping) else {}
    injected_ap = metrics.get("injected_average_precision", {}) if isinstance(metrics.get("injected_average_precision"), Mapping) else {}
    all_truth = metrics.get("all_truth_detection", {}) if isinstance(metrics.get("all_truth_detection"), Mapping) else {}
    lines = [
        "# Synthetic Injection Validation",
        "",
        f"- Checkpoint: {payload.get('checkpoint', '')}",
        f"- Dataset: {payload.get('dataset', '')}",
        f"- Seed: {payload.get('seed', '')}",
        f"- Gate: {payload.get('claim_gate', {}).get('status', '') if isinstance(payload.get('claim_gate'), Mapping) else ''}",
        "",
        "## Injected Sources",
        "",
        f"- Precision: {injected.get('precision')}",
        f"- Recall: {injected.get('recall')}",
        f"- F1: {injected.get('f1')}",
        f"- AP: {injected_ap.get('ap')}",
        "",
        "## All Truth",
        "",
        f"- Precision: {all_truth.get('precision')}",
        f"- Recall: {all_truth.get('recall')}",
        f"- F1: {all_truth.get('f1')}",
        "",
        "## Recall By Injected mag_r",
        "",
    ]
    for key, row in (metrics.get("injected_recall_by_mag_r", {}) or {}).items():
        lines.append(f"- {key}: n={row.get('n')} recall={row.get('recall')}")
    by_mag_label = metrics.get("injected_recall_by_mag_r_and_label", {}) or {}
    if by_mag_label:
        lines.extend(["", "## Recall By Injected mag_r And Label", ""])
        for key, label_rows in by_mag_label.items():
            if not isinstance(label_rows, Mapping):
                continue
            for label, row in label_rows.items():
                lines.append(f"- {key} {label}: n={row.get('n')} recall={row.get('recall')}")
    lines.append("")
    return "\n".join(lines)
