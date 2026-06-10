from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .experiment import PROTOCOL, validate_experiment_config
from .pilot_loop import load_json, load_pilot_loop_outputs, run_pilot_loop, write_json

RAW_DATA_ROOT = Path("/Data/sdss")
REPORT_SCHEMA_VERSION = 2
MAX_HASH_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class ResearchRunSpec:
    run_id: str
    objective: str
    hypothesis: str
    config: str | Path
    dataset: str | Path
    split: str | Path
    report_dir: str | Path
    checkpoint_dir: str | Path
    epochs: int = 1
    train_limit_samples: int | None = None
    batch_size: int = 16
    learning_rate: float = 1e-3
    base_channels: int = 32
    model_arch: str = "baseline"
    loader_mode: str = "sample"
    shard_cache_size: int = 0
    num_workers: int = 0
    pin_memory: bool | str = "auto"
    device: str = "cpu"
    seed: int = 42
    candidate_threshold: float = 0.2
    nms_radius: int = 2
    max_detections_per_cutout: int | None = None
    predict_limit: int | None = None
    pixel_scale_arcsec: float = 0.396
    radius_arcsec: float = 1.0
    band: str = "r"
    close_pair_arcsec: float = 2.0
    seeing_aware: bool = False
    psf_fraction: float = 0.5
    include_suspect_truth: bool = False
    dry_run: bool = False
    program_id: str | None = None
    variant_id: str | None = None
    parent_run_id: str | None = None
    tags: tuple[str, ...] = ()


def run_research_run(spec: ResearchRunSpec) -> dict[str, Any]:
    """Run or dry-run an auditable research loop and write standard reports."""

    preflight = preflight_research_run(spec)
    run_root = Path(spec.report_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    plan = build_plan_payload(spec, preflight)
    write_json(run_root / "plan.json", plan)

    if spec.dry_run:
        report = build_research_report(
            spec=spec,
            preflight=preflight,
            pilot_outputs=None,
            mode="dry_run",
        )
    else:
        pilot_output_dir = run_root / "pilot_loop"
        run_pilot_loop(
            config=spec.config,
            dataset=spec.dataset,
            split=spec.split,
            output_dir=pilot_output_dir,
            checkpoint_dir=spec.checkpoint_dir,
            epochs=spec.epochs,
            train_limit_samples=spec.train_limit_samples,
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            base_channels=spec.base_channels,
            model_arch=spec.model_arch,
            loader_mode=spec.loader_mode,
            shard_cache_size=spec.shard_cache_size,
            num_workers=spec.num_workers,
            pin_memory=spec.pin_memory,
            device=spec.device,
            seed=spec.seed,
            candidate_threshold=spec.candidate_threshold,
            nms_radius=spec.nms_radius,
            max_detections_per_cutout=spec.max_detections_per_cutout,
            predict_limit=spec.predict_limit,
            pixel_scale_arcsec=spec.pixel_scale_arcsec,
            radius_arcsec=spec.radius_arcsec,
            band=spec.band,
            close_pair_arcsec=spec.close_pair_arcsec,
            seeing_aware=spec.seeing_aware,
            psf_fraction=spec.psf_fraction,
            include_suspect_truth=spec.include_suspect_truth,
        )
        report = build_research_report(
            spec=spec,
            preflight=preflight,
            pilot_outputs=load_pilot_loop_outputs(pilot_output_dir),
            mode="executed",
        )

    write_report_bundle(run_root, report)
    return report


def write_report_from_existing_pilot_loop(
    *,
    pilot_output_dir: str | Path,
    run_id: str,
    report_dir: str | Path,
    objective: str,
    hypothesis: str,
) -> dict[str, Any]:
    """Build the research report bundle from an existing run-pilot-loop output directory."""

    outputs = load_pilot_loop_outputs(pilot_output_dir)
    summary = outputs["summary"]
    checkpoint_dir = Path(str(summary.get("outputs", {}).get("checkpoint", "checkpoint"))).parent
    spec = ResearchRunSpec(
        run_id=run_id,
        objective=objective,
        hypothesis=hypothesis,
        config=str(summary.get("config", "")),
        dataset=str(summary.get("dataset", "")),
        split=str(summary.get("split", "")),
        report_dir=report_dir,
        checkpoint_dir=checkpoint_dir,
        epochs=int(summary.get("training", {}).get("epochs", 0)),
        train_limit_samples=summary.get("training", {}).get("train_limit_samples"),
        batch_size=int(summary.get("training", {}).get("batch_size", 0)),
        base_channels=int(summary.get("training", {}).get("base_channels", 0)),
        model_arch=str(summary.get("training", {}).get("model_arch", "baseline")),
        loader_mode=str(summary.get("training", {}).get("loader", {}).get("mode", "sample")),
        shard_cache_size=int(summary.get("training", {}).get("loader", {}).get("shard_cache_size", 0) or 0),
        num_workers=int(summary.get("training", {}).get("loader", {}).get("num_workers", 0) or 0),
        pin_memory=summary.get("training", {}).get("loader", {}).get("pin_memory", "auto"),
        device=str(summary.get("training", {}).get("device", "")),
        seed=int(summary.get("training", {}).get("seed", 0)),
        candidate_threshold=float(summary.get("decode", {}).get("candidate_threshold", 0.0)),
        nms_radius=int(summary.get("decode", {}).get("nms_radius_pixels", 0)),
        max_detections_per_cutout=summary.get("decode", {}).get("max_detections_per_cutout"),
        predict_limit=summary.get("decode", {}).get("predict_limit"),
        pixel_scale_arcsec=float(summary.get("decode", {}).get("pixel_scale_arcsec", 0.396)),
        radius_arcsec=float(summary.get("matching", {}).get("radius_arcsec", 1.0)),
        seeing_aware=bool(summary.get("matching", {}).get("seeing_aware", False)),
        psf_fraction=float(summary.get("matching", {}).get("psf_fraction", 0.5)),
        include_suspect_truth=bool(summary.get("truth_policy", {}).get("include_suspect_truth", False)),
    )
    report = build_research_report(
        spec=spec,
        preflight={"status": "not_run", "reason": "report built from existing pilot-loop outputs"},
        pilot_outputs=outputs,
        mode="report_existing",
    )
    run_root = Path(report_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "plan.json", build_plan_payload(spec, report["preflight"]))
    write_report_bundle(run_root, report)
    return report


def preflight_research_run(spec: ResearchRunSpec) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    run_root = Path(spec.report_dir)
    checkpoint_dir = Path(spec.checkpoint_dir)

    if not spec.run_id or "/" in spec.run_id or "\\" in spec.run_id:
        errors.append("run_id must be non-empty and must not contain path separators")
    if run_root.exists() and any(run_root.iterdir()):
        errors.append(f"report_dir already exists and is non-empty: {run_root}")
    if is_inside_raw_data(run_root) or is_inside_raw_data(checkpoint_dir):
        errors.append("report_dir and checkpoint_dir must not be under /Data/sdss")
    if spec.epochs < 0:
        errors.append("epochs must be non-negative")
    if spec.batch_size <= 0:
        errors.append("batch_size must be positive")

    config_path = Path(spec.config)
    dataset_dir = Path(spec.dataset)
    split_path = Path(spec.split)
    for label, path in [("config", config_path), ("dataset", dataset_dir), ("split", split_path)]:
        if not path.exists():
            errors.append(f"{label} path does not exist: {path}")

    config_payload: Mapping[str, Any] = {}
    if config_path.exists():
        try:
            config_payload = load_json(config_path)
            validate_experiment_config(config_payload)
        except Exception as exc:  # noqa: BLE001 - preflight reports all validation failures together
            errors.append(f"invalid config: {exc}")

    split_counts: dict[str, int] = {}
    if split_path.exists():
        try:
            split_payload = load_json(split_path)
            if split_payload.get("protocol") not in {None, PROTOCOL}:
                errors.append(f"split protocol must be {PROTOCOL!r}")
            splits = split_payload.get("splits", {})
            for split_name in ("train", "val", "test"):
                ids = splits.get(split_name)
                if not isinstance(ids, list):
                    errors.append(f"split is missing list entry: {split_name}")
                elif not ids:
                    errors.append(f"split {split_name!r} is empty")
                else:
                    split_counts[split_name] = len(ids)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid split: {exc}")

    dataset_files: dict[str, bool] = {}
    if dataset_dir.exists():
        required_files = {
            "manifest": dataset_dir / "manifest.csv",
            "metadata": dataset_dir / "metadata.json",
            "truth_catalog": dataset_dir / "truth_catalog.csv",
            "shards": dataset_dir / "shards",
        }
        dataset_files = {name: path.exists() for name, path in required_files.items()}
        for name, exists in dataset_files.items():
            if not exists:
                errors.append(f"dataset is missing {name}: {required_files[name]}")
        metadata_path = dataset_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = load_json(metadata_path)
                if metadata.get("protocol") not in {None, PROTOCOL}:
                    errors.append(f"dataset metadata protocol must be {PROTOCOL!r}")
                if "smoke" in str(dataset_dir).lower():
                    warnings.append("dataset path looks like a smoke dataset; claim gate will mark engineering_check")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"invalid dataset metadata: {exc}")

    payload = {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "paths": {
            "config": str(config_path),
            "dataset": str(dataset_dir),
            "split": str(split_path),
            "report_dir": str(run_root),
            "checkpoint_dir": str(checkpoint_dir),
        },
        "dataset_files": dataset_files,
        "split_counts": split_counts,
        "data_root": str(config_payload.get("data", {}).get("root", "")) if config_payload else "",
    }
    if errors:
        raise ValueError("research-run preflight failed: " + "; ".join(errors))
    return payload


def build_plan_payload(spec: ResearchRunSpec, preflight: Mapping[str, Any]) -> dict[str, Any]:
    spec_payload = asdict(spec)
    spec_payload.update(
        {
            "config": str(spec.config),
            "dataset": str(spec.dataset),
            "split": str(spec.split),
            "report_dir": str(spec.report_dir),
            "checkpoint_dir": str(spec.checkpoint_dir),
        }
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": "dry_run" if spec.dry_run else "planned",
        "run_id": spec.run_id,
        "objective": spec.objective,
        "hypothesis": spec.hypothesis,
        "spec": spec_payload,
        "preflight": preflight,
        "planned_outputs": [
            "plan.json",
            "pilot_loop/",
            "report.json",
            "report.md",
            "next_actions.json",
            "run_manifest.json",
            "state.json",
        ],
    }


def build_research_report(
    *,
    spec: ResearchRunSpec,
    preflight: Mapping[str, Any],
    pilot_outputs: Mapping[str, dict] | None,
    mode: str,
) -> dict[str, Any]:
    artifacts = collect_artifacts(spec, pilot_outputs)
    metrics = summarize_metrics(pilot_outputs)
    claim_gate = evaluate_claim_gate(spec, pilot_outputs, metrics)
    next_actions = choose_next_actions(claim_gate, metrics)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": mode,
        "run_id": spec.run_id,
        "objective": spec.objective,
        "hypothesis": spec.hypothesis,
        "program_id": spec.program_id,
        "variant_id": spec.variant_id,
        "parent_run_id": spec.parent_run_id,
        "tags": list(spec.tags),
        "inputs": {
            "config": str(spec.config),
            "dataset": str(spec.dataset),
            "split": str(spec.split),
        },
        "environment": environment_snapshot(Path(spec.report_dir).parent),
        "preflight": preflight,
        "run_options": {
            "epochs": spec.epochs,
            "train_limit_samples": spec.train_limit_samples,
            "batch_size": spec.batch_size,
            "learning_rate": spec.learning_rate,
            "base_channels": spec.base_channels,
            "model_arch": spec.model_arch,
            "loader_mode": spec.loader_mode,
            "shard_cache_size": spec.shard_cache_size,
            "num_workers": spec.num_workers,
            "pin_memory": spec.pin_memory,
            "device": spec.device,
            "seed": spec.seed,
            "candidate_threshold": spec.candidate_threshold,
            "nms_radius": spec.nms_radius,
            "max_detections_per_cutout": spec.max_detections_per_cutout,
            "predict_limit": spec.predict_limit,
            "radius_arcsec": spec.radius_arcsec,
            "seeing_aware": spec.seeing_aware,
            "psf_fraction": spec.psf_fraction,
            "include_suspect_truth": spec.include_suspect_truth,
        },
        "metrics": metrics,
        "claim_gate": claim_gate,
        "next_actions": next_actions,
        "artifacts": artifacts,
    }


def collect_artifacts(spec: ResearchRunSpec, pilot_outputs: Mapping[str, dict] | None) -> dict[str, Any]:
    paths: dict[str, str | Path] = {
        "config": spec.config,
        "split": spec.split,
        "dataset_manifest": Path(spec.dataset) / "manifest.csv",
        "dataset_metadata": Path(spec.dataset) / "metadata.json",
        "dataset_truth_catalog": Path(spec.dataset) / "truth_catalog.csv",
    }
    if pilot_outputs:
        for name, value in pilot_outputs["summary"].get("outputs", {}).items():
            paths[f"pilot_{name}"] = value
    return {name: file_fingerprint(path) for name, path in sorted(paths.items())}


def summarize_metrics(pilot_outputs: Mapping[str, dict] | None) -> dict[str, Any]:
    if not pilot_outputs:
        return {"status": "not_run"}
    summary = pilot_outputs["summary"]
    val_sweep = pilot_outputs["val_threshold_sweep"]
    test_metrics = pilot_outputs["test_metrics"]
    detection = test_metrics.get("metrics", {}).get("detection", {})
    average_precision = test_metrics.get("metrics", {}).get("average_precision", {})
    return {
        "status": "available",
        "splits": summary.get("splits", {}),
        "training": summary.get("training", {}),
        "threshold_selection": summary.get("threshold_selection", {}),
        "validation": {
            "best_threshold": val_sweep.get("best_threshold"),
            "best_metrics": val_sweep.get("best_metrics", {}),
            "average_precision": val_sweep.get("average_precision", {}),
        },
        "test": {
            "counts": test_metrics.get("counts", {}),
            "detection": detection,
            "average_precision": average_precision,
            "astrometry": test_metrics.get("metrics", {}).get("astrometry", {}),
            "classification": test_metrics.get("metrics", {}).get("classification", {}),
            "deblending": test_metrics.get("metrics", {}).get("deblending", {}),
        },
    }


def evaluate_claim_gate(
    spec: ResearchRunSpec,
    pilot_outputs: Mapping[str, dict] | None,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not pilot_outputs:
        return {
            "status": "blocked",
            "paper_ready": False,
            "paper_claim_allowed": False,
            "reasons": ["no executed pilot-loop metrics are available"],
        }

    test_counts = metrics.get("test", {}).get("counts", {})
    truth = int(test_counts.get("truth", 0))
    candidates = int(test_counts.get("candidate_predictions", 0))
    detection = metrics.get("test", {}).get("detection", {})
    recall = float(detection.get("recall", 0.0))
    precision = float(detection.get("precision", 0.0))
    ap = float(metrics.get("test", {}).get("average_precision", {}).get("ap", 0.0))
    if truth <= 0:
        reasons.append("test truth count is zero")
    if candidates <= 0:
        reasons.append("test candidate prediction count is zero")
    if reasons:
        return {
            "status": "blocked",
            "paper_ready": False,
            "paper_claim_allowed": False,
            "reasons": reasons,
        }

    engineering_reasons = []
    text_blob = " ".join([str(spec.dataset), str(spec.split), str(spec.report_dir)]).lower()
    if "smoke" in text_blob:
        engineering_reasons.append("smoke dataset or split detected")
    if spec.predict_limit is not None:
        engineering_reasons.append("predict_limit was set")
    if spec.train_limit_samples is not None:
        engineering_reasons.append("train_limit_samples was set")
    if spec.epochs < 5:
        engineering_reasons.append("epochs below paper-scale default")
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
            "reasons": ["nonzero recall and AP on the configured test split"],
            "observed_precision": precision,
            "observed_recall": recall,
            "observed_ap": ap,
        }
    return {
        "status": "blocked",
        "paper_ready": False,
        "paper_claim_allowed": False,
        "reasons": ["test recall or AP is zero on a non-smoke configuration"],
        "observed_precision": precision,
        "observed_recall": recall,
        "observed_ap": ap,
    }


def choose_next_actions(claim_gate: Mapping[str, Any], metrics: Mapping[str, Any]) -> list[dict[str, str]]:
    status = claim_gate.get("status")
    actions: list[dict[str, str]] = []
    if status == "engineering_check":
        actions.append(
            {
                "priority": "high",
                "action": "Run the same loop on the pilot dataset without predict_limit and with paper-scale epochs.",
            }
        )
    elif status == "candidate_evidence":
        actions.append(
            {
                "priority": "high",
                "action": "Schedule the PSF-loss, photometry-weighting, valid-mask, and native-frame ablations.",
            }
        )
    else:
        actions.append(
            {
                "priority": "high",
                "action": "Inspect data alignment, WCS decoding, and threshold selection before scaling the run.",
            }
        )

    detection = metrics.get("test", {}).get("detection", {}) if isinstance(metrics, Mapping) else {}
    recall = float(detection.get("recall", 0.0) or 0.0)
    precision = float(detection.get("precision", 0.0) or 0.0)
    if recall == 0.0:
        actions.append(
            {
                "priority": "medium",
                "action": "Generate a validation overlay panel for matched and unmatched predictions.",
            }
        )
    if precision < 0.2:
        actions.append(
            {
                "priority": "medium",
                "action": "Sweep candidate threshold and NMS radius on validation before another test read.",
            }
        )
    return actions


def write_report_bundle(run_root: Path, report: Mapping[str, Any]) -> None:
    write_json(run_root / "report.json", dict(report))
    write_json(run_root / "next_actions.json", {"run_id": report["run_id"], "next_actions": report["next_actions"]})
    write_json(run_root / "run_manifest.json", build_run_manifest(run_root, report))
    write_json(run_root / "state.json", build_run_state(report))
    (run_root / "report.md").write_text(render_markdown_report(report), encoding="utf-8")


def build_run_manifest(run_root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "run_id": report.get("run_id", ""),
        "program_id": report.get("program_id"),
        "variant_id": report.get("variant_id"),
        "run_dir": str(run_root),
        "status": report.get("status", ""),
        "claim_gate": report.get("claim_gate", {}).get("status", ""),
        "artifacts": report.get("artifacts", {}),
    }


def build_run_state(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": report.get("run_id", ""),
        "status": report.get("status", ""),
        "claim_gate": report.get("claim_gate", {}),
        "metrics_status": report.get("metrics", {}).get("status", "unknown"),
        "next_actions": report.get("next_actions", []),
    }


def render_markdown_report(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics", {})
    test = metrics.get("test", {}) if isinstance(metrics, Mapping) else {}
    detection = test.get("detection", {}) if isinstance(test, Mapping) else {}
    claim_gate = report.get("claim_gate", {})
    lines = [
        f"# Research Run {report.get('run_id', '')}",
        "",
        f"- Status: {report.get('status', '')}",
        f"- Objective: {report.get('objective', '')}",
        f"- Hypothesis: {report.get('hypothesis', '')}",
        f"- Claim gate: {claim_gate.get('status', '')}",
        f"- Paper ready: {claim_gate.get('paper_ready', False)}",
        "",
        "## Test Detection",
        "",
        f"- Precision: {detection.get('precision', 'n/a')}",
        f"- Recall: {detection.get('recall', 'n/a')}",
        f"- F1: {detection.get('f1', 'n/a')}",
        f"- TP/FP/FN: {detection.get('tp', 'n/a')}/{detection.get('fp', 'n/a')}/{detection.get('fn', 'n/a')}",
        "",
        "## Next Actions",
        "",
    ]
    for item in report.get("next_actions", []):
        lines.append(f"- [{item.get('priority', '')}] {item.get('action', '')}")
    lines.append("")
    return "\n".join(lines)


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload: dict[str, Any] = {"path": str(target), "exists": target.exists()}
    if not target.exists():
        return payload
    if target.is_dir():
        payload.update({"kind": "directory"})
        return payload
    size = target.stat().st_size
    payload.update({"kind": "file", "bytes": size})
    if size > MAX_HASH_BYTES:
        payload["sha256"] = None
        payload["hash_skipped_reason"] = f"file exceeds {MAX_HASH_BYTES} bytes"
        return payload
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    payload["sha256"] = digest.hexdigest()
    return payload


def environment_snapshot(cwd: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "git": git_metadata(cwd),
    }
    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:  # noqa: BLE001
        payload["torch"] = {"available": False, "error": str(exc)}
    return payload


def git_metadata(cwd: str | Path) -> dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return {"available": True, "root": root, "commit": commit, "dirty": bool(dirty)}
    except Exception:
        return {"available": False}


def is_inside_raw_data(path: str | Path) -> bool:
    try:
        resolved = Path(path).resolve(strict=False)
        raw = RAW_DATA_ROOT.resolve(strict=False)
        return resolved == raw or raw in resolved.parents
    except OSError:
        return False
