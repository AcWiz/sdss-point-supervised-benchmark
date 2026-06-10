from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .cutouts import write_cutout_manifest
from .dataset_builder import build_dataset
from .experiment import build_dry_run_report
from .io import load_prediction_catalog, load_source_catalog, write_prediction_catalog, write_source_catalog
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
from .schema import BANDS, PredictionRecord, SourceRecord
from .sdss_dr17 import build_sdss_field_manifest, load_sdss_source_catalog, write_field_manifest
from .split import make_region_split


@dataclass(frozen=True)
class SplitTruthSelection:
    records: list[SourceRecord]
    cutout_ids: set[str]
    source_ids: set[str]
    all_truth_count: int
    dropped_truth_count: int
    all_quality_counts: dict[str, int]
    kept_quality_counts: dict[str, int]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sdss-point-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="write a leakage-safe sky-region split")
    split_parser.add_argument("--catalog", required=True, help="CSV source catalog")
    split_parser.add_argument("--output", required=True, help="JSON split output")
    split_parser.add_argument("--train-fraction", type=float, default=0.7)
    split_parser.add_argument("--val-fraction", type=float, default=0.15)
    split_parser.add_argument("--test-fraction", type=float, default=0.15)
    split_parser.add_argument("--ra-bin-deg", type=float, default=1.0)
    split_parser.add_argument("--dec-bin-deg", type=float, default=1.0)
    split_parser.add_argument(
        "--region-mode",
        choices=["sky-bin", "catalog-region"],
        default="sky-bin",
        help="split by computed sky bins or by catalog region_id values",
    )
    split_parser.add_argument("--seed", type=int, default=0)

    manifest_parser = subparsers.add_parser("build-manifest", help="write a field-level SDSS DR17 manifest")
    manifest_parser.add_argument("--data-root", required=True, help="SDSS data root with manifest_frames/catalogs CSVs")
    manifest_parser.add_argument("--output", required=True, help="field manifest CSV output")

    convert_parser = subparsers.add_parser(
        "convert-sdss-catalog",
        help="convert one SDSS PhotoObj CSV to the benchmark source catalog contract",
    )
    convert_parser.add_argument("--input", required=True, help="SDSS PhotoObj CSV")
    convert_parser.add_argument("--output", required=True, help="benchmark source catalog CSV output")
    convert_parser.add_argument("--clean-only", action="store_true")

    source_catalog_parser = subparsers.add_parser(
        "build-source-catalog",
        help="aggregate SDSS PhotoObj CSVs into one benchmark source catalog",
    )
    source_catalog_parser.add_argument("--config", required=True, help="dataset/experiment config JSON")
    source_catalog_parser.add_argument("--output", required=True, help="benchmark source catalog CSV output")
    source_catalog_parser.add_argument("--limit-fields", type=int, default=100)
    source_catalog_parser.add_argument("--clean-only", action="store_true")

    cutout_parser = subparsers.add_parser("prepare-cutouts", help="write a cutout worklist from a source catalog")
    cutout_parser.add_argument("--catalog", required=True, help="benchmark source catalog CSV")
    cutout_parser.add_argument("--output", required=True, help="cutout manifest CSV output")
    cutout_parser.add_argument("--cutout-size", type=int, default=128)
    cutout_parser.add_argument("--limit", type=int)

    dataset_parser = subparsers.add_parser("build-dataset", help="build native-frame SDSS cutout NPZ shards")
    dataset_parser.add_argument("--config", required=True, help="dataset/experiment config JSON")
    dataset_parser.add_argument("--output-dir", required=True, help="dataset output directory")
    dataset_parser.add_argument("--limit-fields", type=int, default=100)
    dataset_parser.add_argument("--cutout-size", type=int, default=128)
    dataset_parser.add_argument("--shard-size", type=int, default=1024)
    dataset_parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    dataset_parser.add_argument("--dry-run", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate predictions against a source catalog")
    evaluate_parser.add_argument("--truth", required=True, help="CSV source truth catalog")
    evaluate_parser.add_argument("--predictions", required=True, help="CSV prediction catalog")
    evaluate_parser.add_argument("--output", required=True, help="JSON metric output")
    evaluate_parser.add_argument("--radius-arcsec", type=float, default=1.0)
    evaluate_parser.add_argument("--band", default="r")
    evaluate_parser.add_argument("--close-pair-arcsec", type=float, default=2.0)
    evaluate_parser.add_argument("--seeing-aware", action="store_true")
    evaluate_parser.add_argument("--psf-fraction", type=float, default=0.5)
    evaluate_parser.add_argument("--min-score", type=float)
    evaluate_parser.add_argument("--filter-truth-to-prediction-cutouts", action="store_true")

    sweep_parser = subparsers.add_parser("sweep-thresholds", help="select a validation score threshold")
    sweep_parser.add_argument("--truth", required=True, help="CSV source truth catalog")
    sweep_parser.add_argument("--predictions", required=True, help="CSV prediction catalog")
    sweep_parser.add_argument("--output", required=True, help="JSON threshold sweep output")
    sweep_parser.add_argument("--radius-arcsec", type=float, default=1.0)
    sweep_parser.add_argument("--seeing-aware", action="store_true")
    sweep_parser.add_argument("--psf-fraction", type=float, default=0.5)
    sweep_parser.add_argument("--filter-truth-to-prediction-cutouts", action="store_true")

    train_parser = subparsers.add_parser("train", help="train the PSF-constrained point-supervised baseline")
    train_parser.add_argument("--config", required=True, help="experiment config JSON")
    train_parser.add_argument("--output", required=True, help="training report JSON")
    train_parser.add_argument("--dataset", help="prepared NPZ dataset directory")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--limit-samples", type=int)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--base-channels", type=int, default=32)
    train_parser.add_argument("--model-arch", default="baseline", choices=["baseline", "unet_lite"])
    train_parser.add_argument("--loader-mode", default="sample", choices=["sample", "shard_grouped"])
    train_parser.add_argument("--shard-cache-size", type=int, default=0)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--pin-memory", default="auto", choices=["auto", "true", "false"])
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--split", help="split JSON used to filter dataset samples")
    train_parser.add_argument("--split-name", help="split name to read from --split")
    train_parser.add_argument("--dry-run", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="write predictions from a trained checkpoint")
    predict_parser.add_argument("--checkpoint", required=True, help="model checkpoint path")
    predict_parser.add_argument("--output", required=True, help="prediction CSV output")
    predict_parser.add_argument("--dataset", help="prepared NPZ dataset directory")
    predict_parser.add_argument("--batch-size", type=int, default=16)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument("--nms-radius", type=int, default=2)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--pixel-scale-arcsec", type=float, default=0.396)
    predict_parser.add_argument("--split", help="split JSON used to filter dataset samples")
    predict_parser.add_argument("--split-name", help="split name to read from --split")
    predict_parser.add_argument("--max-detections-per-cutout", type=int)
    predict_parser.add_argument("--limit-samples", type=int)
    predict_parser.add_argument("--shard-cache-size", type=int, default=0)
    predict_parser.add_argument("--num-workers", type=int, default=0)
    predict_parser.add_argument("--pin-memory", default="auto", choices=["auto", "true", "false"])
    predict_parser.add_argument("--dry-run", action="store_true")

    pilot_loop_parser = subparsers.add_parser(
        "run-pilot-loop",
        help="train on a split, select a validation threshold, and evaluate test predictions",
    )
    pilot_loop_parser.add_argument("--config", required=True, help="experiment config JSON")
    pilot_loop_parser.add_argument("--dataset", required=True, help="prepared NPZ dataset directory")
    pilot_loop_parser.add_argument("--split", required=True, help="fixed split JSON")
    pilot_loop_parser.add_argument("--output-dir", required=True, help="report output directory")
    pilot_loop_parser.add_argument("--checkpoint-dir", required=True, help="checkpoint output directory")
    pilot_loop_parser.add_argument("--train-split-name", default="train")
    pilot_loop_parser.add_argument("--val-split-name", default="val")
    pilot_loop_parser.add_argument("--test-split-name", default="test")
    pilot_loop_parser.add_argument("--epochs", type=int, default=1)
    pilot_loop_parser.add_argument("--train-limit-samples", type=int)
    pilot_loop_parser.add_argument("--batch-size", type=int, default=16)
    pilot_loop_parser.add_argument("--learning-rate", type=float, default=1e-3)
    pilot_loop_parser.add_argument("--base-channels", type=int, default=32)
    pilot_loop_parser.add_argument("--model-arch", default="baseline", choices=["baseline", "unet_lite"])
    pilot_loop_parser.add_argument("--loader-mode", default="sample", choices=["sample", "shard_grouped"])
    pilot_loop_parser.add_argument("--shard-cache-size", type=int, default=0)
    pilot_loop_parser.add_argument("--num-workers", type=int, default=0)
    pilot_loop_parser.add_argument("--pin-memory", default="auto", choices=["auto", "true", "false"])
    pilot_loop_parser.add_argument("--device", default="cpu")
    pilot_loop_parser.add_argument("--seed", type=int, default=42)
    pilot_loop_parser.add_argument("--candidate-threshold", type=float, default=0.2)
    pilot_loop_parser.add_argument("--nms-radius", type=int, default=2)
    pilot_loop_parser.add_argument("--max-detections-per-cutout", type=int)
    pilot_loop_parser.add_argument("--predict-limit", type=int)
    pilot_loop_parser.add_argument("--pixel-scale-arcsec", type=float, default=0.396)
    pilot_loop_parser.add_argument("--radius-arcsec", type=float, default=1.0)
    pilot_loop_parser.add_argument("--band", default="r")
    pilot_loop_parser.add_argument("--close-pair-arcsec", type=float, default=2.0)
    pilot_loop_parser.add_argument("--seeing-aware", action="store_true")
    pilot_loop_parser.add_argument("--psf-fraction", type=float, default=0.5)
    pilot_loop_parser.add_argument(
        "--include-suspect-truth",
        action="store_true",
        help="include suspect or zero-weight truth rows in validation and test metrics",
    )

    experiment_parser = subparsers.add_parser("run-experiment", help="write or run an experiment plan report")
    experiment_parser.add_argument("--config", required=True, help="experiment config JSON")
    experiment_parser.add_argument("--output", required=True, help="experiment report JSON")
    experiment_parser.add_argument("--dry-run", action="store_true")

    research_run_parser = subparsers.add_parser(
        "research-run",
        help="run an auditable automated research loop and write JSON/Markdown reports",
    )
    research_run_parser.add_argument("--config", required=True, help="experiment config JSON")
    research_run_parser.add_argument("--dataset", required=True, help="prepared NPZ dataset directory")
    research_run_parser.add_argument("--split", required=True, help="fixed split JSON")
    research_run_parser.add_argument("--run-id", required=True, help="stable run id used in reports")
    research_run_parser.add_argument("--objective", required=True, help="research objective for this run")
    research_run_parser.add_argument("--hypothesis", required=True, help="hypothesis being audited")
    research_run_parser.add_argument("--report-dir", required=True, help="standard research run report directory")
    research_run_parser.add_argument("--checkpoint-dir", required=True, help="checkpoint output directory")
    research_run_parser.add_argument("--epochs", type=int, default=1)
    research_run_parser.add_argument("--train-limit-samples", type=int)
    research_run_parser.add_argument("--batch-size", type=int, default=16)
    research_run_parser.add_argument("--learning-rate", type=float, default=1e-3)
    research_run_parser.add_argument("--base-channels", type=int, default=32)
    research_run_parser.add_argument("--model-arch", default="baseline", choices=["baseline", "unet_lite"])
    research_run_parser.add_argument("--loader-mode", default="sample", choices=["sample", "shard_grouped"])
    research_run_parser.add_argument("--shard-cache-size", type=int, default=0)
    research_run_parser.add_argument("--num-workers", type=int, default=0)
    research_run_parser.add_argument("--pin-memory", default="auto", choices=["auto", "true", "false"])
    research_run_parser.add_argument("--device", default="cpu")
    research_run_parser.add_argument("--seed", type=int, default=42)
    research_run_parser.add_argument("--candidate-threshold", type=float, default=0.2)
    research_run_parser.add_argument("--nms-radius", type=int, default=2)
    research_run_parser.add_argument("--max-detections-per-cutout", type=int)
    research_run_parser.add_argument("--predict-limit", type=int)
    research_run_parser.add_argument("--pixel-scale-arcsec", type=float, default=0.396)
    research_run_parser.add_argument("--radius-arcsec", type=float, default=1.0)
    research_run_parser.add_argument("--band", default="r")
    research_run_parser.add_argument("--close-pair-arcsec", type=float, default=2.0)
    research_run_parser.add_argument("--seeing-aware", action="store_true")
    research_run_parser.add_argument("--psf-fraction", type=float, default=0.5)
    research_run_parser.add_argument("--include-suspect-truth", action="store_true")
    research_run_parser.add_argument("--dry-run", action="store_true")
    research_run_parser.add_argument("--program-id", help=argparse.SUPPRESS)
    research_run_parser.add_argument("--variant-id", help=argparse.SUPPRESS)
    research_run_parser.add_argument("--parent-run-id", help=argparse.SUPPRESS)
    research_run_parser.add_argument("--tag", action="append", dest="tags", help=argparse.SUPPRESS)
    research_run_parser.add_argument("--claim", action="append", dest="claims", help=argparse.SUPPRESS)
    research_run_parser.add_argument("--claim-gate-policy-json", help=argparse.SUPPRESS)

    research_report_parser = subparsers.add_parser(
        "research-report",
        help="write a research report bundle from an existing run-pilot-loop output directory",
    )
    research_report_parser.add_argument("--pilot-output-dir", required=True, help="directory containing summary/metrics")
    research_report_parser.add_argument("--run-id", required=True, help="stable run id used in reports")
    research_report_parser.add_argument("--report-dir", required=True, help="standard research run report directory")
    research_report_parser.add_argument("--objective", required=True, help="research objective for this report")
    research_report_parser.add_argument("--hypothesis", required=True, help="hypothesis being audited")

    research_program_parser = subparsers.add_parser(
        "research-program",
        help="expand a research program config into an auditable run queue",
    )
    research_program_parser.add_argument("--program", required=True, help="research program JSON")
    research_program_parser.add_argument("--output-dir", required=True, help="directory for program plan output")
    research_program_parser.add_argument("--run-prefix", help="optional prefix for generated run ids")
    research_program_parser.add_argument("--execute", action="store_true", help="execute generated research-run specs")

    research_autopilot_parser = subparsers.add_parser(
        "research-autopilot",
        help="opportunistically execute a research program on available GPUs",
    )
    research_autopilot_parser.add_argument("--program", required=True, help="research program JSON")
    research_autopilot_parser.add_argument("--output-dir", required=True, help="scheduler output directory")
    research_autopilot_parser.add_argument("--run-prefix", help="optional prefix for generated run ids")
    research_autopilot_parser.add_argument("--execute", action="store_true", help="execute generated research-run specs")
    research_autopilot_parser.add_argument("--max-jobs", type=int, default=2)
    research_autopilot_parser.add_argument("--min-free-gb", type=float, default=10.0)
    research_autopilot_parser.add_argument("--max-util", type=float, default=20.0)
    research_autopilot_parser.add_argument("--poll-seconds", type=float, default=30.0)
    research_autopilot_parser.add_argument("--sample-seconds", type=float, default=10.0)
    research_autopilot_parser.add_argument("--max-runtime-minutes", type=float, default=1440.0)

    research_board_parser = subparsers.add_parser(
        "research-board",
        help="summarize research run reports under a root directory",
    )
    research_board_parser.add_argument("--root", default="reports/research_runs", help="research run root")
    research_board_parser.add_argument("--output", help="JSON output path")
    research_board_parser.add_argument("--markdown-output", help="Markdown output path")
    research_board_parser.add_argument("--rebuild-index", action="store_true", help="rewrite root/index.jsonl")

    research_compare_parser = subparsers.add_parser(
        "research-compare",
        help="compare research runs under a root directory",
    )
    research_compare_parser.add_argument("--root", default="reports/research_runs", help="research run root")
    research_compare_parser.add_argument("--run-id", action="append", dest="run_ids", help="run id to include")
    research_compare_parser.add_argument("--output", required=True, help="JSON compare output path")
    research_compare_parser.add_argument("--markdown-output", help="Markdown compare output path")

    research_diagnose_parser = subparsers.add_parser(
        "research-diagnose",
        help="diagnose a research run report",
    )
    research_diagnose_parser.add_argument("--report", required=True, help="report.json path")
    research_diagnose_parser.add_argument("--output", required=True, help="JSON diagnosis output path")
    research_diagnose_parser.add_argument("--markdown-output", help="Markdown diagnosis output path")

    research_next_parser = subparsers.add_parser(
        "research-next",
        help="write an evidence ledger and next recommended actions from all runs",
    )
    research_next_parser.add_argument("--root", default="reports/research_runs", help="research run root")
    research_next_parser.add_argument("--output", required=True, help="JSON evidence ledger output path")

    args = parser.parse_args(argv)
    if args.command == "split":
        return _split_command(args)
    if args.command == "build-manifest":
        return _build_manifest_command(args)
    if args.command == "convert-sdss-catalog":
        return _convert_sdss_catalog_command(args)
    if args.command == "build-source-catalog":
        return _build_source_catalog_command(args)
    if args.command == "prepare-cutouts":
        return _prepare_cutouts_command(args)
    if args.command == "build-dataset":
        return _build_dataset_command(args)
    if args.command == "evaluate":
        return _evaluate_command(args)
    if args.command == "sweep-thresholds":
        return _sweep_thresholds_command(args)
    if args.command == "train":
        return _train_command(args)
    if args.command == "predict":
        return _predict_command(args)
    if args.command == "run-pilot-loop":
        return _run_pilot_loop_command(args)
    if args.command == "run-experiment":
        return _run_experiment_command(args)
    if args.command == "research-run":
        return _research_run_command(args)
    if args.command == "research-report":
        return _research_report_command(args)
    if args.command == "research-program":
        return _research_program_command(args)
    if args.command == "research-autopilot":
        return _research_autopilot_command(args)
    if args.command == "research-board":
        return _research_board_command(args)
    if args.command == "research-compare":
        return _research_compare_command(args)
    if args.command == "research-diagnose":
        return _research_diagnose_command(args)
    if args.command == "research-next":
        return _research_next_command(args)
    raise AssertionError(f"unhandled command: {args.command}")


def _split_command(args: argparse.Namespace) -> int:
    records = load_source_catalog(args.catalog)
    split = make_region_split(
        records,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        ra_bin_deg=args.ra_bin_deg,
        dec_bin_deg=args.dec_bin_deg,
        seed=args.seed,
        region_mode=args.region_mode,
    )
    payload = {
        "protocol": "sdss-point-supervised-v1",
        "source_catalog": str(Path(args.catalog)),
        "region_mode": args.region_mode,
        "region_bin_degrees": {"ra": args.ra_bin_deg, "dec": args.dec_bin_deg},
        "splits": {
            "train": list(split.train_ids),
            "val": list(split.val_ids),
            "test": list(split.test_ids),
        },
        "regions": {
            "train": list(split.train_regions),
            "val": list(split.val_regions),
            "test": list(split.test_regions),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _build_manifest_command(args: argparse.Namespace) -> int:
    records = build_sdss_field_manifest(args.data_root)
    write_field_manifest(records, args.output)
    return 0


def _convert_sdss_catalog_command(args: argparse.Namespace) -> int:
    records = load_sdss_source_catalog(args.input, clean_only=args.clean_only)
    write_source_catalog(records, args.output)
    return 0


def _build_source_catalog_command(args: argparse.Namespace) -> int:
    config = _load_json(args.config)
    data_config = config.get("data", {})
    records = build_sdss_field_manifest(data_config["root"], bands=tuple(data_config.get("bands") or BANDS))
    ready_records = [record for record in records if record.status == "ready"]
    selected_records = ready_records[: args.limit_fields] if args.limit_fields is not None else ready_records
    sources = []
    for record in selected_records:
        sources.extend(load_sdss_source_catalog(record.catalog_path, clean_only=args.clean_only))
    write_source_catalog(sources, args.output)
    return 0


def _prepare_cutouts_command(args: argparse.Namespace) -> int:
    records = load_source_catalog(args.catalog)
    write_cutout_manifest(records, args.output, cutout_size=args.cutout_size, limit=args.limit)
    return 0


def _build_dataset_command(args: argparse.Namespace) -> int:
    build_dataset(
        config_path=args.config,
        output_dir=args.output_dir,
        limit_fields=args.limit_fields,
        cutout_size=args.cutout_size,
        shard_size=args.shard_size,
        dtype=args.dtype,
        dry_run=args.dry_run,
    )
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    truth = load_source_catalog(args.truth)
    predictions = load_prediction_catalog(args.predictions)
    if args.filter_truth_to_prediction_cutouts:
        truth = _filter_truth_to_prediction_cutouts(truth, predictions)
    payload = {
        "protocol": "sdss-point-supervised-v1",
        "truth_catalog": str(Path(args.truth)),
        "prediction_catalog": str(Path(args.predictions)),
        **_evaluate_payload(
            truth,
            predictions,
            radius_arcsec=args.radius_arcsec,
            band=args.band,
            close_pair_arcsec=args.close_pair_arcsec,
            seeing_aware=args.seeing_aware,
            psf_fraction=args.psf_fraction,
            min_score=args.min_score,
            filter_truth_to_prediction_cutouts=args.filter_truth_to_prediction_cutouts,
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _sweep_thresholds_command(args: argparse.Namespace) -> int:
    truth = load_source_catalog(args.truth)
    predictions = load_prediction_catalog(args.predictions)
    if args.filter_truth_to_prediction_cutouts:
        truth = _filter_truth_to_prediction_cutouts(truth, predictions)
    payload = {
        "protocol": "sdss-point-supervised-v1",
        "truth_catalog": str(Path(args.truth)),
        "prediction_catalog": str(Path(args.predictions)),
        **_sweep_thresholds_payload(
            truth,
            predictions,
            radius_arcsec=args.radius_arcsec,
            seeing_aware=args.seeing_aware,
            psf_fraction=args.psf_fraction,
            filter_truth_to_prediction_cutouts=args.filter_truth_to_prediction_cutouts,
        ),
    }
    _write_json(args.output, payload)
    return 0


def _train_command(args: argparse.Namespace) -> int:
    config = _load_json(args.config)
    if not args.dry_run:
        if not args.dataset:
            raise ValueError("--dataset is required for non-dry-run training")
        from .training import train_model

        train_model(
            config_path=args.config,
            dataset_dir=args.dataset,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            base_channels=args.base_channels,
            model_arch=args.model_arch,
            loader_mode=args.loader_mode,
            shard_cache_size=args.shard_cache_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            device=args.device,
            seed=args.seed,
            split_path=args.split,
            split_name=args.split_name,
            limit_samples=args.limit_samples,
        )
        return 0
    payload = {
        "protocol": "sdss-point-supervised-v1",
        "status": "dry_run",
        "config": config,
        "method": "psf-constrained-point-supervised-cataloger",
        "planned_outputs": ["checkpoint", "training_metrics", "validation_predictions"],
        "split": args.split,
        "split_name": args.split_name,
    }
    _write_json(args.output, payload)
    return 0


def _predict_command(args: argparse.Namespace) -> int:
    if not args.dry_run:
        if not args.dataset:
            raise ValueError("--dataset is required for non-dry-run prediction")
        from .training import predict_dataset

        predict_dataset(
            checkpoint_path=args.checkpoint,
            dataset_dir=args.dataset,
            output_path=args.output,
            batch_size=args.batch_size,
            threshold=args.threshold,
            nms_radius=args.nms_radius,
            device=args.device,
            pixel_scale_arcsec=args.pixel_scale_arcsec,
            split_path=args.split,
            split_name=args.split_name,
            max_detections_per_cutout=args.max_detections_per_cutout,
            limit_samples=args.limit_samples,
        )
        return 0
    write_prediction_catalog([], args.output)
    return 0


def _run_pilot_loop_command(args: argparse.Namespace) -> int:
    from .pilot_loop import run_pilot_loop

    run_pilot_loop(
        config=args.config,
        dataset=args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        train_split_name=args.train_split_name,
        val_split_name=args.val_split_name,
        test_split_name=args.test_split_name,
        epochs=args.epochs,
        train_limit_samples=args.train_limit_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_channels=args.base_channels,
        model_arch=args.model_arch,
        loader_mode=args.loader_mode,
        shard_cache_size=args.shard_cache_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        device=args.device,
        seed=args.seed,
        candidate_threshold=args.candidate_threshold,
        nms_radius=args.nms_radius,
        max_detections_per_cutout=args.max_detections_per_cutout,
        predict_limit=args.predict_limit,
        pixel_scale_arcsec=args.pixel_scale_arcsec,
        radius_arcsec=args.radius_arcsec,
        band=args.band,
        close_pair_arcsec=args.close_pair_arcsec,
        seeing_aware=args.seeing_aware,
        psf_fraction=args.psf_fraction,
        include_suspect_truth=args.include_suspect_truth,
    )
    return 0


def _run_experiment_command(args: argparse.Namespace) -> int:
    config = _load_json(args.config)
    if not args.dry_run:
        raise NotImplementedError("non-dry-run experiment execution is intentionally staged behind dry-run reports")
    _write_json(args.output, build_dry_run_report(config, command="run-experiment"))
    return 0


def _research_run_command(args: argparse.Namespace) -> int:
    from .automation import ResearchRunSpec, run_research_run
    from .research_program import append_registry_entry

    report = run_research_run(
        ResearchRunSpec(
            run_id=args.run_id,
            objective=args.objective,
            hypothesis=args.hypothesis,
            config=args.config,
            dataset=args.dataset,
            split=args.split,
            report_dir=args.report_dir,
            checkpoint_dir=args.checkpoint_dir,
            epochs=args.epochs,
            train_limit_samples=args.train_limit_samples,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            base_channels=args.base_channels,
            model_arch=args.model_arch,
            loader_mode=args.loader_mode,
            shard_cache_size=args.shard_cache_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            device=args.device,
            seed=args.seed,
            candidate_threshold=args.candidate_threshold,
            nms_radius=args.nms_radius,
            max_detections_per_cutout=args.max_detections_per_cutout,
            predict_limit=args.predict_limit,
            pixel_scale_arcsec=args.pixel_scale_arcsec,
            radius_arcsec=args.radius_arcsec,
            band=args.band,
            close_pair_arcsec=args.close_pair_arcsec,
            seeing_aware=args.seeing_aware,
            psf_fraction=args.psf_fraction,
            include_suspect_truth=args.include_suspect_truth,
            dry_run=args.dry_run,
            program_id=args.program_id,
            variant_id=args.variant_id,
            parent_run_id=args.parent_run_id,
            tags=tuple(args.tags or ()),
            claims=tuple(args.claims or ()),
            claim_gate_policy=json.loads(args.claim_gate_policy_json) if args.claim_gate_policy_json else None,
        )
    )
    append_registry_entry(Path(args.report_dir).parent / "index.jsonl", report)
    return 0


def _research_report_command(args: argparse.Namespace) -> int:
    from .automation import write_report_from_existing_pilot_loop
    from .research_program import append_registry_entry

    report = write_report_from_existing_pilot_loop(
        pilot_output_dir=args.pilot_output_dir,
        run_id=args.run_id,
        report_dir=args.report_dir,
        objective=args.objective,
        hypothesis=args.hypothesis,
    )
    append_registry_entry(Path(args.report_dir).parent / "index.jsonl", report)
    return 0


def _research_program_command(args: argparse.Namespace) -> int:
    from .automation import run_research_run
    from .research_program import (
        append_registry_entry,
        expand_research_program,
        load_research_program,
        write_program_plan,
    )

    if not args.execute:
        write_program_plan(args.program, output_dir=args.output_dir, run_prefix=args.run_prefix, dry_run=True)
        return 0
    program = load_research_program(args.program)
    specs = expand_research_program(program, run_prefix=args.run_prefix, dry_run=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executed = []
    for spec in specs:
        report = run_research_run(spec)
        append_registry_entry(Path(spec.report_dir).parent / "index.jsonl", report)
        executed.append(report["run_id"])
    _write_json(output_dir / "program_execution.json", {"protocol": "sdss-point-supervised-v1", "runs": executed})
    return 0


def _research_autopilot_command(args: argparse.Namespace) -> int:
    from .autopilot import AutopilotOptions, run_research_autopilot

    payload = run_research_autopilot(
        program_path=args.program,
        output_dir=args.output_dir,
        options=AutopilotOptions(
            run_prefix=args.run_prefix,
            execute=args.execute,
            max_jobs=args.max_jobs,
            min_free_gb=args.min_free_gb,
            max_util=args.max_util,
            poll_seconds=args.poll_seconds,
            sample_seconds=args.sample_seconds,
            max_runtime_minutes=args.max_runtime_minutes,
        ),
    )
    _write_json(Path(args.output_dir) / ("scheduler_execution.json" if args.execute else "scheduler_plan.json"), payload)
    return 0


def _research_board_command(args: argparse.Namespace) -> int:
    from .research_program import build_research_board, rebuild_registry, render_board_markdown

    if args.rebuild_index:
        rebuild_registry(args.root)
    board = build_research_board(args.root)
    if args.output:
        _write_json(args.output, board)
    if args.markdown_output:
        Path(args.markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_output).write_text(render_board_markdown(board), encoding="utf-8")
    if not args.output and not args.markdown_output:
        print(render_board_markdown(board))
    return 0


def _research_compare_command(args: argparse.Namespace) -> int:
    from .research_program import compare_research_runs, render_compare_markdown

    payload = compare_research_runs(args.root, run_ids=args.run_ids)
    _write_json(args.output, payload)
    if args.markdown_output:
        Path(args.markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_output).write_text(render_compare_markdown(payload), encoding="utf-8")
    return 0


def _research_diagnose_command(args: argparse.Namespace) -> int:
    from .research_program import build_diagnosis, render_diagnosis_markdown

    payload = build_diagnosis(args.report)
    _write_json(args.output, payload)
    if args.markdown_output:
        Path(args.markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_output).write_text(render_diagnosis_markdown(payload), encoding="utf-8")
    return 0


def _research_next_command(args: argparse.Namespace) -> int:
    from .research_program import build_evidence_ledger

    build_evidence_ledger(args.root, output_path=args.output)
    return 0


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _filter_predictions_by_score(predictions, min_score: float | None):
    if min_score is None:
        return predictions
    return [prediction for prediction in predictions if prediction.score >= min_score]


def _filter_truth_to_prediction_cutouts(truth, predictions):
    cutout_ids = {prediction.cutout_id for prediction in predictions}
    return [record for record in truth if record.cutout_id in cutout_ids]


def _evaluate_payload(
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
    scored_predictions = _filter_predictions_by_score(predictions, min_score)
    matches = _match_catalogs_for_options(
        truth,
        scored_predictions,
        radius_arcsec=radius_arcsec,
        seeing_aware=seeing_aware,
        psf_fraction=psf_fraction,
    )
    metrics = {
        "detection": detection_metrics(matches),
        "average_precision": _detection_average_precision_for_options(
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


def _sweep_thresholds_payload(
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
        filtered = _filter_predictions_by_score(predictions, threshold)
        matches = _match_catalogs_for_options(
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
        "average_precision": _average_precision_from_points(points, best_metrics["f1"], len(thresholds)),
        "thresholds": rows,
    }


def _average_precision_from_points(
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


def _detection_average_precision_for_options(
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
        filtered = _filter_predictions_by_score(predictions, threshold)
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


def _match_catalogs_for_options(
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


def _write_split_truth_catalog(
    *,
    dataset_dir: str | Path,
    split_path: str | Path,
    split_name: str,
    output_path: str | Path,
    include_suspect: bool,
    limit_samples: int | None = None,
) -> SplitTruthSelection:
    selection = _select_split_truth_records(
        dataset_dir=dataset_dir,
        split_path=split_path,
        split_name=split_name,
        include_suspect=include_suspect,
        limit_samples=limit_samples,
    )
    write_source_catalog(selection.records, output_path)
    return selection


def _select_split_truth_records(
    *,
    dataset_dir: str | Path,
    split_path: str | Path,
    split_name: str,
    include_suspect: bool = False,
    limit_samples: int | None = None,
) -> SplitTruthSelection:
    dataset = Path(dataset_dir)
    source_ids = _load_split_ids(split_path, split_name)
    cutout_ids = _cutout_ids_for_split(dataset / "manifest.csv", source_ids, limit_samples=limit_samples)
    all_records = [record for record in load_source_catalog(dataset / "truth_catalog.csv") if record.cutout_id in cutout_ids]
    if include_suspect:
        kept_records = list(all_records)
    else:
        kept_records = [record for record in all_records if _is_primary_truth(record)]
    return SplitTruthSelection(
        records=kept_records,
        cutout_ids=cutout_ids,
        source_ids=source_ids,
        all_truth_count=len(all_records),
        dropped_truth_count=len(all_records) - len(kept_records),
        all_quality_counts=_quality_counts(all_records),
        kept_quality_counts=_quality_counts(kept_records),
    )


def _split_sample_summary(dataset_dir: str | Path, split_path: str | Path, split_name: str) -> dict[str, int]:
    source_ids = _load_split_ids(split_path, split_name)
    cutout_ids = _cutout_ids_for_split(Path(dataset_dir) / "manifest.csv", source_ids)
    return {"source_ids": len(source_ids), "cutouts": len(cutout_ids)}


def _truth_selection_summary(selection: SplitTruthSelection, split_name: str) -> dict:
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


def _cutout_ids_for_split(
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


def _load_split_ids(path: str | Path, split_name: str) -> set[str]:
    payload = _load_json(path)
    splits = payload.get("splits", {})
    if split_name not in splits:
        raise ValueError(f"split {split_name!r} not found in {path}")
    return {str(source_id) for source_id in splits[split_name]}


def _is_primary_truth(record: SourceRecord) -> bool:
    if (record.label_quality or "").lower() == "suspect":
        return False
    return record.label_weight is None or record.label_weight > 0.0


def _quality_counts(records: Sequence[SourceRecord]) -> dict[str, int]:
    counts = Counter((record.label_quality or "unknown").lower() for record in records)
    return dict(sorted(counts.items()))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
