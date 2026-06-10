from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .automation import ResearchRunSpec, build_plan_payload, evaluate_claim_gate
from .experiment import PROTOCOL
from .pilot_loop import load_json, write_json

PROGRAM_SCHEMA_VERSION = 1
REGISTRY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResearchVariantSpec:
    variant_id: str
    objective: str
    hypothesis: str
    tags: tuple[str, ...]
    run: dict[str, Any]
    claims: tuple[str, ...]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchProgramSpec:
    program_id: str
    objective: str
    config: str | Path
    dataset: str | Path
    split: str | Path
    report_root: str | Path
    checkpoint_root: str | Path
    defaults: dict[str, Any]
    variants: tuple[ResearchVariantSpec, ...]
    claim_gates: dict[str, Any]


def load_research_program(path: str | Path) -> ResearchProgramSpec:
    payload = load_json(path)
    if payload.get("protocol") not in {None, PROTOCOL}:
        raise ValueError(f"program protocol must be {PROTOCOL!r}")
    if payload.get("schema_version") not in {None, PROGRAM_SCHEMA_VERSION}:
        raise ValueError(f"program schema_version must be {PROGRAM_SCHEMA_VERSION}")
    variants = []
    for row in payload.get("variants", []):
        variant_id = str(row.get("variant_id", "")).strip()
        if not variant_id:
            raise ValueError("each research variant must define variant_id")
        variants.append(
            ResearchVariantSpec(
                variant_id=variant_id,
                objective=str(row.get("objective") or payload.get("objective") or ""),
                hypothesis=str(row.get("hypothesis") or ""),
                tags=tuple(str(tag) for tag in row.get("tags", [])),
                run=dict(row.get("run", {})),
                claims=tuple(str(claim) for claim in row.get("claims", [])),
                depends_on=tuple(str(dep) for dep in row.get("depends_on", [])),
            )
        )
    if not variants:
        raise ValueError("research program must define at least one variant")
    return ResearchProgramSpec(
        program_id=str(payload.get("program_id") or ""),
        objective=str(payload.get("objective") or ""),
        config=str(payload.get("config") or ""),
        dataset=str(payload.get("dataset") or ""),
        split=str(payload.get("split") or ""),
        report_root=str(payload.get("report_root") or "reports/research_runs"),
        checkpoint_root=str(payload.get("checkpoint_root") or "artifacts/checkpoints/research_runs"),
        defaults=dict(payload.get("defaults", {})),
        variants=tuple(variants),
        claim_gates=dict(payload.get("claim_gates", {})),
    )


def expand_research_program(
    program: ResearchProgramSpec,
    *,
    run_prefix: str | None = None,
    dry_run: bool = True,
) -> list[ResearchRunSpec]:
    specs = []
    for variant in program.variants:
        merged = dict(program.defaults)
        merged.update(variant.run)
        run_id = str(merged.pop("run_id", "") or default_run_id(program.program_id, variant.variant_id, run_prefix))
        report_dir = merged.pop("report_dir", Path(program.report_root) / run_id)
        checkpoint_dir = merged.pop("checkpoint_dir", Path(program.checkpoint_root) / run_id)
        specs.append(
            ResearchRunSpec(
                run_id=run_id,
                objective=variant.objective or program.objective,
                hypothesis=variant.hypothesis,
                config=merged.pop("config", program.config),
                dataset=merged.pop("dataset", program.dataset),
                split=merged.pop("split", program.split),
                report_dir=report_dir,
                checkpoint_dir=checkpoint_dir,
                epochs=int(merged.pop("epochs", 1)),
                train_limit_samples=optional_int(merged.pop("train_limit_samples", None)),
                batch_size=int(merged.pop("batch_size", 16)),
                learning_rate=float(merged.pop("learning_rate", 1e-3)),
                base_channels=int(merged.pop("base_channels", 32)),
                model_arch=str(merged.pop("model_arch", "baseline")),
                loader_mode=str(merged.pop("loader_mode", "sample")),
                shard_cache_size=int(merged.pop("shard_cache_size", 0)),
                num_workers=int(merged.pop("num_workers", 0)),
                pin_memory=merged.pop("pin_memory", "auto"),
                device=str(merged.pop("device", "cpu")),
                seed=int(merged.pop("seed", 42)),
                candidate_threshold=float(merged.pop("candidate_threshold", 0.2)),
                nms_radius=int(merged.pop("nms_radius", 2)),
                max_detections_per_cutout=optional_int(merged.pop("max_detections_per_cutout", None)),
                predict_limit=optional_int(merged.pop("predict_limit", None)),
                pixel_scale_arcsec=float(merged.pop("pixel_scale_arcsec", 0.396)),
                radius_arcsec=float(merged.pop("radius_arcsec", 1.0)),
                band=str(merged.pop("band", "r")),
                close_pair_arcsec=float(merged.pop("close_pair_arcsec", 2.0)),
                seeing_aware=bool(merged.pop("seeing_aware", False)),
                psf_fraction=float(merged.pop("psf_fraction", 0.5)),
                include_suspect_truth=bool(merged.pop("include_suspect_truth", False)),
                dry_run=dry_run,
                program_id=program.program_id,
                variant_id=variant.variant_id,
                parent_run_id=optional_str(merged.pop("parent_run_id", None)),
                tags=variant.tags,
            )
        )
    return specs


def write_program_plan(
    program_path: str | Path,
    *,
    output_dir: str | Path,
    run_prefix: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    program = load_research_program(program_path)
    specs = expand_research_program(program, run_prefix=run_prefix, dry_run=dry_run)
    plan_rows = []
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for spec, variant in zip(specs, program.variants, strict=True):
        plan_rows.append(
            {
                "run_id": spec.run_id,
                "variant_id": spec.variant_id,
                "report_dir": str(spec.report_dir),
                "checkpoint_dir": str(spec.checkpoint_dir),
                "objective": spec.objective,
                "hypothesis": spec.hypothesis,
                "tags": list(spec.tags),
                "depends_on": list(variant.depends_on),
                "spec": stringify_paths(build_plan_payload(spec, {"status": "not_run"}).get("spec", {})),
            }
        )
    payload = {
        "schema_version": PROGRAM_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "program_id": program.program_id,
        "objective": program.objective,
        "program_config": str(program_path),
        "dry_run": dry_run,
        "runs": plan_rows,
    }
    write_json(output / "program_plan.json", payload)
    return payload


def load_run_reports(root: str | Path) -> list[dict[str, Any]]:
    base = Path(root)
    if not base.exists():
        return []
    reports = []
    for path in sorted(base.glob("**/report.json")):
        try:
            report = load_json(path)
        except Exception:
            continue
        report["_report_path"] = str(path)
        report["_run_dir"] = str(path.parent)
        reports.append(report)
    return reports


def append_registry_entry(registry_path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    entry = build_registry_entry(report)
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def rebuild_registry(root: str | Path, *, registry_path: str | Path | None = None) -> dict[str, Any]:
    base = Path(root)
    target = Path(registry_path) if registry_path else base / "index.jsonl"
    reports = load_run_reports(base)
    entries = [build_registry_entry(report) for report in reports]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries), encoding="utf-8")
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "registry": str(target),
        "runs": len(entries),
        "entries": entries,
    }


def build_registry_entry(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    test = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    detection = test.get("detection", {}) if isinstance(test, dict) else {}
    ap = test.get("average_precision", {}) if isinstance(test, dict) else {}
    claim_gate = report.get("claim_gate", {})
    run_options = report.get("run_options", {})
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "run_id": report.get("run_id", ""),
        "program_id": report.get("program_id"),
        "variant_id": report.get("variant_id"),
        "status": report.get("status", ""),
        "claim_gate": claim_gate.get("status", ""),
        "paper_claim_allowed": bool(claim_gate.get("paper_claim_allowed", False)),
        "objective": report.get("objective", ""),
        "hypothesis": report.get("hypothesis", ""),
        "tags": list(report.get("tags", [])),
        "report_path": report.get("_report_path", ""),
        "run_dir": report.get("_run_dir", ""),
        "metrics": {
            "precision": float_or_none(detection.get("precision")),
            "recall": float_or_none(detection.get("recall")),
            "f1": float_or_none(detection.get("f1")),
            "ap": float_or_none(ap.get("ap")),
            "truth": int_or_none(test.get("counts", {}).get("truth") if isinstance(test.get("counts"), dict) else None),
            "candidate_predictions": int_or_none(
                test.get("counts", {}).get("candidate_predictions") if isinstance(test.get("counts"), dict) else None
            ),
        },
        "run_options": {
            "epochs": run_options.get("epochs"),
            "train_limit_samples": run_options.get("train_limit_samples"),
            "batch_size": run_options.get("batch_size"),
            "base_channels": run_options.get("base_channels"),
            "model_arch": run_options.get("model_arch"),
            "loader_mode": run_options.get("loader_mode"),
            "predict_limit": run_options.get("predict_limit"),
            "device": run_options.get("device"),
        },
    }


def build_research_board(root: str | Path) -> dict[str, Any]:
    reports = load_run_reports(root)
    entries = [build_registry_entry(report) for report in reports]
    by_gate: dict[str, int] = {}
    for entry in entries:
        gate = str(entry.get("claim_gate") or "unknown")
        by_gate[gate] = by_gate.get(gate, 0) + 1
    best_by_f1 = sorted(
        entries,
        key=lambda row: row.get("metrics", {}).get("f1") if row.get("metrics", {}).get("f1") is not None else -1.0,
        reverse=True,
    )[:5]
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "root": str(root),
        "runs": len(entries),
        "claim_gate_counts": by_gate,
        "best_by_f1": best_by_f1,
        "entries": entries,
    }


def compare_research_runs(root: str | Path, *, run_ids: list[str] | None = None) -> dict[str, Any]:
    entries = [build_registry_entry(report) for report in load_run_reports(root)]
    if run_ids:
        wanted = set(run_ids)
        entries = [entry for entry in entries if entry["run_id"] in wanted]
    rows = sorted(entries, key=lambda row: (row.get("program_id") or "", row.get("variant_id") or "", row.get("run_id") or ""))
    baseline = rows[0]["metrics"] if rows else {}
    for row in rows:
        metrics = row["metrics"]
        row["delta_vs_first"] = {
            "f1": delta(metrics.get("f1"), baseline.get("f1")),
            "ap": delta(metrics.get("ap"), baseline.get("ap")),
            "recall": delta(metrics.get("recall"), baseline.get("recall")),
        }
    return {"schema_version": REGISTRY_SCHEMA_VERSION, "root": str(root), "runs": len(rows), "rows": rows}


def build_diagnosis(report_path: str | Path) -> dict[str, Any]:
    report = load_json(report_path)
    metrics = report.get("metrics", {})
    test = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    detection = test.get("detection", {}) if isinstance(test, dict) else {}
    counts = test.get("counts", {}) if isinstance(test, dict) else {}
    validation = metrics.get("validation", {}) if isinstance(metrics, dict) else {}
    reasons = []
    checks = []
    recall = float(detection.get("recall", 0.0) or 0.0)
    precision = float(detection.get("precision", 0.0) or 0.0)
    candidates = int(counts.get("candidate_predictions", 0) or 0)
    truth = int(counts.get("truth", 0) or 0)
    if truth <= 0:
        reasons.append("test truth catalog is empty")
        checks.append("Verify split selection and truth filtering policy.")
    if candidates <= 0:
        reasons.append("prediction decoder emitted no candidates")
        checks.append("Lower candidate threshold and inspect center heatmap ranges.")
    if recall == 0.0 and candidates > 0 and truth > 0:
        reasons.append("zero recall despite nonzero candidates and truth")
        checks.append("Generate matched/unmatched overlays and inspect WCS pixel-to-sky decoding.")
        checks.append("Compare validation best threshold against raw candidate score distribution.")
    if precision < 0.2 and candidates > 0:
        reasons.append("low precision")
        checks.append("Sweep candidate threshold and NMS radius on validation before another test read.")
    if validation.get("best_threshold") in {None, 0, 0.0}:
        reasons.append("validation threshold selection is weak or unavailable")
        checks.append("Inspect validation threshold sweep for degenerate score ordering.")
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "run_id": report.get("run_id", ""),
        "report": str(report_path),
        "status": "needs_attention" if reasons else "no_obvious_issue",
        "reasons": reasons,
        "recommended_checks": checks,
        "claim_gate": report.get("claim_gate", {}),
        "metrics": {"counts": counts, "detection": detection, "validation": validation},
    }


def build_evidence_ledger(root: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    reports = load_run_reports(root)
    claims: dict[str, dict[str, Any]] = {}
    for report in reports:
        tags = list(report.get("tags") or [])
        if not tags:
            tags = ["uncategorized"]
        gate = str(report.get("claim_gate", {}).get("status") or "unknown")
        entry = build_registry_entry(report)
        for tag in tags:
            bucket = claims.setdefault(tag, {"supporting_runs": [], "engineering_runs": [], "blocked_runs": []})
            if gate == "candidate_evidence":
                bucket["supporting_runs"].append(entry)
            elif gate == "engineering_check":
                bucket["engineering_runs"].append(entry)
            else:
                bucket["blocked_runs"].append(entry)
    ledger = {"schema_version": REGISTRY_SCHEMA_VERSION, "root": str(root), "claims": claims}
    if output_path:
        write_json(output_path, ledger)
    return ledger


def render_board_markdown(board: dict[str, Any]) -> str:
    lines = ["# Research Board", "", f"- Root: {board.get('root', '')}", f"- Runs: {board.get('runs', 0)}", ""]
    lines.append("## Claim Gates")
    lines.append("")
    for gate, count in sorted(board.get("claim_gate_counts", {}).items()):
        lines.append(f"- {gate}: {count}")
    lines.extend(["", "## Best Runs", ""])
    for entry in board.get("best_by_f1", []):
        metrics = entry.get("metrics", {})
        lines.append(
            f"- {entry.get('run_id', '')}: F1={metrics.get('f1')} AP={metrics.get('ap')} "
            f"gate={entry.get('claim_gate', '')}"
        )
    lines.append("")
    return "\n".join(lines)


def render_compare_markdown(compare: dict[str, Any]) -> str:
    lines = ["# Research Run Compare", "", "| run_id | variant | gate | precision | recall | f1 | ap |", "|---|---|---|---:|---:|---:|---:|"]
    for row in compare.get("rows", []):
        metrics = row.get("metrics", {})
        lines.append(
            "| {run_id} | {variant} | {gate} | {precision} | {recall} | {f1} | {ap} |".format(
                run_id=row.get("run_id", ""),
                variant=row.get("variant_id") or "",
                gate=row.get("claim_gate", ""),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1=metrics.get("f1"),
                ap=metrics.get("ap"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_diagnosis_markdown(diagnosis: dict[str, Any]) -> str:
    lines = ["# Research Diagnosis", "", f"- Run: {diagnosis.get('run_id', '')}", f"- Status: {diagnosis.get('status', '')}", ""]
    lines.append("## Reasons")
    lines.append("")
    for reason in diagnosis.get("reasons", []):
        lines.append(f"- {reason}")
    lines.extend(["", "## Recommended Checks", ""])
    for check in diagnosis.get("recommended_checks", []):
        lines.append(f"- {check}")
    lines.append("")
    return "\n".join(lines)


def default_run_id(program_id: str, variant_id: str, run_prefix: str | None) -> str:
    prefix = run_prefix or program_id
    return f"{prefix}_{variant_id}" if prefix else variant_id


def stringify_paths(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, default=str))


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def gate_report_with_program_policy(
    report: dict[str, Any],
    *,
    policy: dict[str, Any],
) -> dict[str, Any]:
    spec = ResearchRunSpec(
        run_id=str(report.get("run_id", "")),
        objective=str(report.get("objective", "")),
        hypothesis=str(report.get("hypothesis", "")),
        config=str(report.get("inputs", {}).get("config", "")),
        dataset=str(report.get("inputs", {}).get("dataset", "")),
        split=str(report.get("inputs", {}).get("split", "")),
        report_dir=str(report.get("_run_dir", "reports")),
        checkpoint_dir="artifacts/checkpoints",
        epochs=int(report.get("run_options", {}).get("epochs", 0) or 0),
        train_limit_samples=report.get("run_options", {}).get("train_limit_samples"),
        batch_size=int(report.get("run_options", {}).get("batch_size", 1) or 1),
        model_arch=str(report.get("run_options", {}).get("model_arch", "baseline")),
        loader_mode=str(report.get("run_options", {}).get("loader_mode", "sample")),
        shard_cache_size=int(report.get("run_options", {}).get("shard_cache_size", 0) or 0),
        num_workers=int(report.get("run_options", {}).get("num_workers", 0) or 0),
        pin_memory=report.get("run_options", {}).get("pin_memory", "auto"),
        predict_limit=report.get("run_options", {}).get("predict_limit"),
        tags=tuple(str(tag) for tag in report.get("tags", [])),
    )
    gate = evaluate_claim_gate(spec, {"summary": {}, "val_threshold_sweep": {}, "test_metrics": {}}, report.get("metrics", {}))
    missing = []
    min_epochs = int(policy.get("min_epochs", 5))
    if spec.epochs < min_epochs:
        missing.append(f"epochs below policy minimum {min_epochs}")
    required_tags = set(str(tag) for tag in policy.get("required_tags", []))
    if required_tags and not required_tags.issubset(set(spec.tags)):
        missing.append(f"missing required tags: {sorted(required_tags - set(spec.tags))}")
    if missing:
        gate = dict(gate)
        gate["status"] = "engineering_check"
        gate["paper_ready"] = False
        gate["paper_claim_allowed"] = False
        gate["reasons"] = list(gate.get("reasons", [])) + missing
    return gate
