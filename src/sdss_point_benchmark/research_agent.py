from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .experiment import PROTOCOL
from .pilot_loop import write_json
from .research_program import (
    build_registry_entry,
    build_research_board,
    load_research_program,
    load_run_reports,
)

AGENT_PLAN_SCHEMA_VERSION = 1
DEFAULT_AGENT_PROGRAM_SUFFIX = "agent_loss_diagnosis_v1"
DEFAULT_BATCH_SIZE = 192
COMPLETED_VARIANT_GATES = {"candidate_evidence", "engineering_check"}
DEPENDENCY_GATE_ORDER = {
    "blocked": 0,
    "complete": 1,
    "engineering_check": 2,
    "candidate_evidence": 3,
}
FULL_PSF_E20_VARIANT_ID = "ablation_full_psf_unet_lite_e20_matched_bs192"
CENTER_ONLY_E50_VARIANT_ID = "ablation_center_only_unet_lite_e50_seed42"
BASELINE_E50_VARIANT_ID = "pilot100_baseline_e50"
FAINTEST_SYNTHETIC_MAG_BIN = "[21.5,22.5)"


def build_research_agent_plan(
    program_path: str | Path,
    *,
    root: str | Path = "reports/research_runs",
    autonomy: str = "small_runs_then_gated_long_runs",
    budget: str = "conservative",
    approve_pending: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    program = load_research_program(program_path)
    reports = load_run_reports(root)
    entries = [build_registry_entry(report) for report in reports]
    board = build_research_board(root)
    policy = agent_policy(autonomy=autonomy, budget=budget)
    agent_program_id = f"{program.program_id}_{DEFAULT_AGENT_PROGRAM_SUFFIX}"

    current_state = summarize_current_state(board, entries, reports)
    executable_variants = build_candidate_variants(current_state=current_state)
    pending_approval_variants = build_pending_approval_variants(current_state=current_state)
    recommended_mainline_tasks = build_recommended_mainline_tasks(
        current_state=current_state,
        root=Path(root),
    )
    if approve_pending:
        promoted, pending_approval_variants = promote_pending_approval_variants(
            pending_approval_variants,
            approve_pending=approve_pending,
            current_state=current_state,
        )
        executable_variants.extend(promoted)
    if autonomy == "fully_automatic":
        promoted, pending_approval_variants = promote_pending_approval_variants(
            pending_approval_variants,
            approve_pending=[str(variant.get("variant_id", "")) for variant in pending_approval_variants],
            current_state=current_state,
        )
        executable_variants.extend(promoted)
        pending_approval_variants = []

    defaults = dict(program.defaults)
    defaults.update(
        {
            "batch_size": DEFAULT_BATCH_SIZE,
            "model_arch": "unet_lite",
            "loader_mode": defaults.get("loader_mode", "shard_grouped"),
            "shard_cache_size": defaults.get("shard_cache_size", 4),
            "num_workers": defaults.get("num_workers", 4),
            "pin_memory": defaults.get("pin_memory", "auto"),
        }
    )

    agent_plan = {
        "schema_version": AGENT_PLAN_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source_program": str(program_path),
        "source_report_root": str(root),
        "autonomy": autonomy,
        "budget": budget,
        "policy": policy,
        "current_state": current_state,
        "interpretation": build_interpretation(current_state),
        "recommended_mainline_tasks": recommended_mainline_tasks,
        "blocked_claims": build_blocked_claims(current_state),
        "experiment_queue": executable_variants,
        "pending_approval": pending_approval_variants,
        "commands": build_agent_commands(
            agent_plan_path=Path(root) / "agent_plan.json",
            scheduler_dir=Path("reports/research_scheduler") / agent_program_id,
            policy=policy,
        ),
    }
    return {
        "schema_version": AGENT_PLAN_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "program_id": agent_program_id,
        "objective": (
            "Run the next conservative autonomous research loop for SDSS "
            "point-supervised catalog generation."
        ),
        "config": str(program.config),
        "dataset": str(program.dataset),
        "split": str(program.split),
        "report_root": str(program.report_root),
        "checkpoint_root": str(program.checkpoint_root),
        "defaults": defaults,
        "claim_gates": dict(program.claim_gates),
        "variants": executable_variants,
        "pending_approval_variants": pending_approval_variants,
        "recommended_mainline_tasks": recommended_mainline_tasks,
        "agent_plan": agent_plan,
    }


def write_research_agent_plan(
    program_path: str | Path,
    *,
    root: str | Path = "reports/research_runs",
    output_path: str | Path,
    markdown_output_path: str | Path | None = None,
    autonomy: str = "small_runs_then_gated_long_runs",
    budget: str = "conservative",
    approve_pending: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    payload = build_research_agent_plan(
        program_path,
        root=root,
        autonomy=autonomy,
        budget=budget,
        approve_pending=approve_pending,
    )
    write_json(output_path, payload)
    if markdown_output_path:
        path = Path(markdown_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_research_agent_plan_markdown(payload), encoding="utf-8")
    return payload


def summarize_current_state(
    board: Mapping[str, Any],
    entries: list[Mapping[str, Any]],
    reports: list[Mapping[str, Any]],
) -> dict[str, Any]:
    champion = best_candidate_entry(entries)
    completed_entries = completed_variant_entries(entries)
    full_psf_e20 = find_report_by_variant_id(reports, FULL_PSF_E20_VARIANT_ID)
    center_only_e20 = find_report_by_variant_id(reports, "ablation_center_only_unet_lite_e20")
    no_psf_e20 = find_report_by_variant_id(reports, "ablation_no_psf_unet_lite_e20")
    center_only_e50 = find_report_by_variant_id(reports, CENTER_ONLY_E50_VARIANT_ID)
    baseline_e50 = find_report_by_variant_id(reports, BASELINE_E50_VARIANT_ID)
    synthetic_validation = find_synthetic_validation_report(reports)
    synthetic_faint_diagnostic = find_synthetic_diagnostic_report(reports, "synthetic_faint_recovery")
    synthetic_heatmap_diagnostic = find_synthetic_diagnostic_report(reports, "synthetic_heatmap_response")
    synthetic_morphology_diagnostic = find_synthetic_diagnostic_report(reports, "synthetic_morphology_response")
    return {
        "runs": board.get("runs", 0),
        "claim_gate_counts": dict(board.get("claim_gate_counts", {})),
        "detection_champion": compact_entry(champion),
        "completed_variants": sorted(completed_entries),
        "completed_variant_gates": {
            variant_id: str(entry.get("claim_gate", "")) for variant_id, entry in sorted(completed_entries.items())
        },
        "completed_variant_reports": completed_entries,
        "method_ablation_status": {
            "full_psf_unet_lite_e20": compact_report(full_psf_e20),
            "center_only_unet_lite_e20": compact_report(center_only_e20),
            "no_psf_unet_lite_e20": compact_report(no_psf_e20),
        },
        "mainline_status": {
            "center_only_unet_lite_e50": compact_report(center_only_e50),
            "baseline_e50": compact_report(baseline_e50),
            "synthetic_injection_validation": compact_synthetic_validation(synthetic_validation),
            "synthetic_faint_recovery_diagnostic": compact_synthetic_diagnostic(synthetic_faint_diagnostic),
            "synthetic_heatmap_response_diagnostic": compact_synthetic_diagnostic(synthetic_heatmap_diagnostic),
            "synthetic_morphology_response_diagnostic": compact_synthetic_diagnostic(synthetic_morphology_diagnostic),
        },
        "evidence_audits": list(board.get("evidence_audits", [])),
        "e50_audits": list(board.get("evidence_audits", [])),
    }


def best_candidate_entry(entries: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = [
        entry
        for entry in entries
        if entry.get("claim_gate") == "candidate_evidence"
        and entry.get("metrics", {}).get("f1") is not None
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: float(row.get("metrics", {}).get("f1") or -1.0), reverse=True)[0]


def compact_entry(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not entry:
        return None
    metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), Mapping) else {}
    return {
        "run_id": entry.get("run_id", ""),
        "variant_id": entry.get("variant_id", ""),
        "claim_gate": entry.get("claim_gate", ""),
        "metrics": {
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "ap": metrics.get("ap"),
        },
        "run_options": dict(entry.get("run_options", {})) if isinstance(entry.get("run_options"), Mapping) else {},
        "run_dir": entry.get("run_dir", ""),
        "report_path": entry.get("report_path", ""),
    }


def compact_report(report: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    entry = build_registry_entry(dict(report))
    return compact_entry(entry)


def compact_synthetic_validation(report: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), Mapping) else {}
    injected = metrics.get("injected_detection", {}) if isinstance(metrics.get("injected_detection"), Mapping) else {}
    injected_ap = (
        metrics.get("injected_average_precision", {})
        if isinstance(metrics.get("injected_average_precision"), Mapping)
        else {}
    )
    all_truth = metrics.get("all_truth_detection", {}) if isinstance(metrics.get("all_truth_detection"), Mapping) else {}
    by_mag = metrics.get("injected_recall_by_mag_r", {})
    if not isinstance(by_mag, Mapping):
        by_mag = {}
    validation = metrics.get("validation", {}) if isinstance(metrics.get("validation"), Mapping) else {}
    return {
        "run_id": report.get("run_id") or Path(str(report.get("_run_dir", ""))).name,
        "variant_id": report.get("variant_id") or Path(str(report.get("_run_dir", ""))).name,
        "claim_gate": report.get("claim_gate", {}).get("status", "") if isinstance(report.get("claim_gate"), Mapping) else "",
        "source_run_id": validation.get("source_run_id")
        or (report.get("run_options", {}) if isinstance(report.get("run_options"), Mapping) else {}).get("source_run_id"),
        "source_variant_id": validation.get("source_variant_id")
        or (report.get("run_options", {}) if isinstance(report.get("run_options"), Mapping) else {}).get("source_variant_id"),
        "metrics": {
            "injected_precision": injected.get("precision"),
            "injected_recall": injected.get("recall"),
            "injected_f1": injected.get("f1"),
            "injected_ap": injected_ap.get("ap"),
            "all_truth_precision": all_truth.get("precision"),
            "all_truth_recall": all_truth.get("recall"),
            "all_truth_f1": all_truth.get("f1"),
            "faintest_mag_bin": faintest_mag_bin(by_mag),
            "faintest_mag_bin_by_label": faintest_mag_bin_by_label(
                metrics.get("injected_recall_by_mag_r_and_label", {})
                if isinstance(metrics.get("injected_recall_by_mag_r_and_label"), Mapping)
                else {}
            ),
        },
        "counts": dict(report.get("counts", {})) if isinstance(report.get("counts"), Mapping) else {},
        "report_path": report.get("_report_path", ""),
        "run_dir": report.get("_run_dir", ""),
    }


def compact_synthetic_diagnostic(report: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), Mapping) else {}
    by_mag = metrics.get("by_mag_r", {}) if isinstance(metrics.get("by_mag_r"), Mapping) else {}
    by_mag_label = metrics.get("by_mag_r_and_label", {}) if isinstance(metrics.get("by_mag_r_and_label"), Mapping) else {}
    faint = by_mag.get(FAINTEST_SYNTHETIC_MAG_BIN, {}) if isinstance(by_mag.get(FAINTEST_SYNTHETIC_MAG_BIN), Mapping) else {}
    response_by_condition = (
        metrics.get("response_by_condition", {}) if isinstance(metrics.get("response_by_condition"), Mapping) else {}
    )
    recall_by_condition = (
        metrics.get("recall_by_condition", {}) if isinstance(metrics.get("recall_by_condition"), Mapping) else {}
    )
    return {
        "run_id": report.get("run_id") or Path(str(report.get("_run_dir", ""))).name,
        "variant_id": report.get("variant_id") or Path(str(report.get("_run_dir", ""))).name,
        "diagnostic": report.get("diagnostic", ""),
        "claim_gate": report.get("claim_gate", {}).get("status", "") if isinstance(report.get("claim_gate"), Mapping) else "",
        "validation_run_dir": report.get("validation_run_dir", ""),
        "metrics": {
            "faintest_mag_bin": faint,
            "faintest_mag_bin_by_label": faintest_mag_bin_by_label(
                by_mag_label
            ),
            "overall": metrics.get("overall", {}) if isinstance(metrics.get("overall"), Mapping) else {},
            "overall_response": (
                metrics.get("overall_response", {}) if isinstance(metrics.get("overall_response"), Mapping) else {}
            ),
            "response_by_condition": {
                str(key): dict(value) for key, value in response_by_condition.items() if isinstance(value, Mapping)
            },
            "recall_by_condition": {
                str(key): dict(value) for key, value in recall_by_condition.items() if isinstance(value, Mapping)
            },
        },
        "findings": list(report.get("findings", [])) if isinstance(report.get("findings"), list) else [],
        "next_actions": list(report.get("next_actions", [])) if isinstance(report.get("next_actions"), list) else [],
        "report_path": report.get("_report_path", ""),
        "run_dir": report.get("_run_dir", ""),
    }


def faintest_mag_bin(by_mag: Mapping[str, Any]) -> dict[str, Any] | None:
    rows: list[tuple[float, str, Mapping[str, Any]]] = []
    for key, row in by_mag.items():
        if not isinstance(row, Mapping):
            continue
        hi = parse_mag_bin_upper(str(key))
        if hi is None:
            continue
        rows.append((hi, str(key), row))
    if not rows:
        return None
    _hi, key, row = sorted(rows, key=lambda item: item[0], reverse=True)[0]
    return {"bin": key, **dict(row)}


def faintest_mag_bin_by_label(by_mag_label: Mapping[str, Any]) -> dict[str, Any]:
    if FAINTEST_SYNTHETIC_MAG_BIN in by_mag_label and isinstance(by_mag_label[FAINTEST_SYNTHETIC_MAG_BIN], Mapping):
        return {
            label: dict(row)
            for label, row in by_mag_label[FAINTEST_SYNTHETIC_MAG_BIN].items()
            if isinstance(row, Mapping)
        }
    rows: list[tuple[float, str, Mapping[str, Any]]] = []
    for key, label_rows in by_mag_label.items():
        if not isinstance(label_rows, Mapping):
            continue
        hi = parse_mag_bin_upper(str(key))
        if hi is None:
            continue
        rows.append((hi, str(key), label_rows))
    if not rows:
        return {}
    _hi, _key, label_rows = sorted(rows, key=lambda item: item[0], reverse=True)[0]
    return {label: dict(row) for label, row in label_rows.items() if isinstance(row, Mapping)}


def parse_mag_bin_upper(label: str) -> float | None:
    try:
        return float(label.rsplit(",", 1)[1].rstrip(")"))
    except (IndexError, ValueError):
        return None


def completed_variant_entries(entries: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for entry in sorted(entries, key=lambda row: str(row.get("variant_id") or "")):
        variant_id = str(entry.get("variant_id") or "").strip()
        gate = str(entry.get("claim_gate") or "")
        if not variant_id or gate not in COMPLETED_VARIANT_GATES:
            continue
        current = completed.get(variant_id)
        compact = compact_entry(entry)
        if current is None or compact_entry_rank(compact) > compact_entry_rank(current):
            completed[variant_id] = compact
    return completed


def compact_entry_rank(entry: Mapping[str, Any]) -> tuple[int, float]:
    gate = str(entry.get("claim_gate") or "")
    metrics = entry.get("metrics", {}) if isinstance(entry.get("metrics"), Mapping) else {}
    f1 = metrics.get("f1")
    try:
        f1_value = float(f1) if f1 is not None else -1.0
    except (TypeError, ValueError):
        f1_value = -1.0
    return (DEPENDENCY_GATE_ORDER.get(gate, 0), f1_value)


def find_report_by_variant_id(reports: list[Mapping[str, Any]], variant_id: str) -> Mapping[str, Any] | None:
    matches = [report for report in reports if str(report.get("variant_id") or "") == variant_id]
    if not matches:
        return None
    entries = [build_registry_entry(dict(report)) for report in matches]
    best = best_candidate_entry(entries)
    if best:
        run_id = best.get("run_id")
        for report in matches:
            if report.get("run_id") == run_id:
                return report
    return matches[0]


def find_synthetic_validation_report(reports: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    matches = [report for report in reports if str(report.get("validation") or "") == "synthetic_injection_mainline"]
    if not matches:
        return None
    return sorted(matches, key=synthetic_validation_rank, reverse=True)[0]


def find_synthetic_diagnostic_report(reports: list[Mapping[str, Any]], diagnostic: str) -> Mapping[str, Any] | None:
    matches = [report for report in reports if str(report.get("diagnostic") or "") == diagnostic]
    if not matches:
        return None
    return sorted(matches, key=synthetic_diagnostic_rank, reverse=True)[0]


def synthetic_validation_rank(report: Mapping[str, Any]) -> tuple[int, int, int, float]:
    gate = report.get("claim_gate", {}) if isinstance(report.get("claim_gate"), Mapping) else {}
    status = str(gate.get("status") or "")
    counts = report.get("counts", {}) if isinstance(report.get("counts"), Mapping) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), Mapping) else {}
    injected = metrics.get("injected_detection", {}) if isinstance(metrics.get("injected_detection"), Mapping) else {}
    try:
        recall = float(injected.get("recall", 0.0) or 0.0)
    except (TypeError, ValueError):
        recall = 0.0
    try:
        backgrounds = int(counts.get("backgrounds", 0) or 0)
    except (TypeError, ValueError):
        backgrounds = 0
    try:
        injected_truth = int(counts.get("injected_truth", 0) or 0)
    except (TypeError, ValueError):
        injected_truth = 0
    return (DEPENDENCY_GATE_ORDER.get(status, 0), injected_truth, backgrounds, recall)


def synthetic_diagnostic_rank(report: Mapping[str, Any]) -> tuple[int, int, str]:
    counts = report.get("counts", {}) if isinstance(report.get("counts"), Mapping) else {}
    try:
        truth = int(counts.get("injected_truth", counts.get("truth", 0)) or 0)
    except (TypeError, ValueError):
        truth = 0
    return (1, truth, str(report.get("run_id") or report.get("_run_dir") or ""))


def find_report(
    reports: list[Mapping[str, Any]],
    *,
    model_arch: str,
    loss_variant: str,
    epochs: int,
) -> Mapping[str, Any] | None:
    matches = []
    for report in reports:
        options = report.get("run_options", {}) if isinstance(report.get("run_options"), Mapping) else {}
        if str(options.get("model_arch", "")) != model_arch:
            continue
        if str(options.get("loss_variant", "")) != loss_variant:
            continue
        if int(options.get("epochs", -1) or -1) != epochs:
            continue
        matches.append(report)
    if not matches:
        return None
    entries = [build_registry_entry(dict(report)) for report in matches]
    best = best_candidate_entry(entries)
    if best:
        run_id = best.get("run_id")
        for report in matches:
            if report.get("run_id") == run_id:
                return report
    return matches[0]


def build_candidate_variants(*, current_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    variants = []
    method_status = current_state.get("method_ablation_status", {})
    completed_gates = completed_variant_gates(current_state)
    if (
        not isinstance(method_status, Mapping)
        or method_status.get("full_psf_unet_lite_e20") is None
        and not variant_completed(completed_gates, FULL_PSF_E20_VARIANT_ID)
    ):
        variants.append(full_psf_e20_variant())
    for spec in auxiliary_factorial_specs():
        e5 = auxiliary_e5_variant(spec)
        e20 = auxiliary_e20_variant(spec)
        e5_id = str(e5["variant_id"])
        e20_id = str(e20["variant_id"])
        e5_done = variant_completed(completed_gates, e5_id)
        e20_done = variant_completed(completed_gates, e20_id)
        if not e5_done:
            variants.append(e5)
            if not e20_done:
                variants.append(e20)
            continue
        if not e20_done and variant_gate_satisfied(completed_gates, e5_id, "candidate_evidence"):
            ready_e20 = dict(e20)
            ready_e20.pop("depends_on", None)
            ready_e20.pop("dependency_gate", None)
            variants.append(ready_e20)
    return variants


def completed_variant_gates(current_state: Mapping[str, Any]) -> dict[str, str]:
    gates = current_state.get("completed_variant_gates", {})
    if not isinstance(gates, Mapping):
        return {}
    return {str(variant_id): str(gate) for variant_id, gate in gates.items()}


def variant_completed(completed_gates: Mapping[str, str], variant_id: str) -> bool:
    return str(completed_gates.get(variant_id, "")) in COMPLETED_VARIANT_GATES


def variant_gate_satisfied(completed_gates: Mapping[str, str], variant_id: str, required_gate: str) -> bool:
    required = str(required_gate or "complete")
    observed = str(completed_gates.get(variant_id, ""))
    return DEPENDENCY_GATE_ORDER.get(observed, 0) >= DEPENDENCY_GATE_ORDER.get(required, 0)


def full_psf_e20_variant() -> dict[str, Any]:
    return {
        "variant_id": "ablation_full_psf_unet_lite_e20_matched_bs192",
        "objective": "Run a matched 20-epoch full-loss UNet-lite comparison at the common batch size.",
        "hypothesis": (
            "A matched full PSF-constrained 20-epoch run will show whether the poor e5 "
            "full-loss result was early optimization noise or a persistent detection loss."
        ),
        "success_signal": "F1/AP approach or exceed center_only e20 while preserving validation-selected thresholding.",
        "failure_signal": "F1/AP remain below center_only e20, weakening the PSF-constrained detection claim.",
        "tags": ["pilot", "fixed_split", "native_frame", "unet_lite", "method_ablation", "agent_generated"],
        "claims": ["psf_constrained_method", "ablation_screen", "auxiliary_loss_diagnosis"],
        "run": {
            "epochs": 20,
            "batch_size": DEFAULT_BATCH_SIZE,
            "model_arch": "unet_lite",
            "loss_variant": "full_psf_point_supervised",
        },
    }


def auxiliary_factorial_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "class",
            "variant_stem": "ablation_center_class_unet_lite",
            "objective_name": "center + class",
            "hypothesis": "Class supervision is the auxiliary term that most harms or helps center detection.",
            "weights": {
                "center_loss_weight": 1.0,
                "photometry_loss_weight": 0.0,
                "multiband_loss_weight": 0.0,
                "psf_reconstruction_loss_weight": 0.0,
                "class_loss_weight": 0.5,
            },
        },
        {
            "name": "photometry",
            "variant_stem": "ablation_center_photometry_unet_lite",
            "objective_name": "center + photometry",
            "hypothesis": "Photometry regression is the auxiliary term that most harms or helps center detection.",
            "weights": {
                "center_loss_weight": 1.0,
                "photometry_loss_weight": 1.0,
                "multiband_loss_weight": 0.0,
                "psf_reconstruction_loss_weight": 0.0,
                "class_loss_weight": 0.0,
            },
        },
        {
            "name": "multiband",
            "variant_stem": "ablation_center_multiband_unet_lite",
            "objective_name": "center + multiband",
            "hypothesis": "Multiband consistency is the auxiliary term that most harms or helps center detection.",
            "weights": {
                "center_loss_weight": 1.0,
                "photometry_loss_weight": 0.0,
                "multiband_loss_weight": 0.05,
                "psf_reconstruction_loss_weight": 0.0,
                "class_loss_weight": 0.0,
            },
        },
        {
            "name": "psf",
            "variant_stem": "ablation_center_psf_unet_lite",
            "objective_name": "center + PSF reconstruction",
            "hypothesis": "PSF reconstruction is the auxiliary term that most harms or helps center detection.",
            "weights": {
                "center_loss_weight": 1.0,
                "photometry_loss_weight": 0.0,
                "multiband_loss_weight": 0.0,
                "psf_reconstruction_loss_weight": 0.2,
                "class_loss_weight": 0.0,
            },
        },
    ]


def auxiliary_e5_variant(spec: Mapping[str, Any]) -> dict[str, Any]:
    name = str(spec["objective_name"])
    return {
        "variant_id": f"{spec['variant_stem']}_e5",
        "objective": f"Screen the {name} factorial auxiliary-loss variant.",
        "hypothesis": str(spec["hypothesis"]),
        "success_signal": "The e5 screen reaches candidate_evidence without losing detection AP versus nearby ablations.",
        "failure_signal": "The e5 screen falls to engineering_check or loses most recall/AP.",
        "tags": [
            "pilot",
            "fixed_split",
            "native_frame",
            "unet_lite",
            "method_ablation",
            "factorial_auxiliary",
            "agent_generated",
        ],
        "claims": ["psf_constrained_method", "ablation_screen", "auxiliary_loss_diagnosis"],
        "run": {
            "epochs": 5,
            "batch_size": DEFAULT_BATCH_SIZE,
            "model_arch": "unet_lite",
            "loss_variant": "full_psf_point_supervised",
            **dict(spec["weights"]),
        },
    }


def auxiliary_e20_variant(spec: Mapping[str, Any]) -> dict[str, Any]:
    variant_id = f"{spec['variant_stem']}_e20"
    parent = f"{spec['variant_stem']}_e5"
    name = str(spec["objective_name"])
    return {
        "variant_id": variant_id,
        "depends_on": [parent],
        "dependency_gate": "candidate_evidence",
        "objective": f"Extend the {name} factorial auxiliary-loss variant to 20 epochs if e5 passes.",
        "hypothesis": f"A 20-epoch {name} run will identify whether that auxiliary term explains the center-only gap.",
        "success_signal": "The e20 run narrows or reverses the gap to center_only e20 under the fixed split.",
        "failure_signal": "The e20 run remains below center_only e20, marking this auxiliary as negative detection evidence.",
        "tags": [
            "pilot",
            "fixed_split",
            "native_frame",
            "unet_lite",
            "method_ablation",
            "factorial_auxiliary",
            "agent_generated",
        ],
        "claims": ["psf_constrained_method", "ablation_screen", "auxiliary_loss_diagnosis"],
        "run": {
            "epochs": 20,
            "batch_size": DEFAULT_BATCH_SIZE,
            "model_arch": "unet_lite",
            "loss_variant": "full_psf_point_supervised",
            **dict(spec["weights"]),
        },
    }


def build_pending_approval_variants(*, current_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    champion = current_state.get("detection_champion", {})
    champion_run = champion.get("run_id", "") if isinstance(champion, Mapping) else ""
    completed_gates = completed_variant_gates(current_state)
    variants = [
        {
            "variant_id": "ablation_center_only_unet_lite_e50_seed42_pending",
            "objective": "Run a 50-epoch center-only detection champion check after the diagnostic queue is reviewed.",
            "hypothesis": "The current center-only e20 champion remains strong at e50 under the common batch size.",
            "approval_reason": "e50 is paper-scale pilot evidence and should wait for the auxiliary-loss diagnosis.",
            "parent_evidence_run_id": champion_run,
            "tags": ["pilot", "fixed_split", "native_frame", "unet_lite", "paper_scale_candidate", "pending_approval"],
            "claims": ["architecture_improvement", "ablation_screen", "stratified_metrics"],
            "run": {
                "epochs": 50,
                "batch_size": DEFAULT_BATCH_SIZE,
                "model_arch": "unet_lite",
                "loss_variant": "center_only",
                "seed": 42,
            },
        },
        {
            "variant_id": "ablation_full_psf_unet_lite_e50_matched_bs192_pending",
            "depends_on": ["ablation_full_psf_unet_lite_e20_matched_bs192"],
            "dependency_gate": "candidate_evidence",
            "objective": "Run a matched 50-epoch full-loss comparison only if the e20 full-loss run is credible.",
            "hypothesis": "The full PSF-constrained method can recover detection quality at paper-scale pilot budget.",
            "approval_reason": "The current evidence does not yet justify an automatic e50 full-loss run.",
            "tags": ["pilot", "fixed_split", "native_frame", "unet_lite", "paper_scale_candidate", "pending_approval"],
            "claims": ["psf_constrained_method", "stratified_metrics"],
            "run": {
                "epochs": 50,
                "batch_size": DEFAULT_BATCH_SIZE,
                "model_arch": "unet_lite",
                "loss_variant": "full_psf_point_supervised",
                "seed": 42,
            },
        },
    ]
    return [
        variant
        for variant in variants
        if not variant_completed(completed_gates, canonical_pending_variant_id(str(variant.get("variant_id", ""))))
    ]


def promote_pending_approval_variants(
    pending_variants: list[dict[str, Any]],
    *,
    approve_pending: list[str] | tuple[str, ...],
    current_state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    approved_ids = [str(variant_id).strip() for variant_id in approve_pending if str(variant_id).strip()]
    if not approved_ids:
        return [], pending_variants
    pending_by_id = {str(variant.get("variant_id", "")): variant for variant in pending_variants}
    completed_gates = completed_variant_gates(current_state)
    unknown = sorted(
        variant_id
        for variant_id in set(approved_ids) - set(pending_by_id)
        if not variant_completed(completed_gates, canonical_pending_variant_id(variant_id))
    )
    if unknown:
        raise ValueError(f"unknown pending approval variant(s): {', '.join(unknown)}")

    approved = set(approved_ids)
    promoted: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for variant in pending_variants:
        pending_id = str(variant.get("variant_id", ""))
        canonical_id = canonical_pending_variant_id(pending_id)
        if pending_id not in approved:
            remaining.append(variant)
            continue
        if variant_completed(completed_gates, canonical_id):
            continue
        promoted_variant = canonicalize_pending_variant(
            variant,
            canonical_id=canonical_id,
            current_state=current_state,
        )
        promoted.append(promoted_variant)
    return promoted, remaining


def canonicalize_pending_variant(
    variant: Mapping[str, Any],
    *,
    canonical_id: str,
    current_state: Mapping[str, Any],
) -> dict[str, Any]:
    promoted = dict(variant)
    pending_id = str(variant.get("variant_id", ""))
    promoted["variant_id"] = canonical_id
    tags = [str(tag) for tag in variant.get("tags", []) if str(tag) != "pending_approval"]
    promoted["tags"] = tags
    promoted["approved_from_pending"] = {
        "variant_id": pending_id,
        "approved_at": utc_now(),
    }
    dependency_gate = str(promoted.get("dependency_gate", "complete"))
    completed_gates = completed_variant_gates(current_state)
    dependencies = [str(dep) for dep in promoted.get("depends_on", [])]
    if dependencies and all(variant_gate_satisfied(completed_gates, dep, dependency_gate) for dep in dependencies):
        promoted.pop("depends_on", None)
        promoted.pop("dependency_gate", None)
    return promoted


def canonical_pending_variant_id(variant_id: str) -> str:
    return variant_id.removesuffix("_pending")


def agent_policy(*, autonomy: str, budget: str) -> dict[str, Any]:
    budgets = {
        "conservative": {"max_parallel_gpu_jobs": 2, "max_runtime_minutes": 1440, "min_free_gb": 10.0, "max_util": 20.0},
        "medium": {"max_parallel_gpu_jobs": 2, "max_runtime_minutes": 2880, "min_free_gb": 10.0, "max_util": 20.0},
        "aggressive": {"max_parallel_gpu_jobs": 2, "max_runtime_minutes": 4320, "min_free_gb": 10.0, "max_util": 30.0},
    }
    if budget not in budgets:
        raise ValueError(f"unsupported research agent budget: {budget}")
    if autonomy not in {"small_runs_then_gated_long_runs", "fully_automatic"}:
        raise ValueError(f"unsupported research agent autonomy: {autonomy}")
    return {
        "autonomy": autonomy,
        "budget": budget,
        "automatic_epochs": [5, 20],
        "long_runs": "pending_approval" if autonomy == "small_runs_then_gated_long_runs" else "automatic",
        **budgets[budget],
    }


def build_interpretation(current_state: Mapping[str, Any]) -> list[str]:
    champion = current_state.get("detection_champion", {})
    champion_run = champion.get("run_id") if isinstance(champion, Mapping) else None
    method = current_state.get("method_ablation_status", {})
    mainline = current_state.get("mainline_status", {})
    completed = set(current_state.get("completed_variants", []))
    lines = []
    if champion_run:
        metrics = champion.get("metrics", {}) if isinstance(champion, Mapping) else {}
        lines.append(
            "Current detection champion is "
            f"{champion_run} with F1={metrics.get('f1')} and AP={metrics.get('ap')}."
        )
    if isinstance(method, Mapping) and method.get("center_only_unet_lite_e20") and not method.get("full_psf_unet_lite_e20"):
        lines.append(
            "Center-only e20 is available but matched full-loss e20 is missing, so the PSF-loss claim is not resolved."
        )
    if isinstance(method, Mapping) and method.get("no_psf_unet_lite_e20") and method.get("center_only_unet_lite_e20"):
        lines.append(
            "No-PSF e20 remains below center-only e20, so the issue is broader than the PSF reconstruction term alone."
        )
    if auxiliary_loss_diagnosis_complete(completed):
        lines.append("The auxiliary-loss diagnosis queue is complete; completed diagnostic variants are omitted from the automatic queue.")
    if isinstance(mainline, Mapping) and mainline.get("center_only_unet_lite_e50"):
        e50_metrics = method_metrics(mainline.get("center_only_unet_lite_e50"))
        lines.append(
            "Center-only e50 is now candidate evidence "
            f"with F1={e50_metrics.get('f1')} and AP={e50_metrics.get('ap')}; "
            "the next default step should strengthen the mainline protocol rather than add more seeds."
        )
    if isinstance(mainline, Mapping) and mainline.get("synthetic_injection_validation"):
        synthetic = mainline.get("synthetic_injection_validation")
        synthetic_metrics = method_metrics(synthetic)
        faint = synthetic_metrics.get("faintest_mag_bin")
        faint_by_label = synthetic_metrics.get("faintest_mag_bin_by_label")
        faint_text = ""
        if isinstance(faint, Mapping):
            faint_text = f"; faintest injected bin {faint.get('bin')} recall={faint.get('recall')}"
        if isinstance(faint_by_label, Mapping) and faint_by_label:
            label_text = ", ".join(
                f"{label} recall={row.get('recall')}" for label, row in sorted(faint_by_label.items())
                if isinstance(row, Mapping) and row.get("recall") is not None
            )
            if label_text:
                faint_text += f" ({label_text})"
        lines.append(
            "Synthetic-injection validation is available "
            f"with injected recall={synthetic_metrics.get('injected_recall')} and AP={synthetic_metrics.get('injected_ap')}"
            f"{faint_text}."
        )
    if isinstance(mainline, Mapping) and mainline.get("synthetic_heatmap_response_diagnostic"):
        diagnostic = mainline.get("synthetic_heatmap_response_diagnostic")
        diagnostic_metrics = method_metrics(diagnostic)
        faint = diagnostic_metrics.get("faintest_mag_bin")
        faint_by_label = diagnostic_metrics.get("faintest_mag_bin_by_label")
        if isinstance(faint, Mapping):
            lines.append(
                "Heatmap-response diagnostic shows the faintest injected bin has "
                f"match-radius low-floor response={faint.get('best_within_match_radius_and_low_floor')} "
                f"and best local median score={safe_nested_get(faint, 'best_score', 'median')}."
            )
        if isinstance(faint_by_label, Mapping) and faint_by_label:
            galaxy = faint_by_label.get("galaxy", {}) if isinstance(faint_by_label.get("galaxy"), Mapping) else {}
            star = faint_by_label.get("star", {}) if isinstance(faint_by_label.get("star"), Mapping) else {}
            if galaxy or star:
                lines.append(
                    "Balanced synthetic heatmap response separates source type: "
                    f"faint galaxy low-floor response={galaxy.get('best_within_match_radius_and_low_floor')}, "
                    f"faint star low-floor response={star.get('best_within_match_radius_and_low_floor')}."
                )
    if isinstance(mainline, Mapping) and mainline.get("synthetic_morphology_response_diagnostic"):
        diagnostic = mainline.get("synthetic_morphology_response_diagnostic")
        metrics = method_metrics(diagnostic)
        condition_response = metrics.get("response_by_condition", {})
        if isinstance(condition_response, Mapping):
            star = condition_response.get("mag22_star_r1.3", {})
            galaxy = condition_response.get("mag22_galaxy_r2.4", {})
            if isinstance(star, Mapping) or isinstance(galaxy, Mapping):
                lines.append(
                    "Morphology-response diagnostic is available: "
                    f"mag22 star_r1.3 low-floor response={star.get('best_within_match_radius_and_low_floor') if isinstance(star, Mapping) else None}, "
                    f"mag22 galaxy_r2.4 low-floor response={galaxy.get('best_within_match_radius_and_low_floor') if isinstance(galaxy, Mapping) else None}."
                )
    if isinstance(method, Mapping) and method.get("full_psf_unet_lite_e20") and method.get("center_only_unet_lite_e20"):
        full_metrics = method_metrics(method.get("full_psf_unet_lite_e20"))
        center_metrics = method_metrics(method.get("center_only_unet_lite_e20"))
        if metric_below(full_metrics, center_metrics, "f1") and metric_below(full_metrics, center_metrics, "ap"):
            lines.append(
                "Matched full-PSF e20 is below center-only e20 on F1 and AP, so full-PSF e50 should remain pending approval rather than automatic."
            )
    if not lines:
        lines.append("The evidence board does not yet identify a stable candidate-evidence detection champion.")
    return lines


def build_recommended_mainline_tasks(
    *,
    current_state: Mapping[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    """Return recommended next work that should not be auto-queued as training."""

    tasks: list[dict[str, Any]] = []
    mainline = current_state.get("mainline_status", {})
    method = current_state.get("method_ablation_status", {})
    center_e50 = mainline.get("center_only_unet_lite_e50") if isinstance(mainline, Mapping) else None
    baseline_e50 = mainline.get("baseline_e50") if isinstance(mainline, Mapping) else None
    synthetic_validation = (
        mainline.get("synthetic_injection_validation") if isinstance(mainline, Mapping) else None
    )
    synthetic_faint_diagnostic = (
        mainline.get("synthetic_faint_recovery_diagnostic") if isinstance(mainline, Mapping) else None
    )
    synthetic_heatmap_diagnostic = (
        mainline.get("synthetic_heatmap_response_diagnostic") if isinstance(mainline, Mapping) else None
    )
    synthetic_morphology_diagnostic = (
        mainline.get("synthetic_morphology_response_diagnostic") if isinstance(mainline, Mapping) else None
    )
    full_psf_e20 = method.get("full_psf_unet_lite_e20") if isinstance(method, Mapping) else None
    center_e20 = method.get("center_only_unet_lite_e20") if isinstance(method, Mapping) else None
    audits = current_state.get("evidence_audits", [])

    center_e50_audit_done = evidence_audit_done(
        audits,
        baseline_label="baseline_e50",
        target_label="center_only_e50",
    )

    if isinstance(center_e50, Mapping) and isinstance(baseline_e50, Mapping) and not center_e50_audit_done:
        tasks.append(
            {
                "task_id": "audit_center_only_e50_vs_baseline_e50",
                "priority": "high",
                "kind": "mainline_experiment",
                "status": "ready",
                "objective": "Promote the center-only e50 result from a raw leaderboard entry to paired mainline evidence.",
                "rationale": (
                    "Center-only e50 is candidate evidence; the mainline paper path needs paired deltas, "
                    "bootstrap intervals, and claimable strata against the e50 baseline before more seeds."
                ),
                "expected_evidence": "A validation-threshold, paired existing-run audit with aggregate and stratified deltas.",
                "command": (
                    "PYTHONPATH=src python -m sdss_point_benchmark.cli research-evidence-audit "
                    f"--baseline-run-dir {baseline_e50.get('run_dir')} "
                    f"--target-run-dir {center_e50.get('run_dir')} "
                    f"--baseline-label baseline_e50 --target-label center_only_e50 "
                    f"--output {root / 'center_only_e50_vs_baseline_e50_audit.json'} "
                    f"--markdown-output {root / 'center_only_e50_vs_baseline_e50_audit.md'}"
                ),
            }
        )

    if (
        isinstance(center_e50, Mapping)
        and center_e50_audit_done
        and not isinstance(synthetic_validation, Mapping)
    ):
        tasks.append(
            {
                "task_id": "synthetic_injection_mainline_validation",
                "priority": "high",
                "kind": "paper_validation",
                "status": "needs_design",
                "objective": "Test the center-only e50 mainline on controlled injected sources with known truth.",
                "rationale": (
                    "The paired e50 audit is positive, but PhotoObj agreement is still weak supervision. "
                    "The next mainline task should check recovery against controlled truth before adding more seeds."
                ),
                "expected_evidence": (
                    "A fixed synthetic-injection protocol reporting recovery by magnitude, crowding, and source density "
                    "for the current center-only e50 checkpoint."
                ),
                "command": "Draft and implement a fixed synthetic-injection evaluation protocol using the existing center-only e50 checkpoint.",
            }
        )

    if (
        isinstance(synthetic_validation, Mapping)
        and not isinstance(synthetic_faint_diagnostic, Mapping)
        and not isinstance(synthetic_heatmap_diagnostic, Mapping)
    ):
        synthetic_metrics = method_metrics(synthetic_validation)
        faint = synthetic_metrics.get("faintest_mag_bin")
        faint_bin = faint.get("bin") if isinstance(faint, Mapping) else None
        faint_recall = faint.get("recall") if isinstance(faint, Mapping) else None
        tasks.append(
            {
                "task_id": "diagnose_faint_synthetic_recovery",
                "priority": "high",
                "kind": "paper_validation",
                "status": "ready",
                "objective": "Diagnose the faint-source recovery drop in the controlled synthetic-injection protocol.",
                "rationale": (
                    "The existing center-only e50 checkpoint passes the paired e50 PhotoObj audit and has nonzero "
                    f"synthetic recovery, but the faintest injected mag_r bin {faint_bin} has recall {faint_recall}. "
                    "This is a mainline validation risk that should be resolved before expanding more seeds or long runs."
                ),
                "expected_evidence": (
                    "A fixed-threshold diagnostic comparing injected-source score distributions, threshold sweep behavior, "
                    "and per-magnitude false negatives for the current center-only e50 checkpoint."
                ),
                "command": (
                    "Inspect "
                    f"{synthetic_validation.get('run_dir')}/predictions_candidates.csv and "
                    f"{synthetic_validation.get('run_dir')}/truth_injected.csv; then run a threshold/score diagnostic "
                    "that reports injected recall by mag_r without changing the fixed test split."
                ),
            }
        )

    if isinstance(synthetic_faint_diagnostic, Mapping) and not isinstance(synthetic_heatmap_diagnostic, Mapping):
        findings = synthetic_faint_diagnostic.get("findings", [])
        finding_text = "; ".join(str(item) for item in findings[:2]) if isinstance(findings, list) else ""
        tasks.append(
            {
                "task_id": "measure_faint_heatmap_response",
                "priority": "high",
                "kind": "paper_validation",
                "status": "ready",
                "objective": "Measure raw center-heatmap response at faint injected-source positions.",
                "rationale": (
                    "The faint recovery diagnostic shows decoded candidates do not appear near most faint misses. "
                    f"{finding_text} A heatmap-response check distinguishes weak model response from decode/NMS artifacts."
                ),
                "expected_evidence": (
                    "Per-magnitude center-heatmap score summaries at injected coordinates and local neighborhoods."
                ),
                "command": (
                    "PYTHONPATH=src python -m sdss_point_benchmark.cli research-synthetic-heatmap-diagnostic "
                    f"--validation-dir {synthetic_validation.get('run_dir') if isinstance(synthetic_validation, Mapping) else ''} "
                    f"--output-dir {root / 'synthetic_injection_center_only_e50_heatmap_diagnostic_v1'} "
                    "--device cuda:0 --search-radius-pixels 8 --low-floor 0.05 --shard-cache-size 2"
                ),
            }
        )

    if isinstance(synthetic_heatmap_diagnostic, Mapping) and not isinstance(synthetic_morphology_diagnostic, Mapping):
        heatmap_metrics = method_metrics(synthetic_heatmap_diagnostic)
        faint = heatmap_metrics.get("faintest_mag_bin")
        faint_by_label = heatmap_metrics.get("faintest_mag_bin_by_label")
        response = faint.get("best_within_match_radius_and_low_floor") if isinstance(faint, Mapping) else None
        best_median = safe_nested_get(faint, "best_score", "median") if isinstance(faint, Mapping) else None
        galaxy = faint_by_label.get("galaxy", {}) if isinstance(faint_by_label, Mapping) and isinstance(faint_by_label.get("galaxy"), Mapping) else {}
        star = faint_by_label.get("star", {}) if isinstance(faint_by_label, Mapping) and isinstance(faint_by_label.get("star"), Mapping) else {}
        if galaxy or star:
            task_id = "diagnose_faint_extended_source_response"
            objective = "Design the smallest diagnostic for faint extended-source center response."
            rationale = (
                f"Balanced synthetic validation separates the type confound: at {FAINTEST_SYNTHETIC_MAG_BIN}, "
                f"galaxy low-floor response is {galaxy.get('best_within_match_radius_and_low_floor')} "
                f"with best local median {safe_nested_get(galaxy, 'best_score', 'median')}, while star response is "
                f"{star.get('best_within_match_radius_and_low_floor')} with best local median "
                f"{safe_nested_get(star, 'best_score', 'median')}. The next mainline move should target faint "
                "extended-source signal or morphology handling, not generic threshold tuning."
            )
            expected = (
                "A controlled diagnostic that varies faint extended-source flux/profile or supervision weighting and "
                "reports mag_r x label recall plus raw heatmap response."
            )
        else:
            task_id = "increase_faint_source_signal_or_loss_weight_diagnostic"
            objective = "Design the smallest training-side diagnostic that increases faint-source center response."
            rationale = (
                f"Heatmap response for {FAINTEST_SYNTHETIC_MAG_BIN} is weak: match-radius low-floor response is "
                f"{response} and best local median score is {best_median}. Threshold and max-detection changes "
                "did not recover faint injections, so the next mainline move should test signal strength or loss weighting."
            )
            expected = (
                "A short fixed-split diagnostic that increases faint-source supervision weight or injection signal and "
                "reports heatmap response plus injected recall by mag_r."
            )
        tasks.append(
            {
                "task_id": task_id,
                "priority": "high",
                "kind": "method_rescue",
                "status": "ready",
                "objective": objective,
                "rationale": rationale,
                "expected_evidence": expected,
                "command": (
                    "PYTHONPATH=src python -m sdss_point_benchmark.cli research-synthetic-morphology-diagnostic "
                    f"--validation-dir {synthetic_validation.get('run_dir') if isinstance(synthetic_validation, Mapping) else ''} "
                    f"--output-dir {root / 'synthetic_injection_center_only_e50_morphology_diagnostic_v1'} "
                    "--device cuda:0 --search-radius-pixels 8 --low-floor 0.05 --shard-cache-size 2"
                ),
            }
        )

    if isinstance(synthetic_morphology_diagnostic, Mapping):
        next_actions = (
            synthetic_morphology_diagnostic.get("next_actions", [])
            if isinstance(synthetic_morphology_diagnostic.get("next_actions"), list)
            else []
        )
        action = next_actions[0] if next_actions and isinstance(next_actions[0], Mapping) else {}
        action_id = str(action.get("action") or "review_morphology_diagnostic")
        reason = str(action.get("reason") or "morphology diagnostic completed")
        task_templates = {
            "design_surface_brightness_or_extended_profile_rescue": {
                "task_id": "design_surface_brightness_or_extended_profile_rescue",
                "objective": "Design a short mainline rescue experiment for faint extended-source center response.",
                "expected_evidence": (
                    "A fixed-split short training diagnostic that changes only one supervision/profile lever and reports "
                    "morphology-sweep response plus injected recall."
                ),
                "command": "Draft a short-run experiment card for surface-brightness or extended-profile rescue before any e50 run.",
            },
            "audit_synthetic_profile_or_catalog_label_protocol": {
                "task_id": "audit_synthetic_profile_or_catalog_label_protocol",
                "objective": "Audit the synthetic galaxy profile and weak-label protocol before training changes.",
                "expected_evidence": "A protocol note comparing injected profile assumptions with available catalog morphology fields.",
                "command": "Inspect synthetic profile assumptions and catalog fields, then write a protocol audit note.",
            },
            "expand_synthetic_validation_backgrounds": {
                "task_id": "expand_synthetic_validation_backgrounds",
                "objective": "Scale controlled synthetic validation after the morphology sweep does not isolate the failure mode.",
                "expected_evidence": "A larger-background synthetic validation with the same fixed checkpoint and split.",
                "command": "Run a larger synthetic validation using the existing center-only e50 checkpoint and fixed test split.",
            },
        }
        template = task_templates.get(
            action_id,
            {
                "task_id": "review_morphology_diagnostic",
                "objective": "Review the completed morphology diagnostic and choose the next mainline intervention.",
                "expected_evidence": "A written interpretation of morphology response and recall by condition.",
                "command": "Inspect the morphology diagnostic report and write the next mainline experiment card.",
            },
        )
        tasks.append(
            {
                "priority": "high",
                "kind": "method_rescue",
                "status": "needs_design",
                "rationale": reason,
                **template,
            }
        )

    if missing_claimable_strata(audits):
        tasks.append(
            {
                "task_id": "restore_claimable_stratification",
                "priority": "high",
                "kind": "paper_validation",
                "status": "ready",
                "objective": "Make mainline validation claimable for crowdedness, seeing, and source-quality slices.",
                "rationale": (
                    "Current reports expose mag_r recall, but nearest-neighbor, seeing, and SNR strata are unavailable "
                    "in the default run report. Paper-facing claims need these strata or an explicit limitation."
                ),
                "expected_evidence": (
                    "Existing-run audit or report metadata that includes derived nearest-neighbor/source-density strata, "
                    "plus a clear record of which strata remain unavailable."
                ),
                "command": "Inspect existing evidence-audit strata and backfill derived strata into the mainline comparison report.",
            }
        )
    elif has_unavailable_physical_strata(audits):
        tasks.append(
            {
                "task_id": "write_mainline_strata_limitations",
                "priority": "medium",
                "kind": "paper_validation",
                "status": "ready",
                "objective": "Summarize which mainline strata are claimable now and which require new metadata.",
                "rationale": (
                    "The paired e50 audit has magnitude, label-quality, flags, nearest-neighbor, and source-density strata. "
                    "Seeing and SNR remain unavailable, so they should be documented as validation gaps instead of blocking "
                    "the next experiment."
                ),
                "expected_evidence": "A short mainline validation note that cites the existing e50 audit and records seeing/SNR as unavailable.",
                "command": "Write a mainline validation note from reports/research_runs/center_only_e50_vs_baseline_e50_audit.md.",
            }
        )

    if isinstance(full_psf_e20, Mapping) and isinstance(center_e20, Mapping):
        full_metrics = method_metrics(full_psf_e20)
        center_metrics = method_metrics(center_e20)
        if metric_below(full_metrics, center_metrics, "f1") and metric_below(full_metrics, center_metrics, "ap"):
            tasks.append(
                {
                    "task_id": "diagnose_full_psf_loss_gap",
                    "priority": "medium",
                    "kind": "method_rescue",
                    "status": "ready",
                    "objective": "Diagnose why the nominal PSF-constrained objective underperforms center-only before any e50 full-PSF run.",
                    "rationale": (
                        "Matched full-PSF e20 is below center-only e20 on F1 and AP; a longer full-PSF run is not "
                        "the smallest falsifying experiment."
                    ),
                    "expected_evidence": (
                        "A short, targeted loss-balancing or gradient-scale diagnostic that explains whether auxiliary terms "
                        "suppress center heatmap learning."
                    ),
                    "command": "Plan a cheap loss-scale diagnostic before approving ablation_full_psf_unet_lite_e50_matched_bs192_pending.",
                }
            )

    if not tasks:
        tasks.append(
            {
                "task_id": "define_next_mainline_protocol",
                "priority": "high",
                "kind": "mainline_experiment",
                "status": "needs_design",
                "objective": "Define the next fixed-protocol experiment before adding more pilot-scale runs.",
                "rationale": "The automatic queue is empty, so the next useful step is protocol design rather than opportunistic training.",
                "expected_evidence": "A small, auditable protocol change with a falsifiable success signal.",
                "command": "Draft the next experiment card from the current evidence board.",
            }
        )
    return tasks[:3]


def missing_claimable_strata(audits: Any) -> bool:
    if not isinstance(audits, list) or not audits:
        return True
    for audit in audits:
        if not isinstance(audit, Mapping):
            continue
        available = set(str(item) for item in audit.get("available_strata", []))
        if {"mag_r", "nearest_neighbor_arcsec_derived", "source_density_per_cutout"}.issubset(available):
            return False
    return True


def evidence_audit_done(audits: Any, *, baseline_label: str, target_label: str) -> bool:
    if not isinstance(audits, list):
        return False
    for audit in audits:
        if not isinstance(audit, Mapping):
            continue
        if str(audit.get("baseline_label", "")) != baseline_label:
            continue
        if str(audit.get("target_label", "")) != target_label:
            continue
        if audit.get("delta_f1") is not None and audit.get("delta_ap") is not None:
            return True
    return False


def has_unavailable_physical_strata(audits: Any) -> bool:
    if not isinstance(audits, list):
        return False
    for audit in audits:
        if not isinstance(audit, Mapping):
            continue
        unavailable = audit.get("unavailable_strata", {})
        if not isinstance(unavailable, Mapping):
            continue
        if any(str(field) in unavailable for field in ("seeing", "snr")):
            return True
    return False


def build_blocked_claims(current_state: Mapping[str, Any]) -> list[dict[str, str]]:
    method = current_state.get("method_ablation_status", {})
    blocked = []
    if isinstance(method, Mapping) and not method.get("full_psf_unet_lite_e20"):
        blocked.append(
            {
                "claim": "psf_constrained_method",
                "reason": "matched full-loss e20 evidence is missing while center-only e20 is the detection champion",
            }
        )
    elif isinstance(method, Mapping) and method.get("full_psf_unet_lite_e20") and method.get("center_only_unet_lite_e20"):
        full_metrics = method_metrics(method.get("full_psf_unet_lite_e20"))
        center_metrics = method_metrics(method.get("center_only_unet_lite_e20"))
        if metric_below(full_metrics, center_metrics, "f1") and metric_below(full_metrics, center_metrics, "ap"):
            blocked.append(
                {
                    "claim": "psf_constrained_method",
                    "reason": "matched full-loss e20 is below center-only e20 on F1 and AP",
                }
            )
    blocked.append(
        {
            "claim": "headline_result",
            "reason": "current runs are candidate evidence, not multi-seed or final-protocol paper support",
        }
    )
    return blocked


def auxiliary_loss_diagnosis_complete(completed_variants: set[str]) -> bool:
    required = {FULL_PSF_E20_VARIANT_ID}
    for spec in auxiliary_factorial_specs():
        stem = str(spec["variant_stem"])
        required.add(f"{stem}_e5")
        required.add(f"{stem}_e20")
    return required.issubset(completed_variants)


def method_metrics(entry: Any) -> Mapping[str, Any]:
    if not isinstance(entry, Mapping):
        return {}
    metrics = entry.get("metrics", {})
    return metrics if isinstance(metrics, Mapping) else {}


def metric_below(left: Mapping[str, Any], right: Mapping[str, Any], metric: str) -> bool:
    try:
        return float(left.get(metric)) < float(right.get(metric))
    except (TypeError, ValueError):
        return False


def safe_nested_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def build_agent_commands(*, agent_plan_path: Path, scheduler_dir: Path, policy: Mapping[str, Any]) -> list[str]:
    max_jobs = int(policy.get("max_parallel_gpu_jobs", 2))
    min_free_gb = float(policy.get("min_free_gb", 10.0))
    max_util = float(policy.get("max_util", 20.0))
    max_runtime = float(policy.get("max_runtime_minutes", 1440))
    return [
        "make research-board",
        "make research-compare-latest",
        "make research-agent-plan",
        (
            "PYTHONPATH=src python -m sdss_point_benchmark.cli research-autopilot "
            f"--program {agent_plan_path} --output-dir {scheduler_dir} --execute "
            f"--max-jobs {max_jobs} --min-free-gb {min_free_gb:g} --max-util {max_util:g} "
            f"--max-runtime-minutes {max_runtime:g}"
        ),
    ]


def render_research_agent_plan_markdown(payload: Mapping[str, Any]) -> str:
    agent = payload.get("agent_plan", {}) if isinstance(payload.get("agent_plan"), Mapping) else {}
    current = agent.get("current_state", {}) if isinstance(agent.get("current_state"), Mapping) else {}
    champion = current.get("detection_champion", {}) if isinstance(current.get("detection_champion"), Mapping) else {}
    lines = [
        "# Research Agent Plan",
        "",
        f"- Program: {payload.get('program_id', '')}",
        f"- Source reports: {agent.get('source_report_root', '')}",
        f"- Autonomy: {agent.get('autonomy', '')}",
        f"- Budget: {agent.get('budget', '')}",
        "",
        "## Current State",
        "",
        f"- Runs: {current.get('runs', 0)}",
        f"- Claim gates: {current.get('claim_gate_counts', {})}",
    ]
    if champion:
        metrics = champion.get("metrics", {}) if isinstance(champion.get("metrics"), Mapping) else {}
        lines.append(
            f"- Detection champion: {champion.get('run_id', '')} "
            f"F1={metrics.get('f1')} AP={metrics.get('ap')}"
        )
    completed = current.get("completed_variants", [])
    if completed:
        lines.append(f"- Completed variants: {len(completed)}")
    lines.extend(["", "## Interpretation", ""])
    for item in agent.get("interpretation", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Automatic Queue", ""])
    for variant in payload.get("variants", []):
        run = variant.get("run", {}) if isinstance(variant.get("run"), Mapping) else {}
        deps = variant.get("depends_on", [])
        dep_text = f" depends_on={deps}" if deps else ""
        lines.append(
            f"- {variant.get('variant_id', '')}: epochs={run.get('epochs')} "
            f"loss={run.get('loss_variant')} batch={run.get('batch_size')}{dep_text}"
        )
    lines.extend(["", "## Recommended Mainline Tasks", ""])
    for task in payload.get("recommended_mainline_tasks", []):
        lines.append(
            f"- [{task.get('priority', '')}] {task.get('task_id', '')}: "
            f"{task.get('objective', '')}"
        )
        command = str(task.get("command", "")).strip()
        if command:
            lines.append(f"  command: `{command}`")
    lines.extend(["", "## Pending Approval", ""])
    for variant in payload.get("pending_approval_variants", []):
        run = variant.get("run", {}) if isinstance(variant.get("run"), Mapping) else {}
        lines.append(
            f"- {variant.get('variant_id', '')}: epochs={run.get('epochs')} "
            f"loss={run.get('loss_variant')} reason={variant.get('approval_reason', '')}"
        )
    lines.extend(["", "## Commands", ""])
    for command in agent.get("commands", []):
        lines.append(f"- `{command}`")
    lines.append("")
    return "\n".join(lines)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
