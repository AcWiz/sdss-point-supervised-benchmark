from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment import PROTOCOL
from .pilot_loop import load_json, write_json
from .research_program import (
    build_diagnosis,
    build_evidence_ledger,
    build_research_board,
    compare_research_runs,
    expand_research_program,
    load_research_program,
    rebuild_registry,
    render_board_markdown,
    render_compare_markdown,
    render_diagnosis_markdown,
)

DEPENDENCY_GATE_ORDER = {
    "blocked": 0,
    "complete": 1,
    "engineering_check": 2,
    "candidate_evidence": 3,
}


@dataclass(frozen=True)
class AutopilotOptions:
    run_prefix: str | None = None
    execute: bool = False
    max_jobs: int = 2
    min_free_gb: float = 10.0
    max_util: float = 20.0
    poll_seconds: float = 30.0
    sample_seconds: float = 10.0
    max_runtime_minutes: float = 1440.0


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_gpu: float

    @property
    def free_gb(self) -> float:
        return float(self.memory_total_mb - self.memory_used_mb) / 1024.0


@dataclass(frozen=True)
class RunDependency:
    run_id: str
    gate: str = "complete"


def run_research_autopilot(
    *,
    program_path: str | Path,
    output_dir: str | Path,
    options: AutopilotOptions,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    program = load_research_program(program_path)
    specs = expand_research_program(program, run_prefix=options.run_prefix, dry_run=not options.execute)
    dependencies = build_dependency_map(program, options.run_prefix)
    existing = {spec.run_id: inspect_existing_run(spec) for spec in specs}
    completed_gates: dict[str, str] = {
        run_id: status.get("claim_gate", "")
        for run_id, status in existing.items()
        if status.get("status") == "complete"
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "planned" if not options.execute else "executed",
        "program": str(program_path),
        "program_id": program.program_id,
        "options": asdict(options),
        "runs": [
            {
                "run_id": spec.run_id,
                "variant_id": spec.variant_id,
                "report_dir": str(spec.report_dir),
                "checkpoint_dir": str(spec.checkpoint_dir),
                "device": spec.device,
                "dry_run": spec.dry_run,
                "claims": list(spec.claims),
                "claim_gate_policy": dict(spec.claim_gate_policy or {}),
                "depends_on": [dependency.run_id for dependency in dependencies.get(spec.run_id, [])],
                "dependency_gate": dependency_gate_for_run(dependencies, spec.run_id),
                "dependencies": [
                    {"run_id": dependency.run_id, "gate": dependency.gate}
                    for dependency in dependencies.get(spec.run_id, [])
                ],
                "existing_status": existing[spec.run_id],
            }
            for spec in specs
        ],
    }
    write_json(output / "scheduler_plan.json", payload)
    if not options.execute:
        return payload

    events_path = output / "scheduler_state.jsonl"
    gpu_path = output / "gpu_snapshots.jsonl"
    completed: list[str] = []
    failed: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    pending = {spec.run_id: spec for spec in specs}
    running: dict[str, RunningRun] = {}
    busy_gpu_indices: set[int] = set()
    deadline_seconds = max(0.0, options.max_runtime_minutes * 60.0)
    for run_id, spec in list(pending.items()):
        status = existing[run_id]
        if status["status"] == "complete":
            completed.append(run_id)
            completed_gates[run_id] = status.get("claim_gate", "")
            skipped.append({"run_id": run_id, "report": status.get("report", "")})
            refresh_research_outputs(Path(spec.report_dir).parent, spec)
            _append_jsonl(events_path, {"time": utc_now(), "event": "skipped_existing", "run_id": run_id, "report": status.get("report", "")})
            del pending[run_id]
        elif status["status"] == "blocked":
            blocked.append({"run_id": run_id, "error": status.get("reason", "existing output blocks run")})
            _append_jsonl(
                events_path,
                {
                    "time": utc_now(),
                    "event": "blocked_existing_output",
                    "run_id": run_id,
                    "reason": status.get("reason", "existing output blocks run"),
                },
            )
            del pending[run_id]
    while pending or running:
        for run_id, running_run in list(running.items()):
            return_code = running_run.process.poll()
            elapsed = time.perf_counter() - running_run.started_at
            if return_code is None and deadline_seconds and elapsed > deadline_seconds:
                running_run.process.terminate()
                return_code = running_run.process.wait(timeout=30)
            if return_code is None:
                continue
            busy_gpu_indices.discard(running_run.gpu_index)
            if return_code == 0:
                completed.append(run_id)
                completed_gates[run_id] = inspect_existing_run(running_run.spec).get("claim_gate", "")
                refresh_research_outputs(Path(running_run.spec.report_dir).parent, running_run.spec)
                _append_jsonl(
                    events_path,
                    {
                        "time": utc_now(),
                        "event": "completed",
                        "run_id": run_id,
                        "seconds": elapsed,
                        "return_code": return_code,
                    },
                )
            else:
                error = read_tail(running_run.log_path)
                failed.append({"run_id": run_id, "error": error})
                _append_jsonl(
                    events_path,
                    {
                        "time": utc_now(),
                        "event": "failed",
                        "run_id": run_id,
                        "seconds": elapsed,
                        "return_code": return_code,
                        "log": str(running_run.log_path),
                    },
                )
            del running[run_id]

        failed_or_blocked = {row["run_id"] for row in failed} | {row["run_id"] for row in blocked}
        for run_id in list(pending):
            deps = dependencies.get(run_id, [])
            failed_deps = [dep.run_id for dep in deps if dep.run_id in failed_or_blocked]
            if failed_deps:
                blocked.append({"run_id": run_id, "error": f"dependency failed: {', '.join(failed_deps)}"})
                _append_jsonl(events_path, {"time": utc_now(), "event": "blocked", "run_id": run_id, "dependencies": failed_deps})
                del pending[run_id]
                continue
            insufficient_deps = [
                dependency
                for dependency in deps
                if dependency.run_id in completed and not dependency_gate_satisfied(
                    completed_gates.get(dependency.run_id, ""),
                    dependency.gate,
                )
            ]
            if insufficient_deps:
                blocked.append(
                    {
                        "run_id": run_id,
                        "error": "dependency gate not satisfied: "
                        + ", ".join(
                            f"{dependency.run_id} has {completed_gates.get(dependency.run_id, '') or 'complete'} "
                            f"but requires {dependency.gate}"
                            for dependency in insufficient_deps
                        ),
                    }
                )
                _append_jsonl(
                    events_path,
                    {
                        "time": utc_now(),
                        "event": "blocked_dependency_gate",
                        "run_id": run_id,
                        "dependencies": [
                            {
                                "run_id": dependency.run_id,
                                "required_gate": dependency.gate,
                                "observed_gate": completed_gates.get(dependency.run_id, ""),
                            }
                            for dependency in insufficient_deps
                        ],
                    },
                )
                del pending[run_id]

        capacity = max(0, options.max_jobs - len(running))
        if capacity <= 0:
            time.sleep(options.poll_seconds)
            continue

        ready_specs = [
            spec
            for run_id, spec in pending.items()
            if all(
                dependency.run_id in completed
                and dependency_gate_satisfied(completed_gates.get(dependency.run_id, ""), dependency.gate)
                for dependency in dependencies.get(run_id, [])
            )
        ]
        started_any = False
        for spec in ready_specs[:capacity]:
            gpu = wait_for_available_gpu_once(
                max_util=options.max_util,
                min_free_gb=options.min_free_gb,
                sample_seconds=options.sample_seconds,
                gpu_snapshot_path=gpu_path,
                excluded_indices=busy_gpu_indices,
            )
            if gpu is None and query_gpus():
                break
            gpu_index = gpu.index if gpu is not None else -1
            device = f"cuda:{gpu_index}" if gpu is not None else "cpu"
            scheduled = replace(spec, dry_run=False, device=device)
            log_path = output / "logs" / f"{scheduled.run_id}.log"
            process = start_research_run_process(scheduled, log_path)
            running[scheduled.run_id] = RunningRun(
                spec=scheduled,
                process=process,
                gpu_index=gpu_index,
                started_at=time.perf_counter(),
                log_path=log_path,
            )
            if gpu is not None:
                busy_gpu_indices.add(gpu.index)
            del pending[spec.run_id]
            started_any = True
            _append_jsonl(
                events_path,
                {
                    "time": utc_now(),
                    "event": "started",
                    "run_id": scheduled.run_id,
                    "device": device,
                    "pid": process.pid,
                    "log": str(log_path),
                },
            )
        if not started_any:
            time.sleep(options.poll_seconds)

    result = {
        **payload,
        "status": "completed" if not failed and not blocked else "completed_with_failures",
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "blocked": blocked,
        "scheduler_state": str(events_path),
        "gpu_snapshots": str(gpu_path),
    }
    write_json(output / "scheduler_execution.json", result)
    return result


@dataclass(frozen=True)
class RunningRun:
    spec: Any
    process: subprocess.Popen
    gpu_index: int
    started_at: float
    log_path: Path


def build_dependency_map(program, run_prefix: str | None) -> dict[str, list[RunDependency]]:
    mapping: dict[str, str] = {}
    for variant in program.variants:
        merged = dict(program.defaults)
        merged.update(variant.run)
        run_id = str(merged.get("run_id") or default_run_id(program.program_id, variant.variant_id, run_prefix))
        mapping[variant.variant_id] = run_id
    dependencies: dict[str, list[RunDependency]] = {}
    for variant in program.variants:
        run_id = mapping[variant.variant_id]
        dependency_gate = normalize_dependency_gate(getattr(variant, "dependency_gate", "complete"))
        dependencies[run_id] = [RunDependency(mapping.get(dep, dep), dependency_gate) for dep in variant.depends_on]
    return dependencies


def normalize_dependency_gate(gate: str) -> str:
    normalized = str(gate or "complete").strip()
    if normalized not in DEPENDENCY_GATE_ORDER:
        raise ValueError(f"unsupported dependency_gate: {gate!r}")
    return normalized


def dependency_gate_for_run(dependencies: dict[str, list[RunDependency]], run_id: str) -> str:
    gates = {dependency.gate for dependency in dependencies.get(run_id, [])}
    if not gates:
        return "complete"
    return sorted(gates, key=lambda gate: DEPENDENCY_GATE_ORDER[gate], reverse=True)[0]


def dependency_gate_satisfied(observed_gate: str, required_gate: str) -> bool:
    required = normalize_dependency_gate(required_gate)
    if required == "complete":
        return True
    observed = str(observed_gate or "complete")
    return DEPENDENCY_GATE_ORDER.get(observed, 0) >= DEPENDENCY_GATE_ORDER[required]


def default_run_id(program_id: str, variant_id: str, run_prefix: str | None) -> str:
    prefix = run_prefix or program_id
    return f"{prefix}_{variant_id}" if prefix else variant_id


def start_research_run_process(spec, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "sdss_point_benchmark.cli",
        "research-run",
        "--config",
        str(spec.config),
        "--dataset",
        str(spec.dataset),
        "--split",
        str(spec.split),
        "--run-id",
        spec.run_id,
        "--objective",
        spec.objective,
        "--hypothesis",
        spec.hypothesis,
        "--report-dir",
        str(spec.report_dir),
        "--checkpoint-dir",
        str(spec.checkpoint_dir),
        "--epochs",
        str(spec.epochs),
        "--batch-size",
        str(spec.batch_size),
        "--learning-rate",
        str(spec.learning_rate),
        "--base-channels",
        str(spec.base_channels),
        "--heatmap-sigma",
        str(spec.heatmap_sigma),
        "--model-arch",
        spec.model_arch,
        "--loader-mode",
        spec.loader_mode,
        "--shard-cache-size",
        str(spec.shard_cache_size),
        "--num-workers",
        str(spec.num_workers),
        "--pin-memory",
        str(spec.pin_memory).lower(),
        "--loss-variant",
        spec.loss_variant,
        "--device",
        spec.device,
        "--seed",
        str(spec.seed),
        "--candidate-threshold",
        str(spec.candidate_threshold),
        "--nms-radius",
        str(spec.nms_radius),
        "--radius-arcsec",
        str(spec.radius_arcsec),
        "--band",
        spec.band,
        "--close-pair-arcsec",
        str(spec.close_pair_arcsec),
        "--psf-fraction",
        str(spec.psf_fraction),
    ]
    if spec.train_limit_samples is not None:
        command.extend(["--train-limit-samples", str(spec.train_limit_samples)])
    if spec.center_loss_weight is not None:
        command.extend(["--center-loss-weight", str(spec.center_loss_weight)])
    if spec.photometry_loss_weight is not None:
        command.extend(["--photometry-loss-weight", str(spec.photometry_loss_weight)])
    if spec.multiband_loss_weight is not None:
        command.extend(["--multiband-loss-weight", str(spec.multiband_loss_weight)])
    if spec.psf_reconstruction_loss_weight is not None:
        command.extend(["--psf-reconstruction-loss-weight", str(spec.psf_reconstruction_loss_weight)])
    if spec.class_loss_weight is not None:
        command.extend(["--class-loss-weight", str(spec.class_loss_weight)])
    if spec.max_detections_per_cutout is not None:
        command.extend(["--max-detections-per-cutout", str(spec.max_detections_per_cutout)])
    if spec.predict_limit is not None:
        command.extend(["--predict-limit", str(spec.predict_limit)])
    if spec.seeing_aware:
        command.append("--seeing-aware")
    if spec.include_suspect_truth:
        command.append("--include-suspect-truth")
    if spec.program_id:
        command.extend(["--program-id", spec.program_id])
    if spec.variant_id:
        command.extend(["--variant-id", spec.variant_id])
    if spec.parent_run_id:
        command.extend(["--parent-run-id", spec.parent_run_id])
    for tag in spec.tags:
        command.extend(["--tag", tag])
    for claim in spec.claims:
        command.extend(["--claim", claim])
    if spec.claim_gate_policy:
        command.extend(["--claim-gate-policy-json", json.dumps(spec.claim_gate_policy, sort_keys=True)])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src")) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path.write_text(json.dumps({"command": command}, sort_keys=True) + "\n", encoding="utf-8")
    handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, text=True, env=env)
    if process.__class__.__module__.startswith("unittest.mock"):
        handle.close()
    return process


def wait_for_available_gpu_once(
    *,
    max_util: float,
    min_free_gb: float,
    sample_seconds: float,
    gpu_snapshot_path: Path,
    excluded_indices: set[int],
) -> GpuSnapshot | None:
    first = query_gpus()
    _append_jsonl(gpu_snapshot_path, {"time": utc_now(), "sample": [asdict(gpu) for gpu in first]})
    if not first:
        return None
    if sample_seconds > 0:
        time.sleep(sample_seconds)
    second = query_gpus()
    _append_jsonl(gpu_snapshot_path, {"time": utc_now(), "sample": [asdict(gpu) for gpu in second]})
    by_index = {gpu.index: gpu for gpu in second}
    candidates = [
        gpu
        for gpu in first
        if gpu.index not in excluded_indices
        and gpu.index in by_index
        and gpu.utilization_gpu <= max_util
        and by_index[gpu.index].utilization_gpu <= max_util
        and gpu.free_gb >= min_free_gb
        and by_index[gpu.index].free_gb >= min_free_gb
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: (row.utilization_gpu, -row.free_gb, row.index))[0]


def wait_for_available_gpu(
    *,
    max_util: float,
    min_free_gb: float,
    poll_seconds: float,
    sample_seconds: float,
    gpu_snapshot_path: Path,
) -> GpuSnapshot | None:
    while True:
        gpu = wait_for_available_gpu_once(
            max_util=max_util,
            min_free_gb=min_free_gb,
            sample_seconds=sample_seconds,
            gpu_snapshot_path=gpu_snapshot_path,
            excluded_indices=set(),
        )
        if gpu is not None or not query_gpus():
            return gpu
        time.sleep(poll_seconds)


def query_gpus() -> list[GpuSnapshot]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return []
    rows: list[GpuSnapshot] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            GpuSnapshot(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mb=int(parts[2]),
                memory_used_mb=int(parts[3]),
                utilization_gpu=float(parts[4]),
            )
        )
    return rows


def refresh_research_outputs(root: Path, spec) -> None:
    rebuild_registry(root)
    board = build_research_board(root)
    write_json(root / "board.json", board)
    (root / "board.md").write_text(render_board_markdown(board), encoding="utf-8")
    compare = compare_research_runs(root)
    write_json(root / "compare_latest.json", compare)
    (root / "compare_latest.md").write_text(render_compare_markdown(compare), encoding="utf-8")
    build_evidence_ledger(root, output_path=root / "evidence_ledger.json")
    report_path = Path(spec.report_dir) / "report.json"
    if report_path.exists():
        diagnosis = build_diagnosis(report_path)
        write_json(Path(spec.report_dir) / "diagnosis.json", diagnosis)
        (Path(spec.report_dir) / "diagnosis.md").write_text(render_diagnosis_markdown(diagnosis), encoding="utf-8")


def inspect_existing_run(spec) -> dict[str, str]:
    report_dir = Path(spec.report_dir)
    report_path = report_dir / "report.json"
    if not report_dir.exists():
        return {"status": "absent"}
    if report_path.exists():
        try:
            report = load_json(report_path)
        except Exception as exc:  # noqa: BLE001
            return {"status": "blocked", "reason": f"existing report is unreadable: {exc}", "report": str(report_path)}
        if str(report.get("run_id", "")) != spec.run_id:
            return {
                "status": "blocked",
                "reason": f"existing report run_id {report.get('run_id', '')!r} does not match {spec.run_id!r}",
                "report": str(report_path),
            }
        return {
            "status": "complete",
            "report": str(report_path),
            "claim_gate": str(report.get("claim_gate", {}).get("status", "")) if isinstance(report.get("claim_gate", {}), dict) else "",
        }
    if any(report_dir.iterdir()):
        return {"status": "blocked", "reason": f"report_dir exists but report.json is missing: {report_dir}"}
    return {"status": "empty"}


def read_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
