from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

PROTOCOL = "sdss-point-supervised-v1"


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    """Validate the minimum stable experiment config contract."""

    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config.protocol must be {PROTOCOL!r}")
    data = config.get("data")
    if not isinstance(data, Mapping) or not data.get("root"):
        raise ValueError("config.data.root is required")


def default_artifact_layout() -> dict[str, str]:
    """Return the repository-standard artifact layout used by agents and scripts."""

    return {
        "manifests": "artifacts/manifests/",
        "splits": "artifacts/splits/",
        "checkpoints": "artifacts/checkpoints/",
        "reports": "reports/",
    }


def build_dry_run_report(
    config: Mapping[str, Any],
    command: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated, reproducible dry-run report for experiment commands."""

    validate_experiment_config(config)
    experiments = config.get("experiments", [])
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "dry_run",
        "generated_at": generated_at or datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "command": command,
        "config": dict(config),
        "planned_experiments": [_experiment_id(experiment) for experiment in experiments],
        "artifact_layout": default_artifact_layout(),
        "reproducibility": {
            "data_root": str(config["data"]["root"]),
            "native_frame_policy": config.get("data", {}).get("native_frame_policy", ""),
            "split_seed": config.get("split", {}).get("seed", ""),
        },
    }


def _experiment_id(experiment: Any) -> str:
    if isinstance(experiment, Mapping):
        return str(experiment.get("id") or experiment.get("name") or "")
    return str(experiment)
