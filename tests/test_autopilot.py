import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sdss_point_benchmark.autopilot import (
    AutopilotOptions,
    GpuSnapshot,
    build_dependency_map,
    run_research_autopilot,
    start_research_run_process,
    wait_for_available_gpu,
)
from sdss_point_benchmark.research_program import expand_research_program, load_research_program
from test_training_pipeline import _write_tiny_dataset


class AutopilotTests(unittest.TestCase):
    def test_autopilot_dry_run_writes_scheduler_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            program = _write_program(root, config, dataset_dir)
            output = root / "scheduler"

            payload = run_research_autopilot(
                program_path=program,
                output_dir=output,
                options=AutopilotOptions(run_prefix="dry", execute=False),
            )
            written = json.loads((output / "scheduler_plan.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "planned")
        self.assertEqual(written["runs"][0]["run_id"], "dry_baseline")
        self.assertTrue(written["runs"][0]["dry_run"])
        self.assertEqual(written["runs"][0]["claims"], ["scheduler"])
        self.assertEqual(written["runs"][0]["existing_status"]["status"], "absent")

    def test_dependency_map_resolves_variant_ids_to_prefixed_run_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            program = load_research_program(_write_program(root, config, dataset_dir, include_dependency=True))

            dependencies = build_dependency_map(program, "queue")

        self.assertEqual(dependencies["queue_followup"], ["queue_baseline"])

    def test_start_research_run_process_passes_research_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            program = load_research_program(_write_program(root, config, dataset_dir))
            spec = expand_research_program(program, run_prefix="queue", dry_run=False)[0]
            with patch("sdss_point_benchmark.autopilot.subprocess.Popen") as popen:
                start_research_run_process(spec, root / "logs" / "run.log")
                command = popen.call_args.args[0]

        self.assertIn("--program-id", command)
        self.assertIn("autopilot_tiny", command)
        self.assertIn("--variant-id", command)
        self.assertIn("baseline", command)
        self.assertIn("--tag", command)
        self.assertIn("fixed_split", command)
        self.assertIn("--claim", command)
        self.assertIn("scheduler", command)
        self.assertIn("--claim-gate-policy-json", command)

    def test_wait_for_available_gpu_requires_two_available_samples(self):
        busy = [GpuSnapshot(index=0, name="GPU", memory_total_mb=24576, memory_used_mb=20000, utilization_gpu=0)]
        free = [GpuSnapshot(index=0, name="GPU", memory_total_mb=24576, memory_used_mb=1024, utilization_gpu=0)]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("sdss_point_benchmark.autopilot.query_gpus", side_effect=[busy, busy, busy, free, free]),
            patch("sdss_point_benchmark.autopilot.time.sleep", return_value=None),
        ):
            gpu = wait_for_available_gpu(
                max_util=20,
                min_free_gb=10,
                poll_seconds=0,
                sample_seconds=0,
                gpu_snapshot_path=Path(tmp) / "gpu.jsonl",
            )

        self.assertIsNotNone(gpu)
        self.assertEqual(gpu.index, 0)

    def test_autopilot_execute_skips_existing_matching_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            program = _write_program(root, config, dataset_dir)
            report_dir = root / "reports" / "research_runs" / "queue_baseline"
            report_dir.mkdir(parents=True)
            (report_dir / "report.json").write_text(
                json.dumps(
                    {
                        "run_id": "queue_baseline",
                        "status": "executed",
                        "claim_gate": {"status": "engineering_check"},
                        "metrics": {"test": {"counts": {}, "detection": {}}},
                        "claims": ["scheduler"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("sdss_point_benchmark.autopilot.start_research_run_process") as start:
                payload = run_research_autopilot(
                    program_path=program,
                    output_dir=root / "scheduler",
                    options=AutopilotOptions(run_prefix="queue", execute=True, poll_seconds=0, sample_seconds=0),
                )

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["completed"], ["queue_baseline"])
        self.assertEqual(payload["skipped"][0]["run_id"], "queue_baseline")
        start.assert_not_called()

    def test_autopilot_execute_blocks_nonempty_incomplete_report_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            program = _write_program(root, config, dataset_dir)
            report_dir = root / "reports" / "research_runs" / "queue_baseline"
            report_dir.mkdir(parents=True)
            (report_dir / "partial.txt").write_text("partial\n", encoding="utf-8")

            with patch("sdss_point_benchmark.autopilot.start_research_run_process") as start:
                payload = run_research_autopilot(
                    program_path=program,
                    output_dir=root / "scheduler",
                    options=AutopilotOptions(run_prefix="queue", execute=True, poll_seconds=0, sample_seconds=0),
                )

        self.assertEqual(payload["status"], "completed_with_failures")
        self.assertEqual(payload["blocked"][0]["run_id"], "queue_baseline")
        self.assertIn("report.json is missing", payload["blocked"][0]["error"])
        start.assert_not_called()


def _write_config(root: Path) -> Path:
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "protocol": "sdss-point-supervised-v1",
                "data": {"root": "/Data/sdss/example"},
                "cutout": {"pixel_scale_arcsec": 0.396},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def _write_program(root: Path, config: Path, dataset_dir: Path, include_dependency: bool = False) -> Path:
    variants = [
        {
            "variant_id": "baseline",
            "hypothesis": "Autopilot writes a dry-run queue.",
            "tags": ["smoke", "fixed_split"],
            "claims": ["scheduler"],
            "run": {},
        }
    ]
    if include_dependency:
        variants.append(
            {
                "variant_id": "followup",
                "depends_on": ["baseline"],
                "hypothesis": "Autopilot preserves dependencies.",
                "tags": ["smoke", "fixed_split"],
                "claims": ["scheduler"],
                "run": {},
            }
        )
    program = root / "program.json"
    program.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "sdss-point-supervised-v1",
                "program_id": "autopilot_tiny",
                "objective": "Audit autopilot.",
                "config": str(config),
                "dataset": str(dataset_dir),
                "split": str(root / "split.json"),
                "report_root": str(root / "reports" / "research_runs"),
                "checkpoint_root": str(root / "checkpoints" / "research_runs"),
                "claim_gates": {"min_epochs": 5, "required_tags": ["fixed_split"]},
                "defaults": {
                    "epochs": 1,
                    "batch_size": 1,
                    "base_channels": 4,
                    "model_arch": "baseline",
                    "loader_mode": "shard_grouped",
                    "shard_cache_size": 1,
                    "candidate_threshold": 0.0,
                    "max_detections_per_cutout": 4,
                    "predict_limit": 1,
                    "device": "cpu",
                },
                "variants": variants,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return program


if __name__ == "__main__":
    unittest.main()
