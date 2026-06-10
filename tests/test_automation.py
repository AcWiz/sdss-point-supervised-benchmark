import json
import tempfile
import unittest
from pathlib import Path

from sdss_point_benchmark.automation import (
    ResearchRunSpec,
    evaluate_claim_gate,
    preflight_research_run,
    run_research_run,
    write_report_from_existing_pilot_loop,
)
from sdss_point_benchmark.cli import main
from sdss_point_benchmark.research_program import (
    build_diagnosis,
    build_evidence_ledger,
    build_research_board,
    compare_research_runs,
    expand_research_program,
    load_research_program,
    rebuild_registry,
)
from test_training_pipeline import _write_tiny_dataset


class AutomationTests(unittest.TestCase):
    def test_preflight_rejects_non_empty_report_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            report_dir = root / "reports"
            report_dir.mkdir()
            (report_dir / "old.txt").write_text("old\n", encoding="utf-8")

            spec = ResearchRunSpec(
                run_id="reject_non_empty",
                objective="Audit the loop.",
                hypothesis="The loop should reject reused report directories.",
                config=config,
                dataset=dataset_dir,
                split=root / "split.json",
                report_dir=report_dir,
                checkpoint_dir=root / "checkpoint",
            )

            with self.assertRaisesRegex(ValueError, "non-empty"):
                preflight_research_run(spec)

    def test_research_run_dry_run_writes_plan_and_report_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            report_dir = root / "reports" / "dry"
            checkpoint_dir = root / "checkpoints" / "dry"

            report = run_research_run(
                ResearchRunSpec(
                    run_id="dry",
                    objective="Audit dry-run reporting.",
                    hypothesis="Dry runs should record plan and claim gate.",
                    config=config,
                    dataset=dataset_dir,
                    split=root / "split.json",
                    report_dir=report_dir,
                    checkpoint_dir=checkpoint_dir,
                    dry_run=True,
                )
            )
            plan = json.loads((report_dir / "plan.json").read_text(encoding="utf-8"))
            next_actions = json.loads((report_dir / "next_actions.json").read_text(encoding="utf-8"))
            report_markdown_exists = (report_dir / "report.md").exists()
            run_manifest = json.loads((report_dir / "run_manifest.json").read_text(encoding="utf-8"))
            state = json.loads((report_dir / "state.json").read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["claim_gate"]["status"], "blocked")
        self.assertEqual(plan["preflight"]["status"], "passed")
        self.assertTrue(report_markdown_exists)
        self.assertEqual(next_actions["run_id"], "dry")
        self.assertEqual(run_manifest["run_id"], "dry")
        self.assertEqual(state["metrics_status"], "not_run")

    def test_cli_research_run_executes_tiny_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            report_dir = root / "reports" / "tiny"
            checkpoint_dir = root / "checkpoints" / "tiny"

            code = main(
                [
                    "research-run",
                    "--config",
                    str(config),
                    "--dataset",
                    str(dataset_dir),
                    "--split",
                    str(root / "split.json"),
                    "--run-id",
                    "tiny",
                    "--objective",
                    "Audit tiny end-to-end automation.",
                    "--hypothesis",
                    "The research loop writes auditable reports.",
                    "--report-dir",
                    str(report_dir),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--epochs",
                    "1",
                    "--train-limit-samples",
                    "1",
                    "--batch-size",
                    "1",
                    "--base-channels",
                    "4",
                    "--model-arch",
                    "unet_lite",
                    "--loader-mode",
                    "shard_grouped",
                    "--shard-cache-size",
                    "1",
                    "--candidate-threshold",
                    "0.0",
                    "--max-detections-per-cutout",
                    "4",
                    "--predict-limit",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
            summary_exists = (report_dir / "pilot_loop" / "summary.json").exists()
            checkpoint_exists = (checkpoint_dir / "best.pt").exists()

        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "executed")
        self.assertEqual(report["metrics"]["status"], "available")
        self.assertEqual(report["claim_gate"]["status"], "engineering_check")
        self.assertEqual(report["run_options"]["train_limit_samples"], 1)
        self.assertEqual(report["run_options"]["model_arch"], "unet_lite")
        self.assertEqual(report["run_options"]["loader_mode"], "shard_grouped")
        self.assertTrue(summary_exists)
        self.assertTrue(checkpoint_exists)

    def test_research_report_from_existing_pilot_loop_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            pilot_output = root / "pilot"
            checkpoint_dir = root / "checkpoint"
            main(
                [
                    "run-pilot-loop",
                    "--config",
                    str(config),
                    "--dataset",
                    str(dataset_dir),
                    "--split",
                    str(root / "split.json"),
                    "--output-dir",
                    str(pilot_output),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--base-channels",
                    "4",
                    "--candidate-threshold",
                    "0.0",
                    "--max-detections-per-cutout",
                    "4",
                    "--predict-limit",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            report_dir = root / "reports" / "existing"
            report = write_report_from_existing_pilot_loop(
                pilot_output_dir=pilot_output,
                run_id="existing",
                report_dir=report_dir,
                objective="Audit existing outputs.",
                hypothesis="Existing pilot-loop outputs can be reported.",
            )
            report_exists = (report_dir / "report.json").exists()

        self.assertEqual(report["status"], "report_existing")
        self.assertEqual(report["metrics"]["status"], "available")
        self.assertTrue(report_exists)

    def test_claim_gate_blocks_missing_predictions(self):
        gate = evaluate_claim_gate(
            ResearchRunSpec(
                run_id="missing",
                objective="Audit.",
                hypothesis="No predictions should block.",
                config="config.json",
                dataset="dataset",
                split="split.json",
                report_dir="reports",
                checkpoint_dir="checkpoints",
            ),
            pilot_outputs={"summary": {}, "val_threshold_sweep": {}, "test_metrics": {}},
            metrics={"test": {"counts": {"truth": 1, "candidate_predictions": 0}, "detection": {}}},
        )

        self.assertEqual(gate["status"], "blocked")
        self.assertFalse(gate["paper_claim_allowed"])

    def test_research_program_expands_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            program_path = _write_program(root, config, dataset_dir)

            program = load_research_program(program_path)
            specs = expand_research_program(program, run_prefix="queue", dry_run=True)

        self.assertEqual([spec.run_id for spec in specs], ["queue_baseline", "queue_low_threshold"])
        self.assertEqual(specs[0].program_id, "tiny_program")
        self.assertEqual(specs[1].candidate_threshold, 0.1)
        self.assertEqual(specs[0].model_arch, "baseline")
        self.assertTrue(specs[0].dry_run)

    def test_cli_research_program_writes_queue_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            program_path = _write_program(root, config, dataset_dir)
            output_dir = root / "program_out"

            code = main(
                [
                    "research-program",
                    "--program",
                    str(program_path),
                    "--output-dir",
                    str(output_dir),
                    "--run-prefix",
                    "dry",
                ]
            )
            plan = json.loads((output_dir / "program_plan.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(plan["program_id"], "tiny_program")
        self.assertEqual(len(plan["runs"]), 2)
        self.assertEqual(plan["runs"][0]["run_id"], "dry_baseline")

    def test_registry_board_compare_and_diagnosis_from_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "reports" / "research_runs"
            report_dir = run_root / "dry"
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = _write_config(root)
            run_research_run(
                ResearchRunSpec(
                    run_id="dry",
                    objective="Audit registry.",
                    hypothesis="Dry run should enter board.",
                    config=config,
                    dataset=dataset_dir,
                    split=root / "split.json",
                    report_dir=report_dir,
                    checkpoint_dir=root / "checkpoints" / "dry",
                    dry_run=True,
                    program_id="tiny_program",
                    variant_id="baseline",
                    tags=("smoke", "fixed_split"),
                )
            )

            registry = rebuild_registry(run_root)
            board = build_research_board(run_root)
            compare = compare_research_runs(run_root)
            diagnosis = build_diagnosis(report_dir / "report.json")
            ledger = build_evidence_ledger(run_root)

        self.assertEqual(registry["runs"], 1)
        self.assertEqual(board["claim_gate_counts"], {"blocked": 1})
        self.assertEqual(compare["runs"], 1)
        self.assertEqual(diagnosis["status"], "needs_attention")
        self.assertIn("smoke", ledger["claims"])


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


def _write_program(root: Path, config: Path, dataset_dir: Path) -> Path:
    program = root / "program.json"
    program.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "sdss-point-supervised-v1",
                "program_id": "tiny_program",
                "objective": "Audit tiny research program.",
                "config": str(config),
                "dataset": str(dataset_dir),
                "split": str(root / "split.json"),
                "report_root": str(root / "reports" / "research_runs"),
                "checkpoint_root": str(root / "checkpoints" / "research_runs"),
                "defaults": {
                    "epochs": 1,
                    "batch_size": 1,
                    "base_channels": 4,
                    "candidate_threshold": 0.0,
                    "max_detections_per_cutout": 4,
                    "predict_limit": 1,
                    "device": "cpu",
                },
                "variants": [
                    {
                        "variant_id": "baseline",
                        "hypothesis": "Baseline writes reports.",
                        "tags": ["smoke", "fixed_split"],
                        "claims": ["benchmark_contract"],
                        "run": {},
                    },
                    {
                        "variant_id": "low_threshold",
                        "hypothesis": "Lower threshold changes candidates.",
                        "tags": ["diagnostic"],
                        "claims": ["threshold_policy"],
                        "run": {"candidate_threshold": 0.1},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return program


if __name__ == "__main__":
    unittest.main()
