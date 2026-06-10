import json
import tempfile
import unittest
from pathlib import Path

from sdss_point_benchmark.cli import main
from sdss_point_benchmark.experiment import (
    build_dry_run_report,
    default_artifact_layout,
    validate_experiment_config,
)


class ExperimentContractTests(unittest.TestCase):
    def test_validate_experiment_config_requires_protocol_and_data_root(self):
        with self.assertRaisesRegex(ValueError, "protocol"):
            validate_experiment_config({"data": {"root": "/Data/sdss/example"}})

        with self.assertRaisesRegex(ValueError, "data.root"):
            validate_experiment_config({"protocol": "sdss-point-supervised-v1"})

    def test_build_dry_run_report_includes_reproducibility_metadata(self):
        config = {
            "protocol": "sdss-point-supervised-v1",
            "data": {"root": "/Data/sdss/example"},
            "experiments": [{"id": "E0", "name": "Data Integrity"}],
        }

        report = build_dry_run_report(
            config,
            command="run-experiment",
            generated_at="2026-06-04T00:00:00Z",
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["protocol"], "sdss-point-supervised-v1")
        self.assertEqual(report["status"], "dry_run")
        self.assertEqual(report["generated_at"], "2026-06-04T00:00:00Z")
        self.assertEqual(report["command"], "run-experiment")
        self.assertEqual(report["planned_experiments"], ["E0"])
        self.assertEqual(report["artifact_layout"]["reports"], "reports/")
        self.assertEqual(report["reproducibility"]["data_root"], "/Data/sdss/example")

    def test_default_artifact_layout_is_stable_for_agents(self):
        self.assertEqual(
            default_artifact_layout(),
            {
                "manifests": "artifacts/manifests/",
                "splits": "artifacts/splits/",
                "checkpoints": "artifacts/checkpoints/",
                "reports": "reports/",
            },
        )

    def test_cli_run_experiment_uses_validated_report_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            output = Path(tmp) / "report.json"
            config.write_text(
                json.dumps(
                    {
                        "protocol": "sdss-point-supervised-v1",
                        "data": {"root": "/Data/sdss/example"},
                        "experiments": [{"id": "E0", "name": "Data Integrity"}],
                    }
                ),
                encoding="utf-8",
            )

            exit_code = main(["run-experiment", "--config", str(config), "--output", str(output), "--dry-run"])
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["planned_experiments"], ["E0"])
        self.assertIn("generated_at", payload)
        self.assertEqual(payload["artifact_layout"]["checkpoints"], "artifacts/checkpoints/")


if __name__ == "__main__":
    unittest.main()
