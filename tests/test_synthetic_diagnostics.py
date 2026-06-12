import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sdss_point_benchmark.cli import main
from sdss_point_benchmark.io import write_prediction_catalog, write_source_catalog
from sdss_point_benchmark.schema import PredictionRecord, SourceRecord
from sdss_point_benchmark.synthetic_diagnostics import (
    best_local_score,
    build_synthetic_faint_recovery_diagnostic,
    heatmap_response_by_mag,
    heatmap_response_by_mag_and_label,
    heatmap_response_findings,
    heatmap_response_row,
    mag_bin_for_record,
    morphology_diagnostic_findings,
    morphology_injection_specs_for_background,
    morphology_recall_by_condition,
    morphology_response_by_condition,
    render_synthetic_diagnostic_markdown,
    render_synthetic_heatmap_diagnostic_markdown,
    render_synthetic_morphology_diagnostic_markdown,
)


class SyntheticFaintDiagnosticTests(unittest.TestCase):
    def test_diagnostic_flags_faint_bin_without_nearby_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation = root / "synthetic_validation"
            validation.mkdir()
            truth = [
                SourceRecord("bright0", "c0", 10.0 / 3600.0, 10.0 / 3600.0, "star", x=10.0, y=10.0, mag_r=18.0),
                SourceRecord("faint0", "c0", 20.0 / 3600.0, 20.0 / 3600.0, "star", x=20.0, y=20.0, mag_r=22.0),
                SourceRecord("faint1", "c0", 80.0 / 3600.0, 80.0 / 3600.0, "star", x=80.0, y=80.0, mag_r=22.0),
            ]
            predictions = [
                PredictionRecord("p0", "c0", 10.2 / 3600.0, 10.1 / 3600.0, "star", 0.9, x=10.2, y=10.1),
                PredictionRecord("p1", "c0", 20.1 / 3600.0, 20.1 / 3600.0, "star", 0.8, x=20.1, y=20.1),
                PredictionRecord("p2", "c0", 120.0 / 3600.0, 120.0 / 3600.0, "star", 0.7, x=120.0, y=120.0),
            ]
            write_source_catalog(truth, validation / "truth_injected.csv")
            write_prediction_catalog(predictions, validation / "predictions_candidates.csv")
            validation.joinpath("report.json").write_text(
                json.dumps(
                    {
                        "run_id": "synthetic_validation",
                        "parameters": {"threshold": 0.5, "match_radius_pixels": 2.0},
                        "outputs": {
                            "truth_injected": str(validation / "truth_injected.csv"),
                            "predictions": str(validation / "predictions_candidates.csv"),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            payload = build_synthetic_faint_recovery_diagnostic(
                validation_dir=validation,
                output_dir=root / "diagnostic",
                max_thresholds=8,
            )
            self.assertTrue(Path(payload["outputs"]["false_negatives"]).exists())

            faint = payload["metrics"]["by_mag_r"]["[21.5,22.5)"]["fixed_threshold"]
            self.assertEqual(faint["recall"], 0.5)
            self.assertEqual(faint["false_negatives"], 1.0)
            self.assertEqual(faint["false_negative_nearest_candidate"]["no_candidate_within_match_radius"], 1.0)
            self.assertEqual(payload["next_actions"][0]["action"], "rerun_synthetic_low_candidate_floor_diagnostic")

    def test_mag_bin_and_markdown_helpers(self):
        record = SourceRecord("s", "c", 0.0, 0.0, "star", mag_r=22.0)
        payload = {
            "validation_run_dir": "run",
            "threshold": 0.2,
            "match_radius_pixels": 2.0,
            "claim_gate": {"status": "engineering_check"},
            "metrics": {
                "fixed_threshold_detection": {"precision": 0.5, "recall": 0.25, "f1": 0.33},
                "by_mag_r": {
                    "[21.5,22.5)": {
                        "n": 4.0,
                        "fixed_threshold": {
                            "recall": 0.25,
                            "false_negatives": 3.0,
                            "false_negative_nearest_candidate": {"no_candidate_within_match_radius": 3.0},
                        },
                    }
                },
            },
            "findings": ["low faint recall"],
            "next_actions": [{"action": "rerun", "reason": "candidate floor"}],
        }

        self.assertEqual(mag_bin_for_record(record), "[21.5,22.5)")
        text = render_synthetic_diagnostic_markdown(payload)
        self.assertIn("Synthetic Faint-Recovery Diagnostic", text)
        self.assertIn("low faint recall", text)

    def test_heatmap_response_helpers_summarize_local_peaks(self):
        scores = np.zeros((32, 32), dtype=np.float32)
        scores[10, 10] = 0.8
        scores[22, 22] = 0.03

        bright = heatmap_response_row(
            scores,
            cutout_id="c0",
            source_id="bright",
            x=10.0,
            y=10.0,
            mag_r=18.0,
            kind="psf_star",
            fixed_threshold=0.2,
            low_floor=0.05,
            match_radius_pixels=2.0,
            search_radius_pixels=4.0,
        )
        faint = heatmap_response_row(
            scores,
            cutout_id="c0",
            source_id="faint",
            x=22.0,
            y=22.0,
            mag_r=22.0,
            kind="sersic_galaxy",
            fixed_threshold=0.2,
            low_floor=0.05,
            match_radius_pixels=2.0,
            search_radius_pixels=4.0,
        )
        by_mag = heatmap_response_by_mag([bright, faint])
        by_mag_label = heatmap_response_by_mag_and_label([bright, faint])
        findings, next_actions = heatmap_response_findings(by_mag, low_floor=0.05)
        markdown = render_synthetic_heatmap_diagnostic_markdown(
            {
                "validation_run_dir": "run",
                "threshold": 0.2,
                "low_floor": 0.05,
                "match_radius_pixels": 2.0,
                "search_radius_pixels": 4.0,
                "claim_gate": {"status": "engineering_check"},
                "metrics": {"by_mag_r": by_mag, "by_mag_r_and_label": by_mag_label},
                "findings": findings,
                "next_actions": next_actions,
            }
        )

        self.assertAlmostEqual(best_local_score(scores, x=10.0, y=10.0, radius_pixels=2.0)["score"], 0.8)
        self.assertEqual(by_mag["[17.5,19.0)"]["best_within_match_radius_and_low_floor"], 1.0)
        self.assertEqual(by_mag["[21.5,22.5)"]["best_within_match_radius_and_low_floor"], 0.0)
        self.assertEqual(by_mag_label["[17.5,19.0)"]["star"]["best_within_match_radius_and_low_floor"], 1.0)
        self.assertEqual(by_mag_label["[21.5,22.5)"]["galaxy"]["best_within_match_radius_and_low_floor"], 0.0)
        self.assertEqual(next_actions[0]["action"], "increase_faint_source_signal_or_loss_weight_diagnostic")
        self.assertIn("Synthetic Heatmap-Response Diagnostic", markdown)

    def test_morphology_grid_balances_star_controls_and_galaxy_radii(self):
        specs = morphology_injection_specs_for_background(
            random.Random(42),
            cutout_id="c0",
            shape=(128, 128),
        )

        self.assertEqual(len(specs), 8)
        labels = ["star" if spec.kind == "psf_star" else "galaxy" for spec in specs]
        self.assertEqual(labels.count("star"), 2)
        self.assertEqual(labels.count("galaxy"), 6)
        self.assertEqual(
            sorted((float(spec.metadata["mag_r"]), spec.kind, float(spec.radius)) for spec in specs),
            [
                (21.0, "psf_star", 1.3),
                (21.0, "sersic_galaxy", 1.3),
                (21.0, "sersic_galaxy", 2.4),
                (21.0, "sersic_galaxy", 3.6),
                (22.0, "psf_star", 1.3),
                (22.0, "sersic_galaxy", 1.3),
                (22.0, "sersic_galaxy", 2.4),
                (22.0, "sersic_galaxy", 3.6),
            ],
        )

    def test_morphology_condition_summaries_drive_next_action(self):
        rows = [
            {
                "condition": "mag22_star_r1.3",
                "source_id": "s",
                "cutout_id": "c0",
                "mag_r": 22.0,
                "mag_r_bin": "[21.5,22.5)",
                "kind": "psf_star",
                "label": "star",
                "radius": 1.3,
                "ellipticity": 0.0,
                "center_score": 0.8,
                "best_score": 0.9,
                "best_distance_pixels": 0.5,
                "center_score_ge_low_floor": True,
                "best_score_ge_low_floor": True,
                "best_score_ge_fixed_threshold": True,
                "best_within_match_radius": True,
                "best_within_match_radius_and_low_floor": True,
                "best_within_match_radius_and_fixed_threshold": True,
            },
            {
                "condition": "mag22_galaxy_r2.4",
                "source_id": "g",
                "cutout_id": "c0",
                "mag_r": 22.0,
                "mag_r_bin": "[21.5,22.5)",
                "kind": "sersic_galaxy",
                "label": "galaxy",
                "radius": 2.4,
                "ellipticity": 0.25,
                "center_score": 0.01,
                "best_score": 0.04,
                "best_distance_pixels": 1.0,
                "center_score_ge_low_floor": False,
                "best_score_ge_low_floor": False,
                "best_score_ge_fixed_threshold": False,
                "best_within_match_radius": True,
                "best_within_match_radius_and_low_floor": False,
                "best_within_match_radius_and_fixed_threshold": False,
            },
            {
                "condition": "mag22_galaxy_r3.6",
                "source_id": "g2",
                "cutout_id": "c0",
                "mag_r": 22.0,
                "mag_r_bin": "[21.5,22.5)",
                "kind": "sersic_galaxy",
                "label": "galaxy",
                "radius": 3.6,
                "ellipticity": 0.25,
                "center_score": 0.01,
                "best_score": 0.04,
                "best_distance_pixels": 1.0,
                "center_score_ge_low_floor": False,
                "best_score_ge_low_floor": False,
                "best_score_ge_fixed_threshold": False,
                "best_within_match_radius": True,
                "best_within_match_radius_and_low_floor": False,
                "best_within_match_radius_and_fixed_threshold": False,
            },
        ]
        truth = [
            SourceRecord("s", "c0", 10.0 / 3600.0, 10.0 / 3600.0, "star", mag_r=22.0, size=1.3),
            SourceRecord("g", "c0", 20.0 / 3600.0, 20.0 / 3600.0, "galaxy", mag_r=22.0, size=2.4),
        ]
        predictions = [PredictionRecord("p", "c0", 10.0 / 3600.0, 10.0 / 3600.0, "star", 0.9)]
        matches = __import__("sdss_point_benchmark.matching", fromlist=["match_catalogs"]).match_catalogs(
            truth,
            predictions,
            radius_arcsec=2.0,
        )

        response = morphology_response_by_condition(rows)
        recall = morphology_recall_by_condition(truth, matches)
        findings, next_actions = morphology_diagnostic_findings(response, recall)
        markdown = render_synthetic_morphology_diagnostic_markdown(
            {
                "validation_run_dir": "run",
                "threshold": 0.2,
                "low_floor": 0.05,
                "match_radius_pixels": 2.0,
                "search_radius_pixels": 8.0,
                "claim_gate": {"status": "engineering_check"},
                "metrics": {
                    "fixed_threshold_detection": {"precision": 1.0, "recall": 0.5, "f1": 0.667},
                    "average_precision": {"ap": 0.5},
                    "response_by_condition": response,
                    "recall_by_condition": recall,
                },
                "findings": findings,
                "next_actions": next_actions,
            }
        )

        self.assertEqual(response["mag22_star_r1.3"]["best_within_match_radius_and_low_floor"], 1.0)
        self.assertEqual(response["mag22_galaxy_r2.4"]["best_within_match_radius_and_low_floor"], 0.0)
        self.assertEqual(recall["mag22_star_r1.3"]["recall"], 1.0)
        self.assertEqual(recall["mag22_galaxy_r2.4"]["recall"], 0.0)
        self.assertEqual(next_actions[0]["action"], "design_surface_brightness_or_extended_profile_rescue")
        self.assertIn("Synthetic Morphology-Response Diagnostic", markdown)

    def test_cli_dispatches_morphology_diagnostic_arguments(self):
        with patch(
            "sdss_point_benchmark.synthetic_diagnostics.build_synthetic_morphology_diagnostic",
            return_value={},
        ) as build:
            code = main(
                [
                    "research-synthetic-morphology-diagnostic",
                    "--validation-dir",
                    "validation",
                    "--output-dir",
                    "out",
                    "--device",
                    "cuda:0",
                    "--search-radius-pixels",
                    "6",
                    "--low-floor",
                    "0.04",
                    "--shard-cache-size",
                    "3",
                ]
            )

        self.assertEqual(code, 0)
        build.assert_called_once_with(
            validation_dir="validation",
            output_dir="out",
            device="cuda:0",
            search_radius_pixels=6.0,
            low_floor=0.04,
            shard_cache_size=3,
        )


if __name__ == "__main__":
    unittest.main()
