import json
import tempfile
import unittest
from pathlib import Path

from sdss_point_benchmark.evidence_audit import (
    bootstrap_paired_deltas,
    build_evidence_audit,
    derive_nearest_neighbor_arcsec,
    derive_source_density_per_cutout,
    load_evidence_audits,
    match_catalogs_for_options,
    render_evidence_audit_markdown,
    stratified_recall_audit,
)
from sdss_point_benchmark.research_program import build_research_board
from sdss_point_benchmark.io import write_prediction_catalog, write_source_catalog
from sdss_point_benchmark.schema import PredictionRecord, SourceRecord


class EvidenceAuditTests(unittest.TestCase):
    def test_derived_nearest_neighbor_and_density_bins_are_available(self):
        truth = [
            SourceRecord("t1", "c1", 10.0, 0.0, "star", x=0.0, y=0.0),
            SourceRecord("t2", "c1", 10.0001, 0.0, "star", x=3.0, y=4.0),
            SourceRecord("t3", "c2", 20.0, 0.0, "galaxy", x=1.0, y=1.0),
        ]
        predictions = [
            PredictionRecord("p1", "c1", 10.0, 0.0, "star", score=0.9),
            PredictionRecord("p2", "c2", 20.0, 0.0, "galaxy", score=0.8),
        ]
        matches = match_catalogs_for_options(
            truth,
            predictions,
            {"radius_arcsec": 1.0, "seeing_aware": False, "psf_fraction": 0.5},
        )
        derived = {
            "nearest_neighbor_arcsec": derive_nearest_neighbor_arcsec(truth, pixel_scale_arcsec=0.5),
            "source_density_per_cutout": derive_source_density_per_cutout(truth),
        }

        report = stratified_recall_audit(truth, matches, derived_maps=derived)

        self.assertAlmostEqual(derived["nearest_neighbor_arcsec"]["t1"], 2.5)
        self.assertEqual(report["nearest_neighbor_arcsec_derived"]["status"], "available")
        self.assertEqual(report["nearest_neighbor_arcsec_derived"]["bins"]["[2.0,4.0)"]["n"], 2.0)
        self.assertEqual(report["source_density_per_cutout"]["status"], "available")
        self.assertEqual(report["source_density_per_cutout"]["bins"]["[0.0,3.0)"]["n"], 3.0)

    def test_unavailable_snr_and_seeing_do_not_enter_claimable_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _write_run(
                root / "baseline",
                "baseline",
                predictions=[PredictionRecord("bp1", "c1", 10.0, 0.0, "star", score=0.7)],
                threshold=0.7,
            )
            target = _write_run(
                root / "target",
                "target",
                predictions=[PredictionRecord("tp1", "c1", 10.0, 0.0, "star", score=0.9)],
                threshold=0.9,
            )

            audit = build_evidence_audit(
                baseline_run_dir=baseline,
                target_run_dir=target,
                bootstrap_iterations=2,
                bootstrap_max_thresholds=2,
            )

        self.assertEqual(audit["strata"]["comparison"]["snr"]["status"], "unavailable")
        self.assertEqual(audit["strata"]["comparison"]["seeing"]["status"], "unavailable")
        self.assertNotIn("snr", audit["claim_summary"]["claimable_strata"])
        self.assertNotIn("seeing", audit["claim_summary"]["claimable_strata"])

    def test_bootstrap_comparison_is_reproducible_with_fixed_seed(self):
        truth = [
            SourceRecord("t1", "c1", 10.0, 0.0, "star"),
            SourceRecord("t2", "c2", 20.0, 0.0, "star"),
        ]
        baseline_predictions = [PredictionRecord("bp1", "c1", 10.0, 0.0, "star", score=0.8)]
        target_predictions = [
            PredictionRecord("tp1", "c1", 10.0, 0.0, "star", score=0.8),
            PredictionRecord("tp2", "c2", 20.0, 0.0, "star", score=0.7),
        ]

        first = bootstrap_paired_deltas(
            truth=truth,
            baseline_predictions=baseline_predictions,
            target_predictions=target_predictions,
            baseline_threshold=0.5,
            target_threshold=0.5,
            matching_options={"radius_arcsec": 1.0, "seeing_aware": False, "psf_fraction": 0.5},
            seed=42,
            iterations=10,
            max_thresholds=3,
        )
        second = bootstrap_paired_deltas(
            truth=truth,
            baseline_predictions=baseline_predictions,
            target_predictions=target_predictions,
            baseline_threshold=0.5,
            target_threshold=0.5,
            matching_options={"radius_arcsec": 1.0, "seeing_aware": False, "psf_fraction": 0.5},
            seed=42,
            iterations=10,
            max_thresholds=3,
        )

        self.assertEqual(first["delta_ci"], second["delta_ci"])
        self.assertGreaterEqual(first["delta_ci"]["f1"]["median"], 0.0)

    def test_evidence_audit_uses_validation_threshold_not_test_best_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = _write_run(
                root / "baseline",
                "baseline",
                predictions=[
                    PredictionRecord("bp1", "c1", 10.0, 0.0, "star", score=0.9),
                    PredictionRecord("bp2", "c1", 11.0, 0.0, "star", score=0.8),
                ],
                threshold=0.95,
            )
            target = _write_run(
                root / "target",
                "target",
                predictions=[PredictionRecord("tp1", "c1", 10.0, 0.0, "star", score=0.9)],
                threshold=0.5,
            )

            audit = build_evidence_audit(
                baseline_run_dir=baseline,
                target_run_dir=target,
                bootstrap_iterations=1,
                bootstrap_max_thresholds=2,
            )

        self.assertFalse(audit["threshold_policy"]["uses_test_threshold_tuning"])
        self.assertEqual(audit["threshold_policy"]["baseline"]["selected_threshold"], 0.95)
        self.assertEqual(audit["aggregate"]["baseline"]["predictions"], 0.0)
        self.assertEqual(audit["aggregate"]["baseline"]["recall"], 0.0)

    def test_load_evidence_audits_includes_loss_diagnosis_audit_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_audit_file(root / "e50_evidence_audit.json", "baseline_e50", "unet_lite_e50", 0.1)
            _write_audit_file(
                root / "loss_diagnosis_center_vs_psf_e20_audit.json",
                "center_only_e20",
                "center_psf_e20",
                -0.02,
            )
            (root / "not_evidence_audit.json").write_text(json.dumps({"audit": "diagnosis"}) + "\n", encoding="utf-8")

            audits = load_evidence_audits(root)
            board = build_research_board(root)

        self.assertEqual([Path(audit["_audit_path"]).name for audit in audits], [
            "e50_evidence_audit.json",
            "loss_diagnosis_center_vs_psf_e20_audit.json",
        ])
        self.assertEqual(len(board["evidence_audits"]), 2)
        self.assertEqual(board["evidence_audits"][1]["target_label"], "center_psf_e20")

    def test_render_evidence_audit_markdown_uses_generic_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_audit_file(root / "loss_diagnosis_center_vs_psf_e20_audit.json", "center", "psf", -0.02)
            audit = load_evidence_audits(root)[0]

        markdown = render_evidence_audit_markdown(audit)

        self.assertTrue(markdown.startswith("# Evidence Audit"))
        self.assertNotIn("# E50 Evidence Audit", markdown)


def _write_run(run_dir: Path, run_id: str, *, predictions: list[PredictionRecord], threshold: float) -> Path:
    pilot = run_dir / "pilot_loop"
    pilot.mkdir(parents=True)
    truth = [
        SourceRecord(
            "t1",
            "c1",
            10.0,
            0.0,
            "star",
            x=5.0,
            y=5.0,
            mag_r=20.0,
            quality_flags="BLENDED;INTERP",
            label_quality="weak",
            label_weight=0.5,
        )
    ]
    write_source_catalog(truth, pilot / "truth_test.csv")
    write_prediction_catalog(predictions, pilot / "predictions_test_candidates.csv")
    (pilot / "summary.json").write_text(
        json.dumps(
            {
                "outputs": {
                    "test_truth": str(pilot / "truth_test.csv"),
                    "test_predictions": str(pilot / "predictions_test_candidates.csv"),
                },
                "decode": {"pixel_scale_arcsec": 0.5},
                "matching": {"radius_arcsec": 1.0, "seeing_aware": False, "psf_fraction": 0.5},
                "threshold_selection": {"source": "val", "best_threshold": threshold},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (pilot / "val_threshold_sweep.json").write_text(
        json.dumps({"best_threshold": threshold, "best_metrics": {"f1": 1.0}}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text(json.dumps({"run_id": run_id}) + "\n", encoding="utf-8")
    return run_dir


def _write_audit_file(path: Path, baseline_label: str, target_label: str, delta_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_label": baseline_label,
                "target_label": target_label,
                "threshold_policy": {"source": "validation", "uses_test_threshold_tuning": False},
                "aggregate": {"delta": {"f1": delta_f1, "ap": delta_f1, "recall": delta_f1}},
                "bootstrap": {"delta_ci": {}},
                "claim_summary": {"available_strata": ["mag_r"], "unavailable_strata": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
