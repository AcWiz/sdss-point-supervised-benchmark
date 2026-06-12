import unittest

from sdss_point_benchmark.synthetic_validation import (
    faintest_mag_bin,
    infer_source_loss_variant,
    infer_source_variant_id,
    injection_specs_for_background,
    injected_recall_by_mag_and_label,
    local_pixel_to_radec,
    mag_to_flux,
    parse_mag_bin_upper,
    render_synthetic_validation_markdown,
    source_records_with_injection_metadata,
    synthetic_claim_gate,
    synthetic_next_actions,
)
from sdss_point_benchmark.schema import SourceRecord
from sdss_point_benchmark.synthetic import InjectionSpec


class SyntheticValidationTests(unittest.TestCase):
    def test_injection_specs_are_reproducible_and_inside_cutout(self):
        import random

        rng_a = random.Random(7)
        rng_b = random.Random(7)

        first = injection_specs_for_background(rng_a, cutout_id="c0", count=4, shape=(128, 128))
        second = injection_specs_for_background(rng_b, cutout_id="c0", count=4, shape=(128, 128))

        self.assertEqual([row.source_id for row in first], [row.source_id for row in second])
        self.assertEqual([row.x for row in first], [row.x for row in second])
        self.assertEqual([row.metadata["mag_r"] for row in first], [18.0, 20.0, 21.0, 22.0])
        self.assertTrue(all(12.0 <= row.x <= 115.0 for row in first))
        self.assertTrue(all(12.0 <= row.y <= 115.0 for row in first))
        self.assertGreater(mag_to_flux(18.0), mag_to_flux(22.0))

    def test_injection_specs_balance_magnitude_and_source_type_when_possible(self):
        import random

        specs = injection_specs_for_background(random.Random(7), cutout_id="c0", count=8, shape=(128, 128))

        combos = [(spec.metadata["mag_r"], spec.kind) for spec in specs]
        self.assertEqual(
            combos,
            [
                (18.0, "psf_star"),
                (18.0, "sersic_galaxy"),
                (20.0, "psf_star"),
                (20.0, "sersic_galaxy"),
                (21.0, "psf_star"),
                (21.0, "sersic_galaxy"),
                (22.0, "psf_star"),
                (22.0, "sersic_galaxy"),
            ],
        )

    def test_render_markdown_includes_injected_metrics(self):
        payload = {
            "checkpoint": "checkpoint.pt",
            "dataset": "dataset",
            "seed": 42,
            "claim_gate": {"status": "candidate_evidence"},
            "metrics": {
                "injected_detection": {"precision": 0.5, "recall": 0.75, "f1": 0.6},
                "injected_average_precision": {"ap": 0.7},
                "all_truth_detection": {"precision": 0.4, "recall": 0.6, "f1": 0.48},
                "injected_recall_by_mag_r": {"[17.5,19.0)": {"n": 2.0, "recall": 1.0}},
                "injected_recall_by_mag_r_and_label": {
                    "[17.5,19.0)": {"star": {"n": 1.0, "recall": 1.0}},
                },
            },
        }

        text = render_synthetic_validation_markdown(payload)

        self.assertIn("Synthetic Injection Validation", text)
        self.assertIn("candidate_evidence", text)
        self.assertIn("Recall: 0.75", text)
        self.assertIn("[17.5,19.0)", text)
        self.assertIn("[17.5,19.0) star", text)

    def test_source_records_are_enriched_with_injected_magnitude_and_pixel_position(self):
        records = [SourceRecord("s1", "c1", 10.0, 11.0, "star", flux_r=100.0)]
        specs = [InjectionSpec("s1", 10.0, 11.0, {"r": 100.0}, "psf_star", metadata={"mag_r": 20.0})]

        enriched = source_records_with_injection_metadata(records, specs)

        self.assertEqual(enriched[0].mag_r, 20.0)
        self.assertAlmostEqual(enriched[0].ra, 10.0 / 3600.0)
        self.assertAlmostEqual(enriched[0].dec, 11.0 / 3600.0)
        self.assertEqual(enriched[0].x, 10.0)
        self.assertEqual(enriched[0].y, 11.0)
        self.assertEqual(enriched[0].label_quality, "synthetic")

    def test_local_pixel_to_radec_uses_one_arcsec_per_pixel(self):
        self.assertEqual(local_pixel_to_radec(3.0, 4.0), (3.0 / 3600.0, 4.0 / 3600.0))

    def test_synthetic_metadata_helpers_identify_source_variant_and_faint_bin(self):
        source_run = "sdss_point_catalog_v2_pilot_agent_loss_diagnosis_v1_ablation_center_only_unet_lite_e50_seed42"
        by_mag = {
            "[17.5,19.0)": {"n": 4.0, "recall": 1.0},
            "[21.5,22.5)": {"n": 4.0, "recall": 0.25},
        }

        self.assertEqual(
            infer_source_variant_id(source_run),
            "ablation_center_only_unet_lite_e50_seed42",
        )
        self.assertEqual(infer_source_loss_variant("ablation_center_only_unet_lite_e50_seed42"), "center_only")
        self.assertEqual(parse_mag_bin_upper("[21.5,22.5)"), 22.5)
        self.assertEqual(faintest_mag_bin(by_mag)["bin"], "[21.5,22.5)")
        self.assertEqual(synthetic_next_actions(by_mag)[0]["action"], "diagnose_faint_synthetic_recovery")

    def test_synthetic_claim_gate_marks_smoke_as_engineering_check(self):
        gate = synthetic_claim_gate(
            output_dir=__import__("pathlib").Path("reports/research_runs/synthetic_smoke"),
            num_backgrounds=2,
            injected_truth_count=4,
            thresholded_prediction_count=5,
            injected_detection={"precision": 0.2, "recall": 0.75},
            injected_average_precision={"ap": 0.2},
        )

        self.assertEqual(gate["status"], "engineering_check")
        self.assertIn("smoke synthetic validation output", gate["reasons"])

    def test_injected_recall_by_mag_and_label(self):
        from sdss_point_benchmark.matching import CatalogMatch, MatchResult

        records = [
            SourceRecord("s1", "c", 0.0, 0.0, "star", mag_r=22.0),
            SourceRecord("g1", "c", 0.0, 0.0, "galaxy", mag_r=22.0),
        ]
        matches = MatchResult(
            matches=(CatalogMatch("s1", "p1", 0.0),),
            unmatched_truth_ids=["g1"],
            unmatched_prediction_ids=[],
            truth_by_id={record.source_id: record for record in records},
            prediction_by_id={},
        )

        rows = injected_recall_by_mag_and_label(records, matches)

        self.assertEqual(rows["[21.5,22.5)"]["star"]["recall"], 1.0)
        self.assertEqual(rows["[21.5,22.5)"]["galaxy"]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
