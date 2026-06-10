import unittest

from sdss_point_benchmark.matching import match_catalogs, match_catalogs_seeing_aware
from sdss_point_benchmark.metrics import (
    astrometry_metrics,
    classification_metrics,
    detection_average_precision,
    detection_metrics,
    detection_score_curve,
    photometry_metrics,
)
from sdss_point_benchmark.schema import PredictionRecord, SourceRecord


class MatchingMetricTests(unittest.TestCase):
    def test_match_catalogs_uses_nearest_prediction_within_radius(self):
        truth = [
            SourceRecord("t1", "c1", 10.0, 0.0, "star", mag_r=19.0),
            SourceRecord("t2", "c1", 10.001, 0.0, "galaxy", mag_r=21.0),
        ]
        predictions = [
            PredictionRecord("p1", "c1", 10.00005, 0.0, "star", score=0.9, mag_r=19.1),
            PredictionRecord("p2", "c1", 10.002, 0.0, "galaxy", score=0.7, mag_r=21.4),
            PredictionRecord("p3", "c1", 11.0, 0.0, "star", score=0.8, mag_r=18.0),
        ]

        result = match_catalogs(truth, predictions, radius_arcsec=0.5)

        self.assertEqual([(m.truth_id, m.prediction_id) for m in result.matches], [("t1", "p1")])
        self.assertEqual(result.unmatched_truth_ids, ["t2"])
        self.assertEqual(set(result.unmatched_prediction_ids), {"p2", "p3"})

    def test_detection_and_measurement_metrics_are_computed_from_matches(self):
        truth = [
            SourceRecord("t1", "c1", 10.0, 0.0, "star", mag_r=19.0),
            SourceRecord("t2", "c1", 20.0, 0.0, "galaxy", mag_r=21.0),
        ]
        predictions = [
            PredictionRecord("p1", "c1", 10.0001, 0.0, "star", score=0.9, mag_r=19.2),
            PredictionRecord("p2", "c1", 20.0001, 0.0, "star", score=0.8, mag_r=20.7),
            PredictionRecord("p3", "c1", 21.0, 0.0, "galaxy", score=0.4, mag_r=22.0),
        ]
        matches = match_catalogs(truth, predictions, radius_arcsec=1.0)

        det = detection_metrics(matches)
        self.assertAlmostEqual(det["precision"], 2 / 3)
        self.assertAlmostEqual(det["recall"], 1.0)
        self.assertAlmostEqual(det["f1"], 0.8)

        cls = classification_metrics(truth, predictions, matches)
        self.assertAlmostEqual(cls["accuracy"], 0.5)
        self.assertAlmostEqual(cls["macro_f1"], 1 / 3)

        photo = photometry_metrics(truth, predictions, matches, band="r")
        self.assertAlmostEqual(photo["bias_mag"], -0.05)
        self.assertAlmostEqual(photo["outlier_rate"], 0.0)

        astro = astrometry_metrics(matches)
        self.assertGreater(astro["centroid_rmse_arcsec"], 0.0)

    def test_seeing_aware_matching_uses_source_psf_fwhm_cap(self):
        truth = [
            SourceRecord("t1", "c1", 10.0, 0.0, "star", mag_r=19.0, psf_fwhm=0.6),
            SourceRecord("t2", "c1", 20.0, 0.0, "galaxy", mag_r=21.0, psf_fwhm=3.0),
        ]
        predictions = [
            PredictionRecord("p1", "c1", 10.0002, 0.0, "star", score=0.9),
            PredictionRecord("p2", "c1", 20.0002, 0.0, "galaxy", score=0.8),
        ]

        matches = match_catalogs_seeing_aware(
            truth,
            predictions,
            max_radius_arcsec=1.0,
            psf_fraction=0.5,
        )

        self.assertEqual([(m.truth_id, m.prediction_id) for m in matches.matches], [("t2", "p2")])

    def test_detection_average_precision_sweeps_prediction_scores(self):
        truth = [
            SourceRecord("t1", "c1", 10.0, 0.0, "star"),
            SourceRecord("t2", "c1", 20.0, 0.0, "star"),
        ]
        predictions = [
            PredictionRecord("p1", "c1", 10.0001, 0.0, "star", score=0.9),
            PredictionRecord("p2", "c1", 50.0, 0.0, "star", score=0.8),
            PredictionRecord("p3", "c1", 20.0001, 0.0, "star", score=0.7),
        ]

        ap = detection_average_precision(truth, predictions, radius_arcsec=1.0)

        self.assertGreater(ap["ap"], 0.8)
        self.assertEqual(ap["n_thresholds"], 3.0)
        self.assertAlmostEqual(ap["best_f1"], 0.8)

    def test_detection_score_curve_reuses_matching_candidates(self):
        truth = [
            SourceRecord("t1", "c1", 10.0, 0.0, "star"),
            SourceRecord("t2", "c1", 20.0, 0.0, "star", psf_fwhm=0.6),
            SourceRecord("t3", "c1", 30.0, 0.0, "star", psf_fwhm=3.0),
        ]
        predictions = [
            PredictionRecord("p1", "c1", 10.0001, 0.0, "star", score=0.9),
            PredictionRecord("p2", "c1", 20.0002, 0.0, "star", score=0.8),
            PredictionRecord("p3", "c1", 30.0002, 0.0, "star", score=0.7),
        ]

        curve = detection_score_curve(
            truth,
            predictions,
            max_radius_arcsec=1.0,
            psf_fraction=0.5,
        )

        self.assertEqual(curve["best_threshold"], 0.7)
        self.assertEqual(curve["average_precision"]["n_thresholds"], 3.0)
        self.assertEqual(curve["best_metrics"]["tp"], 2.0)
        self.assertEqual(curve["best_metrics"]["fn"], 1.0)


if __name__ == "__main__":
    unittest.main()
