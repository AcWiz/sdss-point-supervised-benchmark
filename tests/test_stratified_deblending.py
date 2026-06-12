import unittest

from sdss_point_benchmark.matching import match_catalogs
from sdss_point_benchmark.metrics import deblending_metrics, stratified_detection_report
from sdss_point_benchmark.schema import PredictionRecord, SourceRecord


class StratifiedDeblendingTests(unittest.TestCase):
    def test_stratified_detection_report_reports_multiple_fields(self):
        truth = [
            SourceRecord("t1", "c1", 10.0, 0.0, "star", mag_r=18.5, snr=30.0),
            SourceRecord("t2", "c1", 20.0, 0.0, "galaxy", mag_r=21.5, snr=4.0),
            SourceRecord("t3", "c1", 30.0, 0.0, "galaxy", mag_r=22.5, snr=8.0),
        ]
        predictions = [
            PredictionRecord("p1", "c1", 10.0001, 0.0, "star", 0.9),
            PredictionRecord("p2", "c1", 30.0001, 0.0, "galaxy", 0.8),
        ]
        matches = match_catalogs(truth, predictions, radius_arcsec=1.0)

        report = stratified_detection_report(
            truth,
            matches,
            {"mag_r": [18.0, 20.0, 22.0, 24.0], "snr": [0.0, 5.0, 10.0, 100.0]},
        )

        self.assertEqual(report["mag_r"]["status"], "available")
        self.assertAlmostEqual(report["mag_r"]["bins"]["[18.0,20.0)"]["recall"], 1.0)
        self.assertAlmostEqual(report["mag_r"]["bins"]["[20.0,22.0)"]["recall"], 0.0)
        self.assertAlmostEqual(report["snr"]["bins"]["[5.0,10.0)"]["recall"], 1.0)

    def test_stratified_detection_marks_missing_fields_unavailable(self):
        truth = [SourceRecord("t1", "c1", 10.0, 0.0, "star", mag_r=18.5)]
        predictions = [PredictionRecord("p1", "c1", 10.0, 0.0, "star", 0.9)]
        matches = match_catalogs(truth, predictions, radius_arcsec=1.0)

        report = stratified_detection_report(truth, matches, {"seeing": [0.0, 1.0, 2.0]})

        self.assertEqual(report["seeing"]["status"], "unavailable")
        self.assertEqual(report["seeing"]["bins"], {})

    def test_deblending_metrics_focus_on_close_pairs_and_flux_error(self):
        truth = [
            SourceRecord(
                "t1",
                "c1",
                10.0,
                0.0,
                "star",
                flux_r=100.0,
                nearest_neighbor_arcsec=0.7,
            ),
            SourceRecord(
                "t2",
                "c1",
                10.00015,
                0.0,
                "star",
                flux_r=50.0,
                nearest_neighbor_arcsec=0.7,
            ),
            SourceRecord("t3", "c1", 20.0, 0.0, "galaxy", flux_r=80.0, nearest_neighbor_arcsec=5.0),
        ]
        predictions = [
            PredictionRecord("p1", "c1", 10.00002, 0.0, "star", 0.9, flux_r=90.0),
            PredictionRecord("p2", "c1", 20.00002, 0.0, "galaxy", 0.8, flux_r=84.0),
        ]
        matches = match_catalogs(truth, predictions, radius_arcsec=1.0)

        metrics = deblending_metrics(
            truth,
            predictions,
            matches,
            close_pair_arcsec=1.0,
            band="r",
        )

        self.assertEqual(metrics["close_pair_truth"], 2.0)
        self.assertEqual(metrics["close_pair_detected"], 1.0)
        self.assertAlmostEqual(metrics["close_pair_recall"], 0.5)
        self.assertAlmostEqual(metrics["missed_companion_rate"], 0.5)
        self.assertAlmostEqual(metrics["flux_attribution_mae"], 0.1)


if __name__ == "__main__":
    unittest.main()
