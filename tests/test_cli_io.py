import json
import tempfile
import unittest
from pathlib import Path

from sdss_point_benchmark.cli import main
from sdss_point_benchmark.io import load_prediction_catalog, load_source_catalog


class CliIoTests(unittest.TestCase):
    def test_load_source_catalog_reads_required_and_optional_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.csv"
            path.write_text(
                "source_id,cutout_id,ra,dec,x,y,label,mag_r,snr,psf_fwhm,nearest_neighbor_arcsec\n"
                "s1,c1,10.0,1.0,15.5,16.5,star,19.3,12.0,1.4,2.2\n",
                encoding="utf-8",
            )

            records = load_source_catalog(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_id, "s1")
        self.assertEqual(records[0].x, 15.5)
        self.assertEqual(records[0].y, 16.5)
        self.assertEqual(records[0].mag_r, 19.3)
        self.assertEqual(records[0].snr, 12.0)
        self.assertEqual(records[0].psf_fwhm, 1.4)
        self.assertEqual(records[0].nearest_neighbor_arcsec, 2.2)

    def test_load_prediction_catalog_reads_scores_and_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            path.write_text(
                "prediction_id,cutout_id,ra,dec,x,y,label,score,mag_r,flux_r,size,ellipticity\n"
                "p1,c1,10.0,1.0,15.7,16.4,galaxy,0.82,20.1,33.0,1.9,0.2\n",
                encoding="utf-8",
            )

            records = load_prediction_catalog(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].prediction_id, "p1")
        self.assertEqual(records[0].score, 0.82)
        self.assertEqual(records[0].x, 15.7)
        self.assertEqual(records[0].flux_r, 33.0)

    def test_cli_split_writes_protocol_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog.csv"
            output = Path(tmp) / "split.json"
            catalog.write_text(
                "source_id,cutout_id,ra,dec,label,mag_r\n"
                "s1,c1,1,1,star,19\n"
                "s2,c2,12,1,galaxy,20\n"
                "s3,c3,24,1,star,21\n"
                "s4,c4,36,1,galaxy,22\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "split",
                    "--catalog",
                    str(catalog),
                    "--output",
                    str(output),
                    "--ra-bin-deg",
                    "10",
                    "--dec-bin-deg",
                    "5",
                    "--seed",
                    "3",
                ]
            )

            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["protocol"], "sdss-point-supervised-v1")
        self.assertEqual(payload["region_mode"], "sky-bin")
        self.assertEqual(set(payload["splits"]), {"train", "val", "test"})
        assigned = set(payload["splits"]["train"]) | set(payload["splits"]["val"]) | set(payload["splits"]["test"])
        self.assertEqual(assigned, {"s1", "s2", "s3", "s4"})

    def test_cli_split_can_write_catalog_region_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "catalog.csv"
            output = Path(tmp) / "split.json"
            catalog.write_text(
                "source_id,cutout_id,ra,dec,label,mag_r,region_id\n"
                "s1,c1,1,1,star,19,field-a\n"
                "s2,c2,12,1,galaxy,20,field-a\n"
                "s3,c3,24,1,star,21,field-a\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "split",
                    "--catalog",
                    str(catalog),
                    "--output",
                    str(output),
                    "--region-mode",
                    "catalog-region",
                    "--seed",
                    "3",
                ]
            )

            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["region_mode"], "catalog-region")
        all_regions = set(payload["regions"]["train"]) | set(payload["regions"]["val"]) | set(payload["regions"]["test"])
        self.assertEqual(all_regions, {"field-a"})

    def test_cli_evaluate_writes_metric_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "truth.csv"
            predictions = Path(tmp) / "predictions.csv"
            output = Path(tmp) / "metrics.json"
            truth.write_text(
                "source_id,cutout_id,ra,dec,label,mag_r,snr,nearest_neighbor_arcsec\n"
                "t1,c1,10.0,0.0,star,19.0,20,3.0\n"
                "t2,c1,20.0,0.0,galaxy,21.0,6,0.8\n",
                encoding="utf-8",
            )
            predictions.write_text(
                "prediction_id,cutout_id,ra,dec,label,score,mag_r\n"
                "p1,c1,10.0001,0.0,star,0.9,19.1\n"
                "p2,c1,21.0,0.0,galaxy,0.7,21.5\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "evaluate",
                    "--truth",
                    str(truth),
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(output),
                    "--radius-arcsec",
                    "1.0",
                    "--band",
                    "r",
                    "--seeing-aware",
                ]
            )

            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["protocol"], "sdss-point-supervised-v1")
        self.assertAlmostEqual(payload["metrics"]["detection"]["precision"], 0.5)
        self.assertAlmostEqual(payload["metrics"]["detection"]["recall"], 0.5)
        self.assertIn("average_precision", payload["metrics"])
        self.assertIn("photometry_r", payload["metrics"])
        self.assertTrue(payload["matching"]["seeing_aware"])
        self.assertEqual(payload["counts"]["truth"], 2)
        self.assertEqual(payload["counts"]["predictions"], 2)

    def test_cli_evaluate_filters_by_score_and_prediction_cutouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "truth.csv"
            predictions = Path(tmp) / "predictions.csv"
            output = Path(tmp) / "metrics.json"
            truth.write_text(
                "source_id,cutout_id,ra,dec,label,mag_r\n"
                "t1,c1,10.0,0.0,star,19.0\n"
                "t2,c1,10.1,0.0,galaxy,20.0\n"
                "t3,c2,20.0,0.0,star,18.0\n",
                encoding="utf-8",
            )
            predictions.write_text(
                "prediction_id,cutout_id,ra,dec,label,score,mag_r\n"
                "p1,c1,10.0001,0.0,star,0.9,19.1\n"
                "p2,c1,10.1,0.0,galaxy,0.2,20.1\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "evaluate",
                    "--truth",
                    str(truth),
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(output),
                    "--radius-arcsec",
                    "1.0",
                    "--min-score",
                    "0.5",
                    "--filter-truth-to-prediction-cutouts",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["counts"]["truth"], 2)
        self.assertEqual(payload["counts"]["predictions"], 1)
        self.assertAlmostEqual(payload["metrics"]["detection"]["precision"], 1.0)
        self.assertAlmostEqual(payload["metrics"]["detection"]["recall"], 0.5)

    def test_cli_sweep_thresholds_writes_best_validation_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            truth = Path(tmp) / "truth.csv"
            predictions = Path(tmp) / "predictions.csv"
            output = Path(tmp) / "thresholds.json"
            truth.write_text(
                "source_id,cutout_id,ra,dec,label,mag_r\n"
                "t1,c1,10.0,0.0,star,19.0\n"
                "t2,c1,20.0,0.0,galaxy,20.0\n",
                encoding="utf-8",
            )
            predictions.write_text(
                "prediction_id,cutout_id,ra,dec,label,score,mag_r\n"
                "p1,c1,10.0001,0.0,star,0.9,19.1\n"
                "p2,c1,40.0,0.0,galaxy,0.8,20.1\n"
                "p3,c1,20.0001,0.0,galaxy,0.7,20.1\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "sweep-thresholds",
                    "--truth",
                    str(truth),
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(output),
                    "--radius-arcsec",
                    "1.0",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["best_threshold"], 0.7)
        self.assertAlmostEqual(payload["best_metrics"]["f1"], 0.8)
        self.assertEqual([row["threshold"] for row in payload["thresholds"]], [0.9, 0.8, 0.7])


if __name__ == "__main__":
    unittest.main()
