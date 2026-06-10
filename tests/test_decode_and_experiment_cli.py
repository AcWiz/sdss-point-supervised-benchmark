import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from sdss_point_benchmark.baseline import BaselineOutput, decode_predictions
from sdss_point_benchmark.cli import main


class DecodeAndExperimentCliTests(unittest.TestCase):
    def test_decode_predictions_applies_nms_and_writes_catalog_records(self):
        center = torch.full((1, 1, 8, 8), -8.0)
        center[0, 0, 3, 4] = 8.0
        center[0, 0, 3, 5] = 7.0
        class_logits = torch.zeros(1, 2, 8, 8)
        class_logits[0, 1, 3, 4] = 4.0
        flux = torch.zeros(1, 5, 8, 8)
        flux[0, :, 3, 4] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        shape = torch.zeros(1, 3, 8, 8)

        records = decode_predictions(
            BaselineOutput(center, class_logits, flux, shape),
            cutout_ids=["cutout-a"],
            origin_radec=[(10.0, 20.0)],
            pixel_scale_arcsec=0.5,
            threshold=0.5,
            nms_radius=1,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].cutout_id, "cutout-a")
        self.assertEqual(records[0].label, "galaxy")
        self.assertAlmostEqual(records[0].score, float(torch.sigmoid(torch.tensor(8.0))), places=6)
        self.assertEqual(records[0].x, 4.0)
        self.assertEqual(records[0].y, 3.0)
        self.assertAlmostEqual(records[0].ra, 10.0 + 4.0 * 0.5 / 3600.0)
        self.assertAlmostEqual(records[0].dec, 20.0 + 3.0 * 0.5 / 3600.0)
        self.assertEqual(records[0].flux_r, 3.0)

    def test_cli_build_manifest_and_prepare_cutouts_write_reproducible_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sdss"
            root.mkdir()
            (root / "manifest_frames.csv").write_text(
                "run,rerun,camcol,field,band,status,bytes,path,url,error\n"
                "1302,301,2,100,u,exists,1,/data/u,,\n"
                "1302,301,2,100,g,exists,1,/data/g,,\n"
                "1302,301,2,100,r,exists,1,/data/r,,\n"
                "1302,301,2,100,i,exists,1,/data/i,,\n"
                "1302,301,2,100,z,exists,1,/data/z,,\n",
                encoding="utf-8",
            )
            (root / "manifest_catalogs.csv").write_text(
                "run,rerun,camcol,field,status,n_objects,path,error\n"
                "1302,301,2,100,downloaded,42,/data/catalog.csv,\n",
                encoding="utf-8",
            )
            manifest = Path(tmp) / "manifest.csv"
            cutouts = Path(tmp) / "cutouts.csv"
            source_catalog = Path(tmp) / "sources.csv"
            source_catalog.write_text(
                "source_id,cutout_id,ra,dec,x,y,label,mag_r\n"
                "s1,field-a,10,20,11,12,star,18\n",
                encoding="utf-8",
            )

            manifest_code = main(["build-manifest", "--data-root", str(root), "--output", str(manifest)])
            cutout_code = main(
                [
                    "prepare-cutouts",
                    "--catalog",
                    str(source_catalog),
                    "--output",
                    str(cutouts),
                    "--cutout-size",
                    "64",
                    "--limit",
                    "1",
                ]
            )

            with manifest.open(newline="", encoding="utf-8") as handle:
                manifest_rows = list(csv.DictReader(handle))
            with cutouts.open(newline="", encoding="utf-8") as handle:
                cutout_rows = list(csv.DictReader(handle))

        self.assertEqual(manifest_code, 0)
        self.assertEqual(cutout_code, 0)
        self.assertEqual(manifest_rows[0]["status"], "ready")
        self.assertEqual(cutout_rows[0]["cutout_id"], "field-a__s1")
        self.assertEqual(cutout_rows[0]["size_pixels"], "64")

    def test_cli_run_experiment_writes_protocol_report(self):
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
        self.assertEqual(payload["status"], "dry_run")
        self.assertEqual(payload["config"]["protocol"], "sdss-point-supervised-v1")
        self.assertEqual(payload["planned_experiments"], ["E0"])

    def test_cli_train_and_predict_dry_run_write_standard_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            train_report = Path(tmp) / "train.json"
            predictions = Path(tmp) / "predictions.csv"
            config.write_text('{"protocol": "sdss-point-supervised-v1"}', encoding="utf-8")

            train_code = main(["train", "--config", str(config), "--output", str(train_report), "--dry-run"])
            predict_code = main(["predict", "--checkpoint", "checkpoint.pt", "--output", str(predictions), "--dry-run"])

            train_payload = json.loads(train_report.read_text(encoding="utf-8"))
            with predictions.open(newline="", encoding="utf-8") as handle:
                prediction_rows = list(csv.DictReader(handle))
            prediction_header = predictions.read_text(encoding="utf-8").splitlines()[0]

        self.assertEqual(train_code, 0)
        self.assertEqual(predict_code, 0)
        self.assertEqual(train_payload["status"], "dry_run")
        self.assertEqual(train_payload["config"]["protocol"], "sdss-point-supervised-v1")
        self.assertEqual(prediction_rows, [])
        self.assertEqual(
            prediction_header,
            "prediction_id,cutout_id,ra,dec,label,score,x,y,mag_u,mag_g,mag_r,mag_i,mag_z,flux_u,flux_g,flux_r,flux_i,flux_z,size,ellipticity",
        )


if __name__ == "__main__":
    unittest.main()
