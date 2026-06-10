import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from astropy.io.fits import Header
from astropy.wcs import WCS

from sdss_point_benchmark.cli import _evaluate_payload, _select_split_truth_records, main
from sdss_point_benchmark.schema import PredictionRecord, SourceRecord
from sdss_point_benchmark.training import NpzCutoutDataset, ShardBatchSampler, train_model


class TrainingPipelineTests(unittest.TestCase):
    def test_npz_cutout_dataset_builds_point_supervision_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _write_tiny_dataset(Path(tmp) / "dataset")

            dataset = NpzCutoutDataset(dataset_dir, heatmap_sigma=1.0)
            sample = dataset[0]

        self.assertEqual(sample["image"].shape, (5, 8, 8))
        self.assertEqual(sample["target_heatmap"].shape, (1, 8, 8))
        self.assertGreater(float(sample["target_heatmap"][0, 4, 4]), 0.9)
        self.assertEqual(int(sample["target_class_map"][4, 4]), 0)
        self.assertEqual(int(sample["target_class_map"][5, 5]), -100)
        self.assertGreater(float(sample["target_flux"][:, 4, 4].sum()), 0.0)
        self.assertGreater(float(sample["inverse_variance"][:, 4, 4].sum()), 0.0)
        self.assertEqual(float(sample["target_flux"][:, 5, 5].sum()), 0.0)
        self.assertEqual(float(sample["inverse_variance"][:, 5, 5].sum()), 0.0)
        self.assertEqual(sample["cutout_id"], "field-a__center-a")
        self.assertEqual(sample["field_id"], "field-a")
        self.assertEqual(float(sample["center_x"]), 4.0)
        self.assertEqual(float(sample["center_y"]), 4.0)
        self.assertEqual(int(sample["origin_x"]), 100)
        self.assertEqual(int(sample["origin_y"]), 200)

    def test_npz_cutout_dataset_uses_weak_source_weight_for_photometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _write_tiny_dataset(Path(tmp) / "dataset")

            dataset = NpzCutoutDataset(dataset_dir, heatmap_sigma=1.0)
            sample = dataset[1]

        self.assertGreater(float(sample["target_heatmap"][0, 3, 3]), 0.9)
        self.assertEqual(int(sample["target_class_map"][3, 3]), 0)
        self.assertGreater(float(sample["target_flux"][:, 3, 3].sum()), 0.0)
        self.assertTrue(torch.allclose(sample["inverse_variance"][:, 3, 3], torch.full((5,), 0.5)))

    def test_npz_cutout_dataset_treats_missing_source_weight_as_full_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _write_tiny_dataset(Path(tmp) / "dataset", include_source_weight=False)

            dataset = NpzCutoutDataset(dataset_dir, heatmap_sigma=1.0)
            sample = dataset[0]

        self.assertGreater(float(sample["target_heatmap"][0, 5, 5]), 0.9)
        self.assertEqual(int(sample["target_class_map"][5, 5]), 1)
        self.assertGreater(float(sample["inverse_variance"][:, 5, 5].sum()), 0.0)

    def test_npz_cutout_dataset_filters_samples_by_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "protocol": "sdss-point-supervised-v1",
                        "splits": {
                            "train": ["center-b"],
                            "val": ["center-a"],
                            "test": [],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = NpzCutoutDataset(dataset_dir, split_path=split, split_name="train")
            sample = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(sample["cutout_id"], "field-a__center-b")

    def test_train_model_writes_checkpoint_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _write_tiny_dataset(Path(tmp) / "dataset")
            output_dir = Path(tmp) / "checkpoint"

            report = train_model(
                config_path=None,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=1,
                batch_size=1,
                base_channels=4,
                learning_rate=1e-3,
                device="cpu",
                seed=0,
            )

            checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
            report_payload = json.loads((output_dir / "training_report.json").read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "trained")
        self.assertEqual(report_payload["epochs"], 1)
        self.assertEqual(checkpoint["model"]["base_channels"], 4)
        self.assertEqual(checkpoint["model"]["model_arch"], "baseline")
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("samples_per_second", report_payload)

    def test_shard_batch_sampler_groups_samples_without_dropping(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _write_tiny_dataset(Path(tmp) / "dataset")
            dataset = NpzCutoutDataset(dataset_dir)

            sampler = ShardBatchSampler(dataset.samples, batch_size=1, seed=7, shuffle=True)
            batches = list(iter(sampler))

        self.assertEqual(sorted(index for batch in batches for index in batch), [0, 1])
        self.assertEqual(len(sampler), 2)

    def test_train_model_supports_unet_lite_and_shard_grouped_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _write_tiny_dataset(Path(tmp) / "dataset")
            output_dir = Path(tmp) / "checkpoint"

            report = train_model(
                config_path=None,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=1,
                batch_size=1,
                base_channels=4,
                model_arch="unet_lite",
                loader_mode="shard_grouped",
                shard_cache_size=1,
                num_workers=0,
                device="cpu",
                seed=0,
            )
            checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)

        self.assertEqual(report["model_arch"], "unet_lite")
        self.assertEqual(report["loader"]["mode"], "shard_grouped")
        self.assertEqual(report["loader"]["shard_cache_size"], 1)
        self.assertEqual(checkpoint["model"]["model_arch"], "unet_lite")

    def test_cli_train_and_predict_run_on_tiny_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _write_tiny_dataset(Path(tmp) / "dataset")
            config = Path(tmp) / "config.json"
            checkpoint_dir = Path(tmp) / "checkpoint"
            predictions = Path(tmp) / "predictions.csv"
            config.write_text(
                json.dumps(
                    {
                        "protocol": "sdss-point-supervised-v1",
                        "data": {"root": "/Data/sdss/example"},
                        "cutout": {"pixel_scale_arcsec": 0.396},
                    }
                ),
                encoding="utf-8",
            )

            train_code = main(
                [
                    "train",
                    "--config",
                    str(config),
                    "--dataset",
                    str(dataset_dir),
                    "--output",
                    str(checkpoint_dir),
                    "--epochs",
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
                    "--device",
                    "cpu",
                    "--split",
                    str(Path(tmp) / "split.json"),
                    "--split-name",
                    "train",
                ]
            )
            predict_code = main(
                [
                    "predict",
                    "--checkpoint",
                    str(checkpoint_dir / "best.pt"),
                    "--dataset",
                    str(dataset_dir),
                    "--output",
                    str(predictions),
                    "--batch-size",
                    "2",
                    "--threshold",
                    "0.0",
                    "--nms-radius",
                    "2",
                    "--device",
                    "cpu",
                    "--split",
                    str(Path(tmp) / "split.json"),
                    "--split-name",
                    "val",
                ]
            )
            with predictions.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            manifest_by_cutout = _manifest_rows_by_cutout(dataset_dir)
            wcs = _tiny_r_band_wcs()

        self.assertEqual(train_code, 0)
        self.assertEqual(predict_code, 0)
        self.assertGreater(len(rows), 0)
        self.assertEqual(rows[0]["cutout_id"], "field-a__center-a")
        for row in rows:
            manifest_row = manifest_by_cutout[row["cutout_id"]]
            expected = wcs.all_pix2world(
                [
                    [
                        float(manifest_row["origin_x"]) + float(row["x"]),
                        float(manifest_row["origin_y"]) + float(row["y"]),
                    ]
                ],
                1,
            )[0]
            self.assertAlmostEqual(float(row["ra"]), float(expected[0]), places=9)
            self.assertAlmostEqual(float(row["dec"]), float(expected[1]), places=9)

    def test_cli_predict_requires_r_band_wcs_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = _write_tiny_dataset(Path(tmp) / "dataset", include_wcs_metadata=False)
            checkpoint_dir = Path(tmp) / "checkpoint"
            train_model(
                config_path=None,
                dataset_dir=dataset_dir,
                output_dir=checkpoint_dir,
                epochs=1,
                batch_size=1,
                base_channels=4,
                device="cpu",
                seed=0,
            )

            with self.assertRaisesRegex(ValueError, "r-band WCS metadata"):
                main(
                    [
                        "predict",
                        "--checkpoint",
                        str(checkpoint_dir / "best.pt"),
                        "--dataset",
                        str(dataset_dir),
                        "--output",
                        str(Path(tmp) / "predictions.csv"),
                        "--batch-size",
                        "2",
                        "--threshold",
                        "0.0",
                        "--device",
                        "cpu",
                    ]
                )

    def test_split_truth_selection_uses_manifest_cutouts_and_quality_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")

            selection = _select_split_truth_records(
                dataset_dir=dataset_dir,
                split_path=root / "split.json",
                split_name="val",
            )

        self.assertEqual(selection.cutout_ids, {"field-a__center-a"})
        self.assertEqual(selection.all_truth_count, 2)
        self.assertEqual(selection.dropped_truth_count, 1)
        self.assertEqual([record.source_id for record in selection.records], ["truth-a-clean"])
        self.assertEqual(selection.all_quality_counts, {"clean": 1, "suspect": 1})
        self.assertEqual(selection.kept_quality_counts, {"clean": 1})

    def test_evaluation_counts_zero_prediction_cutout_truth_as_false_negative(self):
        truth = [
            SourceRecord(
                source_id="truth-a",
                cutout_id="cutout-a",
                ra=10.0,
                dec=20.0,
                label="star",
                label_quality="clean",
                label_weight=1.0,
            ),
            SourceRecord(
                source_id="truth-b",
                cutout_id="cutout-b",
                ra=11.0,
                dec=21.0,
                label="star",
                label_quality="clean",
                label_weight=1.0,
            ),
        ]
        predictions = [
            PredictionRecord(
                prediction_id="pred-a",
                cutout_id="cutout-a",
                ra=10.0,
                dec=20.0,
                label="star",
                score=0.9,
            )
        ]

        payload = _evaluate_payload(
            truth,
            predictions,
            radius_arcsec=1.0,
            band="r",
            close_pair_arcsec=2.0,
            seeing_aware=False,
            psf_fraction=0.5,
            min_score=0.5,
            filter_truth_to_prediction_cutouts=False,
        )

        self.assertEqual(payload["counts"]["truth"], 2)
        self.assertEqual(payload["metrics"]["detection"]["tp"], 1.0)
        self.assertEqual(payload["metrics"]["detection"]["fn"], 1.0)

    def test_cli_run_pilot_loop_writes_split_outputs_on_tiny_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = _write_tiny_dataset(root / "dataset")
            config = root / "config.json"
            checkpoint_dir = root / "checkpoint"
            report_dir = root / "reports"
            config.write_text(
                json.dumps(
                    {
                        "protocol": "sdss-point-supervised-v1",
                        "data": {"root": "/Data/sdss/example"},
                        "cutout": {"pixel_scale_arcsec": 0.396},
                    }
                ),
                encoding="utf-8",
            )

            code = main(
                [
                    "run-pilot-loop",
                    "--config",
                    str(config),
                    "--dataset",
                    str(dataset_dir),
                    "--split",
                    str(root / "split.json"),
                    "--output-dir",
                    str(report_dir),
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
                    "--nms-radius",
                    "2",
                    "--max-detections-per-cutout",
                    "4",
                    "--predict-limit",
                    "1",
                    "--device",
                    "cpu",
                ]
            )
            checkpoint_exists = (checkpoint_dir / "best.pt").exists()
            summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
            val_sweep = json.loads((report_dir / "val_threshold_sweep.json").read_text(encoding="utf-8"))
            test_metrics = json.loads((report_dir / "test_metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertTrue(checkpoint_exists)
        self.assertEqual(summary["status"], "generated")
        self.assertEqual(summary["splits"]["val"]["truth_all"], 2)
        self.assertEqual(summary["splits"]["val"]["truth_kept"], 1)
        self.assertEqual(summary["splits"]["test"]["truth_kept"], 1)
        self.assertEqual(summary["decode"]["predict_limit"], 1)
        self.assertIn("best_threshold", val_sweep)
        self.assertIn("detection", test_metrics["metrics"])


def _write_tiny_dataset(dataset_dir: Path, include_source_weight: bool = True, include_wcs_metadata: bool = True) -> Path:
    shards_dir = dataset_dir / "shards"
    shards_dir.mkdir(parents=True)
    images = np.zeros((2, 5, 8, 8), dtype=np.float16)
    images[0, :, 4, 4] = np.array([0.1, 0.2, 0.5, 0.3, 0.2], dtype=np.float16)
    images[0, :, 5, 5] = np.array([0.2, 0.4, 0.8, 0.4, 0.2], dtype=np.float16)
    images[1, :, 3, 3] = np.array([0.1, 0.3, 0.7, 0.5, 0.2], dtype=np.float16)
    shard_payload = {
        "images": images,
        "source_offsets": np.array([0, 2, 3], dtype=np.int64),
        "source_x": np.array([4.0, 5.0, 3.0], dtype=np.float32),
        "source_y": np.array([4.0, 5.0, 3.0], dtype=np.float32),
        "source_label": np.array([1, 2, 1], dtype=np.int16),
        "source_flags": np.array([0, 0, 0], dtype=np.uint64),
        "center_index": np.array([0, 0], dtype=np.int64),
        "source_mag_u": np.array([18.0, 19.0, 18.5], dtype=np.float32),
        "source_mag_g": np.array([17.5, 18.5, 18.0], dtype=np.float32),
        "source_mag_r": np.array([17.0, 18.0, 17.5], dtype=np.float32),
        "source_mag_i": np.array([16.8, 17.8, 17.3], dtype=np.float32),
        "source_mag_z": np.array([16.7, 17.7, 17.2], dtype=np.float32),
    }
    if include_source_weight:
        shard_payload["source_quality"] = np.array([2, 0, 1], dtype=np.int8)
        shard_payload["source_weight"] = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    np.savez_compressed(shards_dir / "shard_000000.npz", **shard_payload)
    (dataset_dir / "manifest.csv").write_text(
        "shard,sample_index,shard_sample_index,cutout_id,field_id,center_source_id,ra,dec,x,y,origin_x,origin_y,label,quality_flags,n_sources\n"
        "shard_000000.npz,0,0,field-a__center-a,field-a,center-a,10.0,20.0,4.0,4.0,100,200,star,,2\n"
        "shard_000000.npz,1,1,field-a__center-b,field-a,center-b,10.1,20.1,3.0,3.0,110,210,star,,1\n",
        encoding="utf-8",
    )
    (dataset_dir / "truth_catalog.csv").write_text(
        "source_id,cutout_id,ra,dec,label,x,y,mag_u,mag_g,mag_r,mag_i,mag_z,flux_u,flux_g,flux_r,flux_i,flux_z,size,ellipticity,crowding,snr,seeing,psf_fwhm,nearest_neighbor_arcsec,galactic_latitude,region_id,quality_flags,label_quality,label_weight,raw_source_id\n"
        "truth-a-clean,field-a__center-a,10.0,20.0,star,4.0,4.0,18.0,17.5,17.0,16.8,16.7,,,,,,,,,,,,,,field-a,,clean,1.0,center-a\n"
        "truth-a-suspect,field-a__center-a,10.0001,20.0001,galaxy,5.0,5.0,19.0,18.5,18.0,17.8,17.7,,,,,,,,,,,,,,field-a,SATURATED,suspect,0.0,neighbor-a\n"
        "truth-b-weak,field-a__center-b,10.1,20.1,star,3.0,3.0,18.5,18.0,17.5,17.3,17.2,,,,,,,,,,,,,,field-a,BLENDED,weak,0.5,center-b\n",
        encoding="utf-8",
    )
    metadata = {
        "protocol": "sdss-point-supervised-v1",
        "bands": ["u", "g", "r", "i", "z"],
        "cutout_size": 8,
        "dtype": "float16",
        "sample_count": 2,
    }
    if include_wcs_metadata:
        metadata["fields"] = [{"field_id": "field-a", "headers": {"r": _tiny_r_band_header()}}]
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    (dataset_dir.parent / "split.json").write_text(
        json.dumps(
            {
                "protocol": "sdss-point-supervised-v1",
                "splits": {"train": ["center-a"], "val": ["center-a"], "test": ["center-b"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_dir


def _tiny_r_band_header() -> dict[str, int | float | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 512,
        "NAXIS2": 512,
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CD1_1": -0.0001,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.0001,
    }


def _tiny_r_band_wcs() -> WCS:
    header = Header()
    for key, value in _tiny_r_band_header().items():
        header[key] = value
    return WCS(header)


def _manifest_rows_by_cutout(dataset_dir: Path) -> dict[str, dict[str, str]]:
    with (dataset_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return {row["cutout_id"]: row for row in csv.DictReader(handle)}


if __name__ == "__main__":
    unittest.main()
