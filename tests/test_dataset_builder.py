import bz2
import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from astropy.io import fits

import sdss_point_benchmark.dataset_builder as dataset_builder
from sdss_point_benchmark.cli import main
from sdss_point_benchmark.dataset_builder import build_cutouts_for_field, build_dataset, encode_cutout_batch
from sdss_point_benchmark.fits_io import FieldImageStack, load_field_image_stack
from sdss_point_benchmark.quality import flag_names, source_quality_flags
from sdss_point_benchmark.schema import BANDS, SourceRecord
from sdss_point_benchmark.sdss_dr17 import SdssFieldManifestRecord


class DatasetBuilderTests(unittest.TestCase):
    def test_fits_reader_loads_bz2_ugriz_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_paths = {}
            for band_index, band in enumerate(BANDS):
                frame_paths[band] = _write_bz2_fits(root / f"frame-{band}.fits.bz2", np.full((6, 8), band_index))

            field_stack = load_field_image_stack(frame_paths)

        self.assertEqual(field_stack.images.shape, (5, 6, 8))
        self.assertEqual(field_stack.bands, BANDS)
        self.assertEqual(float(field_stack.images[2, 0, 0]), 2.0)
        self.assertIn("r", field_stack.headers)
        self.assertEqual(field_stack.headers["r"]["NAXIS1"], 8)

    def test_cutout_builder_crops_center_sources_and_encodes_window_labels(self):
        images = np.arange(5 * 16 * 16, dtype=np.float32).reshape(5, 16, 16)
        sources = [
            SourceRecord(
                "center",
                "field-a",
                10.0,
                20.0,
                "star",
                x=8.0,
                y=8.0,
                mag_r=17.1,
                metadata={"clean": "1", "flags": 0},
            ),
            SourceRecord(
                "neighbor",
                "field-a",
                10.1,
                20.1,
                "galaxy",
                x=10.5,
                y=7.5,
                mag_r=19.2,
                metadata={"clean": "0", "flags": "BLENDED"},
            ),
            SourceRecord(
                "suspect-neighbor",
                "field-a",
                10.3,
                20.3,
                "star",
                x=7.0,
                y=9.0,
                mag_r=20.1,
                metadata={"clean": "1", "flags": "SATURATED"},
            ),
            SourceRecord("edge", "field-a", 10.2, 20.2, "star", x=1.0, y=1.0, mag_r=16.0, metadata={"flags": 0}),
        ]

        cutouts, stats = build_cutouts_for_field("field-a", images, sources, cutout_size=6)
        encoded = encode_cutout_batch(cutouts, dtype=np.float16)

        self.assertEqual(stats["candidate_sources"], 4)
        self.assertEqual(stats["suspect_center_excluded"], 1)
        self.assertEqual(stats["edge_unsafe"], 1)
        self.assertEqual(len(cutouts), 2)
        self.assertEqual(cutouts[0].image.shape, (5, 6, 6))
        self.assertEqual(cutouts[0].origin_x, 5)
        self.assertEqual(cutouts[0].origin_y, 5)
        self.assertEqual([label.label for label in cutouts[0].labels], ["star", "galaxy", "star"])
        self.assertEqual([label.label_quality for label in cutouts[0].labels], ["clean", "weak", "suspect"])
        self.assertEqual([label.label_weight for label in cutouts[0].labels], [1.0, 0.5, 0.0])
        self.assertEqual(cutouts[0].center_label_quality, "clean")
        self.assertEqual(cutouts[0].center_label_weight, 1.0)
        self.assertEqual(encoded["images"].shape, (2, 5, 6, 6))
        self.assertEqual(encoded["images"].dtype, np.float16)
        np.testing.assert_array_equal(encoded["source_offsets"], np.array([0, 3, 6], dtype=np.int64))
        np.testing.assert_allclose(encoded["source_x"][:3], np.array([3.0, 5.5, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(encoded["source_quality"][:3], np.array([2, 1, 0], dtype=np.int8))
        np.testing.assert_allclose(encoded["source_weight"][:3], np.array([1.0, 0.5, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(encoded["center_index"], np.array([0, 1], dtype=np.int64))

    def test_cutout_builder_skips_suspect_center_sources(self):
        images = np.zeros((5, 16, 16), dtype=np.float32)
        sources = [
            SourceRecord(
                "suspect-center",
                "field-a",
                10.0,
                20.0,
                "star",
                x=8.0,
                y=8.0,
                metadata={"clean": "1", "flags": "SATURATED"},
            ),
            SourceRecord(
                "weak-center",
                "field-a",
                10.1,
                20.1,
                "galaxy",
                x=9.0,
                y=9.0,
                metadata={"clean": "0", "flags": "BLENDED"},
            ),
        ]

        cutouts, stats = build_cutouts_for_field("field-a", images, sources, cutout_size=6)

        self.assertEqual([cutout.center_source_id for cutout in cutouts], ["weak-center"])
        self.assertEqual(stats["suspect_center_excluded"], 1)

    def test_cutout_builder_can_store_field_buffer_images_as_requested_dtype(self):
        images = np.arange(5 * 16 * 16, dtype=np.float32).reshape(5, 16, 16)
        sources = [
            SourceRecord("center", "field-a", 10.0, 20.0, "star", x=8.0, y=8.0, metadata={"flags": 0})
        ]

        self.assertIn("image_dtype", inspect.signature(build_cutouts_for_field).parameters)
        default_cutouts, _ = build_cutouts_for_field("field-a", images, sources, cutout_size=6)
        half_cutouts, _ = build_cutouts_for_field(
            "field-a",
            images,
            sources,
            cutout_size=6,
            image_dtype=np.float16,
        )

        self.assertEqual(default_cutouts[0].image.dtype, np.float32)
        self.assertEqual(half_cutouts[0].image.dtype, np.float16)

    def test_source_window_index_preserves_source_order_across_cells(self):
        index_class = getattr(dataset_builder, "_SourceWindowIndex", None)
        self.assertIsNotNone(index_class)
        sources = [
            SourceRecord("right-cell", "field-a", 10.2, 20.2, "galaxy", x=6.5, y=5.0, metadata={"flags": 0}),
            SourceRecord("outside-right-edge", "field-a", 10.3, 20.3, "star", x=7.0, y=5.0, metadata={"flags": 0}),
            SourceRecord("center", "field-a", 10.0, 20.0, "star", x=5.0, y=5.0, metadata={"flags": 0}),
            SourceRecord("lower-cell", "field-a", 10.1, 20.1, "galaxy", x=3.5, y=6.5, metadata={"flags": 0}),
            SourceRecord("outside-type", "field-a", 10.4, 20.4, "other", x=4.5, y=4.5, metadata={"flags": 0}),
        ]

        index = index_class([source for source in sources if source.label in dataset_builder.LABEL_TO_INT], cell_size=4)
        matches = index.query(origin_x=3, origin_y=3, cutout_size=4)
        cutouts, _ = build_cutouts_for_field("field-a", np.zeros((5, 12, 12), dtype=np.float32), sources, cutout_size=4)
        center_cutout = next(cutout for cutout in cutouts if cutout.center_source_id == "center")

        self.assertEqual([source.source_id for source in matches], ["right-cell", "center", "lower-cell"])
        self.assertEqual([label.source_id for label in center_cutout.labels], ["right-cell", "center", "lower-cell"])
        self.assertEqual(center_cutout.center_index, 1)

    def test_quality_parser_marks_named_and_numeric_sdss_flags(self):
        self.assertEqual(flag_names((1 << 2) | (1 << 18)), ["EDGE", "SATURATED"])
        self.assertEqual(source_quality_flags({"flags": "SATURATED INTERP_CENTER"}), ["SATURATED", "INTERP_CENTER"])

    def test_cli_build_dataset_dry_run_writes_qa_without_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sdss"
            output = Path(tmp) / "pilot"
            config = _write_fake_sdss_tree(root)

            exit_code = main(
                [
                    "build-dataset",
                    "--config",
                    str(config),
                    "--output-dir",
                    str(output),
                    "--limit-fields",
                    "1",
                    "--cutout-size",
                    "8",
                    "--dry-run",
                ]
            )
            qa = json.loads((output / "qa_report.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertTrue(qa["dry_run"])
        self.assertEqual(qa["field_count"], 1)
        self.assertEqual(qa["sample_count"], 2)
        self.assertEqual(qa["label_quality_counts"], {"clean": 1, "suspect": 1, "weak": 2})
        self.assertEqual(qa["filter_counts"]["suspect_center_excluded"], 1)
        self.assertFalse((output / "shards").exists())
        self.assertFalse((output / "manifest.csv").exists())
        self.assertFalse((output / "truth_catalog.csv").exists())

    def test_cli_build_dataset_writes_manifest_metadata_and_npz_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sdss"
            output = Path(tmp) / "pilot"
            config = _write_fake_sdss_tree(root)

            exit_code = main(
                [
                    "build-dataset",
                    "--config",
                    str(config),
                    "--output-dir",
                    str(output),
                    "--limit-fields",
                    "1",
                    "--cutout-size",
                    "8",
                    "--shard-size",
                    "1",
                    "--dtype",
                    "float16",
                ]
            )
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                manifest_rows = list(csv.DictReader(handle))
            with (output / "truth_catalog.csv").open(newline="", encoding="utf-8") as handle:
                truth_rows = list(csv.DictReader(handle))
            shard = np.load(output / "shards" / "shard_000000.npz")
            second_shard = np.load(output / "shards" / "shard_000001.npz")

        self.assertEqual(exit_code, 0)
        self.assertEqual(metadata["sample_count"], 2)
        self.assertEqual(len(manifest_rows), 2)
        self.assertEqual(metadata["label_quality_policy"]["quality_to_weight"], {"clean": 1.0, "suspect": 0.0, "weak": 0.5})
        self.assertEqual(len(truth_rows), 6)
        self.assertEqual({row["cutout_id"] for row in truth_rows}, {row["cutout_id"] for row in manifest_rows})
        self.assertEqual(len({row["source_id"] for row in truth_rows}), 6)
        self.assertTrue(all(row["source_id"].startswith(row["cutout_id"] + "__") for row in truth_rows))
        self.assertEqual([row["label_quality"] for row in manifest_rows], ["clean", "weak"])
        self.assertEqual([row["label_weight"] for row in manifest_rows], ["1.0", "0.5"])
        self.assertIn("label_quality", truth_rows[0])
        self.assertIn("label_weight", truth_rows[0])
        self.assertIn("raw_source_id", truth_rows[0])
        self.assertEqual({row["label_quality"] for row in truth_rows}, {"clean", "suspect", "weak"})
        self.assertEqual([row["shard"] for row in manifest_rows], ["shard_000000.npz", "shard_000001.npz"])
        self.assertEqual([row["shard_sample_index"] for row in manifest_rows], ["0", "0"])
        self.assertEqual(shard["images"].shape, (1, 5, 8, 8))
        self.assertEqual(second_shard["images"].shape, (1, 5, 8, 8))
        self.assertEqual(shard["images"].dtype, np.float16)
        self.assertEqual(second_shard["images"].dtype, np.float16)
        self.assertEqual(int(shard["source_offsets"][-1]), 3)
        self.assertEqual(int(second_shard["source_offsets"][-1]), 3)
        np.testing.assert_array_equal(shard["source_quality"], np.array([2, 1, 0], dtype=np.int8))
        np.testing.assert_allclose(shard["source_weight"], np.array([1.0, 0.5, 0.0], dtype=np.float32))

    def test_build_dataset_streams_shards_before_loading_all_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pilot"
            config = Path(tmp) / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "protocol": "sdss-point-supervised-v1",
                        "data": {"root": str(Path(tmp) / "sdss"), "bands": list(BANDS)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = [
                _field_record("field-a", "catalog-a.csv", 100),
                _field_record("field-b", "catalog-b.csv", 101),
            ]
            events = []
            original_encode = dataset_builder.encode_cutout_batch

            def load_stack(record, bands):
                return FieldImageStack(
                    images=np.ones((len(bands), 16, 16), dtype=np.float32),
                    bands=tuple(bands),
                    headers={band: {"NAXIS1": 16, "NAXIS2": 16} for band in bands},
                )

            def load_sources(path, clean_only=False):
                events.append(f"load:{path}")
                suffix = "a" if "catalog-a" in str(path) else "b"
                return [
                    SourceRecord(
                        f"source-{suffix}",
                        f"field-{suffix}",
                        10.0,
                        20.0,
                        "star",
                        x=8.0,
                        y=8.0,
                        metadata={"flags": 0},
                    )
                ]

            def encode(cutouts, dtype=np.float16):
                events.append(f"encode:{cutouts[0].field_id}")
                return original_encode(cutouts, dtype=dtype)

            with (
                mock.patch.object(dataset_builder, "build_sdss_field_manifest", return_value=records),
                mock.patch.object(dataset_builder, "_load_stack_for_record", side_effect=load_stack),
                mock.patch.object(dataset_builder, "load_sdss_source_catalog", side_effect=load_sources),
                mock.patch.object(dataset_builder, "encode_cutout_batch", side_effect=encode),
            ):
                build_dataset(config, output, limit_fields=None, cutout_size=6, shard_size=1, dtype="float16")

            self.assertLess(events.index("encode:field-a"), events.index("load:catalog-b.csv"))
            with (output / "manifest.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_cli_build_source_catalog_aggregates_limited_ready_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sdss"
            config = _write_fake_sdss_tree(root)
            second_catalog = root / "catalog-second.csv"
            second_catalog.write_text(
                "objID,run,rerun,camcol,field,ra,dec,l,b,type_name,clean,rowc_r,colc_r,"
                "psfMag_u,psfMag_g,psfMag_r,psfMag_i,psfMag_z,"
                "cModelMag_u,cModelMag_g,cModelMag_r,cModelMag_i,cModelMag_z,"
                "petroR50_r,expAB_r,flags\n"
                "223,1302,301,2,101,122.1,43.5,178.5,31.8,STAR,0,12.0,12.0,"
                "17.0,16.0,15.0,14.8,14.7,"
                "-9999,-9999,-9999,-9999,-9999,"
                "1.2,0.8,0\n",
                encoding="utf-8",
            )
            with (root / "manifest_frames.csv").open("a", encoding="utf-8") as handle:
                for band in BANDS:
                    handle.write(f"1302,301,2,101,{band},exists,1,{root / f'frame-{band}.fits.bz2'},,\n")
            with (root / "manifest_catalogs.csv").open("a", encoding="utf-8") as handle:
                handle.write(f"1302,301,2,101,downloaded,1,{second_catalog},\n")
            output = Path(tmp) / "sources.csv"

            exit_code = main(
                [
                    "build-source-catalog",
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                    "--limit-fields",
                    "1",
                ]
            )
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertEqual({row["cutout_id"] for row in rows}, {"run001302_camcol2_field0100"})
        self.assertEqual({row["label"] for row in rows}, {"star", "galaxy"})


def _write_bz2_fits(path: Path, data: np.ndarray) -> str:
    fits_path = path.with_suffix("")
    header = fits.Header()
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 10.0
    header["CRVAL2"] = 20.0
    fits.PrimaryHDU(data=np.asarray(data, dtype=np.float32), header=header).writeto(fits_path, overwrite=True)
    path.write_bytes(bz2.compress(fits_path.read_bytes()))
    fits_path.unlink()
    return str(path)


def _field_record(field_id: str, catalog_path: str, field: int) -> SdssFieldManifestRecord:
    return SdssFieldManifestRecord(
        field_id=field_id,
        run=1302,
        rerun=301,
        camcol=2,
        field=field,
        status="ready",
        catalog_path=catalog_path,
        n_objects=1,
        frame_paths={band: "" for band in BANDS},
    )


def _write_fake_sdss_tree(root: Path) -> Path:
    root.mkdir(parents=True)
    frame_paths = {}
    for band_index, band in enumerate(BANDS):
        frame_paths[band] = _write_bz2_fits(root / f"frame-{band}.fits.bz2", np.full((24, 24), band_index))

    (root / "manifest_frames.csv").write_text(
        "run,rerun,camcol,field,band,status,bytes,path,url,error\n"
        + "".join(
            f"1302,301,2,100,{band},exists,1,{frame_paths[band]},,\n"
            for band in BANDS
        ),
        encoding="utf-8",
    )
    catalog_path = root / "catalog.csv"
    catalog_path.write_text(
        "objID,run,rerun,camcol,field,ra,dec,l,b,type_name,clean,rowc_r,colc_r,"
        "psfMag_u,psfMag_g,psfMag_r,psfMag_i,psfMag_z,"
        "cModelMag_u,cModelMag_g,cModelMag_r,cModelMag_i,cModelMag_z,"
        "petroR50_r,expAB_r,flags\n"
        "123,1302,301,2,100,121.1,42.5,177.5,30.8,STAR,1,12.0,12.0,"
        "17.0,16.0,15.0,14.8,14.7,"
        "-9999,-9999,-9999,-9999,-9999,"
        "1.2,0.8,0\n"
        "124,1302,301,2,100,121.2,42.6,177.6,30.9,GALAXY,0,14.0,14.0,"
        "-9999,-9999,-9999,-9999,-9999,"
        "20.0,19.0,18.0,17.8,17.7,"
        "2.4,0.5,BLENDED\n"
        "125,1302,301,2,100,121.3,42.7,177.7,31.0,STAR,1,13.0,13.0,"
        "18.0,17.0,16.0,15.8,15.7,"
        "-9999,-9999,-9999,-9999,-9999,"
        "1.1,0.9,262144\n"
        "126,1302,301,2,100,121.4,42.8,177.8,31.1,OTHER,0,15.0,15.0,"
        "-9999,-9999,-9999,-9999,-9999,"
        "-9999,-9999,-9999,-9999,-9999,"
        "-9999,-9999,0\n"
        "127,1302,301,2,100,121.5,42.9,177.9,31.2,STAR,0,1.0,1.0,"
        "17.0,16.0,15.0,14.8,14.7,"
        "-9999,-9999,-9999,-9999,-9999,"
        "1.2,0.8,0\n",
        encoding="utf-8",
    )
    (root / "manifest_catalogs.csv").write_text(
        "run,rerun,camcol,field,status,n_objects,path,error\n"
        f"1302,301,2,100,downloaded,5,{catalog_path},\n",
        encoding="utf-8",
    )
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "protocol": "sdss-point-supervised-v1",
                "data": {"root": str(root), "name": "fake_sdss", "bands": list(BANDS)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return config


if __name__ == "__main__":
    unittest.main()
