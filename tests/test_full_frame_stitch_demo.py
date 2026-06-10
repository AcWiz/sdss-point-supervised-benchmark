from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from reports.full_frame_stitch_demo.stitch_smoke5_full_frame import (
    FULL_OVERLAY_HEADER_HEIGHT,
    TilePrediction,
    TruthPoint,
    _render_full_overlay,
    generate_tile_grid,
    restore_global_predictions,
    suppress_overlapping_predictions,
)

REPORT_DIR = Path("reports/full_frame_stitch_demo")
SUMMARY_JSON = REPORT_DIR / "summary.json"
PREDICTIONS_CSV = REPORT_DIR / "predictions_full_frame.csv"
LOW_PREDICTIONS_CSV = REPORT_DIR / "predictions_full_frame_threshold_0p2.csv"
PNG_PATHS = [
    REPORT_DIR / "full_frame_gt_only.png",
    REPORT_DIR / "full_frame_overlay.png",
    REPORT_DIR / "full_frame_overlay_with_gt.png",
    REPORT_DIR / "zoom_panels.png",
]


class FullFrameStitchDemoTest(unittest.TestCase):
    def test_tile_grid_covers_right_and_bottom_edges(self):
        tiles = generate_tile_grid(width=205, height=149, tile_size=128, stride=64)

        self.assertIn((77, 21), tiles)
        self.assertEqual(tiles[0], (0, 0))
        self.assertEqual(max(x + 128 for x, _ in tiles), 205)
        self.assertEqual(max(y + 128 for _, y in tiles), 149)

    def test_restore_global_predictions_adds_tile_origin(self):
        restored = restore_global_predictions(
            [
                TilePrediction(tile_x=64, tile_y=128, local_x=10.0, local_y=12.5, score=0.9, label="star"),
                TilePrediction(tile_x=128, tile_y=0, local_x=2.25, local_y=3.0, score=0.8, label="galaxy"),
            ]
        )

        self.assertEqual(restored[0].global_x, 74.0)
        self.assertEqual(restored[0].global_y, 140.5)
        self.assertEqual(restored[1].global_x, 130.25)
        self.assertEqual(restored[1].global_y, 3.0)

    def test_suppress_overlapping_predictions_keeps_highest_score(self):
        predictions = restore_global_predictions(
            [
                TilePrediction(tile_x=0, tile_y=0, local_x=50.0, local_y=50.0, score=0.70, label="star"),
                TilePrediction(tile_x=64, tile_y=0, local_x=-13.0, local_y=50.0, score=0.95, label="star"),
                TilePrediction(tile_x=0, tile_y=0, local_x=90.0, local_y=90.0, score=0.60, label="galaxy"),
            ]
        )

        kept = suppress_overlapping_predictions(predictions, radius_pixels=3.0)

        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0].score, 0.95)
        self.assertEqual(kept[0].global_x, 51.0)
        self.assertEqual(kept[1].global_x, 90.0)

    def test_suppress_overlapping_predictions_handles_many_far_apart_points(self):
        predictions = restore_global_predictions(
            [
                TilePrediction(
                    tile_x=0,
                    tile_y=0,
                    local_x=float((index % 40) * 10),
                    local_y=float((index // 40) * 10),
                    score=1.0 - index * 0.0001,
                    label="star",
                )
                for index in range(1600)
            ]
        )

        kept = suppress_overlapping_predictions(predictions, radius_pixels=3.0)

        self.assertEqual(len(kept), len(predictions))

    def test_full_overlay_keeps_header_outside_image_coordinates(self):
        background = Image.new("RGB", (120, 80), (12, 18, 24))
        truth = [TruthPoint(source_id="edge", x=5.0, y=0.0, label="star", clean="1")]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "overlay.png"
            _render_full_overlay(
                path,
                background,
                predictions=[],
                truth=truth,
                title="Header Separation",
                subtitle="test",
                metadata_tokens=("test",),
            )
            with Image.open(path) as image:
                self.assertEqual(image.size, (120, 80 + FULL_OVERLAY_HEADER_HEIGHT))
                header_pixels = [
                    image.getpixel((x, y))
                    for x in range(2, 9)
                    for y in range(0, FULL_OVERLAY_HEADER_HEIGHT)
                ]
                image_start_pixels = [image.getpixel((x, FULL_OVERLAY_HEADER_HEIGHT)) for x in range(2, 9)]

        self.assertNotIn((67, 200, 92), header_pixels)
        self.assertIn((67, 200, 92), image_start_pixels)


def _demo_artifacts_available() -> bool:
    return all(path.exists() for path in [SUMMARY_JSON, PREDICTIONS_CSV, LOW_PREDICTIONS_CSV, *PNG_PATHS])


@unittest.skipUnless(_demo_artifacts_available(), "full-frame stitch demo artifacts are not available")
class FullFrameStitchDemoArtifactTest(unittest.TestCase):
    def test_summary_records_demo_contract(self):
        summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

        self.assertEqual(summary["field_id"], "run001302_camcol2_field0100")
        self.assertEqual(summary["image_shape"], [5, 1489, 2048])
        self.assertEqual(summary["tile"]["size_pixels"], 128)
        self.assertEqual(summary["tile"]["stride_pixels"], 64)
        self.assertIn("SMOKE5 CHECKPOINT DEMO", summary["demo_notice"])
        self.assertGreater(summary["counts"]["truth"], 0)
        self.assertEqual(summary["outputs"]["gt_overlay"], "full_frame_gt_only.png")

    def test_prediction_csv_coordinates_are_in_frame(self):
        rows = _read_csv(PREDICTIONS_CSV)

        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertEqual(row["field_id"], "run001302_camcol2_field0100")
            self.assertGreaterEqual(float(row["global_x"]), 0.0)
            self.assertLess(float(row["global_x"]), 2048.0)
            self.assertGreaterEqual(float(row["global_y"]), 0.0)
            self.assertLess(float(row["global_y"]), 1489.0)
            self.assertGreaterEqual(float(row["score"]), 0.5)

    def test_low_threshold_csv_exists_for_response_inspection(self):
        rows = _read_csv(LOW_PREDICTIONS_CSV)

        self.assertGreater(len(rows), 0)
        self.assertGreaterEqual(float(rows[0]["score"]), 0.2)

    def test_png_outputs_are_nonempty_and_embed_context(self):
        required_tokens = [
            "run001302_camcol2_field0100",
            "GT",
            "DISPLAY ONLY",
        ]

        for path in PNG_PATHS:
            self.assertGreater(path.stat().st_size, 0)
            with Image.open(path) as image:
                self.assertGreater(image.size[0], 0)
                self.assertGreater(image.size[1], 0)
                metadata_text = "\n".join(str(value) for value in image.text.values())
            for token in required_tokens:
                self.assertIn(token, metadata_text)

    def test_full_frame_overlays_keep_header_outside_native_frame(self):
        for path in [
            REPORT_DIR / "full_frame_gt_only.png",
            REPORT_DIR / "full_frame_overlay.png",
            REPORT_DIR / "full_frame_overlay_with_gt.png",
        ]:
            with Image.open(path) as image:
                self.assertEqual(image.size, (2048, 1489 + FULL_OVERLAY_HEADER_HEIGHT))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
