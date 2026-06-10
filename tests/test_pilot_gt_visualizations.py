import csv
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

REPORT_DIR = Path("reports/gt_visualizations/pilot")
DATASET_DIR = Path("artifacts/datasets/sdss_dr17_l1735_1865_b30_40_pilot")
INDEX_CSV = REPORT_DIR / "index.csv"
LABEL_KEY_CSV = REPORT_DIR / "label_key.csv"
MANIFEST_CSV = DATASET_DIR / "manifest.csv"
TRUTH_CSV = DATASET_DIR / "truth_catalog.csv"


def _pilot_artifacts_available() -> bool:
    return all(path.exists() for path in (INDEX_CSV, LABEL_KEY_CSV, MANIFEST_CSV, TRUTH_CSV))


@unittest.skipUnless(_pilot_artifacts_available(), "pilot GT visualization artifacts are not available")
class PilotGtVisualizationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index_rows = _read_csv(INDEX_CSV)
        cls.label_rows = _read_csv(LABEL_KEY_CSV)

    def test_index_has_science_audit_fields_and_unique_origins(self):
        required_fields = {
            "category",
            "png",
            "sample_index",
            "cutout_id",
            "field_id",
            "origin_x",
            "origin_y",
            "center_x",
            "center_y",
            "center_r_visibility",
            "center_i_visibility",
            "center_visibility_score",
            "n_clean",
            "n_weak",
            "n_suspect",
        }
        self.assertTrue(self.index_rows)
        self.assertLessEqual(required_fields, set(self.index_rows[0]))

        self.assertEqual(Counter(row["category"] for row in self.index_rows), {
            "clean_center": 2,
            "weak_center": 2,
            "crowded": 2,
            "suspect_neighbor": 2,
        })

        origins = [(row["field_id"], row["origin_x"], row["origin_y"]) for row in self.index_rows]
        self.assertEqual(len(origins), len(set(origins)))
        for row in self.index_rows:
            self.assertGreater(float(row["center_visibility_score"]), 0.0)

    def test_label_key_matches_catalog_and_has_one_center_per_cutout(self):
        rows_by_cutout = defaultdict(list)
        for row in self.label_rows:
            rows_by_cutout[row["cutout_id"]].append(row)
            self.assertGreaterEqual(float(row["x"]), 0.0)
            self.assertLess(float(row["x"]), 128.0)
            self.assertGreaterEqual(float(row["y"]), 0.0)
            self.assertLess(float(row["y"]), 128.0)
            if row["quality"] == "suspect":
                self.assertEqual(row["weight"], "0.0")

        truth_counts = _truth_counts_for(set(rows_by_cutout))
        for cutout_id, rows in rows_by_cutout.items():
            self.assertEqual(len(rows), truth_counts[cutout_id])
            self.assertEqual(sum(row["is_center"] == "true" for row in rows), 1)

    def test_png_contract_is_consistent_and_embeds_audit_text(self):
        main_pngs = sorted(path for path in REPORT_DIR.glob("*.png") if path.name != "paper_grid.png")
        self.assertEqual(len(main_pngs), 8)
        expected_paths = {REPORT_DIR / row["png"] if not row["png"].startswith("reports/") else Path(row["png"]) for row in self.index_rows}
        self.assertEqual(set(main_pngs), expected_paths)

        sizes = set()
        required_tokens = [
            "u",
            "g",
            "r",
            "i",
            "z",
            "CENTER",
            "Legend",
            "GT OVERLAY AUDIT",
            "DISPLAY ONLY: i/r/g composite, u/z omitted, nonlinear stretch",
        ]
        for path in main_pngs:
            self.assertGreater(path.stat().st_size, 0)
            with Image.open(path) as image:
                sizes.add(image.size)
                metadata_text = "\n".join(str(value) for value in image.text.values())
            for token in required_tokens:
                self.assertIn(token, metadata_text)
        self.assertEqual(len(sizes), 1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _truth_counts_for(cutout_ids: set[str]) -> dict[str, int]:
    counts = {cutout_id: 0 for cutout_id in cutout_ids}
    with TRUTH_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cutout_id = row["cutout_id"]
            if cutout_id in counts:
                counts[cutout_id] += 1
    return counts


if __name__ == "__main__":
    unittest.main()
