import unittest

from sdss_point_benchmark.schema import SourceRecord
from sdss_point_benchmark.split import assign_region_id, make_region_split


class SplitTests(unittest.TestCase):
    def test_assign_region_id_wraps_ra_and_bins_dec(self):
        self.assertEqual(assign_region_id(359.9, -0.1, ra_bin_deg=10, dec_bin_deg=5), "r035_d017")
        self.assertEqual(assign_region_id(360.1, -0.1, ra_bin_deg=10, dec_bin_deg=5), "r000_d017")

    def test_make_region_split_has_no_region_leakage(self):
        records = [
            SourceRecord(source_id=f"s{i}", cutout_id=f"c{i}", ra=ra, dec=dec, label="star", mag_r=20.0)
            for i, (ra, dec) in enumerate(
                [
                    (1.0, 1.0),
                    (1.1, 1.2),
                    (12.0, 1.0),
                    (12.2, 1.3),
                    (25.0, -1.0),
                    (25.1, -1.2),
                    (41.0, 8.0),
                    (42.0, 8.4),
                    (80.0, -4.0),
                    (81.0, -4.4),
                    (130.0, 2.0),
                ]
            )
        ]

        split = make_region_split(
            records,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            ra_bin_deg=10,
            dec_bin_deg=5,
            seed=7,
        )

        all_ids = set(split.train_ids) | set(split.val_ids) | set(split.test_ids)
        self.assertEqual(all_ids, {record.source_id for record in records})
        self.assertFalse(set(split.train_regions) & set(split.val_regions))
        self.assertFalse(set(split.train_regions) & set(split.test_regions))
        self.assertFalse(set(split.val_regions) & set(split.test_regions))
        self.assertEqual(
            split,
            make_region_split(
                records,
                train_fraction=0.5,
                val_fraction=0.25,
                test_fraction=0.25,
                ra_bin_deg=10,
                dec_bin_deg=5,
                seed=7,
            ),
        )

    def test_make_region_split_defaults_to_sky_bins_when_catalog_region_exists(self):
        records = [
            SourceRecord(
                source_id=f"s{i}",
                cutout_id=f"c{i}",
                ra=ra,
                dec=0.0,
                label="star",
                mag_r=20.0,
                region_id="same_catalog_field",
            )
            for i, ra in enumerate([1.0, 12.0, 24.0, 36.0])
        ]

        split = make_region_split(
            records,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            ra_bin_deg=10,
            dec_bin_deg=5,
            seed=3,
        )

        all_regions = set(split.train_regions) | set(split.val_regions) | set(split.test_regions)
        self.assertNotEqual(all_regions, {"same_catalog_field"})
        self.assertEqual(len(all_regions), 4)

    def test_make_region_split_can_use_catalog_region_ids(self):
        records = [
            SourceRecord(
                source_id=f"s{i}",
                cutout_id=f"c{i}",
                ra=ra,
                dec=0.0,
                label="star",
                mag_r=20.0,
                region_id="same_catalog_field",
            )
            for i, ra in enumerate([1.0, 12.0, 24.0, 36.0])
        ]

        split = make_region_split(
            records,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            ra_bin_deg=10,
            dec_bin_deg=5,
            seed=3,
            region_mode="catalog-region",
        )

        all_regions = set(split.train_regions) | set(split.val_regions) | set(split.test_regions)
        self.assertEqual(all_regions, {"same_catalog_field"})
        assigned = set(split.train_ids) | set(split.val_ids) | set(split.test_ids)
        self.assertEqual(assigned, {record.source_id for record in records})


if __name__ == "__main__":
    unittest.main()
