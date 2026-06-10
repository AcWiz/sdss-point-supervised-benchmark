import unittest

from sdss_point_benchmark.quality import (
    FLAG_NAME_TO_BIT,
    source_label_quality,
    source_label_weight,
    source_quality_flags,
)


class QualityPolicyTests(unittest.TestCase):
    def test_clean_source_without_hard_flags_has_full_weight(self):
        metadata = {"clean": "1", "flags": 0}

        self.assertEqual(source_label_quality(metadata), "clean")
        self.assertEqual(source_label_weight(metadata), 1.0)

    def test_soft_flagged_source_is_weak_not_suspect(self):
        metadata = {"clean": "0", "flags": "BLENDED PSF_FLUX_INTERP"}

        self.assertEqual(source_label_quality(metadata), "weak")
        self.assertEqual(source_label_weight(metadata), 0.5)

    def test_hard_flags_mark_source_suspect_with_zero_weight(self):
        for flag_name in ["SATURATED", "EDGE", "LOCAL_EDGE", "BADSKY", "NOTCHECKED", "PEAKS_TOO_CLOSE"]:
            with self.subTest(flag_name=flag_name):
                metadata = {"clean": "1", "flags": flag_name}

                self.assertEqual(source_label_quality(metadata), "suspect")
                self.assertEqual(source_label_weight(metadata), 0.0)

    def test_interp_center_alone_is_not_hard_exclusion_but_cr_combo_is(self):
        self.assertEqual(source_label_quality({"clean": "0", "flags": "INTERP_CENTER"}), "weak")
        self.assertEqual(source_label_quality({"clean": "0", "flags": "INTERP_CENTER CR"}), "suspect")

    def test_numeric_flags_follow_same_policy_as_named_flags(self):
        numeric_flags = FLAG_NAME_TO_BIT["INTERP_CENTER"] | FLAG_NAME_TO_BIT["CR"]

        self.assertEqual(source_quality_flags({"flags": numeric_flags}), ["CR", "INTERP_CENTER"])
        self.assertEqual(source_label_quality({"clean": "0", "flags": numeric_flags}), "suspect")
        self.assertEqual(source_label_weight({"clean": "0", "flags": numeric_flags}), 0.0)


if __name__ == "__main__":
    unittest.main()
