import unittest

import torch

from sdss_point_benchmark.baseline import BaselineOutput
from sdss_point_benchmark.baseline import decode_predictions as baseline_decode_predictions
from sdss_point_benchmark.decode import decode_predictions


class DecodeModuleTests(unittest.TestCase):
    def test_decode_module_is_canonical_and_baseline_import_stays_compatible(self):
        self.assertIs(baseline_decode_predictions, decode_predictions)

    def test_decode_rejects_mismatched_batch_metadata(self):
        outputs = BaselineOutput(
            center_heatmap=torch.zeros(2, 1, 4, 4),
            class_logits=torch.zeros(2, 2, 4, 4),
            flux=torch.zeros(2, 5, 4, 4),
            shape_params=torch.zeros(2, 3, 4, 4),
        )

        with self.assertRaisesRegex(ValueError, "cutout_ids"):
            decode_predictions(outputs, cutout_ids=["only-one"])

    def test_decode_uses_pixel_to_radec_callback_when_provided(self):
        center = torch.full((1, 1, 4, 4), -8.0)
        center[0, 0, 2, 3] = 8.0
        outputs = BaselineOutput(
            center_heatmap=center,
            class_logits=torch.zeros(1, 2, 4, 4),
            flux=torch.zeros(1, 5, 4, 4),
            shape_params=torch.zeros(1, 3, 4, 4),
        )
        calls = []

        def pixel_to_radec(batch_index: int, x: float, y: float) -> tuple[float, float]:
            calls.append((batch_index, x, y))
            return 123.0 + x, 45.0 + y

        records = decode_predictions(
            outputs,
            cutout_ids=["cutout-a"],
            origin_radec=[(10.0, 20.0)],
            pixel_to_radec=pixel_to_radec,
            pixel_scale_arcsec=1000.0,
            threshold=0.5,
        )

        self.assertEqual(calls, [(0, 3.0, 2.0)])
        self.assertEqual(records[0].ra, 126.0)
        self.assertEqual(records[0].dec, 47.0)


if __name__ == "__main__":
    unittest.main()
