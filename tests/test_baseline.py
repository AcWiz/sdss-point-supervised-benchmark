import unittest

import torch

from sdss_point_benchmark.baseline import (
    AstronomyAwareBaseline,
    AstronomyAwareUNetLite,
    BaselineLoss,
    make_catalog_model,
    make_gaussian_heatmap,
    train_step,
)


class BaselineTests(unittest.TestCase):
    def test_baseline_forward_exposes_catalog_generation_heads(self):
        model = AstronomyAwareBaseline(in_channels=5, num_classes=2, base_channels=8)
        outputs = model(torch.zeros(2, 5, 32, 32))

        self.assertEqual(outputs.center_heatmap.shape, (2, 1, 32, 32))
        self.assertEqual(outputs.class_logits.shape, (2, 2, 32, 32))
        self.assertEqual(outputs.flux.shape, (2, 5, 32, 32))
        self.assertEqual(outputs.shape_params.shape, (2, 3, 32, 32))

    def test_unet_lite_forward_uses_same_output_contract(self):
        model = AstronomyAwareUNetLite(in_channels=5, num_classes=2, base_channels=4)
        outputs = model(torch.zeros(2, 5, 32, 32))

        self.assertEqual(outputs.center_heatmap.shape, (2, 1, 32, 32))
        self.assertEqual(outputs.class_logits.shape, (2, 2, 32, 32))
        self.assertEqual(outputs.flux.shape, (2, 5, 32, 32))
        self.assertEqual(outputs.shape_params.shape, (2, 3, 32, 32))

    def test_model_factory_accepts_architecture_names_and_legacy_checkpoint_name(self):
        self.assertIsInstance(make_catalog_model("baseline", base_channels=4), AstronomyAwareBaseline)
        self.assertIsInstance(make_catalog_model("unet_lite", base_channels=4), AstronomyAwareUNetLite)
        self.assertIsInstance(make_catalog_model("AstronomyAwareBaseline", base_channels=4), AstronomyAwareBaseline)

    def test_baseline_loss_combines_point_photometry_and_multiband_terms(self):
        model = AstronomyAwareBaseline(in_channels=5, num_classes=2, base_channels=8)
        outputs = model(torch.zeros(1, 5, 16, 16))
        target_heatmap = make_gaussian_heatmap([(8.0, 8.0)], height=16, width=16, sigma=1.0).unsqueeze(0).unsqueeze(0)
        target_flux = torch.ones(1, 5, 16, 16)
        inverse_variance = torch.full((1, 5, 16, 16), 0.5)

        loss = BaselineLoss()(outputs, target_heatmap, target_flux, inverse_variance)

        self.assertGreater(float(loss.total.detach()), 0.0)
        self.assertIn("center", loss.parts)
        self.assertIn("photometry", loss.parts)
        self.assertIn("multiband_consistency", loss.parts)

    def test_baseline_loss_supports_ignore_masks_class_and_psf_reconstruction(self):
        model = AstronomyAwareBaseline(in_channels=5, num_classes=2, base_channels=8)
        image = torch.zeros(1, 5, 16, 16)
        outputs = model(image)
        target_heatmap = make_gaussian_heatmap([(8.0, 8.0)], height=16, width=16, sigma=1.0).unsqueeze(0).unsqueeze(0)
        target_flux = torch.ones(1, 5, 16, 16)
        inverse_variance = torch.ones(1, 5, 16, 16)
        ignore_mask = torch.ones(1, 1, 16, 16)
        ignore_mask[:, :, :4, :] = 0.0
        target_class_map = torch.full((1, 16, 16), -100, dtype=torch.long)
        target_class_map[:, 8, 8] = 1
        psf_kernel = torch.ones(1, 1, 3, 3) / 9.0

        loss = BaselineLoss(psf_reconstruction_weight=0.2, class_weight=0.5)(
            outputs,
            target_heatmap,
            target_flux,
            inverse_variance,
            observed_image=image,
            psf_kernel=psf_kernel,
            valid_mask=ignore_mask,
            target_class_map=target_class_map,
        )

        self.assertGreater(float(loss.total.detach()), 0.0)
        self.assertIn("psf_reconstruction", loss.parts)
        self.assertIn("classification", loss.parts)

    def test_train_step_updates_model_parameters(self):
        torch.manual_seed(0)
        model = AstronomyAwareBaseline(in_channels=5, num_classes=2, base_channels=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        batch = {
            "image": torch.zeros(1, 5, 16, 16),
            "target_heatmap": make_gaussian_heatmap([(8.0, 8.0)], 16, 16, 1.0).unsqueeze(0).unsqueeze(0),
            "target_flux": torch.ones(1, 5, 16, 16),
            "inverse_variance": torch.ones(1, 5, 16, 16),
        }
        before = model.center_head.weight.detach().clone()

        loss_value = train_step(model, batch, optimizer, BaselineLoss())

        self.assertGreater(loss_value, 0.0)
        self.assertFalse(torch.equal(before, model.center_head.weight.detach()))


if __name__ == "__main__":
    unittest.main()
