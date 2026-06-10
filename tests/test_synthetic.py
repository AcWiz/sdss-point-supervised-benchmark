import unittest

import numpy as np

from sdss_point_benchmark.synthetic import InjectionSpec, inject_sources


class SyntheticInjectionTests(unittest.TestCase):
    def test_inject_sources_adds_truth_catalog_and_flux(self):
        image = np.zeros((5, 32, 32), dtype=np.float32)
        specs = [
            InjectionSpec(source_id="star1", x=16.0, y=16.0, fluxes={"r": 100.0}, kind="psf_star"),
            InjectionSpec(source_id="gal1", x=8.0, y=8.0, fluxes={"r": 50.0}, kind="sersic_galaxy", radius=2.5),
        ]

        injected, truth = inject_sources(image, specs, bands=("u", "g", "r", "i", "z"), psf_sigma=1.2)

        self.assertEqual(injected.shape, image.shape)
        self.assertGreater(injected[2].sum(), 140.0)
        self.assertEqual([row.source_id for row in truth], ["star1", "gal1"])
        self.assertEqual(truth[0].label, "star")
        self.assertEqual(truth[1].label, "galaxy")


if __name__ == "__main__":
    unittest.main()
