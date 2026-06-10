import unittest
from pathlib import Path

import tomllib


class ProjectMetadataTests(unittest.TestCase):
    def test_pyproject_declares_quality_tooling_and_typed_package(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
        self.assertTrue(any(dependency.startswith("ruff") for dependency in dev_dependencies))
        self.assertEqual(pyproject["tool"]["ruff"]["line-length"], 120)
        self.assertEqual(pyproject["tool"]["pytest"]["ini_options"]["testpaths"], ["tests"])
        self.assertEqual(
            pyproject["tool"]["setuptools"]["package-data"]["sdss_point_benchmark"],
            ["py.typed"],
        )

    def test_package_exposes_pep561_marker(self):
        self.assertTrue(Path("src/sdss_point_benchmark/py.typed").exists())

    def test_makefile_exposes_conda_first_verification_targets(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")

        for target in ("test-conda:", "lint-conda:", "smoke-conda:", "verify-conda:"):
            with self.subTest(target=target):
                self.assertIn(target, makefile)


if __name__ == "__main__":
    unittest.main()
