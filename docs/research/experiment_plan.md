# Experiment Plan

## Summary

The first paper target is a Benchmark + Method contribution. The benchmark
defines leakage-safe splits, catalog schemas, matching, and stress tests. The
method contribution is a PSF-constrained point-supervised catalog generator.

## E0 Data Integrity

Input: `/Data/sdss/sdss_dr17_l1735_1865_b30_40`.

Run:

```bash
PYTHONPATH=src python -m sdss_point_benchmark.cli build-manifest \
  --data-root /Data/sdss/sdss_dr17_l1735_1865_b30_40 \
  --output artifacts/manifests/sdss_dr17_l1735_1865_b30_40_fields.csv
```

Report complete ugriz fields, missing frames, missing catalogs, object counts,
and skipped fields.

## E1 Benchmark Contract

Convert SDSS PhotoObj CSVs into the benchmark `SourceRecord` CSV contract, then
generate the fixed split with seed 42 and 1 degree sky bins. No sky region may
appear in more than one split.

## E2 Baselines

Compare:

- SDSS PhotoObj self-consistency where applicable;
- SEP/SExtractor-style classical detection;
- generic heatmap detector baseline;
- current `AstronomyAwareBaseline` without PSF reconstruction.

## E3 Proposed Method

Train the PSF-constrained model on train regions, tune threshold/NMS on val, and
report only final numbers on test. Main metrics are detection AP/F1, centroid
RMSE, photometry bias/scatter, class macro-F1, and close-pair recall.

## E4 Stratified Stress Tests

Use the existing evaluator strata: `mag_r`, SNR, seeing, and nearest-neighbor
distance. The key scientific claim should be supported in faint and blended
regimes, not only in easy isolated bright sources.

## E5 Synthetic Injection

Inject known PSF-star and galaxy profiles into real backgrounds. Report true
flux attribution error, close-pair recall, missed companion rate, and recovery
as a function of magnitude contrast and separation.

## E6 Ablations

Remove one component at a time:

- PSF reconstruction;
- inverse-variance photometry weighting;
- valid masks;
- multiband consistency;
- class head;
- native-frame policy versus optional reprojected products.

## E7 Efficiency

Report inference throughput, GPU memory, catalog decode time, and CPU baseline
runtime on a fixed test subset.
