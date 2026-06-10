# SDSS Point-Supervised Catalog Generation Research Plan

## Summary

This project targets a Benchmark + Method paper for point-supervised source
catalog generation on native SDSS DR17 corrected frames. The benchmark defines
leakage-safe data splits, catalog schemas, matching rules, and stress tests. The
method contribution is a compact PSF-constrained point-supervised cataloger that
uses weak PhotoObj centers, coarse STAR/GALAXY labels, and noisy photometry to
generate source catalogs from multiband cutouts.

The first real dataset is:

```text
/Data/sdss/sdss_dr17_l1735_1865_b30_40
```

Headline experiments use native SDSS corrected-frame `u/g/r/i/z` images. WCS
reprojection is reserved for a separate ablation because it changes
interpolation, noise correlation, and PSF behavior.

## Contributions

- A reproducible SDSS DR17 point-supervised catalog-generation benchmark with
  region-safe splits and fixed CSV/JSON contracts.
- A native-frame pilot dataset builder that writes NPZ cutout shards with
  ragged source labels.
- A PyTorch training and prediction loop for the current
  `AstronomyAwareBaseline`.
- A PSF-constrained learning objective combining point heatmaps, sparse class
  labels, sparse photometry targets, multiband consistency, and Gaussian PSF
  proxy reconstruction.
- A paper evaluation design focused on faint sources, close pairs, crowded
  fields, and synthetic injection truth rather than PhotoObj agreement alone.

## Method And Data Flow

1. Build a pilot dataset with `make dataset-pilot`.
2. Train the compact baseline with `make train-pilot`.
3. Decode center heatmaps into a prediction catalog with `make predict-pilot`.
4. Evaluate predictions with the benchmark matcher and metrics.

The prepared dataset stores images as `(N, 5, H, W)` arrays and labels as
ragged arrays. Training converts label points into Gaussian heatmaps, sparse
class maps, and sparse photometry targets. The current PSF term uses a Gaussian
kernel proxy; real SDSS psField ingestion is a planned paper-scale extension.

## Experiment Matrix

- **E0 Data integrity:** ready/partial fields, missing frames/catalogs, object
  counts, pilot QA statistics.
- **E1 Benchmark contract:** fixed sky-region split, source catalog conversion,
  prediction catalog contract, matching radius policy.
- **E2 Baselines:** PhotoObj sanity checks, SEP/SExtractor-style detector,
  center-only heatmap model, astronomy-aware baseline without PSF loss.
- **E3 Main method:** PSF-constrained model selected on validation thresholds
  and reported once on test.
- **E4 Stress tests:** stratification by `mag_r`, SNR, seeing, and nearest
  neighbor distance.
- **E5 Synthetic injection:** injected PSF stars and simple galaxy profiles on
  real backgrounds with known flux and separation.
- **E6 Ablations:** remove PSF reconstruction, inverse-variance photometry,
  valid masks, multiband consistency, class head, and native-frame policy.
- **E7 Efficiency:** inference throughput, decode time, memory, and CPU
  classical baseline runtime.

## Acceptance Criteria

- `make test` and `make smoke` pass before any completion claim.
- Pilot shards load through `NpzCutoutDataset` and produce finite training
  losses.
- A non-dry-run `train` command writes `best.pt` and `training_report.json`.
- A non-dry-run `predict` command writes a benchmark-compatible prediction CSV.
- Main paper claims are supported by stratified metrics, especially faint and
  blended regimes.

## Current Limits

The v1 loop is intentionally compact. It does not yet download or parse real
psField files, tune thresholds on a persisted validation split, run full
paper-scale jobs, or cross-match to deeper external catalogs. Those are the next
milestones after the pilot training and prediction loop is stable.
