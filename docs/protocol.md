# Benchmark Protocol v1

## Task Definition

Input: fixed-size SDSS-like cutouts with channels ordered as `u/g/r/i/z`.

Weak supervision: source center points, coarse labels, and optional weak
photometric labels from a catalog such as SDSS PhotoObj.

Output catalog:

- source center in sky coordinates;
- class label and confidence;
- per-band flux or magnitude;
- size and shape proxies;
- optional deblending metadata.

## Splits

Splits are made by sky region, not by image row. The default `split` command bins
right ascension and declination and assigns complete bins to train, validation,
or test. This prevents nearby overlapping cutouts from leaking across splits.

Recommended first public split:

- `train`: 70% of sky bins;
- `val`: 15% of sky bins;
- `test`: 15% of sky bins;
- fixed seed and fixed bin widths in the released split JSON.

Report all headline results on the fixed test split. Use validation only for
model selection and threshold tuning.

## Matching

Predictions are matched to reference catalog entries one-to-one within each
cutout. The default matcher sorts candidate pairs by angular distance and greedily
keeps the nearest unused prediction/reference pair.

Use either:

- a fixed angular radius, such as 1 arcsec; or
- a seeing-aware radius `min(max_radius_arcsec, psf_fraction * PSF_FWHM)`.

Publish the chosen radius with every table.

## Metrics

Detection:

- precision, recall, F1;
- AP by sweeping prediction score thresholds;
- completeness and purity by magnitude, SNR, crowding, latitude, and seeing.

Astrometry:

- centroid MAE and RMSE in arcseconds;
- stratified by seeing and magnitude.

Photometry:

- magnitude bias;
- magnitude scatter;
- outlier rate;
- optional calibration curves by magnitude bin.

Classification:

- accuracy;
- macro-F1;
- confusion matrix by class;
- stratified by magnitude and color.

Deblending:

- close-pair recall by nearest-neighbor distance;
- missed companion rate;
- fractional flux attribution error when controlled injection truth or matched
  flux labels are available.

Efficiency:

- GPU and CPU inference time;
- comparison with SEP/SExtractor-style pipelines.

## Validation Sources

Do not use SDSS PhotoObj as the only final truth. Use three complementary
validation regimes:

- SDSS weak labels for training and broad catalog agreement;
- controlled synthetic injections into real SDSS backgrounds for known truth;
- deeper or independent catalog cross-matches for faint and blended sources.

## SDSS Weak-Label Quality

SDSS PhotoObj entries are treated as weak labels with a conservative quality
tier. Clean entries receive full supervision weight, weak entries receive
reduced supervision weight, and suspect entries are excluded as center-source
training targets while remaining auditable in cutout labels with zero weight.
Headline results should report either clean/suspect-excluded subsets or
independent validation sources, not PhotoObj agreement alone.

## Implemented CLI Contract

Create splits:

```bash
PYTHONPATH=src python -m sdss_point_benchmark.cli split \
  --catalog source_catalog.csv \
  --output split.json
```

Evaluate predictions:

```bash
PYTHONPATH=src python -m sdss_point_benchmark.cli evaluate \
  --truth source_catalog.csv \
  --predictions predictions.csv \
  --output metrics.json \
  --radius-arcsec 1 \
  --seeing-aware \
  --psf-fraction 0.5 \
  --band r
```

The metric JSON includes detection, AP, astrometry, classification,
photometry, deblending, and stratified detection sections.

## Baseline Comparisons

Recommended comparison set:

- SEP/SExtractor;
- SDSS PHOTO where available;
- generic detector baselines such as CenterNet, DETR, or YOLO adapted to
  multiband input;
- point-supervised detector baselines;
- the astronomy-aware baseline in this repository;
- deblending pipelines such as Scarlet where feasible.

## Required Paper Tables

- Main detection and measurement table on the fixed test split.
- Stratified table by magnitude/SNR/crowding/seeing.
- Close-pair deblending stress table.
- Synthetic injection table with true flux attribution.
- Ablation table covering PSF reconstruction, noise weighting, multiband input,
  incompleteness handling, and generic versus astronomy-aware models.
