# SDSS Point-Supervised Source Measurement Benchmark

This repository implements a first reproducible scaffold for **Point-Supervised
Source Catalog Generation** on SDSS-style multiband cutouts. The benchmark starts
from weak catalog supervision, usually source centers plus coarse labels, and
evaluates whether a model can generate a usable source catalog:

- source detection and centroid recovery;
- star/galaxy classification, with quasar as an optional extension;
- multiband flux or magnitude measurement;
- deblending-aware stress tests for close pairs and crowded fields.

The goal is not to treat SDSS PhotoObj as final truth. The protocol is designed
to combine SDSS weak labels with synthetic injection tests and deeper external
catalog cross-matches for the hard regimes where SDSS catalog incompleteness
matters.

## What Is Implemented

- Region-safe train/validation/test split generation with no sky-bin leakage.
- Catalog schemas for truth and predictions.
- One-to-one angular matching with fixed or seeing-aware match radius.
- Detection, astrometry, photometry, classification, and binned completeness
  metrics.
- Average precision, stratified reporting, and close-pair deblending metrics.
- Simple synthetic source injection utilities for controlled tests on real
  backgrounds.
- A compact PyTorch baseline with center heatmap, class, flux, and morphology
  heads plus point, class, noise-weighted photometry, valid-mask, multiband, and
  PSF-aware reconstruction losses.
- A CLI for writing benchmark split JSON and metric reports from CSV catalogs.
- An audit-style automated research loop that runs fixed pilot experiments,
  records provenance, writes JSON/Markdown reports, and gates paper claims.
- A research-program layer for queueing experiment variants, indexing runs,
  comparing evidence, diagnosing failures, and tracking paper-claim support.
- A Codex research-agent playbook for repeatable start-of-session inspection,
  evidence review, experiment critique, and next-step planning.

## Quick Start

For autonomous research-agent sessions, read
[`docs/research/codex_research_agent.md`](docs/research/codex_research_agent.md)
before launching experiments or changing automation.

Create the recommended Python 3.11 conda environment:

```bash
make env-create
```

If the environment already exists, update it instead:

```bash
make env-update
```

Run the conda-first verification checks:

```bash
make verify-conda
```

Run the tests outside conda when needed:

```bash
make test
```

If `make` is unavailable, run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests
```

Create a leakage-safe split:

```bash
PYTHONPATH=src python -m sdss_point_benchmark.cli split \
  --catalog path/to/source_catalog.csv \
  --output splits/sdss_regions_v1.json \
  --region-mode sky-bin \
  --ra-bin-deg 1 \
  --dec-bin-deg 1 \
  --seed 42
```

Build a field-level manifest for the local DR17 region:

```bash
PYTHONPATH=src python -m sdss_point_benchmark.cli build-manifest \
  --data-root /Data/sdss/sdss_dr17_l1735_1865_b30_40 \
  --output artifacts/manifests/sdss_dr17_l1735_1865_b30_40_fields.csv
```

Convert one SDSS PhotoObj CSV to the benchmark source-catalog contract:

```bash
PYTHONPATH=src python -m sdss_point_benchmark.cli convert-sdss-catalog \
  --input /Data/sdss/sdss_dr17_l1735_1865_b30_40/catalogs/catalog_run001302_camcol2_field0100.csv \
  --output artifacts/catalogs/source_catalog_field0100.csv
```

Create a cutout worklist from a benchmark source catalog:

```bash
PYTHONPATH=src python -m sdss_point_benchmark.cli prepare-cutouts \
  --catalog artifacts/catalogs/source_catalog.csv \
  --output artifacts/manifests/cutouts.csv \
  --cutout-size 128
```

Write a dry-run experiment report from the DR17 protocol config:

```bash
PYTHONPATH=src python -m sdss_point_benchmark.cli run-experiment \
  --config configs/sdss_dr17_l1735_1865_b30_40.json \
  --output reports/sdss_dr17_l1735_1865_b30_40_dry_run.json \
  --dry-run
```

Write an automated research dry-run plan without training:

```bash
make research-smoke-dry-run
```

Run the audit-style smoke research loop:

```bash
make research-smoke RESEARCH_DEVICE=cpu
```

This creates:

```text
reports/research_runs/<run_id>/plan.json
reports/research_runs/<run_id>/pilot_loop/
reports/research_runs/<run_id>/report.json
reports/research_runs/<run_id>/report.md
reports/research_runs/<run_id>/next_actions.json
artifacts/checkpoints/research_runs/<run_id>/
```

The research loop is conservative by design. Smoke runs and limited prediction
runs are marked as engineering checks, not paper-ready evidence.

Expand the default research program into an auditable queue:

```bash
make research-program-dry-run
```

After runs exist, build the cross-run board and evidence ledger:

```bash
make research-board
make research-next
make research-agent-plan
```

`research-agent-plan` writes `reports/research_runs/agent_plan.json/.md`. The
JSON is an autopilot-compatible program for the next conservative diagnostic
queue; e50 and multi-seed runs stay in `pending_approval_variants` by default.

Compare all runs or diagnose a specific report:

```bash
make research-compare-latest
make research-diagnose-latest RESEARCH_REPORT_DIR=reports/research_runs/<run_id>
```

Build the 100-field native-frame pilot dataset:

```bash
conda run -n sdss_point_py311 make dataset-pilot
```

Train the compact point-supervised baseline on prepared NPZ shards:

```bash
conda run -n sdss_point_py311 python -m sdss_point_benchmark.cli train \
  --config configs/sdss_dr17_l1735_1865_b30_40.json \
  --dataset artifacts/datasets/sdss_dr17_l1735_1865_b30_40_pilot \
  --output artifacts/checkpoints/sdss_pilot_baseline \
  --epochs 50 \
  --batch-size 32
```

Write predictions from a saved checkpoint:

```bash
conda run -n sdss_point_py311 python -m sdss_point_benchmark.cli predict \
  --checkpoint artifacts/checkpoints/sdss_pilot_baseline/best.pt \
  --dataset artifacts/datasets/sdss_dr17_l1735_1865_b30_40_pilot \
  --output reports/predictions/sdss_pilot_baseline.csv
```

For prepared NPZ datasets, prediction catalog `ra`/`dec` values are decoded from
local cutout pixels through the dataset `metadata.json` r-band native-frame WCS.
The `--pixel-scale-arcsec` flag is retained only for direct decoder fallback
paths that do not provide WCS metadata.

Expected CSV columns:

```text
prediction_id,cutout_id,ra,dec,label,score,x,y,mag_u,mag_g,mag_r,mag_i,mag_z,flux_u,flux_g,flux_r,flux_i,flux_z,size,ellipticity
```

Optional columns include `mag_u`, `mag_g`, `mag_i`, `mag_z`, `flux_u`,
`flux_g`, `flux_r`, `flux_i`, `flux_z`, `size`, `ellipticity`, `crowding`,
`snr`, `seeing`, `psf_fwhm`, `nearest_neighbor_arcsec`,
`galactic_latitude`, `x`, `y`, and `region_id`.

Evaluate predictions:

```bash
PYTHONPATH=src python -m sdss_point_benchmark.cli evaluate \
  --truth path/to/source_catalog.csv \
  --predictions path/to/prediction_catalog.csv \
  --output reports/metrics.json \
  --radius-arcsec 1 \
  --seeing-aware \
  --band r
```

Prediction CSV columns:

```text
prediction_id,cutout_id,ra,dec,label,score,mag_r
```

## Baseline Sketch

`sdss_point_benchmark.baseline.AstronomyAwareBaseline` consumes a 5-channel
`u/g/r/i/z` tensor and predicts:

- center heatmap logits;
- per-pixel class logits;
- non-negative per-band flux maps;
- three morphology proxy channels.

The included `BaselineLoss` combines:

- point-supervised center heatmap loss;
- optional class-map cross entropy at labeled source points;
- inverse-variance-weighted photometry loss;
- a small adjacent-band consistency regularizer.
- optional valid masks so unlabeled pixels are not forced to background;
- optional PSF-convolved reconstruction loss against the observed image.

The code is intentionally compact so that stronger baselines can replace the
encoder, add PSF-conditioned reconstruction, or move to set prediction while
keeping the same catalog and metric contract.

## Repository Layout

```text
src/sdss_point_benchmark/
  baseline.py       PyTorch baseline model and losses
  decode.py         heatmap-to-catalog decoding
  cli.py            split-generation command
  experiment.py     config validation and reproducible dry-run reports
  io.py             CSV catalog loader
  matching.py       angular source matching
  metrics.py        benchmark metrics
  schema.py         catalog dataclasses
  split.py          sky-region split protocol
  synthetic.py      controlled source injection helpers
  sdss_dr17.py      SDSS DR17 manifest and PhotoObj adapter helpers
  cutouts.py        cutout worklist writer
configs/
  benchmark_v1.json example benchmark protocol knobs
  sdss_dr17_l1735_1865_b30_40.json local DR17 experiment config
docs/
  protocol.md       benchmark/evaluation details
  research/         method design, experiment plan, automation runbook
tests/
  *.py              unittest coverage for core behavior
```

## Engineering Standards

The package is configured as a typed Python package with a PEP 561 marker
(`py.typed`). `pyproject.toml` also declares pytest and ruff settings so the same
code-quality contract can be used locally and in automation.

Common development commands:

```bash
make test-conda
make smoke-conda
make lint-conda
```

See `CONTRIBUTING.md` and `AGENTS.md` for repository conventions, artifact
locations, and automation rules.

## Current Limits

This scaffold does not download SDSS data, publish fixed splits, run full
paper-scale training, ingest real psField files, or perform real survey
cross-matches. It includes local DR17 manifest/adaptation helpers, native-frame
pilot dataset generation, compact NPZ dataloader training, checkpoint
serialization, and prediction decoding from saved weights.
