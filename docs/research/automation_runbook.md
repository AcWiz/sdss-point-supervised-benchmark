# Automation Runbook

For Codex sessions acting as autonomous research co-authors, first read
`docs/research/codex_research_agent.md`. That playbook defines the required
session preflight, evidence ladder, GPU policy, review checklist, and
end-of-session summary format.

## Directory Policy

Raw SDSS data stays outside the repository at:

```text
/Data/sdss/sdss_dr17_l1735_1865_b30_40
```

Generated files should go to:

- `artifacts/manifests/` for manifest and cutout worklists;
- `artifacts/splits/` for fixed split JSON;
- `artifacts/checkpoints/` for model weights;
- `reports/` for metrics and experiment reports.

## Baseline Verification

```bash
make verify-conda
```

## Build Field Manifest

```bash
conda run -n sdss_point_py311 python -m sdss_point_benchmark.cli build-manifest \
  --data-root /Data/sdss/sdss_dr17_l1735_1865_b30_40 \
  --output artifacts/manifests/sdss_dr17_l1735_1865_b30_40_fields.csv
```

## Build Pilot Source Catalog

```bash
conda run -n sdss_point_py311 python -m sdss_point_benchmark.cli build-source-catalog \
  --config configs/sdss_dr17_l1735_1865_b30_40.json \
  --output artifacts/manifests/sdss_dr17_l1735_1865_b30_40_pilot100_source_catalog.csv \
  --limit-fields 100 \
  --clean-only
```

## Freeze Pilot Split

```bash
conda run -n sdss_point_py311 python -m sdss_point_benchmark.cli split \
  --catalog artifacts/manifests/sdss_dr17_l1735_1865_b30_40_pilot100_source_catalog.csv \
  --output artifacts/splits/sdss_dr17_l1735_1865_b30_40_pilot100_seed42_skybin_v1.json \
  --region-mode sky-bin \
  --ra-bin-deg 1 \
  --dec-bin-deg 1 \
  --seed 42
```

## Prepare Cutout Worklist

```bash
conda run -n sdss_point_py311 python -m sdss_point_benchmark.cli prepare-cutouts \
  --catalog artifacts/catalogs/source_catalog.csv \
  --output artifacts/manifests/cutouts_trainvaltest.csv \
  --cutout-size 128
```

## Dry-Run Experiment Report

```bash
conda run -n sdss_point_py311 python -m sdss_point_benchmark.cli run-experiment \
  --config configs/sdss_dr17_l1735_1865_b30_40.json \
  --output reports/sdss_dr17_l1735_1865_b30_40_dry_run.json \
  --dry-run
```

## Audit-Style Automated Research Loop

The repository now includes a conservative automated research loop. It follows a
fixed sequence:

```text
preflight -> plan.json -> train -> validation threshold sweep -> test metrics
-> report.json/report.md -> next_actions.json
```

This loop is inspired by automated research systems, but it is deliberately
auditable and repo-native. It does not modify model code, change public splits,
or mark any run as paper-ready automatically.

Dry-run a smoke research plan:

```bash
make research-smoke-dry-run
```

Run the smoke research loop:

```bash
make research-smoke RESEARCH_DEVICE=cpu
```

Run the paper-scale pilot once the pilot dataset and split are prepared:

```bash
make research-pilot RESEARCH_DEVICE=cuda:0
```

To turn an existing `run-pilot-loop` directory into a research report without
retraining:

```bash
make research-report-existing
```

Standard outputs are:

```text
reports/research_runs/<run_id>/plan.json
reports/research_runs/<run_id>/pilot_loop/
reports/research_runs/<run_id>/report.json
reports/research_runs/<run_id>/report.md
reports/research_runs/<run_id>/next_actions.json
artifacts/checkpoints/research_runs/<run_id>/
```

Claim gates are intentionally conservative:

- `blocked`: missing metrics or failed evidence checks;
- `engineering_check`: smoke, limited, or short training run;
- `candidate_evidence`: non-smoke run with nonzero test evidence, still not
  paper-ready.

## Research Program Layer

The v2 automation layer adds a repo-native research-program manager. It does
not rewrite model code or create new public splits. It expands configuration
variants, tracks runs, compares evidence, and records what claims still need
support.

Expand the default program queue without training:

```bash
make research-program-dry-run
```

This writes:

```text
reports/research_program_v1/program_plan.json
```

Execute the configured variants only when the datasets, split files, and
hardware budget are ready:

```bash
make research-program-execute RESEARCH_DEVICE=cuda:0
```

Build cross-run state:

```bash
make research-board
make research-next
make research-compare-latest
make research-agent-plan
```

These write:

```text
reports/research_runs/index.jsonl
reports/research_runs/board.json
reports/research_runs/board.md
reports/research_runs/evidence_ledger.json
reports/research_runs/compare_latest.json
reports/research_runs/compare_latest.md
reports/research_runs/agent_plan.json
reports/research_runs/agent_plan.md
```

The agent plan is a conservative, autopilot-compatible program generated from
the current evidence board. It queues e5/e20 diagnostics automatically and keeps
paper-scale e50 or multi-seed runs under `pending_approval_variants` unless the
autonomy mode is explicitly changed.

Diagnose a specific run:

```bash
make research-diagnose-latest RESEARCH_REPORT_DIR=reports/research_runs/<run_id>
```

The diagnosis report focuses on failure modes that matter for this project:
empty truth, empty predictions, zero recall with nonzero candidates, low
precision, and degenerate validation threshold selection.

Each single run now also writes:

```text
run_manifest.json
state.json
```

These files are machine-readable summaries for schedulers, dashboards, and
future paper-writing tools.

## Build Pilot Dataset

```bash
conda run -n sdss_point_py311 make dataset-pilot-dry-run
conda run -n sdss_point_py311 make dataset-pilot
```

This writes native-frame NPZ cutout shards under:

```text
artifacts/datasets/sdss_dr17_l1735_1865_b30_40_pilot/
```

The pilot defaults are 100 fields, 128 pixel cutouts, 1024 samples per shard,
and float16 images.

## Train Pilot Baseline

```bash
conda run -n sdss_point_py311 python -m sdss_point_benchmark.cli train \
  --config configs/sdss_dr17_l1735_1865_b30_40.json \
  --dataset artifacts/datasets/sdss_dr17_l1735_1865_b30_40_pilot \
  --output artifacts/checkpoints/sdss_pilot_baseline \
  --epochs 50 \
  --batch-size 32 \
  --device cpu
```

The output directory contains:

- `best.pt` checkpoint;
- `training_report.json` with loss history and config snapshot.

Use `--device cuda` on GPU machines after confirming the PyTorch environment has
CUDA support.

## Predict Pilot Catalog

```bash
conda run -n sdss_point_py311 python -m sdss_point_benchmark.cli predict \
  --checkpoint artifacts/checkpoints/sdss_pilot_baseline/best.pt \
  --dataset artifacts/datasets/sdss_dr17_l1735_1865_b30_40_pilot \
  --output reports/predictions/sdss_pilot_baseline.csv \
  --batch-size 32 \
  --threshold 0.5 \
  --nms-radius 2
```

## Evaluate Predictions

```bash
conda run -n sdss_point_py311 python -m sdss_point_benchmark.cli evaluate \
  --truth artifacts/catalogs/test_truth.csv \
  --predictions reports/predictions/sdss_pilot_baseline.csv \
  --output reports/metrics_test.json \
  --radius-arcsec 1.0 \
  --seeing-aware \
  --band r
```

## Current Execution Boundary

The current training loop is intentionally compact: it trains the baseline on
prepared NPZ shards, uses Gaussian heatmaps from catalog points, sparse
photometry targets from catalog magnitudes, and a Gaussian PSF proxy for the
reconstruction loss. The next research stage is real psField ingestion,
validation split threshold tuning, and full paper-scale experiment orchestration.
