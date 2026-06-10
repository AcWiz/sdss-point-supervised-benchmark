.PHONY: test lint smoke test-conda lint-conda smoke-conda verify-conda check-cuda-conda env-create env-update dataset-pilot-dry-run dataset-pilot train-pilot predict-pilot pilot-loop-smoke research-smoke-dry-run research-smoke research-pilot research-report-existing research-program-dry-run research-program-execute research-autopilot-dry-run research-autopilot research-board research-next research-compare-latest research-diagnose-latest

PYTHON ?= python
CONDA_ENV ?= sdss_point_py311
DATASET_CONFIG ?= configs/sdss_dr17_l1735_1865_b30_40.json
PILOT_OUTPUT ?= artifacts/datasets/sdss_dr17_l1735_1865_b30_40_pilot
CHECKPOINT_DIR ?= artifacts/checkpoints/sdss_pilot_baseline
PREDICTIONS_OUTPUT ?= reports/predictions/sdss_pilot_baseline.csv
SMOKE_DATASET ?= artifacts/datasets/sdss_dr17_l1735_1865_b30_40_smoke5
SMOKE_SPLIT ?= artifacts/splits/sdss_dr17_l1735_1865_b30_40_smoke5_seed42_skybin_v1.json
PILOT_LOOP_SMOKE_CHECKPOINT ?= artifacts/checkpoints/sdss_pilot_loop_smoke5
PILOT_LOOP_SMOKE_REPORT ?= reports/pilot_split_loop_smoke5
PILOT_LOOP_DEVICE ?= cuda:0
RESEARCH_RUN_ID ?= smoke_$(shell date -u +%Y%m%dT%H%M%SZ)
RESEARCH_DEVICE ?= cpu
RESEARCH_EPOCHS ?= 1
RESEARCH_BATCH_SIZE ?= 16
RESEARCH_BASE_CHANNELS ?= 8
RESEARCH_REPORT_DIR ?= reports/research_runs/$(RESEARCH_RUN_ID)
RESEARCH_CHECKPOINT_DIR ?= artifacts/checkpoints/research_runs/$(RESEARCH_RUN_ID)
RESEARCH_OBJECTIVE ?= Audit the point-supervised SDSS catalog-generation loop.
RESEARCH_HYPOTHESIS ?= The current PSF-constrained baseline can produce measurable validation/test evidence under the fixed split.
RESEARCH_PROGRAM ?= configs/research_program_v1.json
RESEARCH_AUTOPILOT_PROGRAM ?= configs/research_program_v2_pilot.json
RESEARCH_PROGRAM_OUT ?= reports/research_program_v1
RESEARCH_AUTOPILOT_OUT ?= reports/research_scheduler/pilot_v2
RESEARCH_ROOT ?= reports/research_runs
RESEARCH_COMPARE_OUT ?= reports/research_runs/compare_latest.json
RESEARCH_DIAGNOSE_REPORT ?= $(RESEARCH_REPORT_DIR)/report.json

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m unittest discover -s tests

lint:
	$(PYTHON) -m ruff check src tests

smoke:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli run-experiment \
		--config configs/sdss_dr17_l1735_1865_b30_40.json \
		--output /tmp/sdss_point_benchmark_smoke.json \
		--dry-run

test-conda:
	conda run -n $(CONDA_ENV) make test PYTHON=python

lint-conda:
	conda run -n $(CONDA_ENV) make lint PYTHON=python

smoke-conda:
	conda run -n $(CONDA_ENV) make smoke PYTHON=python

verify-conda: test-conda smoke-conda lint-conda

check-cuda-conda:
	conda run -n $(CONDA_ENV) python -c "import torch; print('torch.cuda.is_available=', torch.cuda.is_available())" || true

env-create:
	conda env create -f environment.yml

env-update:
	conda env update -n $(CONDA_ENV) -f environment.yml --prune

dataset-pilot-dry-run:
	conda run -n $(CONDA_ENV) python -m sdss_point_benchmark.cli build-dataset \
		--config $(DATASET_CONFIG) \
		--output-dir $(PILOT_OUTPUT) \
		--limit-fields 100 \
		--cutout-size 128 \
		--shard-size 1024 \
		--dtype float16 \
		--dry-run

dataset-pilot:
	conda run -n $(CONDA_ENV) python -m sdss_point_benchmark.cli build-dataset \
		--config $(DATASET_CONFIG) \
		--output-dir $(PILOT_OUTPUT) \
		--limit-fields 100 \
		--cutout-size 128 \
		--shard-size 1024 \
		--dtype float16

train-pilot:
	conda run -n $(CONDA_ENV) python -m sdss_point_benchmark.cli train \
		--config $(DATASET_CONFIG) \
		--dataset $(PILOT_OUTPUT) \
		--output $(CHECKPOINT_DIR) \
		--epochs 50 \
		--batch-size 32 \
		--device cpu

predict-pilot:
	conda run -n $(CONDA_ENV) python -m sdss_point_benchmark.cli predict \
		--checkpoint $(CHECKPOINT_DIR)/best.pt \
		--dataset $(PILOT_OUTPUT) \
		--output $(PREDICTIONS_OUTPUT) \
		--batch-size 32 \
		--threshold 0.5 \
		--nms-radius 2 \
		--device cpu

pilot-loop-smoke:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli run-pilot-loop \
		--config $(DATASET_CONFIG) \
		--dataset $(SMOKE_DATASET) \
		--split $(SMOKE_SPLIT) \
		--output-dir $(PILOT_LOOP_SMOKE_REPORT) \
		--checkpoint-dir $(PILOT_LOOP_SMOKE_CHECKPOINT) \
		--epochs 1 \
		--batch-size 16 \
		--base-channels 8 \
		--candidate-threshold 0.95 \
		--nms-radius 2 \
		--max-detections-per-cutout 16 \
		--predict-limit 64 \
		--radius-arcsec 1.0 \
		--seeing-aware \
		--device $(PILOT_LOOP_DEVICE)

research-smoke-dry-run:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-run \
		--config $(DATASET_CONFIG) \
		--dataset $(SMOKE_DATASET) \
		--split $(SMOKE_SPLIT) \
		--run-id $(RESEARCH_RUN_ID) \
		--objective "$(RESEARCH_OBJECTIVE)" \
		--hypothesis "$(RESEARCH_HYPOTHESIS)" \
		--report-dir $(RESEARCH_REPORT_DIR) \
		--checkpoint-dir $(RESEARCH_CHECKPOINT_DIR) \
		--epochs $(RESEARCH_EPOCHS) \
		--batch-size $(RESEARCH_BATCH_SIZE) \
		--base-channels $(RESEARCH_BASE_CHANNELS) \
		--candidate-threshold 0.95 \
		--nms-radius 2 \
		--max-detections-per-cutout 16 \
		--predict-limit 64 \
		--radius-arcsec 1.0 \
		--seeing-aware \
		--device $(RESEARCH_DEVICE) \
		--dry-run

research-smoke:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-run \
		--config $(DATASET_CONFIG) \
		--dataset $(SMOKE_DATASET) \
		--split $(SMOKE_SPLIT) \
		--run-id $(RESEARCH_RUN_ID) \
		--objective "$(RESEARCH_OBJECTIVE)" \
		--hypothesis "$(RESEARCH_HYPOTHESIS)" \
		--report-dir $(RESEARCH_REPORT_DIR) \
		--checkpoint-dir $(RESEARCH_CHECKPOINT_DIR) \
		--epochs $(RESEARCH_EPOCHS) \
		--batch-size $(RESEARCH_BATCH_SIZE) \
		--base-channels $(RESEARCH_BASE_CHANNELS) \
		--candidate-threshold 0.95 \
		--nms-radius 2 \
		--max-detections-per-cutout 16 \
		--predict-limit 64 \
		--radius-arcsec 1.0 \
		--seeing-aware \
		--device $(RESEARCH_DEVICE)

research-pilot:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-run \
		--config $(DATASET_CONFIG) \
		--dataset $(PILOT_OUTPUT) \
		--split artifacts/splits/sdss_dr17_l1735_1865_b30_40_pilot100_seed42_skybin_v1.json \
		--run-id $(RESEARCH_RUN_ID) \
		--objective "$(RESEARCH_OBJECTIVE)" \
		--hypothesis "$(RESEARCH_HYPOTHESIS)" \
		--report-dir $(RESEARCH_REPORT_DIR) \
		--checkpoint-dir $(RESEARCH_CHECKPOINT_DIR) \
		--epochs 50 \
		--batch-size 32 \
		--base-channels 32 \
		--candidate-threshold 0.2 \
		--nms-radius 2 \
		--radius-arcsec 1.0 \
		--seeing-aware \
		--device $(RESEARCH_DEVICE)

research-report-existing:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-report \
		--pilot-output-dir $(PILOT_LOOP_SMOKE_REPORT) \
		--run-id $(RESEARCH_RUN_ID) \
		--report-dir $(RESEARCH_REPORT_DIR) \
		--objective "$(RESEARCH_OBJECTIVE)" \
		--hypothesis "$(RESEARCH_HYPOTHESIS)"

research-program-dry-run:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-program \
		--program $(RESEARCH_PROGRAM) \
		--output-dir $(RESEARCH_PROGRAM_OUT)

research-program-execute:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-program \
		--program $(RESEARCH_PROGRAM) \
		--output-dir $(RESEARCH_PROGRAM_OUT) \
		--execute

research-autopilot-dry-run:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-autopilot \
		--program $(RESEARCH_AUTOPILOT_PROGRAM) \
		--output-dir $(RESEARCH_AUTOPILOT_OUT)

research-autopilot:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-autopilot \
		--program $(RESEARCH_AUTOPILOT_PROGRAM) \
		--output-dir $(RESEARCH_AUTOPILOT_OUT) \
		--execute

research-board:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-board \
		--root $(RESEARCH_ROOT) \
		--output $(RESEARCH_ROOT)/board.json \
		--markdown-output $(RESEARCH_ROOT)/board.md \
		--rebuild-index

research-next:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-next \
		--root $(RESEARCH_ROOT) \
		--output $(RESEARCH_ROOT)/evidence_ledger.json

research-compare-latest:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-compare \
		--root $(RESEARCH_ROOT) \
		--output $(RESEARCH_COMPARE_OUT) \
		--markdown-output $(RESEARCH_COMPARE_OUT:.json=.md)

research-diagnose-latest:
	PYTHONPATH=src $(PYTHON) -m sdss_point_benchmark.cli research-diagnose \
		--report $(RESEARCH_DIAGNOSE_REPORT) \
		--output $(RESEARCH_REPORT_DIR)/diagnosis.json \
		--markdown-output $(RESEARCH_REPORT_DIR)/diagnosis.md
