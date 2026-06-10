# Contributing

## Development Commands

Create or update the recommended Python 3.11 conda environment:

```bash
make env-create
# or, if sdss_point_py311 already exists:
make env-update
```

Run the conda-first verification checks:

```bash
make verify-conda
```

Equivalent individual checks:

```bash
make test-conda
make smoke-conda
make lint-conda
```

## Code Standards

- Keep public catalog contracts stable unless the protocol version changes.
- Add or update tests before changing behavior.
- Keep raw survey data outside the repository.
- Write generated outputs to `artifacts/` or `reports/`.
- Prefer small modules with one responsibility over expanding `cli.py` or model files.

## Research Reproducibility

Every experiment report should include:

- protocol version;
- config snapshot;
- generated timestamp;
- planned experiment IDs;
- artifact layout;
- data root and split seed.

The canonical helper for dry-run reports is
`sdss_point_benchmark.experiment.build_dry_run_report`.
