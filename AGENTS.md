# Agent Operating Notes

This project is a research benchmark and method scaffold for SDSS
point-supervised source catalog generation.

## Ground Rules

- Do not move, rewrite, or delete raw data under `/Data/sdss`.
- Keep raw-data paths in config files; keep generated artifacts out of source
  modules.
- Preserve fixed split files once created. Changing a public split requires a
  new protocol/config version.
- Use native SDSS corrected-frame data for headline results unless an experiment
  explicitly says it is a reprojection ablation.
- Before claiming completion, run:

```bash
make test
```

## Research Defaults

- Main target: Benchmark + Method paper.
- Method: PSF-constrained point-supervised catalog generation.
- Hardware assumption: single machine with 1-2 GPUs.
- First real dataset: `/Data/sdss/sdss_dr17_l1735_1865_b30_40`.
- For research-agent sessions, first read
  `docs/research/codex_research_agent.md` and follow its start/end protocol.

## Artifact Locations

- Manifests: `artifacts/manifests/`
- Splits: `artifacts/splits/`
- Checkpoints: `artifacts/checkpoints/`
- Reports: `reports/`
