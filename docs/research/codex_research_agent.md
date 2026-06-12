# Codex Research Agent Playbook

This document is the project memory for Codex sessions that act as a research
co-author rather than a narrow code executor. Use it before changing research
automation, launching experiments, interpreting metrics, or proposing the next
paper step.

The goal is to keep each session auditable: every run should start from the
current evidence state, execute a small defensible research move, and end with a
review that a future agent can trust.

## Design Principles

- Start from evidence, not ambition. Read the latest reports before proposing a
  new experiment.
- Treat SDSS PhotoObj as weak supervision, not final truth. Paper claims need
  stratified metrics, synthetic injection, or independent catalog checks.
- Separate engineering checks from scientific evidence. A smoke run can prove
  the loop works; it cannot support a paper claim.
- Prefer fixed protocols over clever one-off runs. The fixed split, native-frame
  policy, and claim gates are part of the method.
- Make expensive GPU work earn its cost. Run the smallest diagnostic that can
  falsify the current hypothesis before launching long jobs.
- Preserve negative results. A failed run is useful when it has a manifest,
  diagnosis, and concrete next action.

These principles are adapted from successful automated-research systems:
end-to-end idea, experiment, paper, and review loops as in AI Scientist
(https://arxiv.org/abs/2408.06292), agentic experiment managers as in AI
Scientist-v2 (https://arxiv.org/abs/2504.08066), tool-grounded autonomous
scientific execution as in Coscientist
(https://www.nature.com/articles/s41586-023-06792-0), and benchmark-harness
discipline from software-agent evaluation such as SWE-bench
(https://www.swebench.com/).

## Session Start Protocol

Run this read-only preflight at the beginning of a research session:

```bash
git status --short --branch
sed -n '1,220p' AGENTS.md
sed -n '1,260p' docs/research/codex_research_agent.md
sed -n '1,220p' reports/research_runs/board.md 2>/dev/null || true
sed -n '1,220p' reports/research_runs/compare_latest.md 2>/dev/null || true
```

If the task may use GPU, also run:

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
conda run -n sdss_point_py311 python -c "import torch; print('cuda=', torch.cuda.is_available()); print('devices=', torch.cuda.device_count())"
```

Before launching an experiment, refresh the local evidence index:

```bash
make research-board
make research-next
make research-compare-latest
```

Do not overwrite or remove unrelated partial runs. If a run is stale, document
why it is being superseded and keep its report unless the user explicitly asks
for cleanup.

## Research Loop

Use this loop for each autonomous session:

1. Observe: inspect code, configs, latest reports, GPU state, and git status.
2. State hypothesis: write the specific claim being tested in one sentence.
3. Choose intervention: prefer the smallest code/config/experiment change that
   can test the hypothesis.
4. Execute: run the command with fixed config paths and record output locations.
5. Evaluate: compare against the current board, not just a single metric.
6. Critique: look for leakage, threshold tuning mistakes, degenerate decoding,
   resource bottlenecks, and unsupported claims.
7. Plan next: leave one to three ranked next actions with expected evidence.

The loop should produce one of these outcomes:

- code change with tests;
- completed research run with report, diagnosis, and board update;
- blocked state with the exact missing dependency or hardware condition;
- planning-only memo when the user explicitly asked for planning.

## Evidence Ladder

Use this ladder when interpreting claim gates:

- `blocked`: missing files, invalid metrics, failed run, or no usable evidence.
- `engineering_check`: proves the pipeline, loader, GPU path, or report writing.
- `candidate_evidence`: non-smoke fixed-split evidence with nonzero meaningful
  metrics; useful for deciding the next experiment.
- paper support: requires repeated runs or stronger validation, stratified
  metrics, fixed validation-selected thresholds, and no known leakage.
- headline result: requires the final fixed test protocol, paper baselines,
  stress tests, and a written limitation analysis.

Never describe `candidate_evidence` as paper-ready. It is a decision point, not
a final claim.

## Experiment Card

Before starting a nontrivial run, construct this card in the session notes or in
the run objective:

```text
Question:
Hypothesis:
Variant/config:
Dataset and split:
GPU/resource budget:
Success signal:
Failure signal:
Claims touched:
Expected artifact paths:
```

Good success signals are concrete, for example "UNet-lite e50 improves AP over
baseline e50 under the same fixed split" or "batch size 96 raises GPU memory use
without reducing throughput". Avoid vague signals like "better performance".

## GPU And Throughput Policy

The project assumes one to two GPUs. GPU experiments should be explicit about
batch size, cache size, worker count, device, and memory use.

When GPU utilization is low and memory use is small, adjust in this order:

1. Increase `batch_size` until memory is meaningfully used but stable.
2. Increase `shard_cache_size` if data loading rereads shards too often.
3. Increase `num_workers` when CPU loading limits throughput.
4. Enable or verify `pin_memory` for CUDA runs.
5. Only then consider model-size changes.

Record the before/after throughput and memory in the report or final summary.
Do not kill unrelated GPU processes. If another process owns the GPU, either
choose a free device or wait.

## Claim And Paper Discipline

Each run should map to one or more claims in the research program config. A
claim is only strengthened when the run satisfies all relevant protocol
conditions:

- fixed split exists and is unchanged;
- native corrected-frame data is used unless the run is an explicit reprojection
  ablation;
- validation choices are not tuned on the test set;
- metrics include precision, recall, F1, AP, and relevant stratification;
- the run has `plan.json`, `run_manifest.json`, `report.json`, `report.md`,
  `state.json`, and `next_actions.json`;
- the board and evidence ledger are refreshed after the run.

For method claims, compare to a meaningful baseline under the same data,
training budget, threshold policy, and decode settings.

## Review Checklist

Before ending a session, review these failure modes:

- Data: raw files under `/Data/sdss` were not moved, rewritten, or deleted.
- Split: no fixed split was silently regenerated or replaced.
- Protocol: native-frame policy is preserved for headline runs.
- Metrics: test metrics were not used to select thresholds or variants.
- Decode: threshold/NMS/max detections are recorded and not accidentally
  degenerate.
- Matching: cutout-aware matching is used and radius policy is reported.
- Claims: gates match the evidence level; no smoke run supports a paper claim.
- Artifacts: generated files stay in `artifacts/` or `reports/`.
- Git: unrelated user changes are not reverted.
- Verification: `make test` passes before claiming completion.

## End-Of-Session Summary Template

End every substantial Codex session with this structure:

```text
What changed:
Evidence:
Verification:
Current interpretation:
Risks or gaps:
Next actions:
```

`Evidence` should include exact run IDs and metrics when available. `Next
actions` should be ranked and limited to one to three items.

## Recommended Next-Step Heuristics

Choose the next action with this priority order:

1. Correctness bugs that could invalidate metrics.
2. Missing automation that prevents reproducible runs.
3. Cheap diagnostics that clarify a failure mode.
4. Pilot-scale comparisons that decide architecture or loss direction.
5. Paper-scale runs only after pilot evidence justifies them.
6. Paper writing and figures only when the evidence ledger is coherent.

For the current project state, a typical next sequence is:

1. Keep the e5 candidate evidence as the pilot architecture screen.
2. Run matched e50 baseline and UNet-lite jobs only after GPU settings are sized
   for stable utilization.
3. Add or verify gate-aware scheduling so dependent e50 runs do not launch only
   because a parent process finished; they should launch because the parent
   evidence passed the intended gate.

## Agent Anti-Patterns

Avoid these behaviors:

- launching a long run before reading the latest board;
- changing split files to improve metrics;
- reporting only aggregate F1 when the claim is about faint or blended sources;
- treating PhotoObj agreement as final truth;
- hiding failed runs by deleting reports;
- expanding the automation layer without tests;
- starting paper-scale jobs while the GPU path is clearly underutilized;
- ending with "works" but no exact verification command.
