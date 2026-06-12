# Mainline Validation Status

Current mainline candidate:

- Run: `sdss_point_catalog_v2_pilot_agent_loss_diagnosis_v1_ablation_center_only_unet_lite_e50_seed42`
- Variant: `ablation_center_only_unet_lite_e50_seed42`
- Gate: `candidate_evidence`
- Test metrics: F1 `0.7808180406495979`, AP `0.6869429906582407`, precision `0.8498234980684695`, recall `0.722177413649163`

Paired e50 audit:

- Audit: `reports/research_runs/center_only_e50_vs_baseline_e50_audit.md`
- Baseline: `sdss_point_catalog_v2_pilot_pilot100_baseline_e50`
- Delta vs baseline e50: F1 `+0.1356885708592156`, AP `+0.17523711030937705`, recall `+0.1484201440477706`
- Bootstrap CI is positive for F1, AP, and recall under validation-selected thresholds.

Claimable strata from the audit:

- `mag_r`
- `label`
- `label_quality`
- `photoobj_flags`
- `nearest_neighbor_arcsec_derived`
- `source_density_per_cutout`

Known validation gaps:

- `seeing` is unavailable in the current truth catalog.
- `snr` is unavailable in the current truth catalog.
- Catalog-provided nearest-neighbor/crowding fields are unavailable, but derived nearest-neighbor and source-density strata are available.
- These are still PhotoObj weak-supervision results, not final paper truth.

Synthetic-injection validation:

- Run: `reports/research_runs/synthetic_injection_center_only_e50_v1`
- Checkpoint: `artifacts/checkpoints/research_runs/sdss_point_catalog_v2_pilot_agent_loss_diagnosis_v1_ablation_center_only_unet_lite_e50_seed42/best.pt`
- Gate: `candidate_evidence`
- Protocol: fixed split test backgrounds, 16 backgrounds, 4 injected sources per background, fixed threshold `0.2`, NMS radius `2`, match radius `2` pixels.
- Injected-source metrics: precision `0.35877862595419846`, recall `0.734375`, F1 `0.482051282051282`, AP `0.295266554928252`.
- All-truth metrics including existing weak background labels: precision `0.7099236641221374`, recall `0.808695652173913`, F1 `0.7560975609756097`.
- Injected recall by `mag_r`:
  - `[17.5,19.0)`: `0.875`
  - `[19.0,20.5)`: `0.9375`
  - `[20.5,21.5)`: `1.0`
  - `[21.5,22.5)`: `0.125`

Current interpretation:

The paired e50 audit supports center-only e50 as the current mainline candidate under weak PhotoObj agreement, and synthetic injection confirms nonzero controlled-source recovery. The weak point is the faintest injected bin: recall drops to `0.125` for `[21.5,22.5)`, which is now the highest-priority mainline validation risk.

Faint-source diagnostics:

- Faint diagnostic: `reports/research_runs/synthetic_injection_center_only_e50_faint_diagnostic_v1`
- Low-floor diagnostic: `reports/research_runs/synthetic_injection_center_only_e50_low_floor_diagnostic_v1`
- Heatmap-response diagnostic: `reports/research_runs/synthetic_injection_center_only_e50_heatmap_diagnostic_v1`
- Balanced synthetic validation: `reports/research_runs/synthetic_injection_center_only_e50_balanced_v2`
- Balanced heatmap-response diagnostic: `reports/research_runs/synthetic_injection_center_only_e50_balanced_heatmap_diagnostic_v2`
- Morphology-response diagnostic: `reports/research_runs/synthetic_injection_center_only_e50_morphology_diagnostic_v1`
- Lowering the candidate floor from `0.2` to `0.05` and raising `max_detections_per_cutout` from `16` to `64` increases candidates from `131` to `156`, but `[21.5,22.5)` recall stays at `0.125`.
- The faint diagnostic finds `14/16` false negatives in `[21.5,22.5)`, and all 14 have no decoded candidate within the `2` pixel match radius.
- The heatmap-response diagnostic confirms this is not just score-thresholding: for `[21.5,22.5)`, center-score median is `0.00039810704765841365`, best local score median within `8` pixels is `0.010757192969322205`, and only `0.125` of sources have match-radius local response above low floor `0.05`.
- The balanced protocol removes the initial magnitude/type confound by injecting both `star` and `galaxy` at each magnitude. Under this protocol, `[21.5,22.5)` aggregate recall improves to `0.40625`, but the split is highly asymmetric: `star` recall is `0.75`, while `galaxy` recall is `0.0625`.
- Balanced heatmap response shows the same asymmetry: for `[21.5,22.5)`, `star` match-radius low-floor response is `0.75` with best local median `0.7933900654315948`; `galaxy` response is `0.125` with best local median `0.1650799959897995`.
- The morphology sweep isolates the failure to extended low-surface-brightness profiles. At `mag_r=22`, compact galaxy radius `1.3` has recall `0.9375` and match-radius low-floor response `0.9375`, while galaxy radius `2.4` has recall `0.0625` and response `0.125`, and galaxy radius `3.6` has recall `0.0` and response `0.0`. The star control at radius `1.3` has recall `0.75` and response `0.75`.

Recommended mainline next step:

Design a short mainline rescue experiment for faint extended-source center response before approving more long training runs or multi-seed expansion. The smallest useful next move should target surface-brightness or extended-profile handling and report both the morphology sweep response and injected recall by condition.
