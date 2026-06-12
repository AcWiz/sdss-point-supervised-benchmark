import json
import tempfile
import unittest
from pathlib import Path

from sdss_point_benchmark.cli import main
from sdss_point_benchmark.research_agent import build_research_agent_plan


class ResearchAgentPlanTests(unittest.TestCase):
    def test_agent_plan_identifies_center_only_champion_and_missing_full_e20(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_report(
                reports,
                run_id="pilot_center_only_e20",
                variant_id="ablation_center_only_unet_lite_e20",
                loss_variant="center_only",
                epochs=20,
                f1=0.78,
                ap=0.68,
            )
            _write_report(
                reports,
                run_id="pilot_no_psf_e20",
                variant_id="ablation_no_psf_unet_lite_e20",
                loss_variant="no_psf_reconstruction",
                epochs=20,
                f1=0.67,
                ap=0.50,
            )

            payload = build_research_agent_plan(program, root=reports)

        agent = payload["agent_plan"]
        queue_ids = [variant["variant_id"] for variant in payload["variants"]]
        pending_ids = [variant["variant_id"] for variant in payload["pending_approval_variants"]]

        self.assertEqual(agent["current_state"]["detection_champion"]["run_id"], "pilot_center_only_e20")
        self.assertIn("ablation_full_psf_unet_lite_e20_matched_bs192", queue_ids)
        self.assertIn("ablation_center_only_unet_lite_e50_seed42_pending", pending_ids)
        self.assertNotIn("ablation_center_only_unet_lite_e50_seed42_pending", queue_ids)
        self.assertIn("psf_constrained_method", {row["claim"] for row in agent["blocked_claims"]})

    def test_factorial_auxiliary_variants_only_enable_one_auxiliary_term(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_report(
                reports,
                run_id="pilot_center_only_e20",
                variant_id="ablation_center_only_unet_lite_e20",
                loss_variant="center_only",
                epochs=20,
                f1=0.78,
                ap=0.68,
            )

            payload = build_research_agent_plan(program, root=reports)

        variants = {variant["variant_id"]: variant for variant in payload["variants"]}
        class_run = variants["ablation_center_class_unet_lite_e5"]["run"]
        photometry_run = variants["ablation_center_photometry_unet_lite_e5"]["run"]
        psf_run = variants["ablation_center_psf_unet_lite_e5"]["run"]

        self.assertEqual(class_run["center_loss_weight"], 1.0)
        self.assertEqual(class_run["class_loss_weight"], 0.5)
        self.assertEqual(class_run["photometry_loss_weight"], 0.0)
        self.assertEqual(class_run["multiband_loss_weight"], 0.0)
        self.assertEqual(class_run["psf_reconstruction_loss_weight"], 0.0)
        self.assertEqual(photometry_run["photometry_loss_weight"], 1.0)
        self.assertEqual(photometry_run["class_loss_weight"], 0.0)
        self.assertEqual(psf_run["psf_reconstruction_loss_weight"], 0.2)
        self.assertEqual(psf_run["class_loss_weight"], 0.0)

    def test_agent_plan_omits_completed_parent_but_keeps_unfinished_ready_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_report(
                reports,
                run_id="pilot_center_only_e20",
                variant_id="ablation_center_only_unet_lite_e20",
                loss_variant="center_only",
                epochs=20,
                f1=0.78,
                ap=0.68,
            )
            _write_report(
                reports,
                run_id="tiny_agent_loss_diagnosis_v1_ablation_center_class_unet_lite_e5",
                variant_id="ablation_center_class_unet_lite_e5",
                loss_variant="full_psf_point_supervised",
                epochs=5,
                f1=0.7,
                ap=0.6,
            )

            payload = build_research_agent_plan(program, root=reports)

        completed = payload["agent_plan"]["current_state"]["completed_variants"]
        queue_ids = [variant["variant_id"] for variant in payload["variants"]]
        class_e20 = {variant["variant_id"]: variant for variant in payload["variants"]}[
            "ablation_center_class_unet_lite_e20"
        ]
        self.assertIn("ablation_center_class_unet_lite_e5", completed)
        self.assertNotIn("ablation_center_class_unet_lite_e5", queue_ids)
        self.assertIn("ablation_center_class_unet_lite_e20", queue_ids)
        self.assertNotIn("depends_on", class_e20)

    def test_agent_plan_does_not_queue_child_when_completed_parent_misses_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_report(
                reports,
                run_id="pilot_center_only_e20",
                variant_id="ablation_center_only_unet_lite_e20",
                loss_variant="center_only",
                epochs=20,
                f1=0.78,
                ap=0.68,
            )
            _write_report(
                reports,
                run_id="tiny_agent_loss_diagnosis_v1_ablation_center_class_unet_lite_e5",
                variant_id="ablation_center_class_unet_lite_e5",
                loss_variant="full_psf_point_supervised",
                epochs=5,
                f1=0.1,
                ap=0.1,
                claim_gate="engineering_check",
            )

            payload = build_research_agent_plan(program, root=reports)

        queue_ids = [variant["variant_id"] for variant in payload["variants"]]
        self.assertNotIn("ablation_center_class_unet_lite_e5", queue_ids)
        self.assertNotIn("ablation_center_class_unet_lite_e20", queue_ids)

    def test_agent_plan_omits_completed_diagnosis_queue_and_uses_matched_full_psf_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_report(
                reports,
                run_id="pilot_center_only_e20",
                variant_id="ablation_center_only_unet_lite_e20",
                loss_variant="center_only",
                epochs=20,
                f1=0.78,
                ap=0.68,
            )
            _write_report(
                reports,
                run_id="pilot_no_psf_e20",
                variant_id="ablation_no_psf_unet_lite_e20",
                loss_variant="no_psf_reconstruction",
                epochs=20,
                f1=0.67,
                ap=0.50,
            )
            for name, f1, ap in [
                ("class", 0.72, 0.59),
                ("photometry", 0.70, 0.55),
                ("multiband", 0.779, 0.678),
                ("psf", 0.769, 0.657),
            ]:
                for epochs in (5, 20):
                    _write_report(
                        reports,
                        run_id=f"tiny_agent_loss_diagnosis_v1_ablation_center_{name}_unet_lite_e{epochs}",
                        variant_id=f"ablation_center_{name}_unet_lite_e{epochs}",
                        loss_variant="full_psf_point_supervised",
                        epochs=epochs,
                        f1=f1,
                        ap=ap,
                    )
            _write_report(
                reports,
                run_id="tiny_agent_loss_diagnosis_v1_ablation_full_psf_unet_lite_e20_matched_bs192",
                variant_id="ablation_full_psf_unet_lite_e20_matched_bs192",
                loss_variant="full_psf_point_supervised",
                epochs=20,
                f1=0.69,
                ap=0.59,
            )

            payload = build_research_agent_plan(program, root=reports)

        agent = payload["agent_plan"]
        queue_ids = [variant["variant_id"] for variant in payload["variants"]]
        blocked_claims = {row["claim"]: row["reason"] for row in agent["blocked_claims"]}
        completed = set(agent["current_state"]["completed_variants"])

        self.assertEqual(queue_ids, [])
        self.assertIn("ablation_full_psf_unet_lite_e20_matched_bs192", completed)
        self.assertEqual(
            agent["current_state"]["method_ablation_status"]["full_psf_unet_lite_e20"]["run_id"],
            "tiny_agent_loss_diagnosis_v1_ablation_full_psf_unet_lite_e20_matched_bs192",
        )
        self.assertIn("matched full-loss e20 is below center-only e20", blocked_claims["psf_constrained_method"])
        self.assertTrue(any("auxiliary-loss diagnosis queue is complete" in item for item in agent["interpretation"]))

    def test_default_agent_plan_keeps_seed_variants_out_of_pending_when_queue_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_completed_diagnosis_reports(reports)

            payload = build_research_agent_plan(program, root=reports)

        self.assertEqual(payload["variants"], [])
        pending_ids = [variant["variant_id"] for variant in payload["pending_approval_variants"]]
        self.assertEqual(
            pending_ids,
            [
                "ablation_center_only_unet_lite_e50_seed42_pending",
                "ablation_full_psf_unet_lite_e50_matched_bs192_pending",
            ],
        )
        self.assertTrue(all("seed7" not in variant_id and "seed123" not in variant_id for variant_id in pending_ids))

    def test_completed_center_only_e50_recommends_mainline_audit_before_more_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_completed_diagnosis_reports(reports)
            _write_report(
                reports,
                run_id="tiny_pilot100_baseline_e50",
                variant_id="pilot100_baseline_e50",
                loss_variant="full_psf_point_supervised",
                epochs=50,
                f1=0.64,
                ap=0.51,
            )
            _write_report(
                reports,
                run_id="tiny_ablation_center_only_unet_lite_e50_seed42",
                variant_id="ablation_center_only_unet_lite_e50_seed42",
                loss_variant="center_only",
                epochs=50,
                f1=0.78,
                ap=0.69,
            )

            payload = build_research_agent_plan(program, root=reports)

        self.assertEqual(payload["variants"], [])
        tasks = payload["recommended_mainline_tasks"]
        self.assertEqual(tasks[0]["task_id"], "audit_center_only_e50_vs_baseline_e50")
        self.assertEqual(tasks[0]["kind"], "mainline_experiment")
        self.assertIn("research-evidence-audit", tasks[0]["command"])
        self.assertTrue(all("multi_seed" not in task["task_id"] for task in tasks))
        pending_ids = [variant["variant_id"] for variant in payload["pending_approval_variants"]]
        self.assertNotIn("ablation_center_only_unet_lite_e50_seed42_pending", pending_ids)
        self.assertTrue(all("seed7" not in variant_id and "seed123" not in variant_id for variant_id in pending_ids))

    def test_completed_mainline_audit_advances_to_synthetic_validation_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_completed_diagnosis_reports(reports)
            _write_report(
                reports,
                run_id="tiny_pilot100_baseline_e50",
                variant_id="pilot100_baseline_e50",
                loss_variant="full_psf_point_supervised",
                epochs=50,
                f1=0.64,
                ap=0.51,
            )
            _write_report(
                reports,
                run_id="tiny_ablation_center_only_unet_lite_e50_seed42",
                variant_id="ablation_center_only_unet_lite_e50_seed42",
                loss_variant="center_only",
                epochs=50,
                f1=0.78,
                ap=0.69,
            )
            _write_evidence_audit(
                reports,
                baseline_label="baseline_e50",
                target_label="center_only_e50",
                delta_f1=0.13,
                delta_ap=0.17,
                unavailable_strata={"seeing": "no finite values", "snr": "no finite values"},
            )

            payload = build_research_agent_plan(program, root=reports)

        task_ids = [task["task_id"] for task in payload["recommended_mainline_tasks"]]
        self.assertNotIn("audit_center_only_e50_vs_baseline_e50", task_ids)
        self.assertEqual(task_ids[0], "synthetic_injection_mainline_validation")
        self.assertIn("write_mainline_strata_limitations", task_ids)
        self.assertTrue(all("multi_seed" not in task_id for task_id in task_ids))

    def test_completed_synthetic_validation_recommends_faint_source_diagnosis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_completed_diagnosis_reports(reports)
            _write_report(
                reports,
                run_id="tiny_pilot100_baseline_e50",
                variant_id="pilot100_baseline_e50",
                loss_variant="full_psf_point_supervised",
                epochs=50,
                f1=0.64,
                ap=0.51,
            )
            _write_report(
                reports,
                run_id="tiny_ablation_center_only_unet_lite_e50_seed42",
                variant_id="ablation_center_only_unet_lite_e50_seed42",
                loss_variant="center_only",
                epochs=50,
                f1=0.78,
                ap=0.69,
            )
            _write_evidence_audit(
                reports,
                baseline_label="baseline_e50",
                target_label="center_only_e50",
                delta_f1=0.13,
                delta_ap=0.17,
                unavailable_strata={"seeing": "no finite values", "snr": "no finite values"},
            )
            _write_synthetic_validation_report(reports)

            payload = build_research_agent_plan(program, root=reports)

        tasks = payload["recommended_mainline_tasks"]
        task_ids = [task["task_id"] for task in tasks]
        synthetic = payload["agent_plan"]["current_state"]["mainline_status"]["synthetic_injection_validation"]

        self.assertEqual(task_ids[0], "diagnose_faint_synthetic_recovery")
        self.assertNotIn("synthetic_injection_mainline_validation", task_ids)
        self.assertEqual(synthetic["metrics"]["faintest_mag_bin"]["recall"], 0.125)
        self.assertIn("[21.5,22.5)", tasks[0]["rationale"])
        self.assertTrue(all("multi_seed" not in task_id for task_id in task_ids))

    def test_completed_faint_diagnostic_recommends_heatmap_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_completed_mainline_reports(reports)
            _write_synthetic_validation_report(reports)
            _write_synthetic_faint_diagnostic_report(reports)

            payload = build_research_agent_plan(program, root=reports)

        task_ids = [task["task_id"] for task in payload["recommended_mainline_tasks"]]
        self.assertEqual(task_ids[0], "measure_faint_heatmap_response")
        self.assertIn("research-synthetic-heatmap-diagnostic", payload["recommended_mainline_tasks"][0]["command"])
        self.assertNotIn("diagnose_faint_synthetic_recovery", task_ids)

    def test_completed_heatmap_diagnostic_recommends_morphology_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_completed_mainline_reports(reports)
            _write_synthetic_validation_report(reports)
            _write_synthetic_faint_diagnostic_report(reports)
            _write_synthetic_heatmap_diagnostic_report(reports)

            payload = build_research_agent_plan(program, root=reports)

        task_ids = [task["task_id"] for task in payload["recommended_mainline_tasks"]]
        heatmap = payload["agent_plan"]["current_state"]["mainline_status"]["synthetic_heatmap_response_diagnostic"]
        self.assertEqual(task_ids[0], "diagnose_faint_extended_source_response")
        self.assertIn("research-synthetic-morphology-diagnostic", payload["recommended_mainline_tasks"][0]["command"])
        self.assertEqual(heatmap["metrics"]["faintest_mag_bin"]["best_within_match_radius_and_low_floor"], 0.125)
        self.assertEqual(
            heatmap["metrics"]["faintest_mag_bin_by_label"]["galaxy"]["best_within_match_radius_and_low_floor"],
            0.125,
        )
        self.assertNotIn("measure_faint_heatmap_response", task_ids)

    def test_completed_morphology_diagnostic_recommends_rescue_design_not_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_completed_mainline_reports(reports)
            _write_synthetic_validation_report(reports)
            _write_synthetic_faint_diagnostic_report(reports)
            _write_synthetic_heatmap_diagnostic_report(reports)
            _write_synthetic_morphology_diagnostic_report(reports)

            payload = build_research_agent_plan(program, root=reports)

        task_ids = [task["task_id"] for task in payload["recommended_mainline_tasks"]]
        morphology = payload["agent_plan"]["current_state"]["mainline_status"]["synthetic_morphology_response_diagnostic"]
        self.assertEqual(task_ids[0], "design_surface_brightness_or_extended_profile_rescue")
        self.assertNotIn("diagnose_faint_extended_source_response", task_ids)
        self.assertEqual(
            morphology["metrics"]["response_by_condition"]["mag22_galaxy_r2.4"][
                "best_within_match_radius_and_low_floor"
            ],
            0.125,
        )
        self.assertTrue(all("seed" not in task_id for task_id in task_ids))

    def test_agent_plan_rejects_unknown_pending_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_completed_diagnosis_reports(reports)

            with self.assertRaisesRegex(ValueError, "unknown pending approval variant"):
                build_research_agent_plan(program, root=reports, approve_pending=["unknown_pending"])

    def test_agent_plan_rejects_seed_approval_while_seed_route_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_completed_diagnosis_reports(reports)

            with self.assertRaisesRegex(ValueError, "unknown pending approval variant"):
                build_research_agent_plan(
                    program,
                    root=reports,
                    approve_pending=["ablation_center_only_unet_lite_e20_seed7_pending"],
                )

    def test_cli_writes_agent_plan_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            program = _write_program(root)
            reports = root / "reports" / "research_runs"
            _write_report(
                reports,
                run_id="pilot_center_only_e20",
                variant_id="ablation_center_only_unet_lite_e20",
                loss_variant="center_only",
                epochs=20,
                f1=0.78,
                ap=0.68,
            )
            output = root / "agent_plan.json"
            markdown = root / "agent_plan.md"

            code = main(
                [
                    "research-agent-plan",
                    "--program",
                    str(program),
                    "--root",
                    str(reports),
                    "--output",
                    str(output),
                    "--markdown-output",
                    str(markdown),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            markdown_exists = markdown.exists()

        self.assertEqual(code, 0)
        self.assertTrue(markdown_exists)
        self.assertEqual(payload["program_id"], "tiny_agent_loss_diagnosis_v1")


def _write_program(root: Path) -> Path:
    program = root / "program.json"
    program.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "sdss-point-supervised-v1",
                "program_id": "tiny",
                "objective": "Tiny program.",
                "config": str(root / "config.json"),
                "dataset": str(root / "dataset"),
                "split": str(root / "split.json"),
                "report_root": str(root / "reports" / "research_runs"),
                "checkpoint_root": str(root / "checkpoints" / "research_runs"),
                "claim_gates": {"min_epochs": 5, "required_tags": ["fixed_split", "native_frame"]},
                "defaults": {
                    "epochs": 5,
                    "batch_size": 128,
                    "learning_rate": 0.001,
                    "base_channels": 32,
                    "model_arch": "baseline",
                    "loader_mode": "shard_grouped",
                    "shard_cache_size": 4,
                    "num_workers": 4,
                    "pin_memory": "auto",
                    "device": "cuda:0",
                    "seed": 42,
                    "candidate_threshold": 0.2,
                    "nms_radius": 2,
                    "max_detections_per_cutout": 16,
                    "radius_arcsec": 1.0,
                    "seeing_aware": True,
                },
                "variants": [
                    {
                        "variant_id": "baseline",
                        "objective": "Baseline.",
                        "hypothesis": "Baseline.",
                        "tags": ["pilot", "fixed_split", "native_frame"],
                        "claims": ["benchmark_contract"],
                        "run": {},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return program


def _write_report(
    root: Path,
    *,
    run_id: str,
    variant_id: str,
    loss_variant: str,
    epochs: int,
    f1: float,
    ap: float,
    claim_gate: str = "candidate_evidence",
) -> None:
    report_dir = root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "program_id": "tiny",
        "variant_id": variant_id,
        "status": "executed",
        "objective": "Synthetic report.",
        "hypothesis": "Synthetic report.",
        "tags": ["pilot", "fixed_split", "native_frame", "unet_lite"],
        "claims": ["ablation_screen"],
        "claim_gate": {"status": claim_gate, "paper_claim_allowed": False},
        "run_options": {
            "epochs": epochs,
            "batch_size": 192,
            "base_channels": 32,
            "model_arch": "unet_lite",
            "loader_mode": "shard_grouped",
            "loss_variant": loss_variant,
            "device": "cuda:0",
        },
            "metrics": {
                "test": {
                "counts": {"truth": 100, "candidate_predictions": 90},
                "detection": {"precision": 0.8, "recall": 0.7, "f1": f1},
                "average_precision": {"ap": ap},
            }
        },
    }
    report_dir.joinpath("report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")


def _write_completed_diagnosis_reports(reports: Path) -> None:
    _write_report(
        reports,
        run_id="pilot_center_only_e20",
        variant_id="ablation_center_only_unet_lite_e20",
        loss_variant="center_only",
        epochs=20,
        f1=0.78,
        ap=0.68,
    )
    _write_report(
        reports,
        run_id="pilot_no_psf_e20",
        variant_id="ablation_no_psf_unet_lite_e20",
        loss_variant="no_psf_reconstruction",
        epochs=20,
        f1=0.67,
        ap=0.50,
    )
    for name, f1, ap in [
        ("class", 0.72, 0.59),
        ("photometry", 0.70, 0.55),
        ("multiband", 0.779, 0.678),
        ("psf", 0.769, 0.657),
    ]:
        for epochs in (5, 20):
            _write_report(
                reports,
                run_id=f"tiny_agent_loss_diagnosis_v1_ablation_center_{name}_unet_lite_e{epochs}",
                variant_id=f"ablation_center_{name}_unet_lite_e{epochs}",
                loss_variant="full_psf_point_supervised",
                epochs=epochs,
                f1=f1,
                ap=ap,
            )
    _write_report(
        reports,
        run_id="tiny_agent_loss_diagnosis_v1_ablation_full_psf_unet_lite_e20_matched_bs192",
        variant_id="ablation_full_psf_unet_lite_e20_matched_bs192",
        loss_variant="full_psf_point_supervised",
        epochs=20,
        f1=0.69,
        ap=0.59,
    )


def _write_completed_mainline_reports(reports: Path) -> None:
    _write_completed_diagnosis_reports(reports)
    _write_report(
        reports,
        run_id="tiny_pilot100_baseline_e50",
        variant_id="pilot100_baseline_e50",
        loss_variant="full_psf_point_supervised",
        epochs=50,
        f1=0.64,
        ap=0.51,
    )
    _write_report(
        reports,
        run_id="tiny_ablation_center_only_unet_lite_e50_seed42",
        variant_id="ablation_center_only_unet_lite_e50_seed42",
        loss_variant="center_only",
        epochs=50,
        f1=0.78,
        ap=0.69,
    )
    _write_evidence_audit(
        reports,
        baseline_label="baseline_e50",
        target_label="center_only_e50",
        delta_f1=0.13,
        delta_ap=0.17,
        unavailable_strata={"seeing": "no finite values", "snr": "no finite values"},
    )


def _write_evidence_audit(
    root: Path,
    *,
    baseline_label: str,
    target_label: str,
    delta_f1: float,
    delta_ap: float,
    unavailable_strata: dict[str, str] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath(f"{target_label}_vs_{baseline_label}_audit.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_label": baseline_label,
                "target_label": target_label,
                "aggregate": {
                    "delta": {
                        "f1": delta_f1,
                        "ap": delta_ap,
                        "recall": 0.1,
                    }
                },
                "threshold_policy": {"source": "validation", "uses_test_threshold_tuning": False},
                "bootstrap": {"delta_ci": {}},
                "claim_summary": {
                    "available_strata": [
                        "mag_r",
                        "nearest_neighbor_arcsec_derived",
                        "source_density_per_cutout",
                    ],
                    "unavailable_strata": unavailable_strata or {},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_synthetic_validation_report(root: Path) -> None:
    run_id = "synthetic_injection_center_only_e50_v1"
    report_dir = root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "protocol": "sdss-point-supervised-v1",
        "run_id": run_id,
        "program_id": "synthetic_injection_mainline",
        "variant_id": run_id,
        "status": "executed",
        "objective": "Synthetic validation.",
        "hypothesis": "Synthetic validation.",
        "tags": ["paper_validation", "mainline_validation", "synthetic_injection", "fixed_split", "native_frame"],
        "claims": ["synthetic_injection", "stratified_metrics"],
        "validation": "synthetic_injection_mainline",
        "claim_gate": {"status": "candidate_evidence", "paper_claim_allowed": False},
        "counts": {"backgrounds": 16, "all_truth": 115, "injected_truth": 64, "candidate_predictions": 131},
        "run_options": {
            "epochs": 50,
            "model_arch": "unet_lite",
            "loader_mode": "synthetic_injection",
            "loss_variant": "center_only",
            "source_run_id": "tiny_ablation_center_only_unet_lite_e50_seed42",
            "source_variant_id": "ablation_center_only_unet_lite_e50_seed42",
        },
        "metrics": {
            "test": {
                "counts": {"truth": 64, "candidate_predictions": 131},
                "detection": {"precision": 0.36, "recall": 0.73, "f1": 0.48},
                "average_precision": {"ap": 0.29},
            },
            "validation": {
                "type": "synthetic_injection_mainline",
                "source_run_id": "tiny_ablation_center_only_unet_lite_e50_seed42",
                "source_variant_id": "ablation_center_only_unet_lite_e50_seed42",
            },
            "all_truth_detection": {"precision": 0.71, "recall": 0.81, "f1": 0.76},
            "injected_detection": {"precision": 0.36, "recall": 0.734375, "f1": 0.48},
            "injected_average_precision": {"ap": 0.295},
            "injected_recall_by_mag_r": {
                "[17.5,19.0)": {"n": 16.0, "detected": 14.0, "recall": 0.875},
                "[19.0,20.5)": {"n": 16.0, "detected": 15.0, "recall": 0.9375},
                "[20.5,21.5)": {"n": 16.0, "detected": 16.0, "recall": 1.0},
                "[21.5,22.5)": {"n": 16.0, "detected": 2.0, "recall": 0.125},
            },
            "injected_recall_by_mag_r_and_label": {
                "[21.5,22.5)": {
                    "galaxy": {"n": 16.0, "detected": 1.0, "recall": 0.0625},
                    "star": {"n": 16.0, "detected": 12.0, "recall": 0.75},
                }
            },
        },
    }
    report_dir.joinpath("report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")


def _write_synthetic_faint_diagnostic_report(root: Path) -> None:
    run_id = "synthetic_injection_center_only_e50_faint_diagnostic_v1"
    report_dir = root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "program_id": "synthetic_injection_mainline",
        "variant_id": run_id,
        "status": "executed",
        "diagnostic": "synthetic_faint_recovery",
        "validation": "synthetic_injection_mainline",
        "validation_run_dir": str(root / "synthetic_injection_center_only_e50_v1"),
        "claim_gate": {"status": "engineering_check", "paper_claim_allowed": False},
        "metrics": {
            "by_mag_r": {
                "[21.5,22.5)": {
                    "n": 16.0,
                    "fixed_threshold": {
                        "recall": 0.125,
                        "false_negatives": 14.0,
                        "false_negative_nearest_candidate": {"no_candidate_within_match_radius": 14.0},
                    },
                }
            },
        },
        "findings": ["Most faint false negatives have no decoded candidate within the match radius."],
        "next_actions": [{"action": "rerun_synthetic_low_candidate_floor_diagnostic", "reason": "candidate floor"}],
    }
    report_dir.joinpath("report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")


def _write_synthetic_heatmap_diagnostic_report(root: Path) -> None:
    run_id = "synthetic_injection_center_only_e50_heatmap_diagnostic_v1"
    report_dir = root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "program_id": "synthetic_injection_mainline",
        "variant_id": run_id,
        "status": "executed",
        "diagnostic": "synthetic_heatmap_response",
        "validation": "synthetic_injection_mainline",
        "validation_run_dir": str(root / "synthetic_injection_center_only_e50_v1"),
        "claim_gate": {"status": "engineering_check", "paper_claim_allowed": False},
        "metrics": {
            "overall": {"n": 64.0},
            "by_mag_r": {
                "[21.5,22.5)": {
                    "n": 16.0,
                    "center_score": {"median": 0.0004},
                    "best_score": {"median": 0.0108},
                    "best_within_match_radius_and_low_floor": 0.125,
                }
            },
            "by_mag_r_and_label": {
                "[21.5,22.5)": {
                    "galaxy": {
                        "n": 16.0,
                        "best_score": {"median": 0.165},
                        "best_within_match_radius_and_low_floor": 0.125,
                    },
                    "star": {
                        "n": 16.0,
                        "best_score": {"median": 0.793},
                        "best_within_match_radius_and_low_floor": 0.75,
                    },
                }
            },
        },
        "findings": ["Faintest injected bin has weak local response."],
        "next_actions": [{"action": "increase_faint_source_signal_or_loss_weight_diagnostic", "reason": "weak response"}],
    }
    report_dir.joinpath("report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")


def _write_synthetic_morphology_diagnostic_report(root: Path) -> None:
    run_id = "synthetic_injection_center_only_e50_morphology_diagnostic_v1"
    report_dir = root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "program_id": "synthetic_injection_mainline",
        "variant_id": run_id,
        "status": "executed",
        "diagnostic": "synthetic_morphology_response",
        "validation": "synthetic_injection_mainline",
        "validation_run_dir": str(root / "synthetic_injection_center_only_e50_v1"),
        "claim_gate": {"status": "engineering_check", "paper_claim_allowed": False},
        "metrics": {
            "overall_response": {"n": 128.0},
            "response_by_condition": {
                "mag22_star_r1.3": {
                    "n": 16.0,
                    "best_score": {"median": 0.79},
                    "best_within_match_radius_and_low_floor": 0.75,
                },
                "mag22_galaxy_r1.3": {
                    "n": 16.0,
                    "best_score": {"median": 0.42},
                    "best_within_match_radius_and_low_floor": 0.5,
                },
                "mag22_galaxy_r2.4": {
                    "n": 16.0,
                    "best_score": {"median": 0.16},
                    "best_within_match_radius_and_low_floor": 0.125,
                },
                "mag22_galaxy_r3.6": {
                    "n": 16.0,
                    "best_score": {"median": 0.08},
                    "best_within_match_radius_and_low_floor": 0.0625,
                },
            },
            "recall_by_condition": {
                "mag22_star_r1.3": {"n": 16.0, "detected": 12.0, "recall": 0.75},
                "mag22_galaxy_r2.4": {"n": 16.0, "detected": 1.0, "recall": 0.0625},
                "mag22_galaxy_r3.6": {"n": 16.0, "detected": 0.0, "recall": 0.0},
            },
        },
        "findings": ["mag22 response separates compact controls from extended galaxies."],
        "next_actions": [
            {
                "action": "design_surface_brightness_or_extended_profile_rescue",
                "reason": "faint galaxy response worsens with extended radius while star controls remain stronger",
            }
        ],
    }
    report_dir.joinpath("report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
