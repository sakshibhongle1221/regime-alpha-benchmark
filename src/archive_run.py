import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BASE_DIR / "runs"

CURATED_ARTIFACTS = [
    "README.md",
    "reproduce.ps1",
    "reproduce.sh",
    "scripts/phases/run_phase35_crypto20_guided.ps1",
    "scripts/phases/run_phase35_crypto20_guided.sh",
    "scripts/phases/run_phase36_crypto20_alpha.ps1",
    "scripts/phases/run_phase36_crypto20_alpha.sh",
    "scripts/phases/run_phase37_crypto20_statistics.ps1",
    "scripts/phases/run_phase37_crypto20_statistics.sh",
    "scripts/phases/run_phase39_fold_local_encoder.ps1",
    "scripts/phases/run_phase39_fold_local_encoder.sh",
    "scripts/phases/run_phase39r_classical_baseline.ps1",
    "scripts/phases/run_phase39r_classical_baseline.sh",
    "scripts/phases/run_phase41_inner_validation_candidates.ps1",
    "scripts/phases/run_phase41_inner_validation_candidates.sh",
    "scripts/phases/run_phase42_interpretation_execution.ps1",
    "scripts/phases/run_phase42_interpretation_execution.sh",
    "scripts/phases/run_phase43b_locked_holdout_registration.ps1",
    "scripts/phases/run_phase43b_locked_holdout_freeze.ps1",
    "scripts/phases/run_phase43b_locked_holdout_adjudication.ps1",
    "scripts/phases/run_phase44_paper_readiness_package.ps1",
    "scripts/phases/run_phase45_venue_manuscript_package.ps1",
    "scripts/phases/run_phase46_final_research_completion.ps1",
    "scripts/phases/run_phase47_submission_build.ps1",
    "scripts/phases/run_phase48_paper_polish.ps1",
    "scripts/phases/run_phase49_submission_closure.ps1",
    "requirements.txt",
    "requirements-research.txt",
    "configs/crypto_universe_candidates.csv",
    "reports/regime_alpha_benchmark_report.md",
    "reports/model_card.md",
    "reports/compute_budget.md",
    "reports/environment.md",
    "reports/artifact_manifest.md",
    "reports/reproduction_checklist.md",
    "reports/related_work.md",
    "reports/paper_protocol.md",
    "reports/hypotheses.md",
    "reports/claim_registry.md",
    "reports/experiment_manifest.md",
    "reports/multiasset_universe_plan.md",
    "reports/crypto20_pipeline_plan.md",
    "reports/crypto20_regime_benchmark_plan.md",
    "reports/crypto20_guided_readiness.md",
    "reports/crypto20_alpha_generalization.md",
    "reports/crypto20_statistical_protocol.md",
    "reports/phase38_master_protocol.md",
    "reports/data_role_registry.csv",
    "reports/experiment_ledger.csv",
    "reports/publication_acceptance_gates.md",
    "reports/phase39_fold_local_encoder_design.md",
    "reports/phase39_fold_local_results.md",
    "reports/phase39r_neural_fold_local_results.md",
    "reports/phase40_repaired_statistical_adjudication.md",
    "reports/phase41_inner_validation_candidate_run.md",
    "reports/phase42_interpretation_execution_hardening.md",
    "reports/phase43_locked_holdout_freeze.md",
    "reports/phase43b_locked_holdout_registration.md",
    "reports/phase43b_locked_external_adjudication.md",
    "reports/phase44_paper_readiness_package.md",
    "reports/phase45_venue_manuscript_package.md",
    "reports/phase46_final_research_completion.md",
    "reports/phase47_submission_build_report.md",
    "reports/phase48_paper_polish_report.md",
    "reports/phase48_reviewer_reading_guide.md",
    "reports/phase48_latex_warning_resolution.md",
    "reports/phase49_final_submission_closure.md",
    "reports/phase49_pdf_metadata_audit.md",
    "reports/phase49_artifact_release_guide.md",
    "reports/paper_artifact_map.csv",
    "reports/paper_submission_checklist.md",
    "reports/literature_matrix.csv",
    "paper/main.md",
    "paper/phase48_review_ready_manuscript.tex",
    "paper/phase48_references.bib",
    "paper/phase48_review_ready_manuscript.pdf",
    "paper/phase49_submission_ready_manuscript.tex",
    "paper/phase49_references.bib",
    "paper/phase49_submission_ready_manuscript.pdf",
    "paper/figures/phase49_f1_validation_repair.svg",
    "paper/figures/phase49_f1_validation_repair.pdf",
    "paper/figures/phase49_f1_validation_repair.png",
    "paper/figures/phase49_f2_system_overview.svg",
    "paper/figures/phase49_f2_system_overview.pdf",
    "paper/figures/phase49_f2_system_overview.png",
    "models/experiment_results.csv",
    "models/strategy_comparison.csv",
    "models/regime_benchmark_summary.csv",
    "models/regime_stability_summary.csv",
    "models/regime_quality_summary.csv",
    "models/regime_agreement_matrix.csv",
    "models/compute_profile.csv",
    "models/ablation_budget.csv",
    "models/ablation_results.csv",
    "models/ablation_summary.csv",
    "models/compute_budget_summary.csv",
    "models/guided_encoder_summary.csv",
    "models/guided_encoder_loss.csv",
    "models/guided_encoder_comparison.csv",
    "models/guided_alpha_comparison.csv",
    "models/time_frequency_encoder_summary.csv",
    "models/time_frequency_encoder_loss.csv",
    "models/time_frequency_encoder_comparison.csv",
    "models/feature_importance_global.csv",
    "models/feature_importance_by_regime.csv",
    "models/feature_family_summary.csv",
    "models/per_regime_stats.csv",
    "models/validation_audit.csv",
    "models/fold_audit.csv",
    "models/walkforward_experiment_results.csv",
    "models/walkforward_comparison.csv",
    "models/walkforward_regime_summary.csv",
    "models/robustness_results.csv",
    "models/robustness_summary.csv",
    "models/robustness_wins.csv",
    "models/robustness_stress_results.csv",
    "models/robustness_stress_summary.csv",
    "models/robustness_stress_wins.csv",
    "models/statistical_fold_metrics.csv",
    "models/statistical_method_summary.csv",
    "models/statistical_pairwise_tests.csv",
    "models/statistical_test_summary.csv",
    "models/statistical_multiple_testing.csv",
    "models/statistical_claims.csv",
    "models/statistical_sharpe_diagnostics.csv",
    "models/paper_claim_tests.csv",
    "models/paper_statistical_summary.csv",
    "models/target_distribution.csv",
    "models/target_quality.csv",
    "models/asset_universe_candidates_scored.csv",
    "models/asset_universe_crypto20.csv",
    "models/asset_universe_crypto50.csv",
    "models/asset_universe_exclusions.csv",
    "models/asset_universe_summary.csv",
    "models/crypto20_data_quality.csv",
    "models/crypto20_pipeline_summary.csv",
    "models/crypto20_target_distribution.csv",
    "models/crypto20_target_quality.csv",
    "models/crypto20_regime_benchmark_summary.csv",
    "models/crypto20_per_regime_stats.csv",
    "models/crypto20_regime_symbol_summary.csv",
    "models/crypto20_guided_symbol_readiness.csv",
    "models/crypto20_guided_pair_summary.csv",
    "models/crypto20_guided_compute_plan.csv",
    "models/crypto20_guided_gate.csv",
    "models/crypto20_guided_encoder_summary.csv",
    "models/crypto20_guided_encoder_loss.csv",
    "models/crypto20_guided_encoder_comparison.csv",
    "models/crypto20_walkforward_experiment_results.csv",
    "models/crypto20_walkforward_regime_summary.csv",
    "models/crypto20_walkforward_comparison.csv",
    "models/crypto20_walkforward_guided_alpha_comparison.csv",
    "models/crypto20_statistical_fold_metrics.csv",
    "models/crypto20_statistical_asset_metrics.csv",
    "models/crypto20_statistical_method_summary.csv",
    "models/crypto20_statistical_pairwise_tests.csv",
    "models/crypto20_statistical_asset_pairwise_tests.csv",
    "models/crypto20_statistical_test_summary.csv",
    "models/crypto20_statistical_multiple_testing.csv",
    "models/crypto20_statistical_claims.csv",
    "models/crypto20_statistical_sharpe_diagnostics.csv",
    "models/crypto20_fold_local_encoder_manifest.csv",
    "models/crypto20_fold_local_encoder_loss.csv",
    "models/crypto20_fold_local_encoder_coverage.csv",
    "models/crypto20_fold_local_fold_metrics.csv",
    "models/crypto20_fold_local_experiment_results.csv",
    "models/crypto20_fold_local_method_comparison.csv",
    "models/crypto20_fold_local_regime_summary.csv",
    "models/crypto20_fold_local_guided_comparison.csv",
    "models/crypto20_fold_local_statistical_fold_metrics.csv",
    "models/crypto20_fold_local_statistical_asset_metrics.csv",
    "models/crypto20_fold_local_statistical_method_summary.csv",
    "models/crypto20_fold_local_statistical_pairwise_tests.csv",
    "models/crypto20_fold_local_statistical_asset_pairwise_tests.csv",
    "models/crypto20_fold_local_statistical_test_summary.csv",
    "models/crypto20_fold_local_statistical_multiple_testing.csv",
    "models/crypto20_fold_local_statistical_claims.csv",
    "models/crypto20_fold_local_statistical_sharpe_diagnostics.csv",
    "models/phase48_paper_quality_audit.csv",
    "models/phase48_claim_traceability.csv",
    "models/phase48_latex_warning_audit.csv",
    "models/phase48_section_budget.csv",
    "models/phase49_submission_closure_audit.csv",
    "models/phase49_figure_manifest.csv",
    "models/phase49_citation_audit.csv",
    "models/phase49_pdf_metadata_audit.csv",
    "models/phase49_latex_warning_audit.csv",
    "models/phase49_artifact_release_checklist.csv",
    "models/crypto20_fold_local_statistical_ic_confidence_intervals.png",
    "models/crypto20_fold_local_statistical_multiple_testing.png",
    "models/crypto20_fold_local_statistical_sharpe_diagnostics.png",
    "models/equity_curve.png",
    "models/loss_curve.png",
    "models/phase4_dashboard.png",
    "models/regime_stability.png",
    "models/regime_quality_heatmap.png",
    "models/regime_agreement_heatmap.png",
    "models/compute_budget_plan.png",
    "models/ablation_heatmap.png",
    "models/guided_encoder_loss_curve.png",
    "models/guided_encoder_transition_hmm_guided_gmm.png",
    "models/guided_encoder_transition_hmm_guided_hmm.png",
    "models/time_frequency_encoder_loss_curve.png",
    "models/time_frequency_encoder_transition_tf_hmm_guided_gmm.png",
    "models/time_frequency_encoder_transition_tf_hmm_guided_hmm.png",
    "models/feature_importance_by_regime.png",
    "models/feature_family_importance.png",
    "models/regime_timeline.png",
    "models/robustness_heatmap.png",
    "models/robustness_stress_heatmap.png",
    "models/statistical_ic_confidence_intervals.png",
    "models/statistical_multiple_testing.png",
    "models/statistical_sharpe_diagnostics.png",
    "models/paper_claim_tests.png",
    "models/target_distribution.png",
    "models/crypto20_target_distribution.png",
    "models/crypto20_transition_matrix_hmm.png",
    "models/crypto20_transition_matrix_kmeans.png",
    "models/crypto20_transition_matrix_vol_bucket.png",
    "models/crypto20_guided_compute_gate.png",
    "models/crypto20_guided_encoder_loss_curve.png",
    "models/crypto20_guided_encoder_transition_hmm_guided_gmm.png",
    "models/crypto20_guided_encoder_transition_hmm_guided_hmm.png",
    "models/crypto20_walkforward_equity_curve.png",
    "models/crypto20_statistical_ic_confidence_intervals.png",
    "models/crypto20_statistical_multiple_testing.png",
    "models/crypto20_statistical_sharpe_diagnostics.png",
    "models/transition_matrix_contrastive.png",
    "models/transition_matrix_contrastive_hmm.png",
    "models/transition_matrix_hmm.png",
    "models/transition_matrix_kmeans.png",
    "models/transition_matrix_vol_bucket.png",
    "models/umap_improved.png",
    "models/umap_regimes.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive curated research artifacts into a versioned runs/ snapshot."
    )
    parser.add_argument("--phase", default="phase15_baseline", help="Human-readable phase label.")
    parser.add_argument("--run-id", default="", help="Optional run id. Defaults to timestamp + phase.")
    parser.add_argument("--source-ref", default="HEAD", help="Git ref that produced the archived artifacts.")
    parser.add_argument("--notes", default="", help="Short notes stored in the manifest and run index.")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Overwrite an existing run directory with the same run id.",
    )
    return parser.parse_args()


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=BASE_DIR, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def safe_phase(value: str) -> str:
    keep = []
    for char in value.strip().lower():
        keep.append(char if char.isalnum() else "_")
    return "_".join("".join(keep).split("_")).strip("_") or "run"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_artifact(relative_path: str, run_dir: Path) -> dict | None:
    source = BASE_DIR / relative_path
    if not source.exists() or not source.is_file():
        return None

    target = run_dir / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "source_path": relative_path.replace("\\", "/"),
        "archived_path": str(target.relative_to(BASE_DIR)).replace("\\", "/"),
        "bytes": int(target.stat().st_size),
        "sha256": sha256_file(target),
    }


def write_artifact_manifest(run_dir: Path, artifacts: list[dict]) -> None:
    path = run_dir / "artifact_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "archived_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(artifacts)


def append_run_index(row: dict) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    index_path = RUNS_DIR / "run_index.csv"
    fieldnames = [
        "run_id",
        "created_at_utc",
        "phase",
        "source_ref",
        "source_commit",
        "git_branch",
        "dirty_worktree",
        "artifact_count",
        "missing_artifact_count",
        "notes",
    ]
    rows = []
    if index_path.exists():
        with index_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [existing for existing in rows if existing.get("run_id") != row["run_id"]]
    rows.append(row)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_phase(args.phase)}"
    run_dir = RUNS_DIR / run_id

    if run_dir.exists():
        if not args.allow_overwrite:
            raise SystemExit(f"Run directory already exists: {run_dir}. Pass --allow-overwrite to replace it.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    artifacts = []
    missing = []
    for relative_path in CURATED_ARTIFACTS:
        copied = copy_artifact(relative_path, run_dir)
        if copied is None:
            missing.append(relative_path)
        else:
            artifacts.append(copied)

    source_commit = run_git(["rev-parse", args.source_ref])
    branch = run_git(["branch", "--show-current"])
    status_short = run_git(["status", "--short"])
    dirty = bool(status_short)

    manifest = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "phase": args.phase,
        "notes": args.notes,
        "source_ref": args.source_ref,
        "source_commit": source_commit,
        "git_branch": branch,
        "git_tags_pointing_at_source": run_git(["tag", "--points-at", source_commit]).splitlines() if source_commit else [],
        "dirty_worktree_at_archive_time": dirty,
        "git_status_short_at_archive_time": status_short.splitlines() if status_short else [],
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "artifact_count": len(artifacts),
        "missing_artifacts": missing,
        "artifacts": artifacts,
    }

    with (run_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    write_artifact_manifest(run_dir, artifacts)

    index_row = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "phase": args.phase,
        "source_ref": args.source_ref,
        "source_commit": source_commit,
        "git_branch": branch,
        "dirty_worktree": str(dirty),
        "artifact_count": str(len(artifacts)),
        "missing_artifact_count": str(len(missing)),
        "notes": args.notes,
    }
    append_run_index(index_row)

    latest_path = RUNS_DIR / "latest_run.json"
    with latest_path.open("w", encoding="utf-8") as handle:
        json.dump(index_row, handle, indent=2)
        handle.write("\n")

    print(f"Archived run: {run_id}")
    print(f"Artifacts copied: {len(artifacts)}")
    print(f"Missing optional artifacts: {len(missing)}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"Run index: {RUNS_DIR / 'run_index.csv'}")


if __name__ == "__main__":
    main()
