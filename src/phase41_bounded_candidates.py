from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import BASE_DIR, SAVE_DIR


CONFIG_PATH = Path(BASE_DIR) / "configs" / "phase41_bounded_candidates_v1.json"
CANDIDATE_REGISTRY_PATH = Path(SAVE_DIR) / "phase41_candidate_registry.csv"
SELECTION_RULES_PATH = Path(SAVE_DIR) / "phase41_selection_rules.csv"
REPORT_PATH = Path(BASE_DIR) / "reports" / "phase41_bounded_improvement_protocol.md"

CLASS_ORDER = ["down", "neutral", "up"]
PROB_COLUMNS = ["prob_down", "prob_neutral", "prob_up"]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    description: str
    selection_scope: str
    grid: str
    primary_metric: str
    hard_guardrail: str
    status: str = "registered_not_selected"


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError("Probability array must be two-dimensional.")
    clipped = np.clip(probs, 1e-12, np.inf)
    row_sum = clipped.sum(axis=1, keepdims=True)
    return clipped / row_sum


def temperature_scale_probabilities(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    probs = normalize_probabilities(probs)
    logits = np.log(probs) / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return normalize_probabilities(scaled)


def blend_with_prior(probs: np.ndarray, prior: np.ndarray, weight: float) -> np.ndarray:
    if not 0 <= weight <= 1:
        raise ValueError("Prior blend weight must be in [0, 1].")
    probs = normalize_probabilities(probs)
    prior = normalize_probabilities(np.asarray(prior, dtype=float).reshape(1, -1))
    if prior.shape[1] != probs.shape[1]:
        raise ValueError("Prior width must match probability width.")
    return normalize_probabilities((1.0 - weight) * probs + weight * prior)


def posterior_temperature_weights(posteriors: np.ndarray, temperature: float) -> np.ndarray:
    return temperature_scale_probabilities(posteriors, temperature)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, class_to_index: dict[int, int]) -> float:
    probs = normalize_probabilities(probs)
    labels = np.asarray(labels)
    indices = np.array([class_to_index[int(label)] for label in labels], dtype=int)
    selected = probs[np.arange(len(indices)), indices]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def select_by_inner_validation(candidates: pd.DataFrame) -> pd.Series:
    required = {
        "candidate_id",
        "inner_validation_nll",
        "coverage_ok",
        "turnover_increase_vs_baseline",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table missing columns: {sorted(missing)}")
    valid = candidates[
        candidates["coverage_ok"].astype(bool)
        & (pd.to_numeric(candidates["turnover_increase_vs_baseline"], errors="coerce") <= 0.25)
    ].copy()
    if valid.empty:
        raise ValueError("No candidate satisfies Phase 41 hard guardrails.")
    valid["inner_validation_nll"] = pd.to_numeric(valid["inner_validation_nll"], errors="coerce")
    valid = valid.sort_values(["inner_validation_nll", "candidate_id"], ascending=[True, True])
    return valid.iloc[0]


def load_config(path: Path = CONFIG_PATH) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("selection_boundary") != "inner_chronological_validation_only":
        raise ValueError("Phase 41 config must select only on inner chronological validation.")
    if not config.get("hard_constraints", {}).get("outer_test_selection_forbidden", False):
        raise ValueError("Phase 41 config must forbid outer-test selection.")
    return config


def build_candidate_registry(config: dict) -> pd.DataFrame:
    grids = config["candidate_grids"]
    candidates = [
        Candidate(
            "p41_prob_temperature",
            "probability_calibration",
            "Apply temperature scaling to class probabilities selected only by inner-validation NLL.",
            "inner_validation_only",
            f"temperature={grids['probability_temperature']}",
            "inner_validation_nll",
            "No outer-test probability table may be used for selecting temperature.",
        ),
        Candidate(
            "p41_prior_blend",
            "probability_calibration",
            "Blend predicted class probabilities toward the inner-training class prior.",
            "inner_validation_only",
            f"prior_blend_weight={grids['prior_blend_weight']}",
            "inner_validation_nll",
            "Prior must be computed from inner-training labels only.",
        ),
        Candidate(
            "p41_posterior_temperature",
            "soft_regime_gating",
            "Sharpen or smooth regime posterior weights before blending regime-specific alpha models.",
            "inner_validation_only",
            f"posterior_temperature={grids['posterior_temperature']}",
            "inner_validation_nll",
            "Regime posterior temperature must be selected before outer-test prediction.",
        ),
        Candidate(
            "p41_global_regime_shrinkage",
            "soft_regime_gating",
            "Shrink regime-conditioned probabilities toward the global model when inner validation favors the safer pooled expert.",
            "inner_validation_only",
            f"global_regime_shrinkage={grids['global_regime_shrinkage']}",
            "inner_validation_nll",
            "Shrinkage weight must be selected inside each outer fold.",
        ),
        Candidate(
            "p41_score_threshold",
            "execution_control",
            "Pre-specified score threshold grid to reduce weak low-confidence trades; registered for a later execution-focused run.",
            "inner_validation_only",
            f"score_threshold={grids['score_threshold']}",
            "inner_validation_nll_then_turnover",
            "Threshold selection must not use Phase 40 Sharpe or outer-test returns.",
            "registered_deferred",
        ),
    ]
    return pd.DataFrame([candidate.__dict__ for candidate in candidates])


def build_selection_rules(config: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rule_id": "p41_no_outer_test_selection",
                "rule": "Do not select candidate parameters from repaired outer-test predictions or Phase 40 statistical outputs.",
                "reason": "Prevents post-hoc tuning against already inspected development test evidence.",
                "status": "mandatory",
            },
            {
                "rule_id": "p41_inner_nll_primary",
                "rule": f"Use {config['primary_selection_metric']} as the primary candidate selector.",
                "reason": "Phase 40 showed calibration/NLL weakness; NLL is a proper scoring rule for probability calibration.",
                "status": "mandatory",
            },
            {
                "rule_id": "p41_turnover_guardrail",
                "rule": "Reject candidates that increase turnover by more than 25% versus the fold baseline.",
                "reason": "Prevents a candidate from appearing better only by trading more aggressively.",
                "status": "mandatory",
            },
            {
                "rule_id": "p41_equal_coverage",
                "rule": "Reject candidates whose outer-test coverage differs from the repaired baseline.",
                "reason": "Maintains equal method coverage for fair statistical adjudication.",
                "status": "mandatory",
            },
            {
                "rule_id": "p41_freeze_before_locked_test",
                "rule": "Freeze exactly one candidate before any future locked external evaluation.",
                "reason": "Keeps the final test confirmatory rather than exploratory.",
                "status": "mandatory",
            },
        ]
    )


def report_text(config: dict, candidates: pd.DataFrame, rules: pd.DataFrame) -> str:
    candidate_lines = "\n".join(
        f"| `{row.candidate_id}` | {row.family} | {row.primary_metric} | {row.status} |"
        for row in candidates.itertuples(index=False)
    )
    rule_lines = "\n".join(
        f"| `{row.rule_id}` | {row.rule} | {row.status} |" for row in rules.itertuples(index=False)
    )
    forbidden = "\n".join(f"- `{item}`" for item in config["forbidden_selection_inputs"])
    return f"""# Phase 41 Bounded Calibration And Soft-Gating Protocol

## Purpose

Phase 41 is a controlled improvement phase. It registers calibration and soft-gating candidates motivated by Phase 40, but it does **not** tune against Phase 40 outer-test results.

The current data role remains `{config['data_role']}`. Candidate selection is restricted to `{config['selection_boundary']}`.

## Why This Phase Exists

Phase 40 found no corrected IC/Sharpe superiority claim and showed weak probability/portfolio behavior. The right response is not to search the already-inspected outer-test table for a nicer result. The right response is to define bounded candidates, select them inside each outer fold using inner validation, and then evaluate the frozen choices once on the outer fold.

Execution-control score-threshold candidates are registered but deferred. They are not part of the Phase 41B probability-calibration/soft-gating run because they alter trade execution signals rather than probability calibration; they require a separate execution-focused protocol.

## Forbidden Selection Inputs

{forbidden}

## Registered Candidates

| Candidate | Family | Primary selector | Status |
|---|---|---|---|
{candidate_lines}

## Mandatory Rules

| Rule | Requirement | Status |
|---|---|---|
{rule_lines}

## Paper-Safe Interpretation

This phase is infrastructure and protocol, not a performance claim. It makes future improvement attempts auditable by separating:

1. candidate definition,
2. inner-validation selection,
3. outer-fold evaluation,
4. final locked-test evaluation.

If Phase 41 candidates do not improve inner-validation calibration or repaired outer-fold diagnostics, that negative result remains part of the research record.
"""


def write_artifacts(config_path: Path = CONFIG_PATH) -> None:
    config = load_config(config_path)
    candidates = build_candidate_registry(config)
    rules = build_selection_rules(config)
    CANDIDATE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(CANDIDATE_REGISTRY_PATH, index=False)
    rules.to_csv(SELECTION_RULES_PATH, index=False)
    REPORT_PATH.write_text(report_text(config, candidates, rules), encoding="utf-8")
    print(f"Saved: {CANDIDATE_REGISTRY_PATH}")
    print(f"Saved: {SELECTION_RULES_PATH}")
    print(f"Saved: {REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Phase 41 bounded candidate protocol artifacts.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_artifacts(Path(args.config))


if __name__ == "__main__":
    main()
