from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import BASE_DIR, SAVE_DIR


DEFAULT_CONFIG = Path(BASE_DIR) / "configs" / "phase43_locked_holdout_freeze_v1.json"
MANIFEST_PATH = Path(SAVE_DIR) / "phase43_locked_candidate_manifest.csv"
CLAIM_RULES_PATH = Path(SAVE_DIR) / "phase43_locked_claim_rules.csv"
HOLDOUT_RULES_PATH = Path(SAVE_DIR) / "phase43_locked_holdout_rules.csv"
REPORT_PATH = Path(BASE_DIR) / "reports" / "phase43_locked_holdout_freeze.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "freeze_id",
        "data_role",
        "final_candidate",
        "primary_references",
        "locked_holdout_rule",
        "locked_evaluation_protocol",
        "claim_rules",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Phase 43A config missing keys: {missing}")
    if config["data_role"] != "locked_unobserved_until_phase43b":
        raise ValueError("Phase 43A must not mark locked data as inspected.")
    if config["final_candidate"]["method"] != "regime_lgbm_hmm_guided_hmm":
        raise ValueError("Phase 43A freezes the guided-HMM mechanism as the final candidate.")
    excluded = set(config.get("excluded_from_final_candidate", []))
    forbidden_exclusions = {
        "phase41b_probability_calibration",
        "phase41b_soft_gating",
        "score_threshold_execution_control",
    }
    if not forbidden_exclusions.issubset(excluded):
        raise ValueError("Phase 43A must explicitly exclude calibration, soft-gating, and threshold rescue tuning.")
    return config


def artifact_hash_rows(config_path: Path, config: dict[str, Any]) -> list[dict[str, str]]:
    paths = [config_path]
    paths.extend(Path(BASE_DIR) / item for item in config["frozen_development_basis"]["primary_repaired_artifacts"])
    paths.extend(
        [
            Path(BASE_DIR) / "models" / "crypto20_development_freeze_manifest.json",
            Path(BASE_DIR) / "models" / "phase42_execution_stress_summary.csv",
            Path(BASE_DIR) / "models" / "phase42_cross_asset_alpha_diagnostics.csv",
        ]
    )
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required freeze-support artifact is missing: {path}")
        rows.append(
            {
                "artifact": str(path.relative_to(BASE_DIR)),
                "sha256": sha256_file(path),
                "role": "phase43a_freeze_support",
            }
        )
    return rows


def build_candidate_manifest(config_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "freeze_id": config["freeze_id"],
            "item_type": "final_candidate",
            "name": config["final_candidate"]["method"],
            "status": "frozen",
            "role": "primary locked-holdout candidate",
            "notes": config["final_candidate"]["reason"],
        }
    ]
    rows.extend(
        {
            "freeze_id": config["freeze_id"],
            "item_type": "primary_reference",
            "name": method,
            "status": "frozen",
            "role": "primary comparison",
            "notes": "Must be evaluated on the same locked holdout protocol.",
        }
        for method in config["primary_references"]
    )
    rows.extend(
        {
            "freeze_id": config["freeze_id"],
            "item_type": "secondary_diagnostic_reference",
            "name": method,
            "status": "frozen",
            "role": "secondary diagnostic comparison",
            "notes": "May contextualize the result but does not define the primary success condition.",
        }
        for method in config["secondary_diagnostic_references"]
    )
    rows.extend(
        {
            "freeze_id": config["freeze_id"],
            "item_type": "excluded_rescue_path",
            "name": item,
            "status": "excluded",
            "role": "forbidden before locked evaluation",
            "notes": "Excluded to prevent post-development result chasing.",
        }
        for item in config["excluded_from_final_candidate"]
    )
    rows.extend(
        {
            "freeze_id": config["freeze_id"],
            "item_type": "support_artifact_hash",
            "name": row["artifact"],
            "status": "hashed",
            "role": row["role"],
            "notes": row["sha256"],
        }
        for row in artifact_hash_rows(config_path, config)
    )
    return pd.DataFrame(rows)


def build_claim_rules(config: dict[str, Any]) -> pd.DataFrame:
    rules = config["claim_rules"]
    rows = [
        ("success_condition", rules["success_condition"], "required"),
        ("failure_condition", rules["failure_condition"], "required"),
        ("allowed_claim_if_success", rules["allowed_claim_if_success"], "allowed"),
        ("allowed_claim_if_failure", rules["allowed_claim_if_failure"], "allowed"),
    ]
    rows.extend((f"forbidden_claim_{idx}", claim, "forbidden") for idx, claim in enumerate(rules["forbidden_claims"], start=1))
    return pd.DataFrame(
        [
            {
                "freeze_id": config["freeze_id"],
                "rule_id": rule_id,
                "rule": rule,
                "status": status,
            }
            for rule_id, rule, status in rows
        ]
    )


def build_holdout_rules(config: dict[str, Any]) -> pd.DataFrame:
    holdout = config["locked_holdout_rule"]
    protocol = config["locked_evaluation_protocol"]
    quality = holdout["quality_gates"]
    rows = [
        ("preferred_holdout", holdout["preferred_holdout"], "locked_unobserved"),
        ("source_universe", holdout["source_universe"], "prewritten_selection_source"),
        ("selection_rule", holdout["selection_rule"], "mandatory"),
        ("minimum_assets", holdout["minimum_assets"], "mandatory"),
        ("fallback_holdout", holdout["fallback_holdout"], "fallback_only"),
        ("fallback_rule", holdout["fallback_rule"], "fallback_only"),
        ("minimum_hourly_bars", quality["minimum_hourly_bars"], "quality_gate"),
        ("maximum_gap_hours", quality["maximum_gap_hours"], "quality_gate"),
        ("stable_or_synthetic_assets_forbidden", quality["stable_or_synthetic_assets_forbidden"], "quality_gate"),
        ("coverage_and_hash_manifest_required", quality["coverage_and_hash_manifest_required"], "quality_gate"),
        ("validation", protocol["validation"], "locked_protocol"),
        ("target", protocol["target"], "locked_protocol"),
        ("horizon_hours", protocol["horizon_hours"], "locked_protocol"),
        ("transaction_cost_bps", protocol["transaction_cost_bps"], "locked_protocol"),
        ("candidate_selection_on_holdout", protocol["candidate_selection_on_holdout"], "forbidden"),
        ("threshold_selection_on_holdout", protocol["threshold_selection_on_holdout"], "forbidden"),
        ("rerun_after_failure", protocol["rerun_after_failure"], "forbidden"),
    ]
    return pd.DataFrame(
        [
            {
                "freeze_id": config["freeze_id"],
                "rule_id": rule_id,
                "rule_value": value,
                "status": status,
            }
            for rule_id, value, status in rows
        ]
    )


def report_text(config: dict[str, Any], manifest: pd.DataFrame, claims: pd.DataFrame, holdout: pd.DataFrame) -> str:
    candidate_rows = "\n".join(
        f"| `{row.item_type}` | `{row.name}` | {row.status} | {row.role} |"
        for row in manifest[manifest["item_type"].isin(["final_candidate", "primary_reference", "secondary_diagnostic_reference", "excluded_rescue_path"])].itertuples(index=False)
    )
    claim_rows = "\n".join(
        f"| `{row.rule_id}` | {row.status} | {row.rule} |"
        for row in claims.itertuples(index=False)
    )
    holdout_rows = "\n".join(
        f"| `{row.rule_id}` | {row.status} | {row.rule_value} |"
        for row in holdout.itertuples(index=False)
    )
    hash_count = int((manifest["item_type"] == "support_artifact_hash").sum())
    return f"""# Phase 43A Locked Holdout Freeze

## Status

Phase 43A freezes the final confirmatory protocol before any locked-holdout outcome is inspected.

- Freeze ID: `{config['freeze_id']}`
- Data role: `{config['data_role']}`
- Final candidate: `{config['final_candidate']['method']}`
- Support artifacts hashed: {hash_count}

No locked/final data is evaluated in Phase 43A.

## Frozen Candidate And References

| Type | Name | Status | Role |
|---|---|---|---|
{candidate_rows}

## Locked Holdout Rules

| Rule | Status | Value |
|---|---|---|
{holdout_rows}

## Claim Rules

| Rule | Status | Text |
|---|---|---|
{claim_rows}

## Paper-Safe Interpretation

Phase 43A does not improve the model and does not produce a performance result. It prevents future result-chasing by freezing exactly one final mechanism path before the locked evaluation.

Allowed wording:

```text
We froze the guided-HMM mechanism before locked holdout evaluation; the next result will be reported whether positive or negative.
```

Forbidden wording:

```text
Phase 43A proves the model generalizes.
Phase 43A selects a winner from locked data.
Phase 43A permits threshold tuning after holdout inspection.
```
"""


def write_artifacts(config_path: Path = DEFAULT_CONFIG) -> None:
    config = load_config(config_path)
    manifest = build_candidate_manifest(config_path, config)
    claims = build_claim_rules(config)
    holdout = build_holdout_rules(config)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    claims.to_csv(CLAIM_RULES_PATH, index=False)
    holdout.to_csv(HOLDOUT_RULES_PATH, index=False)
    REPORT_PATH.write_text(report_text(config, manifest, claims, holdout), encoding="utf-8")
    print(f"Saved: {MANIFEST_PATH}")
    print(f"Saved: {CLAIM_RULES_PATH}")
    print(f"Saved: {HOLDOUT_RULES_PATH}")
    print(f"Saved: {REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze Phase 43A locked-holdout protocol artifacts.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_artifacts(Path(args.config))


if __name__ == "__main__":
    main()
