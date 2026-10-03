import argparse
import math
import os
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import BASE_DIR, DB_PATH, N_REGIMES, SAVE_DIR, SYMBOLS, WINDOW_SIZE
from universe import add_symbol_args, resolve_symbols


RANDOM_STATE = 42
DEFAULT_PREFIX = "crypto20_guided_"
HMM_METHOD = "hmm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the Crypto-20 HMM-guided encoder experiment is ready "
            "before launching expensive multi-asset training."
        )
    )
    add_symbol_args(parser)
    parser.set_defaults(universe="crypto20")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-positive-gap", type=int, default=24)
    parser.add_argument("--hard-negative-gap", type=int, default=24)
    parser.add_argument("--max-train-hours", type=float, default=24.0)
    parser.add_argument(
        "--assignment-path",
        default=os.path.join(SAVE_DIR, "crypto20_regime_assignments.csv"),
        help="Row-level Phase 33 regime assignment file. This remains ignored by Git.",
    )
    parser.add_argument("--output-prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def load_feature_counts(symbols: list[str]) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    counts = con.execute(
        """
        SELECT symbol, COUNT(*) AS feature_rows
        FROM features
        WHERE symbol IN ({})
        GROUP BY symbol
        ORDER BY symbol
        """.format(",".join(["?"] * len(symbols))),
        symbols,
    ).df()
    con.close()
    counts["feature_rows"] = counts["feature_rows"].astype(int)
    return counts


def load_hmm_assignments(path: Path, symbols: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(
            f"{path} is missing. Run src/crypto20_regime_benchmark.py --universe crypto20 first."
        )
    assignments = pd.read_csv(path)
    required = {"method", "symbol", "feat_idx", "regime"}
    missing = required - set(assignments.columns)
    if missing:
        raise RuntimeError(f"{path.name} missing columns: {sorted(missing)}")
    hmm = assignments[
        (assignments["method"] == HMM_METHOD) & (assignments["symbol"].isin(symbols))
    ][["symbol", "feat_idx", "regime"]].copy()
    if hmm.empty:
        raise RuntimeError(f"{path.name} has no HMM assignments for the requested symbols.")
    hmm["feat_idx"] = hmm["feat_idx"].astype(int)
    hmm["regime"] = hmm["regime"].astype(int)
    return hmm.sort_values(["symbol", "feat_idx"]).reset_index(drop=True)


def avg_run_length(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    lengths = []
    current = labels[0]
    count = 1
    for label in labels[1:]:
        if label == current:
            count += 1
        else:
            lengths.append(count)
            current = label
            count = 1
    lengths.append(count)
    return float(np.mean(lengths))


def directed_hard_negative_count(labels: np.ndarray, gap: int) -> int:
    total = 0
    n = len(labels)
    for offset in range(1, gap + 1):
        if offset >= n:
            break
        different = labels[:-offset] != labels[offset:]
        total += int(different.sum()) * 2
    return total


def build_symbol_readiness(
    hmm: pd.DataFrame,
    feature_counts: pd.DataFrame,
    min_positive_gap: int,
    hard_negative_gap: int,
) -> pd.DataFrame:
    rows = []
    count_map = feature_counts.set_index("symbol")["feature_rows"].to_dict()
    for symbol, group in hmm.groupby("symbol", sort=True):
        labels = group["regime"].to_numpy(dtype=int)
        eligible = group[
            (group["feat_idx"] >= WINDOW_SIZE - 1)
            & (group["feat_idx"] <= int(count_map.get(symbol, 0)) - 2)
        ]
        eligible_labels = eligible["regime"].to_numpy(dtype=int)
        regime_counts = eligible["regime"].value_counts().reindex(range(N_REGIMES), fill_value=0)
        transitions = int((labels[1:] != labels[:-1]).sum()) if len(labels) > 1 else 0
        hard_negatives = directed_hard_negative_count(eligible_labels, hard_negative_gap)
        rows.append(
            {
                "symbol": symbol,
                "feature_rows": int(count_map.get(symbol, 0)),
                "hmm_rows": int(len(group)),
                "eligible_windows": int(len(eligible)),
                "regimes_present": int((regime_counts > 0).sum()),
                "min_regime_windows": int(regime_counts.min()),
                "max_regime_windows": int(regime_counts.max()),
                "dominant_regime_pct": float(regime_counts.max() / max(len(eligible), 1)),
                "transition_count": transitions,
                "transition_rate": float(transitions / max(len(labels) - 1, 1)),
                "avg_regime_duration": avg_run_length(labels),
                "directed_hard_negative_pairs": hard_negatives,
                "hard_negative_pairs_per_window": float(hard_negatives / max(len(eligible), 1)),
                "positive_gap_hours": int(min_positive_gap),
                "hard_negative_gap_hours": int(hard_negative_gap),
            }
        )
    return pd.DataFrame(rows)


def build_pair_summary(symbol_readiness: pd.DataFrame, hmm: pd.DataFrame, feature_counts: pd.DataFrame) -> pd.DataFrame:
    count_map = feature_counts.set_index("symbol")["feature_rows"].to_dict()
    eligible = hmm[
        (hmm["feat_idx"] >= WINDOW_SIZE - 1)
        & hmm.apply(lambda row: row["feat_idx"] <= int(count_map.get(row["symbol"], 0)) - 2, axis=1)
    ].copy()
    global_counts = eligible["regime"].value_counts().reindex(range(N_REGIMES), fill_value=0)
    total_windows = int(len(eligible))
    total_hard_negatives = int(symbol_readiness["directed_hard_negative_pairs"].sum())
    min_state_count = int(global_counts.min())
    max_state_pct = float(global_counts.max() / max(total_windows, 1))
    positive_coverage = float((eligible["regime"].map(global_counts) > 1).mean()) if total_windows else 0.0
    rows = [
        {"metric": "symbols", "value": int(symbol_readiness["symbol"].nunique()), "status": "pass"},
        {"metric": "eligible_windows", "value": total_windows, "status": "pass" if total_windows >= 100_000 else "warn"},
        {"metric": "regimes_present_global", "value": int((global_counts > 0).sum()), "status": "pass"},
        {"metric": "min_global_regime_windows", "value": min_state_count, "status": "pass" if min_state_count >= 10_000 else "warn"},
        {"metric": "max_global_regime_pct", "value": max_state_pct, "status": "pass" if max_state_pct <= 0.50 else "warn"},
        {"metric": "positive_anchor_coverage_pct", "value": positive_coverage, "status": "pass" if positive_coverage >= 0.99 else "warn"},
        {
            "metric": "directed_hard_negative_pairs",
            "value": total_hard_negatives,
            "status": "pass" if total_hard_negatives >= total_windows else "warn",
        },
        {
            "metric": "hard_negative_pairs_per_window",
            "value": float(total_hard_negatives / max(total_windows, 1)),
            "status": "pass" if total_hard_negatives >= total_windows else "warn",
        },
    ]
    for regime, count in global_counts.items():
        rows.append(
            {
                "metric": f"global_regime_{int(regime)}_windows",
                "value": int(count),
                "status": "pass" if int(count) >= 10_000 else "warn",
            }
        )
    return pd.DataFrame(rows)


def load_compute_profile() -> pd.DataFrame:
    path = Path(SAVE_DIR) / "compute_profile.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def build_compute_plan(
    pair_summary: pd.DataFrame,
    args: argparse.Namespace,
    compute_profile: pd.DataFrame,
) -> pd.DataFrame:
    total_windows = int(pair_summary[pair_summary["metric"] == "eligible_windows"]["value"].iloc[0])
    batches_per_epoch = int(math.ceil(total_windows / args.batch_size))
    rows = []
    if not compute_profile.empty and "measured_step_seconds" in compute_profile:
        step_seconds = float(compute_profile["measured_step_seconds"].iloc[0])
        source = "phase17_measured_step_seconds"
    else:
        step_seconds = float("nan")
        source = "missing_compute_profile"
    epoch_minutes = batches_per_epoch * step_seconds / 60 if not math.isnan(step_seconds) else float("nan")
    full_minutes = epoch_minutes * args.epochs if not math.isnan(epoch_minutes) else float("nan")
    full_hours = full_minutes / 60 if not math.isnan(full_minutes) else float("nan")
    rows.append(
        {
            "scenario": "full_crypto20_guided_encoder",
            "symbols": 20,
            "eligible_windows": total_windows,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "batches_per_epoch": batches_per_epoch,
            "step_seconds_source": source,
            "estimated_epoch_minutes": epoch_minutes,
            "estimated_train_hours": full_hours,
            "max_train_hours": args.max_train_hours,
            "decision": "run_full" if full_hours <= args.max_train_hours else "stage_prototype_first",
        }
    )
    for cap in [50_000, 100_000, 150_000]:
        capped_batches = int(math.ceil(min(total_windows, cap) / args.batch_size))
        capped_epoch_minutes = capped_batches * step_seconds / 60 if not math.isnan(step_seconds) else float("nan")
        capped_hours = capped_epoch_minutes * min(args.epochs, 5) / 60 if not math.isnan(capped_epoch_minutes) else float("nan")
        rows.append(
            {
                "scenario": f"stratified_prototype_{cap}_windows",
                "symbols": 20,
                "eligible_windows": min(total_windows, cap),
                "batch_size": args.batch_size,
                "epochs": min(args.epochs, 5),
                "batches_per_epoch": capped_batches,
                "step_seconds_source": source,
                "estimated_epoch_minutes": capped_epoch_minutes,
                "estimated_train_hours": capped_hours,
                "max_train_hours": args.max_train_hours,
                "decision": "run_first",
            }
        )
    return pd.DataFrame(rows)


def build_gate(pair_summary: pd.DataFrame, compute_plan: pd.DataFrame, symbol_readiness: pd.DataFrame) -> pd.DataFrame:
    hard_failures = []
    if int(pair_summary[pair_summary["metric"] == "symbols"]["value"].iloc[0]) != 20:
        hard_failures.append("not_all_crypto20_symbols_present")
    if int(pair_summary[pair_summary["metric"] == "regimes_present_global"]["value"].iloc[0]) != N_REGIMES:
        hard_failures.append("not_all_regimes_present")
    if float(pair_summary[pair_summary["metric"] == "positive_anchor_coverage_pct"]["value"].iloc[0]) < 0.99:
        hard_failures.append("positive_anchor_coverage_below_99pct")
    if int((symbol_readiness["regimes_present"] < 2).sum()) > 0:
        hard_failures.append("symbol_with_fewer_than_two_regimes")

    full_decision = compute_plan[compute_plan["scenario"] == "full_crypto20_guided_encoder"]["decision"].iloc[0]
    if hard_failures:
        recommendation = "do_not_train_until_data_issue_fixed"
    elif full_decision == "run_full":
        recommendation = "run_full_crypto20_guided_encoder"
    else:
        recommendation = "run_stratified_prototype_before_full_training"

    return pd.DataFrame(
        [
            {
                "gate": "crypto20_guided_encoder_readiness",
                "status": "pass" if not hard_failures else "fail",
                "recommendation": recommendation,
                "hard_failures": ",".join(hard_failures) if hard_failures else "",
                "notes": (
                    "Pair-mining coverage is sufficient; compute estimate decides whether to "
                    "run a staged prototype or full Crypto-20 guided training."
                ),
            }
        ]
    )


def save_compute_plot(compute_plan: pd.DataFrame, path: Path) -> None:
    plot_df = compute_plan.copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(plot_df["scenario"], plot_df["estimated_train_hours"], color=["#2563eb", "#059669", "#059669", "#059669"])
    ax.axvline(float(plot_df["max_train_hours"].iloc[0]), color="#dc2626", linestyle="--", label="budget")
    ax.set_xlabel("Estimated train hours")
    ax.set_title("Crypto-20 Guided Encoder Compute Gate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(
    report_path: Path,
    pair_summary: pd.DataFrame,
    compute_plan: pd.DataFrame,
    gate: pd.DataFrame,
) -> None:
    total_windows = int(pair_summary[pair_summary["metric"] == "eligible_windows"]["value"].iloc[0])
    hard_pairs = int(pair_summary[pair_summary["metric"] == "directed_hard_negative_pairs"]["value"].iloc[0])
    full = compute_plan[compute_plan["scenario"] == "full_crypto20_guided_encoder"].iloc[0]
    recommendation = gate["recommendation"].iloc[0]
    report_path.write_text(
        "\n".join(
            [
                "# Phase 34 Crypto-20 Guided Encoder Readiness",
                "",
                "Phase 34 is a reviewer-facing gate before running the expensive learned-regime expansion.",
                "It checks whether the Phase 33 raw-feature HMM states provide enough weak-supervision signal",
                "for HMM-guided contrastive learning across the full Crypto-20 universe.",
                "",
                "## Current Readiness Result",
                "",
                f"- Eligible HMM-labeled encoder windows: {total_windows:,}",
                f"- Directed in-trajectory hard-negative pairs within the configured gap: {hard_pairs:,}",
                f"- Estimated full 30-epoch CPU training time: {float(full['estimated_train_hours']):.2f} hours",
                f"- Gate recommendation: `{recommendation}`",
                "",
                "## Why This Matters",
                "",
                "The paper should not jump from a two-asset guided encoder to a multi-asset claim without",
                "showing that the weak-supervision signal scales. This phase checks regime coverage,",
                "positive-anchor availability, hard-negative availability near state boundaries, and compute cost.",
                "",
                "## Next Phase",
                "",
                "If the gate recommends a prototype, run a stratified Crypto-20 guided encoder prototype first.",
                "If that prototype preserves structural alignment, then promote it to a full 30-epoch run and",
                "compare learned regimes against the Phase 33 frozen classical baseline.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    args = parse_args()
    symbols = resolve_symbols(args)
    if len(symbols) != 20:
        print(f"NOTE: expected 20 symbols for Crypto-20, got {len(symbols)}.")

    assignment_path = Path(args.assignment_path)
    feature_counts = load_feature_counts(symbols)
    hmm = load_hmm_assignments(assignment_path, symbols)
    symbol_readiness = build_symbol_readiness(
        hmm,
        feature_counts,
        min_positive_gap=args.min_positive_gap,
        hard_negative_gap=args.hard_negative_gap,
    )
    pair_summary = build_pair_summary(symbol_readiness, hmm, feature_counts)
    compute_plan = build_compute_plan(pair_summary, args, load_compute_profile())
    gate = build_gate(pair_summary, compute_plan, symbol_readiness)

    prefix = args.output_prefix
    out_dir = Path(SAVE_DIR)
    symbol_readiness.to_csv(out_dir / f"{prefix}symbol_readiness.csv", index=False)
    pair_summary.to_csv(out_dir / f"{prefix}pair_summary.csv", index=False)
    compute_plan.to_csv(out_dir / f"{prefix}compute_plan.csv", index=False)
    gate.to_csv(out_dir / f"{prefix}gate.csv", index=False)
    save_compute_plot(compute_plan, out_dir / f"{prefix}compute_gate.png")
    write_report(
        Path(BASE_DIR) / "reports" / "crypto20_guided_readiness.md",
        pair_summary,
        compute_plan,
        gate,
    )

    print("\nCrypto-20 guided encoder readiness:")
    print(gate.to_string(index=False))
    print("\nPair summary:")
    print(pair_summary.to_string(index=False))
    print("\nCompute plan:")
    print(compute_plan.to_string(index=False))
    print(f"\nOK: Phase 34 readiness artifacts saved with prefix '{prefix}'.")


if __name__ == "__main__":
    main()
