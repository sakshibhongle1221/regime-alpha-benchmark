import argparse
import math
import os
import platform
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import torch

from config import DB_PATH, FEATURE_COLS, LATENT_DIM, N_FEATURES, SAVE_DIR, SYMBOLS, WINDOW_SIZE
from encoder import NTXentLoss, TemporalEncoder
from train_encoder import BATCH_SIZE, LR


RANDOM_STATE = 42
DEFAULT_EPOCHS = 30
LOCAL_BUDGET_HOURS = 24.0
EVAL_OVERHEAD_MINUTES = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile current encoder cost and plan the next ablation budget."
    )
    parser.add_argument("--symbols", nargs="*", default=SYMBOLS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip synthetic encoder timing and write static sizing estimates only.",
    )
    return parser.parse_args()


def load_feature_counts(symbols: list[str]) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        counts = con.execute(
            f"""
            SELECT
                symbol,
                COUNT(*) AS feature_rows,
                MIN(open_time) AS earliest,
                MAX(open_time) AS latest
            FROM features
            WHERE symbol IN ({",".join(["?"] * len(symbols))})
            GROUP BY symbol
            ORDER BY symbol
            """,
            symbols,
        ).df()
    finally:
        con.close()

    counts["training_windows"] = (counts["feature_rows"] - WINDOW_SIZE - 1).clip(lower=0).astype(int)
    return counts


def profile_encoder_step(batch_size: int, warmup_steps: int, profile_steps: int) -> dict:
    if profile_steps <= 0:
        return {
            "device": "not_profiled",
            "profile_steps": 0,
            "warmup_steps": warmup_steps,
            "measured_step_seconds": float("nan"),
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(RANDOM_STATE)
    model = TemporalEncoder(n_features=N_FEATURES, latent_dim=LATENT_DIM).to(device)
    criterion = NTXentLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    anchor = torch.randn(batch_size, WINDOW_SIZE, N_FEATURES, device=device)
    positive = torch.randn(batch_size, WINDOW_SIZE, N_FEATURES, device=device)
    timings = []

    model.train()
    for step in range(warmup_steps + profile_steps):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(anchor), model(positive))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= warmup_steps:
            timings.append(elapsed)

    return {
        "device": device,
        "profile_steps": profile_steps,
        "warmup_steps": warmup_steps,
        "measured_step_seconds": float(sum(timings) / len(timings)),
    }


def model_parameter_count() -> int:
    model = TemporalEncoder(n_features=N_FEATURES, latent_dim=LATENT_DIM)
    return int(sum(parameter.numel() for parameter in model.parameters()))


def build_profile(args: argparse.Namespace, counts: pd.DataFrame) -> pd.DataFrame:
    total_windows = int(counts["training_windows"].sum())
    batches_per_epoch = total_windows // args.batch_size
    params = model_parameter_count()
    profile = (
        {
            "device": "not_profiled",
            "profile_steps": 0,
            "warmup_steps": args.warmup_steps,
            "measured_step_seconds": float("nan"),
        }
        if args.skip_profile
        else profile_encoder_step(args.batch_size, args.warmup_steps, args.profile_steps)
    )

    step_seconds = profile["measured_step_seconds"]
    epoch_minutes = step_seconds * batches_per_epoch / 60.0 if pd.notna(step_seconds) else float("nan")
    full_train_minutes = epoch_minutes * args.epochs if pd.notna(epoch_minutes) else float("nan")
    ablation_runs = 12
    per_run_minutes = full_train_minutes + EVAL_OVERHEAD_MINUTES if pd.notna(full_train_minutes) else float("nan")
    ablation_hours = per_run_minutes * ablation_runs / 60.0 if pd.notna(per_run_minutes) else float("nan")
    recommended_runs = (
        max(1, min(ablation_runs, math.floor(LOCAL_BUDGET_HOURS / (per_run_minutes / 60.0))))
        if pd.notna(per_run_minutes) and per_run_minutes > 0
        else 0
    )
    budget_status = "green"
    if pd.notna(ablation_hours):
        if ablation_hours > LOCAL_BUDGET_HOURS * 2:
            budget_status = "red"
        elif ablation_hours > LOCAL_BUDGET_HOURS:
            budget_status = "yellow"

    row = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").replace(microsecond=0).isoformat(),
        "symbols": "+".join(args.symbols),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 0,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": profile["device"],
        "feature_count": len(FEATURE_COLS),
        "window_size": WINDOW_SIZE,
        "latent_dim": LATENT_DIM,
        "encoder_parameters": params,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "feature_rows": int(counts["feature_rows"].sum()),
        "training_windows": total_windows,
        "batches_per_epoch": int(batches_per_epoch),
        "warmup_steps": profile["warmup_steps"],
        "profile_steps": profile["profile_steps"],
        "measured_step_seconds": step_seconds,
        "estimated_epoch_minutes": epoch_minutes,
        "estimated_full_train_minutes": full_train_minutes,
        "planned_ablation_runs": ablation_runs,
        "estimated_ablation_hours": ablation_hours,
        "local_budget_hours": LOCAL_BUDGET_HOURS,
        "recommended_initial_runs": int(recommended_runs),
        "budget_status": budget_status,
        "profile_note": "Synthetic encoder forward/backward timing; use as planning estimate, not a formal benchmark.",
    }
    return pd.DataFrame([row])


def build_ablation_budget(profile: pd.DataFrame) -> pd.DataFrame:
    full_train_minutes = float(profile.iloc[0]["estimated_full_train_minutes"])
    eval_minutes = EVAL_OVERHEAD_MINUTES
    losses = ["nt_xent", "info_nce", "hmm_guided"]
    augmentations = ["time_only", "time_frequency"]
    assignments = ["gmm", "hmm"]
    priority_order = {
        ("hmm_guided", "time_only", "hmm"): 1,
        ("hmm_guided", "time_only", "gmm"): 2,
        ("hmm_guided", "time_frequency", "hmm"): 3,
        ("nt_xent", "time_only", "hmm"): 4,
        ("info_nce", "time_only", "hmm"): 5,
        ("nt_xent", "time_frequency", "hmm"): 6,
    }

    rows = []
    for loss in losses:
        for augmentation in augmentations:
            for assignment in assignments:
                priority = priority_order.get((loss, augmentation, assignment), 99)
                planned_phase = "phase18" if loss == "hmm_guided" else "phase21"
                if augmentation == "time_frequency":
                    planned_phase = "phase21"
                if loss == "info_nce":
                    planned_phase = "phase21"
                decision = "hold_until_signal"
                if (loss, augmentation, assignment) in {
                    ("hmm_guided", "time_only", "hmm"),
                    ("hmm_guided", "time_only", "gmm"),
                }:
                    decision = "complete"
                elif (loss, augmentation, assignment) == ("hmm_guided", "time_frequency", "hmm"):
                    decision = "prototype_complete_full_pending"
                total_minutes = full_train_minutes + eval_minutes if pd.notna(full_train_minutes) else float("nan")
                rows.append(
                    {
                        "priority": priority,
                        "planned_phase": planned_phase,
                        "loss": loss,
                        "augmentation": augmentation,
                        "assignment_method": assignment,
                        "epochs": int(profile.iloc[0]["epochs"]),
                        "estimated_train_minutes": full_train_minutes,
                        "estimated_eval_minutes": eval_minutes,
                        "estimated_total_minutes": total_minutes,
                        "decision": decision,
                    }
                )

    budget = pd.DataFrame(rows).sort_values(
        ["priority", "loss", "augmentation", "assignment_method"]
    )
    budget["priority"] = range(1, len(budget) + 1)
    budget["estimated_cumulative_hours"] = budget["estimated_total_minutes"].cumsum() / 60.0
    return budget


def build_summary(profile: pd.DataFrame, budget: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    first = profile.iloc[0]
    summary_rows = [
        {
            "metric": "training_windows",
            "value": int(first["training_windows"]),
            "unit": "windows",
            "notes": "Sliding windows across selected symbols.",
        },
        {
            "metric": "batches_per_epoch",
            "value": int(first["batches_per_epoch"]),
            "unit": "batches",
            "notes": "Uses drop-last batching to match train_encoder.py.",
        },
        {
            "metric": "single_encoder_train",
            "value": round(float(first["estimated_full_train_minutes"]), 2),
            "unit": "minutes",
            "notes": "Estimated from synthetic forward/backward timing.",
        },
        {
            "metric": "initial_12_run_ablation",
            "value": round(float(first["estimated_ablation_hours"]), 2),
            "unit": "hours",
            "notes": "3 losses x 2 augmentations x 2 assignment methods.",
        },
        {
            "metric": "recommended_initial_runs",
            "value": int(first["recommended_initial_runs"]),
            "unit": "runs",
            "notes": f"Cap under a {LOCAL_BUDGET_HOURS:.0f} hour local budget.",
        },
        {
            "metric": "budget_status",
            "value": first["budget_status"],
            "unit": "status",
            "notes": "Green <= budget, yellow <= 2x budget, red > 2x budget.",
        },
    ]
    for row in counts.itertuples(index=False):
        summary_rows.append(
            {
                "metric": f"{row.symbol}_feature_rows",
                "value": int(row.feature_rows),
                "unit": "rows",
                "notes": f"{row.earliest} to {row.latest}",
            }
        )
    return pd.DataFrame(summary_rows)


def plot_budget(budget: pd.DataFrame, output_path: Path) -> None:
    plot_data = budget.sort_values("priority").copy()
    labels = (
        plot_data["loss"]
        + " | "
        + plot_data["augmentation"]
        + " | "
        + plot_data["assignment_method"]
    )
    colors = plot_data["decision"].map(
        {
            "complete": "#16A34A",
            "active_next": "#2563EB",
            "prototype_complete_full_pending": "#F59E0B",
            "hold_until_signal": "#94A3B8",
        }
    ).fillna("#94A3B8")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels, plot_data["estimated_total_minutes"], color=colors)
    ax.invert_yaxis()
    ax.set_title("Phase 17 Compute Plan - Initial Encoder Ablation Budget")
    ax.set_xlabel("Estimated minutes per run")
    ax.set_ylabel("Experiment")
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(plot_data["estimated_total_minutes"]):
        if pd.notna(value):
            ax.text(value, idx, f" {value:.1f}m", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(SAVE_DIR)
    out_dir.mkdir(exist_ok=True)

    counts = load_feature_counts(args.symbols)
    profile = build_profile(args, counts)
    budget = build_ablation_budget(profile)
    summary = build_summary(profile, budget, counts)

    profile.to_csv(out_dir / "compute_profile.csv", index=False)
    budget.to_csv(out_dir / "ablation_budget.csv", index=False)
    summary.to_csv(out_dir / "compute_budget_summary.csv", index=False)
    plot_budget(budget, out_dir / "compute_budget_plan.png")

    print("\nCompute profile:")
    print(profile.to_string(index=False))
    print("\nAblation budget:")
    print(budget.to_string(index=False))
    print("\nOK: Phase 17 compute-plan artifacts saved.")


if __name__ == "__main__":
    main()
