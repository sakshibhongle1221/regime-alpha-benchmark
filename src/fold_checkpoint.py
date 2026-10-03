from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


CHECKPOINT_SCHEMA_VERSION = 1
FRAME_FILES = {
    "predictions": "predictions.csv",
    "assignments": "assignments.csv",
    "implementations": "implementations.csv",
    "manifest": "manifest.csv",
    "losses": "losses.csv",
    "coverage": "coverage.csv",
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def initialize_run_state(
    run_dir: Path,
    state: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"
    if not state_path.exists():
        atomic_write_json(state_path, state)
        return state

    existing = json.loads(state_path.read_text(encoding="utf-8"))
    if not resume:
        raise RuntimeError(
            f"Run state already exists at {state_path}. Pass --resume with the identical "
            "configuration, or choose a new --run-name."
        )
    if existing.get("protocol_hash") != state.get("protocol_hash"):
        changed = [
            key
            for key in ["config_hash", "data_hash", "source_hash", "fold_hash"]
            if existing.get(key) != state.get(key)
        ]
        raise RuntimeError(
            "Resume rejected because experiment lineage changed: "
            f"{changed or ['protocol_hash']}. Use a new --run-name; do not mix runs."
        )
    return existing


def fold_checkpoint_dir(run_dir: Path, fold: int) -> Path:
    return run_dir / "checkpoints" / f"fold_{fold:02d}"


def checkpoint_exists(run_dir: Path, fold: int) -> bool:
    return (fold_checkpoint_dir(run_dir, fold) / "checkpoint.json").exists()


def write_fold_checkpoint(
    run_dir: Path,
    fold: int,
    protocol_hash: str,
    bounds: dict[str, Any],
    frames: dict[str, pd.DataFrame],
) -> Path:
    missing = sorted(set(FRAME_FILES) - set(frames))
    if missing:
        raise ValueError(f"Fold checkpoint is missing frames: {missing}")
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final_dir = fold_checkpoint_dir(run_dir, fold)
    if final_dir.exists():
        raise RuntimeError(f"Refusing to overwrite completed checkpoint: {final_dir}")
    temporary = checkpoint_root / f"fold_{fold:02d}.writing"
    if temporary.exists():
        if temporary.resolve().parent != checkpoint_root.resolve():
            raise RuntimeError(f"Unsafe incomplete checkpoint path: {temporary}")
        shutil.rmtree(temporary)
    temporary.mkdir()

    file_manifest: dict[str, dict[str, Any]] = {}
    for key, filename in FRAME_FILES.items():
        path = temporary / filename
        frames[key].to_csv(path, index=False)
        file_manifest[key] = {
            "filename": filename,
            "rows": int(len(frames[key])),
            "sha256": sha256_file(path),
        }
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fold": int(fold),
        "protocol_hash": protocol_hash,
        "bounds": bounds,
        "files": file_manifest,
    }
    (temporary / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, final_dir)
    return final_dir


def load_fold_checkpoint(
    run_dir: Path,
    fold: int,
    protocol_hash: str,
    expected_bounds: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    directory = fold_checkpoint_dir(run_dir, fold)
    metadata_path = directory / "checkpoint.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No completed checkpoint for fold {fold}: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        failures.append("schema version")
    if metadata.get("fold") != fold:
        failures.append("fold id")
    if metadata.get("protocol_hash") != protocol_hash:
        failures.append("protocol hash")
    if metadata.get("bounds") != expected_bounds:
        failures.append("fold bounds")

    frames: dict[str, pd.DataFrame] = {}
    file_metadata = metadata.get("files", {})
    for key, filename in FRAME_FILES.items():
        path = directory / filename
        saved = file_metadata.get(key, {})
        if not path.exists():
            failures.append(f"missing {key}")
            continue
        if sha256_file(path) != saved.get("sha256"):
            failures.append(f"hash {key}")
            continue
        frame = pd.read_csv(path)
        if len(frame) != saved.get("rows"):
            failures.append(f"row count {key}")
            continue
        frames[key] = frame
    if failures:
        raise RuntimeError(
            f"Checkpoint validation failed for fold {fold}: {sorted(set(failures))}. "
            "Do not resume from modified or partial artifacts."
        )
    return frames
