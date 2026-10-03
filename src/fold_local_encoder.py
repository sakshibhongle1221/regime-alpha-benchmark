from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from hmmlearn.hmm import GaussianHMM
from torch.utils.data import DataLoader, Dataset

from baselines import HMM_FEATURES
from config import FEATURE_COLS, LATENT_DIM, N_REGIMES, WINDOW_SIZE
from encoder import NTXentLoss, TemporalEncoder
from guided_encoder import HMMGuidedContrastiveLoss
from train_encoder import LR
from walkforward_regimes import filter_hmm_posteriors


FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_COLS)}
HMM_FEATURE_INDEX = [FEATURE_INDEX[name] for name in HMM_FEATURES]


@dataclass(frozen=True)
class FoldEncoderConfig:
    epochs: int = 30
    batch_size: int = 128
    max_windows: int = 0
    inner_validation_bars: int = 720
    inner_embargo_bars: int = 120
    label_purge_bars: int = 8
    temperature: float = 0.07
    min_positive_gap: int = 24
    hard_negative_gap: int = 24
    hard_negative_weight: float = 2.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    early_stopping_patience: int = 3
    minimum_selection_epochs: int = 3


@dataclass(frozen=True)
class FoldEncoderBounds:
    fold: int
    outer_train_end: int
    inner_train_end: int
    inner_validation_start: int
    inner_validation_end: int
    outer_test_start: int
    outer_test_end: int


@dataclass
class FittedScaler:
    mean: np.ndarray
    std: np.ndarray
    training_rows: int

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return ((matrix - self.mean) / self.std).astype(np.float32)


@dataclass
class FoldEncoderResult:
    method: str
    model: TemporalEncoder
    scaler: FittedScaler
    selected_epoch: int
    selection_history: pd.DataFrame
    refit_history: pd.DataFrame
    train_windows: int
    validation_windows: int
    runtime_seconds: float
    input_hash: str
    model_hash: str


class VanillaWindowDataset(Dataset):
    def __init__(self, matrices: dict[str, np.ndarray], samples: pd.DataFrame):
        self.matrices = matrices
        self.samples = samples.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        start = int(row.start_idx)
        matrix = self.matrices[str(row.symbol)]
        anchor = matrix[start : start + WINDOW_SIZE]
        positive = matrix[start + 1 : start + WINDOW_SIZE + 1]
        return torch.tensor(anchor, dtype=torch.float32), torch.tensor(positive, dtype=torch.float32)


class GuidedFoldWindowDataset(Dataset):
    def __init__(self, matrices: dict[str, np.ndarray], samples: pd.DataFrame):
        self.matrices = matrices
        self.samples = samples.reset_index(drop=True).copy()
        symbol_ids = {symbol: idx for idx, symbol in enumerate(sorted(matrices))}
        self.samples["symbol_id"] = self.samples["symbol"].map(symbol_ids).astype(int)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples.iloc[idx]
        start = int(row.start_idx)
        window = self.matrices[str(row.symbol)][start : start + WINDOW_SIZE]
        return (
            torch.tensor(window, dtype=torch.float32),
            torch.tensor(int(row.hmm_regime), dtype=torch.long),
            torch.tensor(int(row.symbol_id), dtype=torch.long),
            torch.tensor(int(row.feat_idx), dtype=torch.long),
        )


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def make_fold_bounds(
    fold: int,
    train_end: int,
    test_start: int,
    test_end: int,
    config: FoldEncoderConfig,
) -> FoldEncoderBounds:
    validation_end = train_end
    validation_start = validation_end - config.inner_validation_bars
    inner_train_end = validation_start - config.inner_embargo_bars - config.label_purge_bars
    minimum = WINDOW_SIZE + 2
    if inner_train_end <= minimum or validation_start <= minimum:
        raise ValueError(
            f"Fold {fold} does not have enough history for inner validation: "
            f"inner_train_end={inner_train_end}, validation_start={validation_start}."
        )
    if not (inner_train_end < validation_start < validation_end <= train_end < test_start < test_end):
        raise ValueError(f"Invalid fold-local encoder bounds for fold {fold}.")
    return FoldEncoderBounds(
        fold=fold,
        outer_train_end=train_end,
        inner_train_end=inner_train_end,
        inner_validation_start=validation_start,
        inner_validation_end=validation_end,
        outer_test_start=test_start,
        outer_test_end=test_end,
    )


def fit_training_scaler(matrices: dict[str, np.ndarray], train_end: int) -> FittedScaler:
    parts = []
    for symbol in sorted(matrices):
        matrix = matrices[symbol]
        if train_end > len(matrix):
            raise ValueError(f"Training boundary {train_end} exceeds {symbol} rows {len(matrix)}.")
        parts.append(np.asarray(matrix[:train_end], dtype=np.float64))
    combined = np.vstack(parts)
    if not np.isfinite(combined).all():
        raise ValueError("Non-finite values found in fold-local scaler training rows.")
    mean = combined.mean(axis=0)
    std = combined.std(axis=0) + 1e-8
    return FittedScaler(mean=mean, std=std, training_rows=int(len(combined)))


def normalize_with_scaler(
    matrices: dict[str, np.ndarray], scaler: FittedScaler
) -> dict[str, np.ndarray]:
    return {symbol: scaler.transform(matrix) for symbol, matrix in matrices.items()}


def fit_training_hmm(
    normalized: dict[str, np.ndarray], train_end: int, seed: int
) -> tuple[GaussianHMM, pd.DataFrame, str]:
    set_deterministic_seed(seed)
    parts = []
    lengths = []
    symbols = []
    for symbol in sorted(normalized):
        part = normalized[symbol][:train_end, HMM_FEATURE_INDEX]
        parts.append(part)
        lengths.append(len(part))
        symbols.append(symbol)
    train_x = np.vstack(parts)
    model = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="full",
        n_iter=150,
        random_state=seed,
        min_covar=1e-4,
    )
    model.fit(train_x, lengths)

    rows = []
    offset = 0
    for symbol, length in zip(symbols, lengths):
        sequence = train_x[offset : offset + length]
        labels = model.predict(sequence).astype(int)
        rows.extend(
            {"symbol": symbol, "feat_idx": idx, "hmm_regime": int(label)}
            for idx, label in enumerate(labels)
        )
        offset += length
    hmm_hash = array_hash(
        train_x,
        np.asarray(lengths, dtype=np.int64),
        np.asarray([seed], dtype=np.int64),
    )
    return model, pd.DataFrame(rows), hmm_hash


def filter_hmm_labels(
    model: GaussianHMM,
    normalized: dict[str, np.ndarray],
    start: int,
    end: int,
) -> pd.DataFrame:
    rows = []
    for symbol in sorted(normalized):
        sequence = normalized[symbol][start:end, HMM_FEATURE_INDEX]
        probabilities = filter_hmm_posteriors(model, sequence)
        labels = probabilities.argmax(axis=1)
        rows.extend(
            {"symbol": symbol, "feat_idx": start + idx, "hmm_regime": int(label)}
            for idx, label in enumerate(labels)
        )
    return pd.DataFrame(rows)


def cap_samples(samples: pd.DataFrame, max_windows: int, seed: int, stratify: bool) -> pd.DataFrame:
    if max_windows <= 0 or len(samples) <= max_windows:
        return samples.sort_values(["symbol", "feat_idx"]).reset_index(drop=True)
    if stratify and "hmm_regime" in samples.columns:
        first = (
            samples.groupby(["symbol", "hmm_regime"], group_keys=False)
            .sample(frac=min(1.0, max_windows / len(samples)), random_state=seed)
        )
    else:
        first = samples.sample(frac=min(1.0, max_windows / len(samples)), random_state=seed)
    if len(first) > max_windows:
        first = first.sample(n=max_windows, random_state=seed)
    elif len(first) < max_windows:
        remaining = samples.drop(index=first.index)
        add = remaining.sample(n=min(max_windows - len(first), len(remaining)), random_state=seed)
        first = pd.concat([first, add])
    return first.sort_values(["symbol", "feat_idx"]).reset_index(drop=True)


def build_training_windows(
    matrices: dict[str, np.ndarray],
    start: int,
    end: int,
    max_windows: int,
    seed: int,
    hmm_labels: pd.DataFrame | None = None,
    vanilla_positive: bool = False,
) -> pd.DataFrame:
    rows = []
    label_maps = {}
    if hmm_labels is not None:
        label_maps = {
            symbol: group.set_index("feat_idx")["hmm_regime"].astype(int).to_dict()
            for symbol, group in hmm_labels.groupby("symbol")
        }
    for symbol in sorted(matrices):
        first_end = max(start, WINDOW_SIZE - 1)
        last_end_exclusive = min(end, len(matrices[symbol]))
        if vanilla_positive:
            last_end_exclusive -= 1
        for feat_idx in range(first_end, last_end_exclusive):
            start_idx = feat_idx - WINDOW_SIZE + 1
            if start_idx < start:
                continue
            row = {"symbol": symbol, "start_idx": start_idx, "feat_idx": feat_idx}
            if hmm_labels is not None:
                label = label_maps.get(symbol, {}).get(feat_idx)
                if label is None:
                    continue
                row["hmm_regime"] = int(label)
            rows.append(row)
    samples = pd.DataFrame(rows)
    if samples.empty:
        raise RuntimeError(f"No authorized windows for interval [{start}, {end}).")
    return cap_samples(samples, max_windows=max_windows, seed=seed, stratify=hmm_labels is not None)


def validate_fold_encoder_boundaries(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    bounds: FoldEncoderBounds,
    vanilla_positive: bool,
) -> None:
    positive_extra = 1 if vanilla_positive else 0
    train_start = train_samples["start_idx"].astype(int)
    train_end = train_samples["feat_idx"].astype(int) + positive_extra
    validation_start = validation_samples["start_idx"].astype(int)
    validation_end = validation_samples["feat_idx"].astype(int) + positive_extra
    failures = []
    if (train_start < 0).any() or (train_end >= bounds.inner_train_end).any():
        failures.append("inner-training window crosses its authorized boundary")
    if (validation_start < bounds.inner_validation_start).any():
        failures.append("inner-validation window uses inner-training or embargo context")
    if (validation_end >= bounds.inner_validation_end).any():
        failures.append("inner-validation window crosses outer-training end")
    if failures:
        raise AssertionError("; ".join(failures))


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def _run_epoch(
    model: TemporalEncoder,
    loader: DataLoader,
    criterion: torch.nn.Module,
    method: str,
    device: str,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    stats_sum: dict[str, float] = {}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if method == "vanilla":
                anchor, positive = [item.to(device) for item in batch]
                loss = criterion(model(anchor), model(positive))
                stats = {}
            else:
                windows, labels, symbol_ids, feat_idx = [item.to(device) for item in batch]
                loss, stats = criterion(model(windows), labels, symbol_ids, feat_idx)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            for key, value in stats.items():
                stats_sum[key] = stats_sum.get(key, 0.0) + float(value)
    if total_batches == 0:
        raise RuntimeError("Encoder dataloader is empty.")
    return total_loss / total_batches, {
        key: value / total_batches for key, value in stats_sum.items()
    }


def train_with_inner_validation(
    method: str,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    config: FoldEncoderConfig,
    seed: int,
) -> tuple[int, pd.DataFrame]:
    set_deterministic_seed(seed)
    model = TemporalEncoder(n_features=len(FEATURE_COLS), latent_dim=LATENT_DIM).to(config.device)
    criterion = (
        NTXentLoss(config.temperature)
        if method == "vanilla"
        else HMMGuidedContrastiveLoss(
            temperature=config.temperature,
            min_positive_gap=config.min_positive_gap,
            hard_negative_gap=config.hard_negative_gap,
            hard_negative_weight=config.hard_negative_weight,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    train_loader = make_loader(train_dataset, config.batch_size, True, seed)
    validation_loader = make_loader(validation_dataset, config.batch_size, False, seed + 1)
    rows = []
    best_epoch = 1
    best_loss = np.inf
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_stats = _run_epoch(
            model, train_loader, criterion, method, config.device, optimizer
        )
        validation_loss, validation_stats = _run_epoch(
            model, validation_loader, criterion, method, config.device, None
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"validation_{key}": value for key, value in validation_stats.items()},
        }
        rows.append(row)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            epoch >= config.minimum_selection_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            break
    return best_epoch, pd.DataFrame(rows)


def refit_encoder(
    method: str,
    dataset: Dataset,
    epochs: int,
    config: FoldEncoderConfig,
    seed: int,
) -> tuple[TemporalEncoder, pd.DataFrame]:
    set_deterministic_seed(seed)
    model = TemporalEncoder(n_features=len(FEATURE_COLS), latent_dim=LATENT_DIM).to(config.device)
    criterion = (
        NTXentLoss(config.temperature)
        if method == "vanilla"
        else HMMGuidedContrastiveLoss(
            temperature=config.temperature,
            min_positive_gap=config.min_positive_gap,
            hard_negative_gap=config.hard_negative_gap,
            hard_negative_weight=config.hard_negative_weight,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loader = make_loader(dataset, config.batch_size, True, seed)
    rows = []
    for epoch in range(1, epochs + 1):
        loss, stats = _run_epoch(model, loader, criterion, method, config.device, optimizer)
        rows.append({"epoch": epoch, "loss": loss, **stats})
    return model, pd.DataFrame(rows)


def fit_fold_encoder(
    method: str,
    raw_matrices: dict[str, np.ndarray],
    bounds: FoldEncoderBounds,
    config: FoldEncoderConfig,
) -> FoldEncoderResult:
    if method not in {"vanilla", "guided"}:
        raise ValueError(f"Unknown fold encoder method: {method}")
    started = time.perf_counter()
    fold_seed = config.seed + bounds.fold * 100 + (0 if method == "vanilla" else 10_000)

    inner_scaler = fit_training_scaler(raw_matrices, bounds.inner_train_end)
    inner_normalized = normalize_with_scaler(raw_matrices, inner_scaler)
    inner_labels = validation_labels = None
    if method == "guided":
        inner_hmm, inner_labels, _ = fit_training_hmm(
            inner_normalized, bounds.inner_train_end, fold_seed
        )
        validation_labels = filter_hmm_labels(
            inner_hmm,
            inner_normalized,
            bounds.inner_validation_start,
            bounds.inner_validation_end,
        )

    vanilla = method == "vanilla"
    train_samples = build_training_windows(
        inner_normalized,
        start=0,
        end=bounds.inner_train_end,
        max_windows=config.max_windows,
        seed=fold_seed,
        hmm_labels=inner_labels,
        vanilla_positive=vanilla,
    )
    validation_samples = build_training_windows(
        inner_normalized,
        start=bounds.inner_validation_start,
        end=bounds.inner_validation_end,
        max_windows=config.max_windows,
        seed=fold_seed + 1,
        hmm_labels=validation_labels,
        vanilla_positive=vanilla,
    )
    validate_fold_encoder_boundaries(train_samples, validation_samples, bounds, vanilla)
    dataset_cls = VanillaWindowDataset if vanilla else GuidedFoldWindowDataset
    selected_epoch, selection_history = train_with_inner_validation(
        method,
        dataset_cls(inner_normalized, train_samples),
        dataset_cls(inner_normalized, validation_samples),
        config,
        fold_seed,
    )

    outer_scaler = fit_training_scaler(raw_matrices, bounds.outer_train_end)
    outer_normalized = normalize_with_scaler(raw_matrices, outer_scaler)
    outer_labels = None
    hmm_hash = "not_applicable"
    if method == "guided":
        _, outer_labels, hmm_hash = fit_training_hmm(
            outer_normalized, bounds.outer_train_end, fold_seed + 1
        )
    outer_samples = build_training_windows(
        outer_normalized,
        start=0,
        end=bounds.outer_train_end,
        max_windows=config.max_windows,
        seed=fold_seed + 2,
        hmm_labels=outer_labels,
        vanilla_positive=vanilla,
    )
    model, refit_history = refit_encoder(
        method,
        dataset_cls(outer_normalized, outer_samples),
        selected_epoch,
        config,
        fold_seed + 2,
    )
    scaler_hash = array_hash(outer_scaler.mean, outer_scaler.std)
    sample_hash = hashlib.sha256(
        outer_samples.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    input_hash = hashlib.sha256(
        json.dumps(
            {
                "bounds": asdict(bounds),
                "config": asdict(config),
                "scaler_hash": scaler_hash,
                "hmm_hash": hmm_hash,
                "sample_hash": sample_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return FoldEncoderResult(
        method=method,
        model=model,
        scaler=outer_scaler,
        selected_epoch=selected_epoch,
        selection_history=selection_history,
        refit_history=refit_history,
        train_windows=int(len(outer_samples)),
        validation_windows=int(len(validation_samples)),
        runtime_seconds=float(time.perf_counter() - started),
        input_hash=input_hash,
        model_hash=model_state_hash(model),
    )


def encode_causal_rows(
    model: TemporalEncoder,
    raw_matrices: dict[str, np.ndarray],
    scaler: FittedScaler,
    frame: pd.DataFrame,
    device: str,
    batch_size: int = 512,
    total_rows: int | None = None,
) -> np.ndarray:
    normalized = normalize_with_scaler(raw_matrices, scaler)
    output_rows = int(total_rows if total_rows is not None else len(frame))
    embeddings = np.zeros((output_rows, LATENT_DIM), dtype=np.float32)
    encoded_rows = []
    model.eval()
    with torch.no_grad():
        for symbol, group in frame.groupby("symbol", sort=False):
            pending_windows = []
            pending_rows = []
            for row in group.itertuples(index=False):
                feat_idx = int(row.feat_idx)
                start = feat_idx - WINDOW_SIZE + 1
                if start < 0 or feat_idx >= len(normalized[symbol]):
                    raise ValueError(
                        f"Unauthorized embedding endpoint {symbol}:{feat_idx} with start={start}."
                    )
                window = normalized[symbol][start : feat_idx + 1]
                if len(window) != WINDOW_SIZE:
                    raise ValueError(f"Incomplete causal window for {symbol}:{feat_idx}.")
                pending_windows.append(window)
                pending_rows.append(int(row.row_id))
                encoded_rows.append(int(row.row_id))
                if len(pending_windows) >= batch_size:
                    tensor = torch.tensor(np.stack(pending_windows), dtype=torch.float32).to(device)
                    embeddings[pending_rows] = model(tensor).cpu().numpy()
                    pending_windows, pending_rows = [], []
            if pending_windows:
                tensor = torch.tensor(np.stack(pending_windows), dtype=torch.float32).to(device)
                embeddings[pending_rows] = model(tensor).cpu().numpy()
    requested_rows = frame["row_id"].astype(int).to_numpy()
    if len(set(encoded_rows)) != len(requested_rows):
        raise RuntimeError("Fold-local causal encoding did not encode every requested row exactly once.")
    if not np.isfinite(embeddings[requested_rows]).all():
        raise RuntimeError("Fold-local causal encoding produced non-finite requested embeddings.")
    return embeddings
