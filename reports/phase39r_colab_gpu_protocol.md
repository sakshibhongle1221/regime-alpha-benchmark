# Phase 39R-D Colab GPU Protocol

## Decision

Use Google Colab GPU for the repaired fold-local neural/guided run.

Reason:

- the local environment has CPU-only PyTorch (`torch 2.11.0+cpu`);
- the local NVIDIA GPU has about 2GB VRAM, which is risky for the full encoder run;
- Colab usually provides a stronger CUDA runtime, but it is not guaranteed and can disconnect.

This is a compute migration only. It must not change the frozen dataset, folds, model-selection protocol, or evaluation definitions.

## Required Upload

Upload the generated archive:

```text
runs/phase39r_colab_package/regime_alpha_benchmark_phase39r_colab_package.zip
```

to Google Drive, preferably:

```text
MyDrive/regime_alpha_benchmark_phase39r/
```

## Colab Notebook Cells

### 1. Select GPU

In Colab:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

### 2. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Unpack package to local Colab disk

```python
from pathlib import Path
import zipfile, shutil

drive_root = Path('/content/drive/MyDrive/regime_alpha_benchmark_phase39r')
archive = drive_root / 'regime_alpha_benchmark_phase39r_colab_package.zip'
work = Path('/content/regime-alpha-benchmark')

if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(archive, 'r') as z:
    z.extractall(work)

print('unpacked to', work)
```

### 4. Install dependencies without replacing Colab CUDA PyTorch

```python
%cd /content/regime-alpha-benchmark
!python -m pip install -q -r requirements-colab.txt
```

Do not install `torch` from `requirements-colab.txt`. Colab's CUDA-enabled PyTorch should be used.

### 5. Verify GPU and frozen data

```python
import torch, subprocess, sys
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')

!python src/freeze_development_dataset.py --verify-only
!python -m unittest tests.test_calendar_folds tests.test_development_freeze tests.test_evaluation tests.test_fold_checkpoint tests.test_repaired_classical_baseline -q
```

Continue only if:

```text
cuda available: True
freeze verification: OK
tests: OK
```

### 6. Run a Colab GPU smoke

This checks the runtime before spending many hours.

```python
!python -u src/fold_local_encoder_walkforward.py \
  --universe crypto20 \
  --epochs 1 \
  --batch-size 128 \
  --max-windows 256 \
  --max-folds 1 \
  --run-name phase39r_colab_smoke_v1 \
  --heavy-dir /content/drive/MyDrive/regime_alpha_benchmark_phase39r/checkpoints \
  --output-dir /content/drive/MyDrive/regime_alpha_benchmark_phase39r/smoke_outputs \
  --output-prefix smoke_ \
  --report-path /content/drive/MyDrive/regime_alpha_benchmark_phase39r/smoke_outputs/smoke_report.md
```

### 7. Run the full repaired neural experiment

```python
!python -u src/fold_local_encoder_walkforward.py \
  --universe crypto20 \
  --epochs 30 \
  --batch-size 128 \
  --max-windows 5000 \
  --run-name phase39r_neural_full_v1 \
  --heavy-dir /content/drive/MyDrive/regime_alpha_benchmark_phase39r/checkpoints \
  --output-dir /content/drive/MyDrive/regime_alpha_benchmark_phase39r/final_outputs \
  --output-prefix crypto20_repaired_fold_local_ \
  --report-path /content/drive/MyDrive/regime_alpha_benchmark_phase39r/final_outputs/phase39r_neural_fold_local_results.md
```

### 8. Resume after disconnect

If Colab disconnects, reconnect with GPU, rerun cells 2 to 5, then:

```python
!python -u src/fold_local_encoder_walkforward.py \
  --universe crypto20 \
  --epochs 30 \
  --batch-size 128 \
  --max-windows 5000 \
  --run-name phase39r_neural_full_v1 \
  --heavy-dir /content/drive/MyDrive/regime_alpha_benchmark_phase39r/checkpoints \
  --output-dir /content/drive/MyDrive/regime_alpha_benchmark_phase39r/final_outputs \
  --output-prefix crypto20_repaired_fold_local_ \
  --report-path /content/drive/MyDrive/regime_alpha_benchmark_phase39r/final_outputs/phase39r_neural_fold_local_results.md \
  --resume
```

## Interpretation Rules

- Smoke output is not scientific evidence.
- Full Colab output is still development evidence, not locked final evidence.
- If the run finishes, copy `final_outputs/` and the matching checkpoint metadata back into the local project before interpreting results.
- Do not change epochs, windows, symbols, folds, or evaluation after seeing partial results.
