XTRAIN
an powerful, easy CPU AI training framework
<img width="273" height="100" alt="logo" src="https://github.com/user-attachments/assets/fb07aeb1-fdc2-4c46-954a-cd85aac062aa" />

Overview
XTRAIN is a compact, CPU-first machine learning training framework designed for efficiency, transparency, and ease of use. It provides a clean Python API and a lightweight native core (Rust/C/Assembly) for performance-critical paths. XTRAIN is built to let researchers and engineers train models on CPU-only machines without sacrificing throughput or control.

Key goals

Fast, predictable CPU training for research and lightweight production workloads
Minimal dependencies and clear control over performance knobs
Easy-to-read Python API for experiments and pipelines
Native components (Rust/C/Assembly) where highly optimized code matters
Languages

Primary: Python (≈85%)
Native performance code: Assembly, Rust, C
Tooling scripts: Batchfile
Features
Simple Trainer API for common training loops (train/validate/checkpoint)
Config-driven runs (YAML/JSON) and CLI entrypoints
Dataset loaders with streaming and memory-mapped options for large corpora on CPU
Optimized tensor ops and small-kernel assembly for critical compute paths
Mixed execution: pure-Python fallback + optional native acceleration
Deterministic run helpers and reproducibility utilities
Lightweight checkpointing and export to ONNX / NumPy-friendly formats
Quick start — install
Development install (recommended):

Python 3.9+ (3.10/3.11 recommended)
pip, virtualenv or venv
Basic:

bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e .
Optional native acceleration:

Rust toolchain (cargo) if building Rust extensions
C build tools (make, gcc/clang) if building C modules
If present, native modules are built automatically during install; otherwise XTRAIN falls back to pure-Python implementations.
Minimal example — Python API
This example demonstrates a concise training loop using the XTRAIN API.

Python
import xtrain
from xtrain import datasets, models, optim

# Prepare dataset (streaming or in-memory)
train_ds = datasets.TextDataset("data/train.txt", batch_size=64, seq_len=256, streaming=True)
val_ds = datasets.TextDataset("data/val.txt", batch_size=64, seq_len=256, streaming=True)

# Create model (simple example)
model = models.SimpleTransformer(vocab_size=32000, dim=512, depth=8)

# Optimizer and trainer
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
trainer = xtrain.Trainer(
    model=model,
    optimizer=optimizer,
    device="cpu",
    grad_accum_steps=4,
    log_interval=100,
    checkpoint_dir="checkpoints/"
)

# Fit
trainer.fit(train_ds, val_ds, epochs=10)
CLI / Config-driven runs
Example config (config.yaml):

YAML
model:
  name: SimpleTransformer
  vocab_size: 32000
  dim: 512
  depth: 8

training:
  batch_size: 64
  seq_len: 256
  epochs: 10
  lr: 3e-4
  grad_accumulation: 4

data:
  train_path: data/train.txt
  val_path: data/val.txt
  streaming: true
Run from CLI:

bash
xtrain train --config config.yaml
Data format and loaders
TextDataset: line-based text or pre-tokenized sequences (token IDs per line)
NumpyDataset: memory-mapped .npy files for large numeric datasets
StreamingDataset: sequential read + on-the-fly shuffling buffer for datasets that don't fit in RAM
Tips:

For very large corpora, use memory-mapped or streaming modes.
Tune worker/thread counts and prefetch sizes to match your CPU cores and IO subsystem.
Performance tuning (CPU)
Use the native extension build if available (cargo/make).
Enable optimized BLAS (OpenBLAS, MKL) if performing dense linear algebra — configure via environment variables.
Use gradient accumulation to increase effective batch size without raising memory pressure.
Prefer int8/quantized or lower-precision ops if supported by native backends to reduce memory bandwidth.
Pin worker threads and bind processes to CPU cores for stable performance in multi-tenant environments.
Reproducibility
Set random seeds using xtrain.utils.set_seed(seed)
Use deterministic flags in native libraries if available
Log full config + git commit hash + environment snapshot (python packages, OS) for each run
Checkpointing & export
Checkpoints store model weights, optimizer state, and config metadata
Automatic periodic checkpoints and best-validation checkpointing
Export to:
NumPy-based weight files for inspection
ONNX for interoperability (model export requires optional onnx package)
Project architecture
xtrain/ — Python package, core API
xtrain/native/ — Rust/C/Assembly performance kernels (optional)
examples/ — Example configs, data loaders, minimal training scripts
benchmarks/ — Microbenchmarks for CPU kernels
tests/ — Unit and integration tests
Safety & security
Validate and sanitize untrusted datasets before training (avoid code injection in custom collate functions)
Prefer isolated environments for running third-party models and data
Contributing
Read CONTRIBUTING.md for branching, commit message, and PR guidelines
Run tests:
bash
pytest -q
Add new CPU kernels under xtrain/native and include corresponding Python bindings
Performance changes should include benchmarks under benchmarks/ and before/after numbers
Maintainers
Maintainer: @old-droid
Repo: https://github.com/old-droid/XTRAIN
License
See the LICENSE file in this repository for license details.

Acknowledgements & references
This project blends Python ergonomics with lightweight native kernels (Rust/C/Assembly) for CPU-first model training.
If you integrate third-party toolchains (BLAS, onnx, etc.), follow their licenses and guidelines.
Appendix — recommended workflow
Start with the provided example config and a small dataset to validate end-to-end.
Iterate on model and batch settings locally, measure CPU utilization.
Build native extensions and rerun benchmarks.
Scale batch size using gradient accumulation before increasing model size.
Record config + environment + checkpoints for reproducibility.

