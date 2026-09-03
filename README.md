<img width="273" height="100" alt="logo" src="https://github.com/user-attachments/assets/fb07aeb1-fdc2-4c46-954a-cd85aac062aa" />

> Train ML models efficiently on CPUs — no GPU required.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  Wiki:https://app.devin.ai/org/byte-sized-labs-7705b5f7bf994413988290584a26230e/wiki/MangalanLabs/XTRAIN?branch=Master

## Overview

XTRAIN (CPUWARP-ML) is a CPU-native machine learning training framework. It has no GPU
dependency and no deep learning framework dependency — models, forward passes and
backpropagation are implemented directly on top of NumPy and Numba.

The core engine is `CPUWarpML` in `cpuwarp_ml.py`, which exposes `matmul`, `conv2d`, `relu`
and `softmax` (also available as module-level functions) and combines:

- **WARP scheduling** — `WARPScheduler` in `cpuwarp_ml.py` performs Workload-Aware Resource
  Partitioning: `CPUInfo` detects the CPU vendor, core/thread counts, SIMD features and cache
  sizes, `WorkloadAnalyzer` classifies each operation as compute- or memory-bound, and the
  scheduler allocates compute threads, memory threads and cache accordingly.
- **Numba JIT kernels** — `numba_kernels.py` provides `@jit(nopython=True, parallel=True)`
  kernels (matmul, conv2d, activations, layer norm, softmax, gradient kernels) using `prange`
  for multi-threaded execution. If Numba is unavailable the engine falls back to NumPy
  implementations; `numba_kernels.get_numba_status()` reports what is active.
- **Mixed precision** — `set_mixed_precision()`, `to_float16()` / `to_float32()` and
  `matmul_fp16` / `softmax_fp16` helpers in `cpuwarp_ml.py`.

Supporting modules: `nn_layers.py` (`ReGLU`, `Dense`, `RobustNeuralNet`),
`backpropagation_optimized.py` (fused backward kernels, gradient checkpointing and the SGD /
Adam / AdamW / Lamb optimizers), `advanced_features.py` (`enable_advanced_features`,
distributed training, mixed precision, `ModelExporter`), `dataset_loaders.py` (CIFAR-10,
ImageNet, text, audio and multimodal loaders plus batching and augmentation) and `config.py`.

## Supported models

| Model | Defined in | Description |
|---|---|---|
| `CPUWarpCNN` | `train_cnn.py` | Convolutional network built from `Conv2D`, `BatchNorm2D`, `ReLU`, `MaxPool2D` and `Dense` layers. Constructor: `CPUWarpCNN(input_shape=(C, H, W), num_classes=N)`. |
| `CPUWarpTransformer` | `train_llm.py` | Transformer built from `MultiHeadAttention`, `FeedForward`, `LayerNorm` and `TransformerBlock`. Constructor: `CPUWarpTransformer(vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_len=...)`. |
| `SimpleVLM` | `run_model.py` (inside `ModelRunner._create_multimodal_model`) | Vision-language model that pairs a `CPUWarpCNN` vision encoder with a `CPUWarpTransformer` text encoder and fuses them using the `VLM_FUSION_METHOD` from `.env` (`attention`, `concat`, or additive). Used via `--model multimodal`; it is not importable as a top-level class. |
| `ChapatiLM` | not present on this branch | The `ChapatiLM` implementation lives in `chapati_core.py` on the `chapatiLM` branch. On `Master` only the driver scripts `test_enhanced_chapati.py` and `debug_chapati.py` exist, and they fail with `ModuleNotFoundError: chapati_core` unless `chapati_core.py` is made importable. |

`train_cnn.py` and `train_llm.py` also contain `ReGLU` and `RobustNeuralNet` — a small network
with strict shape/NaN validation, numerical gradient checking (`check_backprop`) and
checkpointing.

## Installation

```bash
git clone https://github.com/MangalanLabs/XTRAIN.git
cd XTRAIN
pip install -r requirements.txt
```

Dependencies (`requirements.txt`): `flask`, `numpy`, `numba`, `psutil`.
`demo_request.py` additionally needs `requests` (`pip install requests`).

## Configuration

All configuration lives in the `.env` file at the repo root and is parsed by `config.py`
(`load_env_file` / `get_config`, returning a `CPUWarpMLConfig`). `${VAR}` references inside
`.env` are expanded. Key groups:

- **Dataset paths** — `DATASET_ROOT` and per-dataset paths (CIFAR-10/100, MNIST, ImageNet,
  COCO, WikiText, LibriSpeech, VQA, ...).
- **Model dimensions** — `CNN_INPUT_SIZE`, `CNN_INPUT_CHANNELS`, `CNN_NUM_CLASSES`,
  `CNN_BATCH_SIZE`, `CNN_EPOCHS`, `LLM_VOCAB_SIZE`, `LLM_D_MODEL`, `LLM_NUM_HEADS`,
  `LLM_NUM_LAYERS`, `LLM_D_FF`, `LLM_MAX_SEQ_LEN`, plus ViT and audio settings.
- **Multimodal** — `ENABLE_MULTIMODAL`, `VLM_FUSION_METHOD`, `VLM_HIDDEN_DIM`,
  `CROSS_MODAL_ATTENTION`.
- **WARP scheduler** — `WARP_COMPUTE_THREADS` (`auto` or a number), `WARP_MEMORY_THREADS`,
  `WARP_CACHE_ALLOCATION`, `WARP_ENABLE_PROFILING`, `WARP_ADAPTIVE_SCHEDULING`.
- **Memory** — `MEMORY_EFFICIENT`, `GRADIENT_CHECKPOINTING`, `MAX_MEMORY_GB`, `NUM_WORKERS`.
- **Training / precision** — `MIXED_PRECISION`, `GRADIENT_CLIPPING`, `DROPOUT_RATE`,
  `LABEL_SMOOTHING`, data augmentation flags, and logging/checkpoint settings.

Run `python config.py` to print and validate the loaded configuration (it also writes a
snapshot to `test_config.txt`).

## Usage

### Interactive CLI

```bash
python xtrain_cli.py
```

Menu options: `1` train a CNN, `2` train an LLM, `3` run inference, `4` exit. A positional
argument selects an option non-interactively, e.g. `python xtrain_cli.py 1`.

> Note: `xtrain_cli.py` shells out to `XTRAIN/train_cnn.py`, `XTRAIN/train_llm.py` and
> `XTRAIN/run_model.py`, so it must be run from the directory *containing* the `XTRAIN`
> checkout. From inside the repo, call the scripts directly instead.

### Model runner

```bash
python run_model.py --model {auto,cnn,llm,multimodal} \
                    --dataset auto \
                    --mode {train,evaluate,benchmark,train_and_eval} \
                    [--epochs N] [--batch-size N] [--config .env]
```

Defaults: `--model auto` (multimodal if `ENABLE_MULTIMODAL=true`, otherwise CNN),
`--dataset auto`, `--mode train`, `--config .env`. `--epochs` and `--batch-size` override the
`.env` values.

### Training scripts directly

```bash
python train_cnn.py [--input-size 32] [--input-channels 3] [--num-classes 10] \
                    [--batch-size 16] [--num-epochs 3] [--batches-per-epoch 20] \
                    [--learning-rate 0.01] [--benchmark]

python train_llm.py [--vocab-size 10000] [--d-model 512] [--num-heads 8] [--num-layers 6] \
                    [--d-ff 2048] [--batch-size 32] [--seq-len 128] [--num-epochs 3] \
                    [--batches-per-epoch 50] [--benchmark]
```

### Inference API

```bash
python api_service.py
```

Starts a Flask server on `0.0.0.0:3434` exposing `POST /infer/<model_type>` where
`<model_type>` is `cnn`, `llm` or `multimodal`. The request body is JSON with a `data` field
containing the input tensor as nested lists; the response is `{"prediction": [...]}`. Model
runners are constructed on first use and cached.

Send a sample request with:

```bash
python demo_request.py {cnn,llm,multimodal}
```

## Testing

The repository contains standalone test scripts that are run directly with Python:

```bash
python test_numba_kernels.py    # validates Numba kernels and benchmarks them against NumPy
python test_robust_net.py       # ReGLU and RobustNeuralNet (shape/NaN checks, gradient check)
python test_cleaned_code.py     # core cpuwarp_ml operations after the C-extension removal
python test_enhanced_chapati.py # ChapatiLM tests; requires chapati_core.py (see above)
```

## Performance notes

- The first iterations are slower because Numba compiles the kernels just-in-time; steady-state
  throughput is reached after that warm-up.
- Performance is best on many-core x86-64 CPUs with AVX2/AVX-512 (tested on AMD EPYC and Intel
  Xeon). `cpuwarp_ml.cpuwarp.get_performance_stats()` reports the detected CPU and features.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and coding standards.

## License

MIT — see [LICENSE](LICENSE).
