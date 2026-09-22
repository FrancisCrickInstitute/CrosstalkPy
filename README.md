# CrosstalkPy: A Python Package to Detect CrossTalk in Microscopy Images

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/) [![Built with Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh) ![Commit activity](https://img.shields.io/github/commit-activity/y/djpbarry/Torch-Unet?style=plastic) ![GitHub](https://img.shields.io/github/license/djpbarry/Torch-Unet?color=green&style=plastic)

<img width="45%" alt="test_Pearsons Correlation_plot_2025-12-16_11-37-05" src="https://github.com/user-attachments/assets/3aeb6b4c-2415-4a15-8c12-c233ca11a0a5" /> <img width="45%" alt="test_Predicted_Label_plot_2025-12-16_11-37-05" src="https://github.com/user-attachments/assets/a18fe6b7-86c4-4c4e-be43-7305e45f8b7f" />

CrosstalkPy is a regression-based deep-learning model designed to detect cross-talk (bleed-through) between different channels in fluorescence microscopy images. Trained on a wide selection of approximately 40,000 images from the [Image Data Resource](https://idr.openmicroscopy.org/), CrosstalkPy is microscope- and sample-agnostic.

We developed CrosstalkPy because standard measures of cross-talk, such as Pearson's Correlation Coefficient, are not reliable (see above).

## Setup

The project is managed with [pixi](https://pixi.sh) (`pixi.toml` + `pixi.lock`),
which is the intended environment tool. A `requirements.txt` is also provided
for a plain `conda`/`pip` workflow as a fallback.

### Option A: pixi (recommended)

Install pixi, then from the repo root:

```
pixi install
pixi run -e crosstalk python <script>.py [args...]
```

The `crosstalk` environment provides Python 3.13 and all Python dependencies.

### Option B: conda + pip

```
conda create --name crosstalk-detection python=3.13
conda activate crosstalk-detection
python -m pip install -r <path to this repo>/requirements.txt
```

Replace `<path to this repo>` with the on-disk location of the repository.

### Hardware / GPU note

Training assumes a CUDA-capable GPU. Note two things:

- The default `batch_size` of **256** can exhaust GPU memory during training on an
  A100 (the model materialises an 8 GiB intermediate activation map), causing a
  `CUDACachingAllocator` out-of-memory error. Use `-b 128` (or smaller) instead.
- The bundled torch is built for CUDA 12.6. The training GPU driver must support
  CUDA >= 12.6, otherwise torch silently falls back to CPU (`Using device: cpu`).

## Evaluation

To test the pre-trained model (or your own model - see below) on the test data in this repo, and compare to other metrics such as Pearson's Correlation Coefficient, activate the environment you created above and run the following command:

```
python test-cross-talk-model.py [-h] [-m MIXED_CHANNEL_DATA_DIR] [-s PURE_SOURCE_DATA_DIR] [-p MODEL_PATH] [-j CPU_JOBS] [-o {single,double}]
```

A number of options can be specified:

```
  -h, --help            show this help message and exit
  -m, --mixed_channel_data_dir MIXED_CHANNEL_DATA_DIR
                        Directory for mixed channel data
  -s, --pure_source_data_dir PURE_SOURCE_DATA_DIR
                        Directory for pure source data
  -p, --model_path MODEL_PATH
                        Path to trained model. To use the model in this repository, set this to ./PreTrained_Model/crosstalk_regression_model_v1.0.0_2026-09-22_14-42-47_128_0.0005.pth
  -j, --cpu_jobs CPU_JOBS
                        Number of CPUs to use
  -o, --model_options {single,double}
                        Use single- or double-branch model
```

## Training

To train a model using default settings on your own data, activate the environment you created above and run the following command:

```
python train_model.py [-h] [-m MIXED_CHANNEL_DATA_DIR] [-s PURE_SOURCE_DATA_DIR] [-b BATCH_SIZE] [-l LEARNING_RATE] [-n NUM_EPOCHS] [-t TRAIN_RATIO] [-v VAL_RATIO] [-j CPU_JOBS] [-o {single,double}] [-r {aggressive_plateau,onecycle,cosine_warmup}]
```

A number of options can be specified to control the training process:

```
  -h, --help            show this help message and exit
  -m, --mixed_channel_data_dir MIXED_CHANNEL_DATA_DIR
                        Directory for mixed channel data
  -s, --pure_source_data_dir PURE_SOURCE_DATA_DIR
                        Directory for pure source data
  -b, --batch_size BATCH_SIZE
                        Batch size for training
  -l, --learning_rate LEARNING_RATE
                        Learning rate for training
  -n, --num_epochs NUM_EPOCHS
                        Number of epochs for training
  -t, --train_ratio TRAIN_RATIO
                        Training data ratio
  -v, --val_ratio VAL_RATIO
                        Validation data ratio
  -j, --cpu_jobs CPU_JOBS
                        Number of CPUs to use
  -o, --model_options {single,double}
                        Use single- or double-branch model
  -r, --learning_scheduler {aggressive_plateau,onecycle,cosine_warmup}
                        Use aggressive_plateau, onecycle or cosine_warmup learning scheduler
```

## Model Distribution

Training exports both the raw weights and a self-contained TorchScript artifact,
plus a manifest, into the training run directory:

- `crosstalk_regression_model_v<version>_<timestamp>_<bs>_<lr>.pth` — raw
  `state_dict` weights (for resuming/reloading within this repo).
- `crosstalk_regression_model_v<version>_<timestamp>_<bs>_<lr>.pt` — a
  TorchScript bundle containing both the architecture **and** the weights. This
  is the recommended artifact to hand to other projects (for example,
  `py-bioimage-qc`): it loads with `torch.jit.load()` and needs no knowledge of
  the model class or its constructor arguments.
- `model_manifest_v<version>_<timestamp>.json` — the model version, the input/
  output contract (a 2×256×256 input where channel 0 is the "mixed" image and
  channel 1 is the "source" image, and a scalar `alpha` output in `[0, 1]`),
  the training hyperparameters, and a SHA256 digest of each artifact for
  download verification.
- `sample_input.npy` and `sample_output.npy` — a deterministic input/output pair
  so a consumer can self-verify their loaded model reproduces the reference
  output (both are SHA256-hashed in the manifest).

For convenience, the `.pt`, manifest, and sample tensors are also copied to
`releases/v<version>/`, and `releases/LATEST` points at the current manifest so
consumers can resolve the latest artifacts without parsing timestamps.

### Getting the model weights

The weights are distributed as **GitHub Releases** (not committed to the
repository), so other projects can download them on demand. The current release
is `v1.0.0` and bundles the `.pt` TorchScript model, the `.pth` state_dict, the
`model_manifest_v1.0.0_*.json`, and the `sample_input.npy` / `sample_output.npy`
verification pair.

To download and verify the `.pt` (the recommended handoff artifact):

```bash
# Download
curl -L -o crosstalk_regression_model_v1.0.0.pt \
  https://github.com/djpbarry/Torch-Unet/releases/download/v1.0.0/crosstalk_regression_model_v1.0.0_2026-09-22_14-42-47_128_0.0005.pt

# Verify integrity (compare to the sha256 in the release manifest)
echo "5f29b254b3ead78101d3f896fc69ecfae205d5695ded0a9558341c2a1e4ef8fb  crosstalk_regression_model_v1.0.0.pt" | sha256sum -c -
```

The full manifest (`model_manifest_v1.0.0_*.json`) lists the SHA256 of every
artifact so a consumer can verify any download:

| Artifact | SHA256 |
| --- | --- |
| `crosstalk_regression_model_v1.0.0_..._128_0.0005.pt` | `5f29b254b3ead78101d3f896fc69ecfae205d5695ded0a9558341c2a1e4ef8fb` |
| `crosstalk_regression_model_v1.0.0_..._128_0.0005.pth` | `9e4426919392ef707d8662640d4fe21cf83b85accad52f76a31fe7569db5cfee` |
| `sample_input.npy` | `4c3d7ca2f2c7649c10eef2256c1c2ebee9d03e056c902cd93de701f2b3769853` |
| `sample_output.npy` | `6045d356648bbb754f69c9e8a6ecf66b080bb702ed324fbb8b5d81999d91d834` |

> **Note:** the TorchScript `.pt` is produced via `torch.jit.script`, which is
> deprecated in torch 2.14+ (it emits a `FutureWarning`) in favour of
> `torch.export`. It still works and keeps the batch size dynamic, which is why
> it is used here; revisit if a future PyTorch removes `torch.jit`.

A `CITATION.cff` at the repo root provides machine-readable citation metadata.

### Loading the exported model

A downstream project can load the `.pt` directly, without importing any of this
repo's code:

```python
import torch

model = torch.jit.load("path/to/crosstalk_regression_model_v1.0.0_...pt")
model.eval()

# Input: [batch, 2, 256, 256] tensor, channel 0 = mixed, channel 1 = source.
# Output: [batch, 1] scalar crosstalk alpha in [0, 1].
with torch.no_grad():
    alpha = model(input_tensor)
```

You can confirm your download + load is correct by feeding the reference
`sample_input.npy` and comparing the output to `sample_output.npy`:

```python
import numpy as np
import torch

model = torch.jit.load("crosstalk_regression_model_v1.0.0.pt").eval()
x = torch.from_numpy(np.load("sample_input.npy"))
expected = np.load("sample_output.npy")

with torch.no_grad():
    out = model(x).numpy()

assert np.allclose(out, expected, atol=1e-5), "mismatch — check the model file"
print("Model verified against reference output.")
```

