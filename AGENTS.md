# AGENTS.md

## Project Overview

CrosstalkPy: a PyTorch microscopy-image crosstalk detector. Given a pair of
TIFF images (a "mixed" bleed-through channel and its "pure" source channel),
the model regresses a scalar `alpha` value in `[0,1]` describing the crosstalk
mixing ratio. Labels are **not** stored in any annotation file: they are parsed
from the filename's `alpha_VALUE` field.

Despite the repository name ("Torch-UNet"), the models are **not** U-Nets —
they are CNN feature extractors feeding a regression head that outputs a single
scalar.

## Environment & Dependencies

Two (currently divergent) environment definitions exist:

- **`pixi.toml`** — the intended package manager (Pixi, conda-forge). Defines a
  `crosstalk` environment and pypi deps: `numpy`, `torch`, `matplotlib`,
  `pillow`, `torchvision`, `imageio`, `tqdm`.
- **`requirements.txt`** — README still documents a `conda` + `pip install -r`
  workflow with `python=3.13`.

Note the mismatch: `pixi.toml` pins `python >=3.14.7,<3.15`, while the README
says `python=3.13`. If you touch either file, keep them in sync or flag the
discrepancy.

`requirements.txt` lists **fewer** packages than the code actually imports. The
test script additionally needs `scipy`, `scikit-image`, `scikit-learn`
(`pearsonr`, `ssim`, `normalized_mutual_info_score`), and `analyse_training_results.py`
needs `pandas`. None of these are declared anywhere.

The repo is Windows-targeted (`platforms = ["win-64"]`), but default CLI paths
in the scripts point at Linux/NEMO HPC mounts (`/nemo/...`, `Z:/working/...`),
so running on this machine needs explicit `-m`/`-s` args.

## Commands

There is **no test suite, linter, formatter, or CI**. The only entry points are
three standalone scripts run via `python`:

```bash
# Train (defaults point at ./Training_Data/Mixed and ./Training_Data/Source)
python train_model.py -m ./Training_Data/Mixed -s ./Training_Data/Source

# Evaluate a trained model (defaults point at a NEMO path; override -p)
python test-cross-talk-model.py -m <mixed_dir> -s <source_dir> \
    -p ./PreTrained_Model/crosstalk_regression_model_trained_2025-12-15_18-22-01_256_0.0005.pth \
    -o single

# Ad-hoc analysis script (hardcoded base dir at top of main(); not CLI-driven)
python analyse_training_results.py
```

Key CLI args (see `argparse` in `train_model.py` and `test-cross-talk-model.py`):

- `-o/--model_options` — `single` (default) or `double`. **Must match** the
  architecture the `.pth` was saved from, or `load_state_dict` will fail.
- `-r/--learning_scheduler` — `aggressive_plateau` (default), `onecycle`,
  `cosine_warmup`.
- `-j/--cpu_jobs` — DataLoader `num_workers`.

## Architecture & Data Flow

Two model definitions, selected at runtime by the `-o` flag:

1. **`regression_model.py` → `AdvancedRegressionModel`** (`single`, default).
   A single 2-channel input (`[mixed, source]` stacked). Sequential conv blocks
   (Conv2d → BatchNorm2d → LeakyReLU(0.01) → MaxPool2d), doubling filters up to
   a cap of 512, then Flatten → FC layers → 1 output. Flattened feature size is
   computed dynamically by `_get_conv_output((256,256))` — everything assumes a
   **256×256 input** (`TARGET_IMAGE_SIZE`).

2. **`two_branch_regression.py` → `SimplifiedTwoBranchRegressionModel`
   (`double`)**. Splits the 2×256×256 input into two 1-channel branches
   (one per image), extracts features independently, concatenates along the
   channel dim, then feeds a shared regression head with a final `Sigmoid()`
   scaled by `* 0.5`.

The input is always built by `torch.cat([mixed, source], dim=0)` = 2 channels,
where channel 0 is "mixed" and channel 1 is "source".

### Critical gotcha: model dims are hardcoded inconsistently

The CLI instantiates the model with specific constructor args
(`AdvancedRegressionModel(initial_filters=128, num_conv_blocks=6)` or
`SimplifiedTwoBranchRegressionModel(initial_filters_per_branch=64)`), so the
training, validation-testing, and eval scripts **must all agree** on which model
and which constructor args are used. The `.pth` files store only `state_dict`
(no model class or hyperparameters), so loading requires manually
re-instantiating the matching architecture. If you change constructor args, re-train,
or expect `load_state_dict` to raise a size-mismatch error.

### Data loading

- `CrosstalkDataset` in `train_model.py` (duplicated almost verbatim in
  `test-cross-talk-model.py`) walks mixed and source dirs, matches filenames with
  regex `image_(\d+)_alpha_(\d+\.?\d*)_(mixed|source)\.tif`, and pairs files by
  `(image_id, alpha_str)`. The alpha float is the regression label. Samples are
  **sorted** by `(image_id, scalar_label)` for deterministic splitting.
- `SplitCrosstalkDataset` wraps an already-split list of sample dicts.
- **Note:** the `CrosstalkDataset` class and `normalize_image`,
  `val_test_transforms_fn`, `evaluate_and_save` functions are **copy-pasted**
  into both `train_model.py` and `test-cross-talk-model.py` rather than imported.
  Editing one file does NOT affect the other — apply fixes to both.

### Training pipeline (`train_model.py`)

1. Build `CrosstalkDataset`, then split with `torch.manual_seed(43)` +
   `torch.randperm` into train/val/test by ratio.
2. Per-image min-max normalization (`normalize_image`). Training augments with
   random h/v flips; the affine/noise/erase augmentations are all **commented out**
   but left in the file.
3. Optimizer: Adam with `weight_decay=1e-4`. Loss: `MSELoss`.
4. `train_model()` builds a scheduler from a `scheduler_configs` dict keyed by
   the `-r` flag. `onecycle` steps per-batch; `plateau`/`custom_warmup` step per
   epoch. Early stopping with per-scheduler patience.
5. Outputs are written to a timestamped `training_run_<ts>_B<bs>_LR<lr>` dir:
   `params.txt`, `model_architecture.txt`, `training_log_*.csv`, best/final
   `.pth`, loss plot, and `{train,val,test}_predictions_*.csv` + scatter PNGs.

### Eval pipeline (`test-cross-talk-model.py`)

Produces `eval_run_<ts>/` with `params.txt`, `model_architecture.txt`, and a
`test_predictions_<ts>.csv` containing additional image-similarity metrics
(RMSE, SSIM, histogram correlation, NMI, Pearson) computed between the two input
channels, plus a scatter plot per metric.

### Scheduler naming pitfall

`scheduler_configs` has a key `cosine_warmup` whose `type` is actually
`'custom_warmup'`, but the scheduler-instantiation `if/elif` chain only handles
`'plateau'`, `'onecycle'`, and `'cosine'` — there is **no branch for
`'custom_warmup'`**. Selecting `-r cosine_warmup` therefore creates no scheduler
object, and the later `scheduler.step()` call will crash with an
`UnboundLocalError`. `cosine_warmup` is effectively broken as wired.

## Conventions & Style

- Plain scripts, no `pytest`/`unittest`, no `setup.py`/`pyproject.toml`. Package
  structure is flat: model modules at repo root, imported via `from
  regression_model import *` (star imports).
- Models subclass `torch.nn.Module`; use `nn.Sequential`; `LeakyReLU(0.01)` and
  `BatchNorm` throughout.
- Timestamps use `datetime.now().strftime("%Y-%m-%d_%H-%M-%S")` and are embedded
  in nearly every output filename and directory.
- CSV logs are written with the `csv` module; the training log prepends metadata
  rows (learning rate, batch size, scheduler) **before** the `epoch,...` header,
  which is why `analyse_training_results.py` has a `skip_rows` helper to locate
  the real header.
- Filenames are the source of truth for labels and pairing — never assume a
  separate manifest exists.

## Downstream consumers of the model checkpoint

The trained checkpoint `crosstalk_regression_model_trained_*.pth` is a **shared
contract** with other Crick projects, notably
`github.com/FrancisCrickInstitute/py-bioimage-qc`. That repo does *not* import
this code — it vendors its own `regression_model.py` (class renamed to
`CrossTalkRegressionModel`) and loads the same `.pth` inline with hardcoded
constructor args `initial_filters=128, num_conv_blocks=6`.

Consequences:

- The `.pth` state_dict, the constructor args, and the `single` architecture must
  stay compatible. Changing `SINGLE_MODEL_KWARGS`, layer counts, or the class
  name will silently break the downstream loader (which can't see our factory).
- Do not assume renaming files here (e.g. `regression_model.py`) is safe; the
  downstream repo has its own copy and matches only by checkpoint contract.

## Data Layout

- `Training_Data/Mixed/` and `Training_Data/Source/` — paired `.tif` images
  named `image_<id>_alpha_<value>_{mixed|source}.tif`.
- `PreTrained_Model/` — shipped `.pth` weights for the `single` model.
- `eval_run_*/` and `training_run_*/` — timestamped output artifacts committed
  to the repo (historical runs).
- `transformed_images/`, `*.tif`, `*.png`, `*.pdf` at root — miscellaneous
  example/scratch data.
- `test_predictions.csv`, `training_log_regression.csv` — standalone sample
  outputs.

## Gotchas Summary

1. No tests/CI; validate changes by actually running a short training or eval.
2. `-o` model choice + constructor args must match the `.pth` being loaded.
3. `CrosstalkDataset` and helpers are duplicated across `train_model.py` and
   `test-cross-talk-model.py` — keep in sync manually.
4. `cosine_warmup` scheduler is broken (missing `custom_warmup` branch).
5. `requirements.txt` omits `scipy`, `scikit-image`, `scikit-learn`, `pandas`
   that the scripts import.
6. `pixi.toml` (Python 3.14) vs README/`requirements.txt` (Python 3.13) disagree.
7. Default CLI paths point at NEMO/Linux HPC mounts, not local Windows paths.
8. Image size is hardcoded to 256×256 (`TARGET_IMAGE_SIZE`, and
   `_get_conv_output((256,256))`).
