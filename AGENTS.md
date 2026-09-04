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

Note the mismatch is now resolved: `pixi.toml` pins `python = "3.13.*"`, matching
the README/`requirements.txt` (`python=3.13`). Keep them in sync if either changes.

`requirements.txt` lists **fewer** packages than the code actually imports. The
test script needs `scipy`, `scikit-image`, `scikit-learn` (`pearsonr`, `ssim`,
`normalized_mutual_info_score`); `analyse_training_results.py` and
`examine_large_errors.py` need `pandas`; and `examine_large_errors.py` also needs
`requests` and `zarr` (with `fsspec[http]`, which is already listed). None of
these are declared in `pixi.toml` or `requirements.txt`.

The repo is Windows-targeted (`platforms = ["win-64"]`), but default CLI paths
in the scripts point at Linux/NEMO HPC mounts (`/nemo/...`, `Z:/working/...`),
so running on this machine needs explicit `-m`/`-s` args.

## Commands

There is **no test suite, linter, formatter, or CI**. The only entry points are
four standalone scripts run via `python`:

```bash
# Train (defaults point at ./Training_Data/Mixed and ./Training_Data/Source)
python train_model.py -m ./Training_Data/Mixed -s ./Training_Data/Source

# Evaluate a trained model (defaults point at a NEMO path; override -p)
python test-cross-talk-model.py -m <mixed_dir> -s <source_dir> \
    -p ./PreTrained_Model/crosstalk_regression_model_trained_2025-12-15_18-22-01_256_0.0005.pth \
    -o single

# Ad-hoc analysis script (hardcoded base dir at top of main(); not CLI-driven)
python analyse_training_results.py

# Download top-N worst predictions from IDR (reads an eval CSV; see below)
python examine_large_errors.py --csv_file <test_predictions_*.csv> --top-n 20
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

### Scheduler naming pitfall (resolved)

`scheduler_configs` has a key `cosine_warmup` whose `type` is the literal
`'custom_warmup'`. The instantiation `if/elif` chain must include a
`'custom_warmup'` branch (it now does — a warmup + cosine `LambdaLR`), and the
per-epoch stepping chain lists both `'cosine'` and `'custom_warmup'`. If either
chain ever drops `'custom_warmup'`, `-r cosine_warmup` will crash with an
`UnboundLocalError` on the first `scheduler.step()`.

### Error-analysis pipeline (`examine_large_errors.py`)

Reads an eval CSV (the `test_predictions_<ts>.csv` written by
`test-cross-talk-model.py`), computes `abs(Actual_Label - Predicted_Label)`,
ranks the worst `--top-n`, fetches per-image metadata from the IDR REST API, and
downloads the original images from IDR's public OME-Zarr stores (`zarr` +
`fsspec[http]`, no auth).

Gotchas:

- It requires an `Image_ID` column, which **only** `test-cross-talk-model.py`
  writes. `train_model.py`'s `evaluate_and_save` does NOT emit `Image_ID`, so its
  prediction CSVs can't be fed here.
- The `Image_ID` is the IDR image id (the `image_<id>_alpha_<value>_*.tif` id),
  used directly as the IDR API key — not a local file index.
- Default `--csv_file` points at a NEMO/HPC path (`Z:/working/...`); override
  locally.
- It adds deps `requests` and `zarr` (plus the already-listed `fsspec`, `pandas`).

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

### Self-contained export (TorchScript `.pt`)

`model_factory.save_torchscript(model, path)` exports the model as a single
`.pt` file that bundles architecture + weights. Consumers load it with
`torch.jit.load(path)` and call it directly — no class definition or constructor
args needed, and batch size stays dynamic. This is the recommended artifact to
hand to external projects instead of the raw `.pth` state_dict (which requires
re-instantiating the architecture).

`train_model.py` now writes both the `.pth` (state_dict, for resume/reload) and
a `.pt` (TorchScript, for external consumers) alongside each other.

Gothca: `torch.jit.script` is deprecated in torch 2.14+ (emits a `FutureWarning`
on *all* Python versions, not just 3.14), and upstream recommends `torch.export`
as the replacement. However, `torch.export` shape-specializes its input (batch
becomes fixed at export time) and rejects other batch sizes. For this
PyTorch-only, dynamic-batch use case, TorchScript still works and is the
pragmatic choice. Revisit if a future PyTorch removes `torch.jit`.

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
2. `.pth` files store only `state_dict`; the `-o` choice + constructor args in
   `model_factory.py` must match the checkpoint being loaded.
3. `CrosstalkDataset` and helpers are duplicated across `train_model.py` and
   `test-cross-talk-model.py` — keep in sync manually.
4. Default CLI paths point at NEMO/Linux HPC mounts, not local Windows paths.
5. Image size is hardcoded to 256×256 (`TARGET_IMAGE_SIZE`, and
   `_get_conv_output((256,256))`).

## Known Issues / To-Do

Prioritized in rough order of impact. No test suite exists, so verify each fix
by running the relevant script.

1. **Split-ratio validation bug** (`train_model.py`, `main()`): the guard
   `if not (abs(train_ratio + val_ratio) < 1.0)` is inverted nonsense — it warns
   when ratios sum to 1.0 and never rejects `train_ratio + val_ratio >= 1.0`,
   which yields an empty/negative test split. Should validate the sums are in
   `(0, 1]`.
2. **`l2_regularization` is dead code** (`train_model.py`): defined but never
   called; regularizer is actually `weight_decay=1e-4` in the Adam optimizer.
   Remove or wire it up.
3. **`drop_last=True` silently drops samples**: all three `DataLoader`s (train/
   val/test) drop the final partial batch, but loss is normalized by
   `len(dataloader.dataset)`, so reported losses undercount on small datasets.
   Decide intended behavior.
4. **Remaining code duplication**: `CrosstalkDataset`, `normalize_image`,
   `val_test_transforms_fn`, `evaluate_and_save` are copy-pasted between
   `train_model.py` and `test-cross-talk-model.py` (model instantiation is
   already centralized in `model_factory.py`). Extract shared modules.
5. **Star imports** (`from regression_model import *`, `from two_branch_regression import *`)
   — fragile; replace with explicit imports.
6. **Hardcoded 256×256 input**: `TARGET_IMAGE_SIZE` is defined but not applied
   as a resize; the models assume 256×256 via `_get_conv_output((256,256))` and
   the two-branch dummy input. Non-256 inputs break the FC layer.
7. **Missing declared dependencies**: `scipy`, `scikit-image`, `scikit-learn`
   (used in `test-cross-talk-model.py`), `pandas` (used in
   `analyse_training_results.py` and `examine_large_errors.py`), and `requests` +
   `zarr` (used in `examine_large_errors.py`) are absent from `pixi.toml` and
   `requirements.txt`.
8. **Hardcoded paths**: `analyse_training_results.py` (`base_directory`) and
   `examine_large_errors.py` (`--csv_file` default) both point at
   `Z:/working/barryd/hpc/python/Torch-Unet`; neither is CLI-driven for its base
   path.
9. **Dead code in `evaluate_and_save`** (`train_model.py` version): an unused
   `csv.writer` and `fieldnames` local precede the `DictWriter` call.
10. **`examine_large_errors.py` not yet validated end-to-end**: requires
    `requests` and `zarr` (not currently installed in the `crosstalk` env) and
    network access to IDR; has not been run in this repo.
