import hashlib
import json
import os

import torch

from regression_model import AdvancedRegressionModel
from two_branch_regression import SimplifiedTwoBranchRegressionModel

# Single source of truth for the constructor args used by `single` and `double`.
# Keep these in sync with any change to the model definitions. The .pth files
# store only state_dict weights (no architecture), so these args MUST match the
# checkpoint being loaded.
SINGLE_MODEL_KWARGS = {"initial_filters": 128, "num_conv_blocks": 6}
DOUBLE_MODEL_KWARGS = {"initial_filters_per_branch": 64}

# Semantic version of the exported model artifacts. Bump whenever the weights
# or the input/output contract change, so pip-installable consumers can pin or
# request a specific version instead of relying on the timestamp in the filename.
MODEL_VERSION = "1.0.0"


def build_model(model_selection):
    """Instantiate the model matching the requested architecture.

    Args:
        model_selection (str): 'single' or 'double'.

    Returns:
        torch.nn.Module: An untrained model instance.
    """
    if model_selection == 'double':
        return SimplifiedTwoBranchRegressionModel(**DOUBLE_MODEL_KWARGS)
    if model_selection == 'single':
        return AdvancedRegressionModel(**SINGLE_MODEL_KWARGS)
    raise ValueError(
        f"Unknown model_selection '{model_selection}'. Expected 'single' or 'double'."
    )


def load_model(model_path, model_selection, device):
    """Build a model and load a checkpoint, with a clear error on mismatch.

    Args:
        model_path (str): Path to the .pth state_dict checkpoint.
        model_selection (str): 'single' or 'double'.
        device (torch.device): Device to map the weights onto.

    Returns:
        torch.nn.Module: Model in eval mode with loaded weights.
    """
    model = build_model(model_selection)
    state_dict = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint '{model_path}' with model_selection "
            f"'{model_selection}'. The weights do not match the requested "
            "architecture. Check that -o/--model_options matches the model the "
            "checkpoint was trained with, and that the constructor args in "
            "model_factory.py match the checkpoint's architecture."
        ) from exc
    model.eval()
    model.to(device)
    return model


def save_torchscript(model, save_path, example_input=None):
    """Export the model as a self-contained TorchScript artifact.

    Unlike saving `state_dict`, this bundles the architecture *and* the weights
    into a single file. Consumers load it with `torch.jit.load` in eval mode and
    call it directly, without importing the model class or knowing the
    architecture (they only need to know the input shape). Batch size stays
    dynamic.

    Note: `torch.jit.script` emits a FutureWarning on Python 3.14+ and is slated
    for replacement by `torch.export`, but it still functions correctly and
    supports dynamic batch (which `torch.export` does not without static shape
    specialization). Revisit if/when TorchScript is removed in a future PyTorch.

    Args:
        model (torch.nn.Module): Trained model to export.
        save_path (str): Output path (conventionally ending in `.pt`).
    """
    model.eval()
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, save_path)
    return scripted


def load_torchscript(model_path, device):
    """Load a self-contained TorchScript artifact for inference.

    Unlike `load_model`, this needs no model class or constructor args: the
    architecture is bundled inside the `.pt` file. This is the loader a
    downstream consumer should use.

    Args:
        model_path (str): Path to the `.pt` file written by `save_torchscript`.
        device (torch.device): Device to run the model on.

    Returns:
        torch.jit.ScriptModule: Loaded model in eval mode on the given device.
    """
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    return model
    """Return the hex SHA256 digest of a file on disk."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(manifest_path, artifact_files, model_selection, model_kwargs):
    """Write a JSON manifest describing exported model artifacts.

    Args:
        manifest_path (str): Output path for the JSON manifest.
        artifact_files (dict): Mapping of descriptor -> relative filename for
            each artifact (e.g. {'pth': '...', 'pt': '...'}).
        model_selection (str): 'single' or 'double'.
        model_kwargs (dict): Constructor args used to build the model.
    """
    entries = {}
    for key, filename in artifact_files.items():
        full_path = os.path.join(os.path.dirname(manifest_path), filename)
        entries[key] = {"file": filename, "sha256": hash_file(full_path)}

    manifest = {
        "version": MODEL_VERSION,
        "model_selection": model_selection,
        "model_kwargs": model_kwargs,
        "input": {
            "channels": 2,
            "height": 256,
            "width": 256,
            "channel_order": ["mixed", "source"],
        },
        "output": {"type": "scalar", "description": "crosstalk alpha in [0, 1]"},
        "artifacts": entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
