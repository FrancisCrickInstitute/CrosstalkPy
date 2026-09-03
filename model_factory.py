import torch

from regression_model import AdvancedRegressionModel
from two_branch_regression import SimplifiedTwoBranchRegressionModel

# Single source of truth for the constructor args used by `single` and `double`.
# Keep these in sync with any change to the model definitions. The .pth files
# store only state_dict weights (no architecture), so these args MUST match the
# checkpoint being loaded.
SINGLE_MODEL_KWARGS = {"initial_filters": 128, "num_conv_blocks": 6}
DOUBLE_MODEL_KWARGS = {"initial_filters_per_branch": 64}


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
