"""Export a trained checkpoint without re-training.

Given an existing `.pth` checkpoint (saved by `train_model.py`), this script
regenerates the distribution artifacts that are written *after* training:

- the self-contained TorchScript `.pt` bundle,
- the `sample_input.npy` / `sample_output.npy` self-verification tensors,
- the `model_manifest_*.json` (version + SHA256 + input/output contract),
- copies of the above into `releases/v<version>/`, and
- the `releases/LATEST` pointer.

This is useful when a training run completed the weights but stopped before the
export step (e.g. a post-training device error), or when you want to re-export
an existing checkpoint after a code change without re-training.

Usage:
    python export_model.py --model-path <checkpoint.pth> \
        [--model-selection single|double] [--output-dir <dir>]

The output directory defaults to the checkpoint's parent directory.
"""

import argparse
import json
import os
import shutil

import torch

from model_factory import (
    DOUBLE_MODEL_KWARGS,
    MODEL_VERSION,
    SINGLE_MODEL_KWARGS,
    build_sample_tensors,
    load_model,
    save_torchscript,
    write_manifest,
)


def export_checkpoint(model_path, model_selection, output_dir):
    """Regenerate export artifacts for an already-trained checkpoint.

    Args:
        model_path (str): Path to the `.pth` state_dict checkpoint.
        model_selection (str): 'single' or 'double'.
        output_dir (str): Directory to write artifacts into.

    Returns:
        dict describing the written artifacts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, model_selection, device)

    timestamp = os.path.basename(model_path).rsplit(".", 1)[0]

    # 1. TorchScript bundle
    script_path = os.path.join(output_dir, f"{timestamp}.pt")
    save_torchscript(model, script_path)
    print(f"TorchScript saved to {script_path}")

    # 2. Sample tensors (self-verification input/output pair)
    sample_tensors = build_sample_tensors(model, output_dir)
    print("Sample tensors saved to", output_dir)

    # 3. Manifest
    manifest_path = os.path.join(
        output_dir, f"model_manifest_v{MODEL_VERSION}_{timestamp}.json"
    )
    model_kwargs = SINGLE_MODEL_KWARGS if model_selection == "single" else DOUBLE_MODEL_KWARGS
    write_manifest(
        manifest_path,
        {
            "pth": os.path.basename(model_path),
            "pt": os.path.basename(script_path),
        },
        model_selection,
        model_kwargs,
        sample_tensors=sample_tensors,
    )
    print(f"Manifest saved to {manifest_path}")

    # 4. Copy to releases/ and write LATEST
    releases_dir = os.path.join("releases", f"v{MODEL_VERSION}")
    os.makedirs(releases_dir, exist_ok=True)
    for src in (
        os.path.join(output_dir, os.path.basename(model_path)),
        script_path,
        manifest_path,
        *[os.path.join(output_dir, fn) for fn in sample_tensors.values()],
    ):
        dst = os.path.join(releases_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"Copied {os.path.basename(src)} to {dst}")

    latest_path = os.path.join("releases", "LATEST")
    with open(latest_path, "w") as f:
        json.dump(
            {
                "version": MODEL_VERSION,
                "manifest": os.path.join(f"v{MODEL_VERSION}", os.path.basename(manifest_path)),
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"LATEST pointer written to {latest_path}")

    return {
        "script": script_path,
        "manifest": manifest_path,
        "sample_tensors": sample_tensors,
        "releases_dir": releases_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate distribution artifacts for an existing checkpoint."
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to the .pth state_dict checkpoint written by train_model.py",
    )
    parser.add_argument(
        "-o", "--model-selection", choices=["single", "double"], default="single",
        help="Model architecture the checkpoint was trained with (default: single)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write artifacts into (default: checkpoint's directory)",
    )
    args = parser.parse_args()

    model_path = os.path.abspath(args.model_path)
    output_dir = args.output_dir or os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.dirname(os.path.abspath(model_path)) != os.path.abspath(output_dir):
        # write_manifest hashes artifacts relative to the manifest's directory,
        # so keep the .pth alongside the manifest when they'd otherwise diverge.
        local_pth = os.path.join(output_dir, os.path.basename(model_path))
        shutil.copy2(model_path, local_pth)
        model_path = local_pth

    return export_checkpoint(model_path, args.model_selection, output_dir)


if __name__ == "__main__":
    main()
