"""Re-run evaluation and plotting for an existing checkpoint (no re-training).

A training run writes the model weights *before* it runs evaluation, so a crash
in the export/evaluation phase can leave a run with a saved `.pth` but no
prediction CSVs or plots. This script loads an existing checkpoint, reproduces
the train/val/test split, and regenerates:

- `{train,val,test}_predictions_<ts>.csv` + scatter PNGs (same format as
  `train_model.py`'s `evaluate_and_save`),
- a training/validation loss curve re-plotted from `training_log_*.csv`.

Usage:
    python evaluate_run.py --model-path <checkpoint.pth> \
        -m <mixed_dir> -s <source_dir> \
        [--model-selection single|double] [--train-ratio 0.7] [--val-ratio 0.15] \
        [--batch-size 128] [--cpu-jobs 6] [--output-dir <dir>]

The output directory defaults to the checkpoint's directory.
"""

import argparse
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_factory import load_model

# Reuse the shared dataset/split/transform code from train_model rather than
# duplicating it (these are importable because train_model guards its main body
# behind `if __name__ == "__main__"`).
from train_model import (
    CrosstalkDataset,
    SplitCrosstalkDataset,
    val_test_transforms_fn,
)


def _skip_rows(file_path, colname):
    """Find how many leading rows to skip before the real CSV header."""
    for skiprows in range(101):
        try:
            df = pd.read_csv(file_path, skiprows=skiprows)
            if colname in df.columns:
                return skiprows
        except Exception:
            continue
    return 0


def evaluate(model, dataloader, dataset_name, output_dir, timestamp, batch_size,
             learning_rate, device):
    """Evaluate one split and write CSV + scatter PNG (mirrors train_model.py)."""
    print(f"\n--- Evaluating on {dataset_name.capitalize()} Set ---")
    predictions = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"{dataset_name.capitalize()}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            actual = labels.cpu().numpy().flatten()
            predicted = outputs.cpu().numpy().flatten()
            predictions.extend(zip(actual, predicted))

    csv_path = os.path.join(
        output_dir, f"{dataset_name}_predictions_{timestamp}_{batch_size}_{learning_rate}.csv"
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Actual_Label", "Predicted_Label"])
        w.writerows(predictions)
    print(f"  predictions saved to {csv_path}")

    if predictions:
        actual = [a for a, _ in predictions]
        predicted = [p for _, p in predictions]
        plt.figure(figsize=(8, 8))
        plt.scatter(actual, predicted, alpha=0.6, s=10)
        plt.plot([min(actual), max(actual)], [min(actual), max(actual)],
                 "--r", label="Ideal (y=x)")
        plt.xlabel("Actual Label")
        plt.ylabel("Predicted Label")
        plt.title(f"{dataset_name.capitalize()} Set: Actual vs. Predicted Labels")
        plt.legend()
        plot_path = os.path.join(
            output_dir,
            f"{dataset_name}_predictions_plot_{timestamp}_{batch_size}_{learning_rate}.png",
        )
        plt.savefig(plot_path)
        plt.close()
        print(f"  scatter plot saved to {plot_path}")


def plot_loss_curve(output_dir, timestamp, batch_size, learning_rate):
    """Re-plot train/val loss from an existing training_log_*.csv."""
    import glob
    logs = glob.glob(os.path.join(output_dir, "training_log_*.csv"))
    if not logs:
        print("\nNo training_log_*.csv found; skipping loss-curve plot.")
        return
    log_file = sorted(logs)[-1]  # latest
    skip = _skip_rows(log_file, "epoch")
    df = pd.read_csv(log_file, skiprows=skip)
    if "epoch" not in df.columns or "train_loss" not in df.columns:
        print(f"\nCould not parse {log_file}; skipping loss-curve plot.")
        return
    epochs = df["epoch"].values
    train_loss = df["train_loss"].values
    val_loss = df["val_loss"].values
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(bottom=0)
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    path = os.path.join(
        output_dir, f"training_validation_loss_{timestamp}_{batch_size}_{learning_rate}.png"
    )
    plt.savefig(path)
    plt.close()
    print(f"Loss curve saved to {path}")


def main():
    p = argparse.ArgumentParser(
        description="Re-run evaluation and plotting for an existing checkpoint."
    )
    p.add_argument("--model-path", required=True, help="Path to the .pth checkpoint")
    p.add_argument("-m", "--mixed-dir", required=True, help="Mixed channel data directory")
    p.add_argument("-s", "--source-dir", required=True, help="Pure source data directory")
    p.add_argument("-o", "--model-selection", choices=["single", "double"], default="single")
    p.add_argument("-t", "--train-ratio", type=float, default=0.7)
    p.add_argument("-v", "--val-ratio", type=float, default=0.15)
    p.add_argument("-b", "--batch-size", type=int, default=128)
    p.add_argument("-l", "--learning-rate", type=float, default=1e-4,
                   help="Learning rate of the trained run (for output filenames)")
    p.add_argument("-j", "--cpu-jobs", type=int, default=6)
    p.add_argument("--output-dir", default=None, help="Output dir (default: checkpoint dir)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.model_path))
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Build and split the dataset identically to train_model.py.
    full = CrosstalkDataset(args.mixed_dir, args.source_dir, transform=val_test_transforms_fn)
    all_samples = full.samples
    torch.manual_seed(43)
    shuffled = torch.randperm(len(all_samples)).tolist()
    train_size = int(args.train_ratio * len(all_samples))
    val_size = int(args.val_ratio * len(all_samples))
    train_samples = [all_samples[i] for i in shuffled[:train_size]]
    val_samples = [all_samples[i] for i in shuffled[train_size:train_size + val_size]]
    test_samples = [all_samples[i] for i in shuffled[train_size + val_size:]]
    print(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

    splits = {
        "train": SplitCrosstalkDataset(args.mixed_dir, args.source_dir,
                                        val_test_transforms_fn, train_samples),
        "val": SplitCrosstalkDataset(args.mixed_dir, args.source_dir,
                                      val_test_transforms_fn, val_samples),
        "test": SplitCrosstalkDataset(args.mixed_dir, args.source_dir,
                                       val_test_transforms_fn, test_samples),
    }

    model = load_model(args.model_path, args.model_selection, device)

    for name, ds in splits.items():
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.cpu_jobs, pin_memory=True, drop_last=False)
        evaluate(model, loader, name, output_dir, timestamp,
                 args.batch_size, args.learning_rate, device)

    plot_loss_curve(output_dir, timestamp, args.batch_size, args.learning_rate)


if __name__ == "__main__":
    main()
