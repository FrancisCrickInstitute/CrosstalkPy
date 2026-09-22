import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def skip_rows(file_path, colname):
    skiprows_val = -1
    while True:
        skiprows_val += 1
        try:
            df = pd.read_csv(file_path, skiprows=skiprows_val)
            if colname in df.columns:
                print(f"Successfully read with skiprows={skiprows_val}")
                return skiprows_val
        except Exception as e:
            continue

    print("Could not find a valid header configuration.")
    return -1


def extract_run_info_from_dirname(dirname):
    """Extract batch size and learning rate from directory name."""
    # Pattern to match directory names like: training_run_2025-08-16_09-11-06_B256_LR1e-06
    pattern = r'training_run_.*_B(\d+)_LR([\d\.e\-]+)'
    match = re.search(pattern, dirname)

    if match:
        batch_size = int(match.group(1))
        learning_rate = float(match.group(2))
        return batch_size, learning_rate
    else:
        return None, None


def calculate_mse(actual, predicted):
    """Calculate Mean Squared Error."""
    return np.mean((actual - predicted) ** 2)


def analyze_training_log(log_file_path):
    """Analyze training log to find minimum losses and their epochs."""
    try:
        # Read the CSV file, skipping the first two rows (metadata)
        df = pd.read_csv(log_file_path, skiprows=skip_rows(log_file_path, 'epoch'))

        # Find minimum training loss and its epoch
        min_train_loss_idx = df['train_loss'].idxmin()
        min_train_loss = df.loc[min_train_loss_idx, 'train_loss']
        min_train_epoch = df.loc[min_train_loss_idx, 'epoch']

        # Find minimum validation loss and its epoch
        min_val_loss_idx = df['val_loss'].idxmin()
        min_val_loss = df.loc[min_val_loss_idx, 'val_loss']
        min_val_epoch = df.loc[min_val_loss_idx, 'epoch']

        return {
            'min_train_loss': min_train_loss,
            'min_train_epoch': min_train_epoch,
            'min_val_loss': min_val_loss,
            'min_val_epoch': min_val_epoch
        }
    except Exception as e:
        print(f"Error reading training log {log_file_path}: {e}")
        return None


def analyze_test_predictions(test_file_path):
    """Analyze test predictions to calculate MSE."""
    try:
        # Read the CSV file, skipping the first two rows (metadata)
        df = pd.read_csv(test_file_path, skiprows=skip_rows(test_file_path, 'Actual_Label'))

        # Calculate MSE between actual and predicted labels
        mse = calculate_mse(df['Actual_Label'].values, df['Predicted_Label'].values)

        return mse
    except Exception as e:
        print(f"Error reading test predictions {test_file_path}: {e}")
        return None


def analyze_training_directory(base_directory):
    """Analyze all training run directories and compile results."""

    results = []

    # Find all training run directories
    training_dirs = glob.glob(os.path.join(base_directory, "training_run_*"))

    for training_dir in training_dirs:
        dir_name = os.path.basename(training_dir)
        print(f"Analyzing directory: {dir_name}")

        # Extract batch size and learning rate from directory name
        batch_size, learning_rate = extract_run_info_from_dirname(dir_name)

        # Look for training log file
        training_log_pattern = os.path.join(training_dir, "training_log_*.csv")
        training_log_files = glob.glob(training_log_pattern)

        # Look for test predictions file
        test_pred_pattern = os.path.join(training_dir, "test_predictions_*.csv")
        test_pred_files = glob.glob(test_pred_pattern)

        # Initialize result dictionary
        result = {
            'directory': dir_name,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'min_train_loss': None,
            'min_train_epoch': None,
            'min_val_loss': None,
            'min_val_epoch': None,
            'test_mse': None
        }

        # Analyze training log if found
        if training_log_files:
            training_analysis = analyze_training_log(training_log_files[0])
            if training_analysis:
                result.update(training_analysis)
        else:
            print(f"  Warning: No training log found in {dir_name}")

        # Analyze test predictions if found
        if test_pred_files:
            test_mse = analyze_test_predictions(test_pred_files[0])
            if test_mse is not None:
                result['test_mse'] = test_mse
        else:
            print(f"  Warning: No test predictions found in {dir_name}")

        results.append(result)

    return results


def save_results_to_csv(results, output_file):
    """Save analysis results to CSV file."""
    df = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = [
        'directory',
        'batch_size',
        'learning_rate',
        'min_train_loss',
        'min_train_epoch',
        'min_val_loss',
        'min_val_epoch',
        'test_mse'
    ]

    df = df[column_order]
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyse training run directories and compile a results CSV."
    )
    parser.add_argument(
        "--base-directory",
        default=".",
        help="Directory containing training_run_* subdirectories (default: current dir)",
    )
    parser.add_argument(
        "--output-file",
        default="training_analysis_results.csv",
        help="Output CSV path (default: training_analysis_results.csv)",
    )
    args = parser.parse_args()

    base_directory = args.base_directory
    output_file = args.output_file

    print("Starting training analysis...")
    print(f"Looking for training directories in: {os.path.abspath(base_directory)}")

    # Analyze all training directories
    results = analyze_training_directory(base_directory)

    if not results:
        print("No training directories found!")
        return

    print(f"\nFound {len(results)} training directories")

    # Save results to CSV
    save_results_to_csv(results, output_file)

    # Display summary
    print("\nAnalysis Summary:")
    print("=" * 50)
    for result in results:
        print(f"Directory: {result['directory']}")
        print(f"  Batch Size: {result['batch_size']}, Learning Rate: {result['learning_rate']}")
        print(f"  Min Train Loss: {result['min_train_loss']:.6f} (Epoch {result['min_train_epoch']})")
        print(f"  Min Val Loss: {result['min_val_loss']:.6f} (Epoch {result['min_val_epoch']})")
        print(f"  Test MSE: {result['test_mse']:.6f}")
        print()


if __name__ == "__main__":
    main()
