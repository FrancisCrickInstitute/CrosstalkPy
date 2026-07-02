#!/usr/bin/env python3
"""
Script to identify images with the largest prediction errors and copy them to an output directory.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def find_image_files(image_id, data_root):
    """
    Find the bleed and source image files for a given Image_ID.

    Args:
        image_id: The image ID to search for
        data_root: Root directory containing crosstalk_training_data

    Returns:
        Tuple of (bleed_file, source_file) or (None, None) if not found
    """
    # Convert image_id to integer to remove any decimal points
    image_id_int = int(image_id)

    bleed_dir = Path(data_root) / "crosstalk_training_data" / "bleed"
    source_dir = Path(data_root) / "crosstalk_training_data" / "source"

    # Use glob patterns to find files with variable alpha values
    bleed_files = list(bleed_dir.glob(f"image_{image_id_int}_alpha_*_mixed.tif"))
    source_files = list(source_dir.glob(f"image_{image_id_int}_alpha_*_source.tif"))

    bleed_path = bleed_files[0] if bleed_files else None
    source_path = source_files[0] if source_files else None

    return bleed_path, source_path


def main():
    parser = argparse.ArgumentParser(
        description="Find images with largest prediction errors and copy them to output directory"
    )
    parser.add_argument(
        "--csv_file",
        default="Z:/working/barryd/hpc/python/Torch-Unet/eval_run_2025-12-16_09-45-57/test_predictions_2025-12-16_09-45-58.csv",
        help="Path to the CSV file with predictions"
    )
    parser.add_argument(
        "--data-root",
        default="Z:/working/barryd/IDR",
        help="Root directory containing crosstalk_training_data folder (default: current directory)"
    )
    parser.add_argument(
        "--output-dir",
        default="largest_errors_images",
        help="Output directory to store the copied images (default: largest_errors_images)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of images with largest errors to extract (default: 20)"
    )

    args = parser.parse_args()

    # Load the CSV file
    print(f"Loading CSV file: {args.csv_file}")
    df = pd.read_csv(args.csv_file)

    # Calculate absolute error between Actual_Label and Predicted_Label
    df['Error'] = abs(df['Actual_Label'] - df['Predicted_Label'])

    # Sort by error in descending order and get top N
    df_sorted = df.nlargest(args.top_n, 'Error')

    print(f"\nTop {args.top_n} images with largest prediction errors:")
    print("-" * 70)
    print(f"{'Image_ID':<15} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
    print("-" * 70)

    for idx, row in df_sorted.iterrows():
        print(
            f"{row['Image_ID']:<15} {row['Actual_Label']:<10.4f} {row['Predicted_Label']:<12.4f} {row['Error']:<10.4f}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n\nCreating output directory: {output_path}")

    # Create subdirectories for organization
    bleed_output = output_path / "bleed"
    source_output = output_path / "source"
    bleed_output.mkdir(parents=True, exist_ok=True)
    source_output.mkdir(parents=True, exist_ok=True)

    # Copy image files
    copied_count = 0
    missing_count = 0

    print(f"\nCopying image files...")
    print("-" * 70)

    for idx, row in df_sorted.iterrows():
        image_id = row['Image_ID']
        error = row['Error']

        bleed_file, source_file = find_image_files(image_id, args.data_root)

        files_found = 0

        # Copy bleed file
        if bleed_file:
            try:
                output_file = bleed_output / bleed_file.name
                shutil.copy2(bleed_file, output_file)
                files_found += 1
                copied_count += 1
            except Exception as e:
                print(f"Error copying bleed file for {image_id}: {e}")

        # Copy source file
        if source_file:
            try:
                output_file = source_output / source_file.name
                shutil.copy2(source_file, output_file)
                files_found += 1
                copied_count += 1
            except Exception as e:
                print(f"Error copying source file for {image_id}: {e}")

        if files_found == 0:
            print(f"Image_ID {image_id}: NO FILES FOUND (error: {error:.4f})")
            missing_count += 1
        elif files_found == 1:
            print(f"Image_ID {image_id}: 1 file found (error: {error:.4f})")
        else:
            print(f"Image_ID {image_id}: Both files found (error: {error:.4f})")

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY:")
    print(f"  Total files copied: {copied_count}")
    print(f"  Images with missing files: {missing_count}")
    print(f"  Output directory: {output_path.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
