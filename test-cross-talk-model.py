import argparse
import csv
import os
import re  # Import regex for pattern matching
from datetime import datetime

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_factory import build_model, load_model
from regression_model import *
from two_branch_regression import *

INDEX_PAGE = "https://idr.openmicroscopy.org/webclient/?experimenter=-1"

TARGET_IMAGE_SIZE = (256, 256)


def evaluate_and_save(model, dataloader, dataset_name, output_dir):
    """
    Evaluates the model, saves predictions to a CSV, and plots the results.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): The data loader for the dataset.
        dataset_name (str): The name of the dataset (e.g., 'test', 'train', 'val').
        output_dir (str): The directory to save output files.
    """
    print(f"\n--- Evaluating Model on {dataset_name.capitalize()} Set ---")

    predictions_data = []
    running_loss = 0.0
    fieldnames = ['Image_ID', 'Actual_Label', 'Predicted_Label', 'Root Mean Squared Error',
                  'Structural Similarity Index',
                  'Histogram Correlation', 'Normalized Mutual Information', 'Pearsons Correlation']

    with torch.no_grad():
        for i, (inputs, labels, ids) in enumerate(tqdm(dataloader, desc=f"{dataset_name.capitalize()} Set Evaluation")):
            inputs = inputs.to(device)
            labels = labels.to(device)
            ids = ids.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            images = inputs.cpu().numpy()

            actual_labels = labels.cpu().numpy().flatten()
            image_ids = ids.cpu().numpy().flatten()
            predicted_labels = outputs.cpu().numpy().flatten()

            for j in range(len(actual_labels)):
                img1_flat = images[j][0].flatten()
                img2_flat = images[j][1].flatten()
                if np.std(img1_flat) == 0 or np.std(img2_flat) == 0:
                    image_p = np.nan
                else:
                    image_p, p2 = pearsonr(img1_flat, img2_flat)
                hist1 = np.histogram(images[j][0].flatten(), bins=256)[0]
                hist2 = np.histogram(images[j][1].flatten(), bins=256)[0]
                if np.std(hist1) == 0 or np.std(hist2) == 0:
                    hist_p = np.nan
                else:
                    hist_p, p1 = pearsonr(hist1, hist2)
                img1_binned = np.digitize(images[j][0].flatten(),
                                          bins=np.linspace(images[j][0].min(), images[j][0].max(), 256))
                img2_binned = np.digitize(images[j][1].flatten(),
                                          bins=np.linspace(images[j][1].min(), images[j][1].max(), 256))
                predictions_data.append({
                    fieldnames[0]: image_ids[j],
                    fieldnames[1]: actual_labels[j],
                    fieldnames[2]: predicted_labels[j],
                    fieldnames[3]: np.sqrt(np.mean((images[j][0] - images[j][1]) ** 2)),
                    fieldnames[4]: ssim(images[j][0], images[j][1],
                                        data_range=np.max([images[j][0].max(), images[j][1].max()])
                                                   - np.min([images[j][0].min(), images[j][1].min()])),
                    fieldnames[5]: hist_p,
                    fieldnames[6]: normalized_mutual_info_score(img1_binned, img2_binned),
                    fieldnames[7]: image_p
                })

    final_loss = running_loss / len(dataloader.dataset)
    print(f"Final {dataset_name.capitalize()} Loss: {final_loss:.6f}")

    # --- Save predictions to CSV ---
    output_csv_path = os.path.join(output_dir,
                                   f"{dataset_name}_predictions_{current_time}.csv")
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        dict_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(predictions_data)

    print(f"{dataset_name.capitalize()} predictions saved to {output_csv_path}")

    # --- Plot Results ---
    if predictions_data:
        actual_labels_all = [d['Actual_Label'] for d in predictions_data]
        for f in fieldnames[2:]:
            metric = [d[f] for d in predictions_data]

            plt.figure(figsize=(10, 10))
            plt.scatter(actual_labels_all, metric, alpha=0.05, s=5)
            plt.plot([min(actual_labels_all), max(actual_labels_all)],
                     [min(actual_labels_all), max(actual_labels_all)],
                     '--r', label='Ideal Prediction (y=x)')
            plt.xlabel("Actual Label")
            plt.ylabel(f)
            plt.title(f"{dataset_name.capitalize()} Set: Actual Labels vs. {f}")
            plt.legend()
            plot_path = os.path.join(output_dir,
                                     f"{dataset_name}_{f}_plot_{current_time}.png")
            plt.savefig(plot_path)
            print(f"{dataset_name.capitalize()} {f} plot saved to {plot_path}")
            plt.close()


# --- Custom Dataset Classes (MODIFIED to use (image_id, alpha_value) as key) ---
class CrosstalkDataset(Dataset):
    def __init__(self, mixed_channel_dir, pure_source_dir, transform=None, max_samples=None):
        self.mixed_channel_dir = mixed_channel_dir
        self.pure_source_dir = pure_source_dir
        self.transform = transform

        # Regex to extract the unique ID, the alpha value, and the file type
        # This pattern works for both '_mixed.tif' and '_source.tif' files
        self.file_pattern = re.compile(r'image_(\d+)_alpha_(\d+\.?\d*)_(mixed|source)\.tif')

        # Store mappings: {(image_id, alpha_value_str): {mixed_file: path, source_file: path}}
        # Using alpha_value_str (from regex group) as part of the key to avoid float comparison issues
        all_sample_info = {}

        # Helper to process files from a directory
        def process_files_in_dir(directory, file_type_key):
            for filename in os.listdir(directory):
                if filename.endswith('.tif'):
                    match = self.file_pattern.search(filename)
                    if match:
                        image_id = match.group(1)
                        alpha_value_str = match.group(2)  # Keep as string for the key
                        file_actual_type = match.group(3)

                        # Only process files that match the expected type for this directory
                        if file_actual_type == file_type_key:
                            compound_key = (image_id, alpha_value_str)
                            if compound_key not in all_sample_info:
                                all_sample_info[compound_key] = {}
                            all_sample_info[compound_key][f'{file_type_key}_file'] = filename

        # Process mixed channel files
        process_files_in_dir(mixed_channel_dir, 'mixed')
        # Process pure source files
        process_files_in_dir(pure_source_dir, 'source')

        self.samples = []
        for (image_id, alpha_value_str), info in all_sample_info.items():
            if 'mixed_file' in info and 'source_file' in info:
                self.samples.append({
                    'image_id': image_id,  # Can keep for debugging/sorting
                    'scalar_label': float(alpha_value_str),  # Convert to float for label
                    'mixed_channel_file': info['mixed_file'],
                    'pure_source_file': info['source_file']
                })

        if not self.samples:
            raise ValueError(
                "No matching samples found. Ensure filenames adhere to 'image_ID_alpha_VALUE_(mixed|source).tif' "
                "pattern and corresponding mixed/source files exist for each (ID, Alpha) pair.")

        # Sort samples for consistent order (important for splitting)
        # Sorting by image_id and then scalar_label to ensure stable order
        self.samples.sort(key=lambda x: (x['image_id'], x['scalar_label']))

        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"Found {len(self.samples)} matching samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        mixed_channel_path = os.path.join(self.mixed_channel_dir, sample_info['mixed_channel_file'])
        pure_source_path = os.path.join(self.pure_source_dir, sample_info['pure_source_file'])
        scalar_label = sample_info['scalar_label']
        image_id = sample_info['image_id']

        mixed_image_np = iio.imread(mixed_channel_path).astype(np.float32)
        source_image_np = iio.imread(pure_source_path).astype(np.float32)

        if self.transform:
            mixed_tensor, source_tensor, label_tensor = self.transform(mixed_image_np, source_image_np, scalar_label)
        else:
            mixed_tensor = torch.from_numpy(mixed_image_np).unsqueeze(0)
            source_tensor = torch.from_numpy(source_image_np).unsqueeze(0)
            label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

        id_tensor = torch.tensor(int(image_id), dtype=torch.uint64).unsqueeze(0)
        input_tensor = torch.cat([mixed_tensor, source_tensor], dim=0)
        return input_tensor, label_tensor, id_tensor


def normalize_image(img):
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:  # Avoid division by zero
        return (img - img_min) / (img_max - img_min)
    else:
        return img


def val_test_transforms_fn(mixed_np, source_np, scalar_label):
    mixed_np = normalize_image(mixed_np)
    source_np = normalize_image(source_np)
    mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
    source_tensor = torch.from_numpy(source_np).unsqueeze(0)
    label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

    return mixed_tensor, source_tensor, label_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training with various parameters.")

    parser.add_argument("-m", "--mixed_channel_data_dir", type=str,
                        default="/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data/bleed",
                        help="Directory for mixed channel data")
    parser.add_argument("-s", "--pure_source_data_dir", type=str,
                        default="/nemo/stp/lm/working/barryd/IDR/crosstalk_training_data/source",
                        help="Directory for pure source data")
    parser.add_argument("-p", "--model_path", type=str,
                        default="/nemo/stp/lm/working/barryd/hpc/python/Torch-Unet/training_run_2025-12-15_16-02-16_B256_LR0.0005/crosstalk_regression_model_trained_2025-12-15_18-22-01_256_0.0005.pth",
                        help="Path to pytorch model")
    parser.add_argument("-j", "--cpu_jobs", type=int, default=20, help="Number of CPUs to use")
    parser.add_argument("-o", "--model_options", type=str, default='single', help="Use single- or double-branch model",
                        choices=['single', 'double'])

    args = parser.parse_args()

    mixed_channel_data_dir = args.mixed_channel_data_dir
    pure_source_data_dir = args.pure_source_data_dir
    model_save_path = args.model_path
    ncpus = args.cpu_jobs
    model_selection = args.model_options

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(model_selection)

    # --- Create a unique output directory for this run ---
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Include batch size and learning rate in the folder name for easy identification
    output_dir_name = f"eval_run_{current_time_str}"
    os.makedirs(output_dir_name, exist_ok=True)
    print(f"Saving all outputs to: {output_dir_name}")

    args_dict = vars(args)
    params_list_path = os.path.join(output_dir_name, "params.txt")
    with open(params_list_path, 'w') as f:
        for arg, value in args_dict.items():
            f.write(f'{arg}: {value}\n')

    print(f"Parameters saved to {params_list_path}")

    # --- Save Model Architecture Summary ---
    model_summary_path = os.path.join(output_dir_name, "model_architecture.txt")
    with open(model_summary_path, "w") as f:
        f.write(str(model))  # This prints the __repr__ of the model
    print(f"Model architecture summary saved to {model_summary_path}")

    print("\nCreating dataset instances for initial file listing...")
    try:
        # No label_mapping_file needed, as labels are derived from filenames
        temp_dataset = CrosstalkDataset(
            mixed_channel_data_dir, pure_source_data_dir,
            transform=val_test_transforms_fn
        )
        print(f"Total samples found in directories: {len(temp_dataset)}")
    except Exception as e:
        print(f"Error initializing temporary dataset: {e}")
        exit()

    print("\nSplitting data using filename lists for correct augmentation application...")
    all_samples = temp_dataset.samples  # This now holds the list of dictionaries with all sample info

    test_size = len(all_samples)

    test_samples = all_samples

    test_dataset_final = temp_dataset

    test_dataloader = DataLoader(
        test_dataset_final,
        shuffle=False,
        num_workers=ncpus,
        pin_memory=True,
        drop_last=True
    )

    print("Dataloader created for testing.")
    criterion = torch.nn.MSELoss()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("\n--- Evaluating Model ---")
    loaded_model = load_model(model_save_path, model_selection, device)

    evaluate_and_save(loaded_model, test_dataloader, 'test', output_dir_name)
