import argparse
import csv
import datetime
import os
import re  # Import regex for pattern matching
from datetime import datetime

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_factory import build_model, load_model, save_torchscript
from regression_model import *
from two_branch_regression import *

TARGET_IMAGE_SIZE = (256, 256)


def l2_regularization(model, lambda_l2=1e-5):
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
    return lambda_l2 * l2_norm


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

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"{dataset_name.capitalize()} Set Evaluation")):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            actual_labels = labels.cpu().numpy().flatten()
            predicted_labels = outputs.cpu().numpy().flatten()

            for j in range(len(actual_labels)):
                predictions_data.append({
                    'Actual_Label': actual_labels[j],
                    'Predicted_Label': predicted_labels[j]
                })

    final_loss = running_loss / len(dataloader.dataset)
    print(f"Final {dataset_name.capitalize()} Loss: {final_loss:.6f}")

    # --- Save predictions to CSV ---
    output_csv_path = os.path.join(output_dir,
                                   f"{dataset_name}_predictions_{current_time}_{batch_size}_{learning_rate}.csv")
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        fieldnames = ['Actual_Label', 'Predicted_Label']
        dict_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        dict_writer.writeheader()
        dict_writer.writerows(predictions_data)

    print(f"{dataset_name.capitalize()} predictions saved to {output_csv_path}")

    # --- Plot Results ---
    if predictions_data:
        actual_labels_all = [d['Actual_Label'] for d in predictions_data]
        predicted_labels_all = [d['Predicted_Label'] for d in predictions_data]

        plt.figure(figsize=(8, 8))
        plt.scatter(actual_labels_all, predicted_labels_all, alpha=0.6, s=10)
        plt.plot([min(actual_labels_all), max(actual_labels_all)],
                 [min(actual_labels_all), max(actual_labels_all)],
                 '--r', label='Ideal Prediction (y=x)')
        plt.xlabel("Actual Label")
        plt.ylabel("Predicted Label")
        plt.title(f"{dataset_name.capitalize()} Set: Actual vs. Predicted Labels")
        plt.legend()
        plot_path = os.path.join(output_dir,
                                 f"{dataset_name}_predictions_plot_{current_time}_{batch_size}_{learning_rate}.png")
        plt.savefig(plot_path)
        print(f"{dataset_name.capitalize()} predictions plot saved to {plot_path}")
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

        mixed_image_np = iio.imread(mixed_channel_path).astype(np.float32)
        source_image_np = iio.imread(pure_source_path).astype(np.float32)

        if self.transform:
            mixed_tensor, source_tensor, label_tensor = self.transform(mixed_image_np, source_image_np, scalar_label)
        else:
            mixed_tensor = torch.from_numpy(mixed_image_np).unsqueeze(0)
            source_tensor = torch.from_numpy(source_image_np).unsqueeze(0)
            label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

        input_tensor = torch.cat([mixed_tensor, source_tensor], dim=0)
        return input_tensor, label_tensor


class SplitCrosstalkDataset(Dataset):
    def __init__(self, mixed_channel_dir, pure_source_dir, transform, split_samples):
        self.mixed_channel_dir = mixed_channel_dir
        self.pure_source_dir = pure_source_dir
        self.transform = transform
        self.split_samples = split_samples  # This is already the filtered list of sample dicts

        if not self.split_samples:
            raise ValueError("SplitCrosstalkDataset received no samples.")
        print(f"SplitCrosstalkDataset created with {len(self.split_samples)} samples.")

    def __len__(self):
        return len(self.split_samples)

    def __getitem__(self, idx_in_split):
        sample_info = self.split_samples[idx_in_split]

        mixed_channel_path = os.path.join(self.mixed_channel_dir, sample_info['mixed_channel_file'])
        pure_source_path = os.path.join(self.pure_source_dir, sample_info['pure_source_file'])
        scalar_label = sample_info['scalar_label']

        mixed_image_np = iio.imread(mixed_channel_path).astype(np.float32)
        source_image_np = iio.imread(pure_source_path).astype(np.float32)

        mixed_tensor, source_tensor, label_tensor = self.transform(mixed_image_np, source_image_np, scalar_label)
        input_tensor = torch.cat([mixed_tensor, source_tensor], dim=0)
        return input_tensor, label_tensor


# --- End of Custom Dataset Classes ---

def normalize_image(img):
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:  # Avoid division by zero
        return (img - img_min) / (img_max - img_min)
    else:
        return img


def train_transforms_fn(mixed_np, source_np, scalar_label):
    mixed_np = normalize_image(mixed_np)
    source_np = normalize_image(source_np)
    mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
    source_tensor = torch.from_numpy(source_np).unsqueeze(0)
    label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

    # Horizontal Flip
    if torch.rand(1) < 0.5:
        mixed_tensor = TF.hflip(mixed_tensor)
        source_tensor = TF.hflip(source_tensor)

    # Vertical Flip
    if torch.rand(1) < 0.5:
        mixed_tensor = TF.vflip(mixed_tensor)
        source_tensor = TF.vflip(source_tensor)

    # img_h, img_w = mixed_tensor.shape[-2:]
    #
    # # Affine Transform Parameters:
    # degrees = random.uniform(-15, 15)
    # translate_x = random.uniform(-0.1, 0.1) * img_w
    # translate_y = random.uniform(-0.1, 0.1) * img_h
    # translate = [translate_x, translate_y]
    # scale = 1.0
    # shear = [0.0]
    # fill = [0.0]
    #
    # # Apply the generated affine transform to both tensors
    # mixed_tensor = TF.affine(
    #     mixed_tensor,
    #     angle=degrees,
    #     translate=translate,
    #     scale=scale,
    #     shear=shear,
    #     interpolation=TF.InterpolationMode.BILINEAR,
    #     fill=fill
    # )
    # source_tensor = TF.affine(
    #     source_tensor,
    #     angle=degrees,
    #     translate=translate,
    #     scale=scale,
    #     shear=shear,
    #     interpolation=TF.InterpolationMode.BILINEAR,
    #     fill=fill
    # )

    # # 3. Add Gaussian Noise (applied identically to both images)
    # # Adjust mean and std based on your image intensity range (now 0-1)
    # noise_mean = 0.0
    # noise_std = random.uniform(0.01, 0.05)  # Experiment with this range (e.g., 1-5% of max intensity)
    # gaussian_noise = torch.randn(mixed_tensor.shape) * noise_std + noise_mean
    #
    # mixed_tensor = mixed_tensor + gaussian_noise
    # source_tensor = source_tensor + gaussian_noise
    #
    # # Clip values to ensure they remain within [0, 1] after adding noise
    # mixed_tensor = torch.clamp(mixed_tensor, 0.0, 1.0)
    # source_tensor = torch.clamp(source_tensor, 0.0, 1.0)

    # 4. Add Random Erasing (applied identically to both images for consistency)
    # Generate random parameters for erasing once
    # if random.random() < 0.5:  # Probability of applying Random Erasing
    #     # Get random parameters for erasing block
    #     # i, j: top-left corner coordinates of the erased block
    #     # h, w: height and width of the erased block
    #     # v: value to fill the erased block with
    #     area_ratio = random.uniform(0.02, 0.1)  # Erase 2% to 10% of the image area
    #     aspect_ratio = random.uniform(0.3, 3.3)  # Aspect ratio of the erased block
    #
    #     # Calculate h and w from area_ratio and aspect_ratio
    #     h = int(np.sqrt(area_ratio * img_h * img_w / aspect_ratio))
    #     w = int(np.sqrt(area_ratio * img_h * img_w * aspect_ratio))
    #
    #     # Ensure h and w are not zero
    #     h = max(1, h)
    #     w = max(1, w)
    #
    #     i = random.randint(0, img_h - h)
    #     j = random.randint(0, img_w - w)
    #
    #     mixed_tensor = TF.erase(
    #         mixed_tensor, i, j, h, w, v=0.0
    #     )
    #     source_tensor = TF.erase(
    #         source_tensor, i, j, h, w, v=0.0
    #     )

    return mixed_tensor, source_tensor, label_tensor


def val_test_transforms_fn(mixed_np, source_np, scalar_label):
    mixed_np = normalize_image(mixed_np)
    source_np = normalize_image(source_np)
    mixed_tensor = torch.from_numpy(mixed_np).unsqueeze(0)
    source_tensor = torch.from_numpy(source_np).unsqueeze(0)
    label_tensor = torch.tensor(scalar_label, dtype=torch.float32).unsqueeze(0)

    return mixed_tensor, source_tensor, label_tensor


# --- End of Transform Functions ---

# Replace your existing train_model function with this enhanced version

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, output_dir,
                learning_scheduler):
    """Enhanced training function with comprehensive scheduler support"""

    # Choose your scheduler configuration
    scheduler_configs = {
        'aggressive_plateau': {
            'type': 'plateau',
            'params': {
                'factor': 0.3,
                'patience': 3,
                'threshold': 5e-5,
                'min_lr': 1e-8
            },
            'early_stop_patience': 8
        },

        'onecycle': {
            'type': 'onecycle',
            'params': {
                'max_lr': 1e-3,  # Start with 2x your current LR
                'pct_start': 0.3,
                'anneal_strategy': 'cos',
                'div_factor': 25.0,
                'final_div_factor': 1e4,
                'epochs': num_epochs,
                'steps_per_epoch': len(train_loader)
            },
            'early_stop_patience': 20
        },

        'cosine_warmup': {
            'type': 'custom_warmup',
            'params': {
                'warmup_epochs': 5,
                'max_lr': 1e-4,
                'final_lr': 1e-7,
                'total_epochs': num_epochs
            },
            'early_stop_patience': 15
        }
    }

    # Choose which scheduler to use - EXPERIMENT WITH THESE!
    scheduler_config = scheduler_configs[learning_scheduler]  # Try 'onecycle' or 'cosine_warmup'

    train_losses = []
    val_losses = []
    lr_history = []

    # Create scheduler based on type
    if scheduler_config['type'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_config['params']
        )
    elif scheduler_config['type'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, **scheduler_config['params']
        )
    elif scheduler_config['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-8
        )
    elif scheduler_config['type'] == 'custom_warmup':
        params = scheduler_config['params']
        warmup_epochs = params['warmup_epochs']
        max_lr = params['max_lr']
        final_lr = params['final_lr']
        total_epochs = params['total_epochs']

        def lr_lambda(epoch):
            # Linear warmup from ~0 to max_lr, then cosine decay to final_lr
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return (final_lr + (max_lr - final_lr) * (1 + np.cos(np.pi * progress)) / 2) / max_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stop_patience = scheduler_config.get('early_stop_patience', 15)

    # Create the timestamped log filename with scheduler info
    timestamped_log_file = os.path.join(output_dir,
                                        f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{batch_size}_{learning_rate}_{scheduler_config['type']}.csv")

    # Prepare the CSV log file
    with open(timestamped_log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Learning Rate', learning_rate])
        writer.writerow(['Batch Size', batch_size])
        writer.writerow(['Scheduler Type', scheduler_config['type']])
        writer.writerow(['Scheduler Params', str(scheduler_config['params'])])
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])  # write header

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)

            for batch_idx, (inputs, targets) in enumerate(
                    tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                # Step OneCycleLR after each batch
                if scheduler_config['type'] == 'onecycle':
                    scheduler.step()

            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            # Step scheduler (except OneCycleLR which steps per batch)
            if scheduler_config['type'] == 'plateau':
                scheduler.step(val_loss)
            elif scheduler_config['type'] in ['cosine', 'custom_warmup']:
                scheduler.step()
            # OneCycleLR already stepped in training loop

            # Early stopping and best model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save best model
                best_model_path = os.path.join(output_dir, f"best_model_{scheduler_config['type']}.pth")
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_without_improvement += 1

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

            # Log to CSV
            writer.writerow([epoch + 1, train_loss, val_loss, current_lr])

            # Early stopping
            if epochs_without_improvement >= early_stop_patience:
                print(
                    f"Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stop_patience} epochs)")
                break

    # Plot learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(lr_history)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate Schedule ({scheduler_config["type"]})')
    plt.yscale('log')
    plt.grid(True)
    lr_plot_path = os.path.join(output_dir, f"lr_schedule_{scheduler_config['type']}.png")
    plt.savefig(lr_plot_path)
    plt.close()
    print(f"Learning rate schedule plot saved to {lr_plot_path}")

    return train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training with various parameters.")

    parser.add_argument("-m", "--mixed_channel_data_dir", type=str,
                        default="./Training_Data/Mixed",
                        help="Directory for mixed channel data")
    parser.add_argument("-s", "--pure_source_data_dir", type=str,
                        default="./Training_Data/Source",
                        help="Directory for pure source data")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("-n", "--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("-t", "--train_ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("-v", "--val_ratio", type=float, default=0.15, help="Validation data ratio")
    parser.add_argument("-j", "--cpu_jobs", type=int, default=1, help="Number of CPUs to use")
    parser.add_argument("-o", "--model_options", type=str, default='single', help="Use single- or double-branch model",
                        choices=['single', 'double'])
    parser.add_argument("-r", "--learning_scheduler", type=str, default='aggressive_plateau',
                        help="Use aggressive_plateau, onecycle or cosine_warmup learning scheduler",
                        choices=['aggressive_plateau', 'onecycle', 'cosine_warmup'])

    args = parser.parse_args()

    mixed_channel_data_dir = args.mixed_channel_data_dir
    pure_source_data_dir = args.pure_source_data_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    ncpus = args.cpu_jobs
    model_selection = args.model_options
    learning_scheduler = args.learning_scheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not (abs(train_ratio + val_ratio) < 1.0):
        print("Warning: Sum of TRAIN_RATIO, VAL_RATIO, TEST_RATIO does not equal 1.0.")

    model = build_model(model_selection)
    print(f'Using {ncpus} cpu workers.')

    # --- Create a unique output directory for this run ---
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Include batch size and learning rate in the folder name for easy identification
    output_dir_name = f"training_run_{current_time_str}_B{batch_size}_LR{learning_rate}"
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
    total_samples = len(all_samples)
    torch.manual_seed(43)
    shuffled_indices = torch.randperm(total_samples).tolist()

    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    train_samples = [all_samples[i] for i in shuffled_indices[0:train_size]]
    val_samples = [all_samples[i] for i in shuffled_indices[train_size: train_size + val_size]]
    test_samples = [all_samples[i] for i in shuffled_indices[train_size + val_size:]]

    print(f"Split sizes: Train = {len(train_samples)}, Validation = {len(val_samples)}, Test = {len(test_samples)}")

    train_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir,
        transform=train_transforms_fn,
        split_samples=train_samples
    )

    val_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir,
        transform=val_test_transforms_fn,
        split_samples=val_samples
    )

    test_dataset_final = SplitCrosstalkDataset(
        mixed_channel_data_dir, pure_source_data_dir,
        transform=val_test_transforms_fn,
        split_samples=test_samples
    )

    train_dataloader = DataLoader(
        train_dataset_final,
        batch_size=batch_size,
        shuffle=True,
        num_workers=ncpus,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=ncpus,
        pin_memory=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset_final,
        batch_size=batch_size,
        shuffle=False,
        num_workers=ncpus,
        pin_memory=True,
        drop_last=True
    )

    print("Dataloaders created for training, validation, and testing.")

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print("\nStarting training with validation...")
    train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs,
                                           device, output_dir_name, learning_scheduler)

    print("Training finished!")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path = os.path.join(output_dir_name,
                                   f"crosstalk_regression_model_trained_{current_time}_{batch_size}_{learning_rate}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model weights saved to {model_save_path}")

    # Export a self-contained TorchScript artifact for downstream consumers
    script_path = os.path.join(output_dir_name,
                               f"crosstalk_regression_model_trained_{current_time}_{batch_size}_{learning_rate}.pt")
    save_torchscript(model, script_path)
    print(f"TorchScript model saved to {script_path}")

    # --- Plot Training and Validation Losses ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(train_losses) + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(bottom=0, top=0.02)
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir_name,
                                  f"training_validation_loss_{current_time}_{batch_size}_{learning_rate}.png")
    plt.savefig(loss_plot_path)
    print(f"Training and validation loss plot saved to {loss_plot_path}")
    plt.close()  # Close the plot to free memory

    print("\n--- Evaluating Model ---")
    loaded_model = load_model(model_save_path, model_selection, device)

    evaluate_and_save(loaded_model, test_dataloader, 'test', output_dir_name)
    evaluate_and_save(loaded_model, train_dataloader, 'train', output_dir_name)
    evaluate_and_save(loaded_model, val_dataloader, 'val', output_dir_name)
