# CrosstalkPy: A Python Package to Detect CrossTalk in Microscopy Images

<img width="45%" alt="test_Pearsons Correlation_plot_2025-12-16_11-37-05" src="https://github.com/user-attachments/assets/3aeb6b4c-2415-4a15-8c12-c233ca11a0a5" /> <img width="45%" alt="test_Predicted_Label_plot_2025-12-16_11-37-05" src="https://github.com/user-attachments/assets/a18fe6b7-86c4-4c4e-be43-7305e45f8b7f" />

CrosstalkPy is a regression-based deep-learning model designed to detect cross-talk (bleed-through) between different channels in fluorescence microscopy images. Trained on a wide selection of approximately 40,000 images from the [Image Data Resource](https://idr.openmicroscopy.org/), CrosstalkPy is microscope- and sample-agnostic.

We developed CrosstalkPy because standard measures of cross-talk, such as Pearson's Correlation Coefficient, are not reliable (see above).

## Setup

### Step 1: Install a Python Distribution

We recommend using conda as it's relatively straightforward and makes the management of different Python environments simple. You can install conda from [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) (miniconda will suffice).

### Step 2: Set Up Environment

Once conda is installed, open a terminal (Mac) or AnaConda Prompt (Windows) and run the following series of commands:

```
conda create --name crosstalk-detection python=3.13
conda activate crosstalk-detection
python -m pip install -r <path to this repo>/requirements.txt
```
where you need to replace `<path to this repo>` with the location on your file system where you downloaded this repo. You will be presented with a list of packages to be downloaded and installed. The following prompt will appear:
```
Proceed ([y]/n)?
```
Hit Enter and all necessary packages will be downloaded and installed - this may take some time. When complete, you can deactivate the environment you have created with the following command.

```
conda deactivate
```
You have successfully set up your environment!

## Evaluation

To test the pre-trained model (or your own model - see below) on the test data in this repo, and compare to other metrics such as Pearson's Correlation Coefficient, activate the environment you created above and run the following command:

```
python test-cross-talk-model.py [-h] [-m MIXED_CHANNEL_DATA_DIR] [-s PURE_SOURCE_DATA_DIR] [-p MODEL_PATH] [-j CPU_JOBS] [-o {single,double}]
```

A number of options can be specified:

```
  -h, --help            show this help message and exit
  -m, --mixed_channel_data_dir MIXED_CHANNEL_DATA_DIR
                        Directory for mixed channel data
  -s, --pure_source_data_dir PURE_SOURCE_DATA_DIR
                        Directory for pure source data
  -p, --model_path MODEL_PATH
                        Path to trained model. To use the model in this repository, set this to ./PreTrained_Model/crosstalk_regression_model_trained_2025-12-15_18-22-01_256_0.0005.pth
  -j, --cpu_jobs CPU_JOBS
                        Number of CPUs to use
  -o, --model_options {single,double}
                        Use single- or double-branch model
```

## Training

To train a model using default settings on your own data, activate the environment you created above and run the following command:

```
python train_model.py [-h] [-m MIXED_CHANNEL_DATA_DIR] [-s PURE_SOURCE_DATA_DIR] [-b BATCH_SIZE] [-l LEARNING_RATE] [-n NUM_EPOCHS] [-t TRAIN_RATIO] [-v VAL_RATIO] [-j CPU_JOBS] [-o {single,double}] [-r {aggressive_plateau,onecycle,cosine_warmup}]
```

A number of options can be specified to control the training process:

```
  -h, --help            show this help message and exit
  -m, --mixed_channel_data_dir MIXED_CHANNEL_DATA_DIR
                        Directory for mixed channel data
  -s, --pure_source_data_dir PURE_SOURCE_DATA_DIR
                        Directory for pure source data
  -b, --batch_size BATCH_SIZE
                        Batch size for training
  -l, --learning_rate LEARNING_RATE
                        Learning rate for training
  -n, --num_epochs NUM_EPOCHS
                        Number of epochs for training
  -t, --train_ratio TRAIN_RATIO
                        Training data ratio
  -v, --val_ratio VAL_RATIO
                        Validation data ratio
  -j, --cpu_jobs CPU_JOBS
                        Number of CPUs to use
  -o, --model_options {single,double}
                        Use single- or double-branch model
  -r, --learning_scheduler {aggressive_plateau,onecycle,cosine_warmup}
                        Use aggressive_plateau, onecycle or cosine_warmup learning scheduler
```
