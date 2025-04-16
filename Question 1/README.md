# DA6401 AS2 Q1

This repository contains tools for training convolutional neural network models with extensive configuration options and automated hyperparameter optimization using Weights & Biases (wandb).

## Links

Wandb report link: https://wandb.ai/me21b138-indian-institute-of-technology-madras/AS2/reports/ME21B138-s-DA6401-Assignment-2--VmlldzoxMjEwMDMxMg

Github link: https://github.com/Shrey1617539/DA6401_AS2/tree/main/Question%201

## Overview

The project is designed to simplify the workflow required for neural network experimentation. It includes scripts for model training (`train.py`), hyperparameter sweeps (`sweep.py` and `sweep.yaml`), helper functions for data loading and model definition (`helping_functions.py`), as well as a Jupyter Notebook (`Q1.ipynb`) for exploratory analysis. Each component works together to allow users to easily configure a custom CNN and perform model evaluation and optimization using wandb.

## Features

- **Customizable Architecture:** Easily modify the network by specifying the number of filters, kernel sizes, dense layer size, and activation functions.
- **Hyperparameter Optimization:** Automated sweeps using wandb allow for optimization of parameters such as learning rate, dropout rate, and more.
- **Data Augmentation & Preprocessing:** Integrated support for image transformations (resizing, horizontal flips, rotations, and color adjustments) enhances model robustness.
- **Flexible Training & Evaluation:** Options to train, validate, and test models using single or multiple datasets.
- **Experiment Tracking:** Seamlessly log metrics and visualize training progress with wandb.

## File Structure

Below is the repository’s file structure to help you navigate the project:

```
Question 1/
├── README.md # Project documentation and this file
├── train.py # Main training script
├── sweep.py # Script for running hyperparameter sweeps
├── sweep.yaml # Configuration file for wandb sweeps
├── helping_functions.py # Helper functions (data loading, CNN model definition, etc.)
└── Q1.ipynb # Jupyter Notebook for exploration and testing
```

## Requirements

- Python
- PyTorch & Torchvision
- Weights & Biases (wandb)
- Argparse
- PyYAML
- TQDM
- NumPy

## Installation

1. **Clone the repository:**

```
git clone https://github.com/Shrey1617539/DA6401_AS2
```

2. **Navigate to the project directory:**

```
cd DA6401_AS2/Question 1
```

## Usage

### Training a Model

- **Run with default hyperparameters:**

```
python train.py
```

- **Run with custom settings:**

```
python train.py --wandb_entity your_entity --wandb_project your_project --dataset_train /path/to/train --dataset_test /path/to/test --epochs 20 --batch_size 64 --learning_rate 0.001
```

This will launch a training session with specified parameters and log metrics to wandb.

### Running Hyperparameter Sweeps

- **Start a sweep using the configuration in `sweep.yaml`:**

```
python sweep.py --wandb_entity your_entity --wandb_project your_project --count 50
```

This command initializes a sweep agent that executes multiple runs by adjusting hyperparameters as defined in the YAML file.

### Running the Notebook

Open the Q1.ipynb notebook with Jupyter. The notebook provides an alternative interface to run experiments, including configuring and launching hyperparameter sweeps via wandb.

## Command Line Arguments for Files

### `train.py` Arguments

| **Short Flag** | **Long Flag**           | **Type** | **Default Value**         | **Description**                            |
| -------------- | ----------------------- | -------- | ------------------------- | ------------------------------------------ |
| `-we`          | `--wandb_entity`        | str      | None                      | Wandb entity used for experiment tracking  |
| `-wp`          | `--wandb_project`       | str      | None                      | Project name for wandb dashboard           |
| `-d`           | `--dataset_train`       | str      | `"path/to/training/data"` | Directory of training data                 |
| `-d_test`      | `--dataset_test`        | str      | `"path/to/test/data"`     | Directory of test data                     |
| `-da`          | `--data_augmentation`   | bool     | True                      | Enable data augmentation                   |
| `-is`          | `--input_size`          | int      | 224                       | Input image size (height and width)        |
| `-bs`          | `--batch_size`          | int      | 64                        | Batch size for training and validation     |
| `-nf`          | `--number_of_filters`   | list     | [64, 128, 256, 512, 1024] | List of filters per convolutional layer    |
| `-ks`          | `--kernel_sizes`        | list     | [3, 3, 3, 5, 5]           | Kernel sizes for convolutional layers      |
| `-pk`          | `--pool_kernels`        | list     | [3, 3, 2, 2, 2]           | Pooling kernel sizes                       |
| `-pd`          | `--paddings`            | list     | [1, 1, 1, 1, 1]           | Padding for convolutional layers           |
| `-cs`          | `--conv_strides`        | list     | [1, 1, 1, 1, 1]           | Convolution strides                        |
| `-dl`          | `--dense_layer`         | int      | 256                       | Size of the fully connected layer          |
| `-af`          | `--activation_fn`       | str      | "gelu"                    | Activation function (e.g., "relu", "tanh") |
| `-us`          | `--use_softmax`         | bool     | 1                         | Flag to enable softmax on the output layer |
| `-bn`          | `--batch_norm`          | bool     | 1                         | Enable batch normalization                 |
| `-ep`          | `--epochs`              | int      | 10                        | Number of training epochs                  |
| `-lr`          | `--learning_rate`       | float    | 0.0001329901471283676     | Learning rate for optimizer                |
| `-pa`          | `--patience`            | int      | 5                         | Patience for early stopping                |
| `-dr`          | `--dropout_rate`        | float    | 0.025831017228233916      | Dropout rate for regularization            |
| `-etr`         | `--evaluate_training`   | bool     | False                     | Evaluate on training data                  |
| `-eva`         | `--evaluate_validation` | bool     | False                     | Evaluate on validation data                |
| `-ete`         | `--evaluate_test`       | bool     | False                     | Evaluate on test data                      |

### `sweep.py` Arguments

| **Short Flag** | **Long Flag**     | **Type** | **Default Value** | **Description**                            |
| -------------- | ----------------- | -------- | ----------------- | ------------------------------------------ |
| `-we`          | `--wandb_entity`  | str      | None              | Wandb entity for experiment tracking       |
| `-wp`          | `--wandb_project` | str      | None              | Project name for wandb dashboard           |
| `-c`           | `--count`         | int      | 100               | Maximum number of runs for the sweep agent |
