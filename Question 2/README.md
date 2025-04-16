# DA6401 AS2 Q2

This project implements a convolutional neural network (CNN) for image classification using transfer learning with ResNet50. The project leverages PyTorch and Torchvision for model building and training, and it integrates Weights & Biases (wandb) for experiment tracking and hyperparameter sweeps.

## Links

Wandb report link: https://wandb.ai/me21b138-indian-institute-of-technology-madras/AS2/reports/ME21B138-s-DA6401-Assignment-2--VmlldzoxMjEwMDMxMg

Github link: https://github.com/Shrey1617539/DA6401_AS2/tree/main/Question%202

## Overview

This project provides a complete pipeline for image classification tasks with the following components:

- CNN Model: A custom CNN built upon the pre-trained ResNet50 model with configurable trainable layers.

- Dataset Handling: Classes and helper functions to load images using torchvision's ImageFolder, apply various data augmentation techniques, and perform a stratified train–validation split.

- Training and Evaluation: Functions to train the model with early stopping based on validation loss along with separate testing routines.

- Experiment Tracking: Integration with wandb for live logging of metrics, assignment of run names based on hyperparameters, and support for hyperparameter sweeps.

## Features

- **Transfer Learning with ResNet50**: Only the last few layers are trainable, enabling efficient fine-tuning.

- **Data Augmentation**: Optional augmentation supporting horizontal flips, rotations, and color jitter.

- **Stratified Data Split**: Ensures that each subset contains a balanced distribution of classes.

- **Early Stopping**: Stops training when no significant improvement in validation loss is observed.

- **Wandb Integration**: Automatically logs training and validation metrics, and supports hyperparameter sweeps.

- **Command-line Flexibility**: Customize key hyperparameters and paths via command-line arguments.

## File Structure

Below is the repository’s file structure to help you navigate the project:

```
Question 2/
├── README.md # Project documentation and this file
├── train.py # Main training script
├── helping_functions.py # Helper functions (data loading, CNN model definition, etc.)
└── Q2.ipynb # Jupyter Notebook for exploration and testing
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
cd DA6401_AS2/Question 2
```

## Usage

### Training a Model

- **Run with default hyperparameters:**

```
python train.py
```

- **Run with custom settings:**

```
python train.py --wandb_entity your_entity --wandb_project your_project --dataset_train /path/to/train --dataset_test /path/to/test --data_augmentation True --input_size 224 --trainable_layers 3 --batch_size 64 --epochs 10 --learning_rate 0.001 --patience 4
```

This will launch a training session with specified parameters and log metrics to wandb.

### Running the Notebook

Open the Q2.ipynb notebook with Jupyter. The notebook provides an alternative interface to run experiments, including configuring and launching hyperparameter sweeps via wandb.

## Command Line Arguments for Files

### `train.py` Arguments

| **Flag (Long)**         | **Type** | **Default**             | **Description**                             |
| ----------------------- | -------- | ----------------------- | ------------------------------------------- |
| `--wandb_entity`        | str      | None                    | Wandb entity name for tracking experiments  |
| `--wandb_project`       | str      | None                    | Wandb project name                          |
| `--dataset_train`       | str      | "path/to/training/data" | Directory location for the training dataset |
| `--dataset_test`        | str      | "path/to/test/data"     | Directory location for the test dataset     |
| `--data_augmentation`   | bool     | True                    | Whether to apply data augmentation          |
| `--input_size`          | int      | 224                     | Image input size (height and width)         |
| `--trainable_layers`    | int      | 3                       | Number of trainable layers in the model     |
| `--batch_size`          | int      | 64                      | Batch size for training and evaluation      |
| `--epochs`              | int      | 10                      | Number of training epochs                   |
| `--learning_rate`       | float    | 0.00006239194756311145  | Learning rate for the optimizer             |
| `--patience`            | int      | 4                       | Patience for early stopping                 |
| `--evaluate_training`   | bool     | False                   | Evaluate the model on the training data     |
| `--evaluate_validation` | bool     | False                   | Evaluate the model on the validation data   |
| `--evaluate_test`       | bool     | False                   | Evaluate the model on the test data         |
