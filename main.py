import argparse
import torch.utils
import torch.utils.data
import wandb  
from helping_functions import Dataset, CNN_model
import numpy as np
import torch
import torchvision

def get_config_value(config, args, key, default=None):
    return getattr(config, key, getattr(args, key, default))

def main(args):
    # Initialize wandb with the provided entity and project
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    
    dataset_train_val = Dataset(data_dir=args.dataset_train, input_size=args.input_size, data_augmentation=get_config_value(wandb.config, args, 'data_augmentation'))
    train_subset, val_subset = dataset_train_val.stratified_split(val_ratio=0.2)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)
    

    dataset_test = Dataset(data_dir=args.dataset_test, input_size=args.input_size, data_augmentation=get_config_value(wandb.config, args, 'data_augmentation'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training script that return model weights"
    )
    parser.add_argument(
        '-we',
        '--wandb_entity',
        type=str,
        default='me21b138-indian-institute-of-technology-madras',
        help='Wandb Entity used to track experiments in the Weights & Biases dashboard'
    )
    parser.add_argument(
        '-wp',
        '--wandb_project',
        type=str,
        default='AS2',
        help='Project name used to track experiments in Weights & Biases dashboard'
    )
    parser.add_argument(
        '-d',
        '--dataset_train',
        type=str,
        default="C:/Users/shrey/Desktop/ACAD/DL/nature_12K/inaturalist_12K/train",
        help='give the train directory location of your dataset'
    )
    parser.add_argument(
        '-d_test',
        '--dataset_test',
        type=str,
        default="C:/Users/shrey/Desktop/ACAD/DL/nature_12K/inaturalist_12K/val",
        help='give the test directory location of your dataset'
    )
    parser.add_argument(
        '-da',
        '--data_augmentation',
        type=bool,
        default=True,
        help='Applt data augmentation to the dataset'
    )
    parser.add_argument(
        '-is',
        '--input_size',
        type=int,
        default=224,
        help='Input size of the images (height and weight)'
    )
    
    # Use parse_known_args to ignore extra arguments injected by wandb sweep agent
    args = parser.parse_args()
    main(args)
