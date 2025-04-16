import argparse
import torch.utils
import torch.utils.data
import wandb  
import helping_functions
from helping_functions import Dataset, CNN_model
import torch

# this function is used to get the config value from wandb or command line arguments
def get_config_value(config, args, key, default=None):
    return getattr(config, key, getattr(args, key, default))

def main(args):
    # Initialize wandb with the provided entity and project
    wandb.init(entity=args.wandb_entity, project=args.wandb_project)
    
    # This class is used to load the dataset and apply data augmentation if required
    dataset_train_val = Dataset(
        data_dir=args.dataset_train, 
        input_size=args.input_size, 
        data_augmentation=get_config_value(wandb.config, args, 'data_augmentation')
    )
    
    # This function is used to split the dataset into train and validation set in a stratified manner
    train_subset, val_subset = dataset_train_val.stratified_split(val_ratio=0.2)

    # This function is used to create the data loaders for train and validation set
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=get_config_value(wandb.config, args, "batch_size"), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=get_config_value(wandb.config, args, "batch_size"), shuffle=False)

    # The name is created using the parameters used in the model
    run_name = f"lr_{get_config_value(wandb.config, args, 'learning_rate')}_1filters_{get_config_value(wandb.config, args, 'number_of_filters')[0]}_5filters_{get_config_value(wandb.config, args, 'number_of_filters')[-1]}_dense_{get_config_value(wandb.config, args, 'dense_layer')}_activation_{get_config_value(wandb.config, args, 'activation_fn')}"
    wandb.run.name=run_name
    
    # This class is used to create the CNN model with the given parameters
    model = CNN_model(
        input_size=args.input_size,
        output_size = len(dataset_train_val.get_classes()),
        num_filters = get_config_value(wandb.config, args, 'number_of_filters'),
        kernel_sizes = get_config_value(wandb.config, args, 'kernel_sizes'),
        pool_kernels = get_config_value(wandb.config, args, 'pool_kernels'),
        paddings = get_config_value(wandb.config, args, 'paddings'),
        conv_strides = get_config_value(wandb.config, args, 'conv_strides'),
        dense_layer = get_config_value(wandb.config, args, 'dense_layer'),
        activation_fn = get_config_value(wandb.config, args, 'activation_fn'),
        use_softmax = get_config_value(wandb.config, args, 'use_softmax'),
        batch_norm = get_config_value(wandb.config, args, 'batch_norm'),
        dropout_rate= get_config_value(wandb.config, args, 'dropout_rate')
    )

    # Check if GPU is available and use it, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # This function is used to train the model with the given parameters
    model = helping_functions.train_CNN_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate = get_config_value(wandb.config, args, 'learning_rate'),
        epochs = get_config_value(wandb.config, args, 'epochs'),
        device=device,
        patience=get_config_value(wandb.config, args, 'patience')
    )
    
    # This class is used to load the dataset and apply data augmentation if required
    dataset_test = Dataset(data_dir=args.dataset_test, input_size=args.input_size, data_augmentation=get_config_value(wandb.config, args, 'data_augmentation'))

    # This function is used to create the data loaders for train and validation set
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=get_config_value(wandb.config, args, 'batch_size'), shuffle=False)
    
    # This function is used to test the model with the given parameters
    test_accuracy, test_loss = helping_functions.test_CNN_model(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
if __name__ == "__main__":

    # Argument parser to take command line arguments
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
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for training and validation'
    )
    parser.add_argument(
        '-nf',
        '--number_of_filters',
        nargs="*",
        type=int,
        default=[16, 32, 64, 128, 256],
        help='Number of filters for each convolutional layer'
    )
    parser.add_argument(
        '-ks',
        '--kernel_sizes',
        nargs="*",
        type=int,
        default=[3, 3, 3, 3, 3],
        help='Kernel sizes for each convolutional layer'
    )
    parser.add_argument(
        '-pk',
        '--pool_kernels',
        nargs="*",
        type=int,
        default=[2, 2, 2, 2, 2],
        help='Pooling kernels for each convolutional layer'
    )
    parser.add_argument(
        '-pd',
        '--paddings',
        nargs="*",
        type=int,
        default=[1, 1, 1, 1, 1],
        help='Paddings for each convolutional layer'
    )
    parser.add_argument(
        '-cs',
        '--conv_strides',
        nargs="*",
        type=int,
        default=[1, 1, 1, 1, 1],
        help='Convolutional strides for each convolutional layer'
    )
    parser.add_argument(
        '-dl',
        '--dense_layer',
        type=int,
        default=512,
        help='Number of neurons in the dense layer'
    )
    parser.add_argument(
        '-af',
        '--activation_fn',
        type=str,
        default='relu',
        help='Activation function to be used in the model'
    )
    parser.add_argument(
        '-us',
        '--use_softmax',
        type=bool,
        default=0,
        help='Use of softmax layer in the model'
    )
    parser.add_argument(
        '-bn',
        '--batch_norm',
        type=bool,
        default=1,
        help='Use of batch normalization in the model'
    )
    parser.add_argument(
        '-ep',
        '--epochs',
        type=int,
        default=10,
        help='number of epochs for training'
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=0.01,
        help='Learning rate for the optimizer'
    )
    parser.add_argument(
        '-pa',
        '--patience',
        type=int,
        default=3,
        help='Patience for early stopping'
    )
    parser.add_argument(
        '-dr',
        '--dropout_rate',
        type=float,
        default=0.3,
        help='Dropout rate for model'
    )

    args = parser.parse_args()
    main(args)