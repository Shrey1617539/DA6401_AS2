import torch
import torchvision 
import wandb
from tqdm import tqdm

class CNN_model(torch.nn.Module):
    """
    A Convolutional Neural Network (CNN) model for image classification.
    This model consists of several convolutional blocks followed by a fully connected layer.
    The architecture is flexible and can be customized with different parameters.
    """
    def __init__(
            self, input_size, output_size, 
            num_filters, kernel_sizes, pool_kernels, 
            paddings, conv_strides, dense_layer, 
            activation_fn, use_softmax=False, 
            batch_norm=True, dropout_rate=0.0
        ):
        
        super(CNN_model, self).__init__()

        self.conv_blocks = torch.nn.ModuleList()
        self.dropout_rate = dropout_rate

        # This function is used to identify the activation function based on the name provided
        def identify_activation(function_name):
            activation = {
                'relu': torch.nn.ReLU(),
                'sigmoid': torch.nn.Sigmoid(),
                'tanh': torch.nn.Tanh(),
                'selu': torch.nn.SELU(),
                'gelu': torch.nn.GELU(),
                'mish': torch.nn.Mish(),
                'leakyrelu': torch.nn.LeakyReLU()
            }
            return activation.get(function_name.lower(), torch.nn.ReLU())
        
        # This function is used to convert the input into a tuple if it is not already a tuple
        def make_tuple(a):
            if isinstance(a, tuple):
                return a
            else:
                return (a,a)

        for i in range(len(num_filters)):
            # Create a convolutional block with the specified parameters
            block = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=3 if i == 0 else num_filters[i-1], 
                    out_channels=num_filters[i], 
                    kernel_size=make_tuple(kernel_sizes[i]), 
                    padding=make_tuple(paddings[i]), 
                    stride=make_tuple(conv_strides[i])
                ),
                torch.nn.BatchNorm2d(num_filters[i]) if batch_norm else torch.nn.Identity(),
                identify_activation(activation_fn),
                torch.nn.Dropout2d(p=dropout_rate) if dropout_rate>0 else torch.nn.Identity(),
                torch.nn.MaxPool2d(kernel_size=make_tuple(pool_kernels[i]), stride=make_tuple(pool_kernels[i]))
            )
            self.conv_blocks.append(block)
        
        # Calculate the output size after all convolutional and pooling layers
        h, w = make_tuple(input_size)
        for i in range(len(num_filters)):
            f_h, f_w = make_tuple(kernel_sizes[i])
            s_h, s_w = make_tuple(conv_strides[i])
            p_h, p_w = make_tuple(paddings[i])

            h = ((h - f_h + 2*p_h)//s_h) + 1
            w = ((w - f_w + 2*p_w)//s_w) + 1

            pp_h, pp_w = make_tuple(pool_kernels[i])
            ps_h, ps_w = make_tuple(pool_kernels[i])

            h = ((h - pp_h)//ps_h) + 1
            w = ((w - pp_w)//ps_w) + 1
        
        # Create the fully connected layer with the specified parameters
        self.dense_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=num_filters[-1]*h*w, out_features=dense_layer, bias=True),
            torch.nn.BatchNorm1d(dense_layer) if batch_norm else torch.nn.Identity(),
            identify_activation(activation_fn),
            torch.nn.Dropout(p=dropout_rate) if dropout_rate>0 else torch.nn.Identity(),
            torch.nn.Linear(in_features=dense_layer, out_features=output_size),
        )
        # Set the activation function for the output layer
        self.use_softmax = use_softmax
        if use_softmax:
            self.softmax_layer = torch.nn.Softmax(dim=1)
    
    # Forward pass through the model
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        
        x = self.dense_layer(x)

        if self.use_softmax:
            x = self.softmax_layer(x)
        
        return x

class Dataset(torch.utils.data.Dataset):
    """
    This class is used to load the dataset and apply data augmentation if required.
    It uses the ImageFolder class from torchvision to load the images and their labels.
    """
    def __init__(self, data_dir, input_size=(224,224), data_augmentation=False):

        super(Dataset, self).__init__()
        # This function is used to convert the input into a tuple if it is not already a tuple
        def make_tuple(a):
            if isinstance(a, tuple):
                return a
            else:
                return (a,a)
            
        # This function is used to apply data augmentation if required
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(make_tuple(input_size)),
            torchvision.transforms.RandomHorizontalFlip() if data_augmentation else torchvision.transforms.Lambda(lambda x: x),
            torchvision.transforms.RandomRotation(20) if data_augmentation else torchvision.transforms.Lambda(lambda x: x),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) if data_augmentation else torchvision.transforms.Lambda(lambda x: x),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.4716, 0.4602, 0.3899], std=[0.2382, 0.2273, 0.2361])
        ])
        self.data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    def get_classes(self):
        return self.data.classes

    def stratified_split(self, val_ratio=0.2, seed=42):
        # Create a dictionary to map each class to its sample indices
        class_to_indices = {}
        for idx, (_, label) in enumerate(self.data.samples):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        
        # Split indices for each class
        train_indices = []
        val_indices = []
        
        generator = torch.Generator().manual_seed(seed)
        
        for cls in class_to_indices:
            indices = class_to_indices[cls]
            n = len(indices)
            n_val = int(n * val_ratio)
            
            # Shuffle indices for this class
            permuted_indices = torch.randperm(n, generator=generator).tolist()
            
            # Assign training and validation indices
            val_indices.extend([indices[i] for i in permuted_indices[:n_val]])
            train_indices.extend([indices[i] for i in permuted_indices[n_val:]])
        
        # Create Subsets
        train_subset = torch.utils.data.Subset(self.data, train_indices)
        val_subset = torch.utils.data.Subset(self.data, val_indices)
        
        return train_subset, val_subset
    
def test_CNN_model(model, test_loader, device, test_logging = True):
    """
    This function is used to test the model on the test set.
    It calculates the accuracy and loss of the model on the test set.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient calculation for testing
    with torch.no_grad():
        # Iterate through the test data
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Accumulate loss and correct predictions
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total

    # Log test results to wandb if required
    if test_logging:
        wandb.log({
            "test_accuracy": accuracy, 
            "test_loss": test_loss / len(test_loader) 
        })
    
    return accuracy, test_loss / len(test_loader)

def train_CNN_model(model, train_loader, val_loader, learning_rate, epochs, device, patience = 3):
    """
    This function is used to train the model with the given parameters.
    It uses the Adam optimizer and CrossEntropy loss function.
    It also implements early stopping based on validation loss.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate through the training data
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            scores = model(images)
            loss = criterion(scores, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Get the predicted class
            _, predicted = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total

        # Validate the model on the validation set
        val_accuracy, val_loss = test_CNN_model(model, val_loader, device, test_logging=False)
        
        # Log training and validation metrics to wandb
        wandb.log({
            "train_loss": running_loss / len(train_loader),
            "train_accuracy": train_accuracy,
            "validation_accuracy":val_accuracy,
            "validation_loss": val_loss,
            "epoch": epoch+1
        })

        # Early stopping based on validation loss
        if val_loss < best_val_loss - 1e-3:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Load the best model state if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model