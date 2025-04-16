import torch
import torchvision
from torchvision import models
from tqdm import tqdm
import wandb

class CNN_model(torch.nn.Module):
    """
    CNN model using ResNet50 as the base model.
    The last 'trainable_layers' layers are trainable, the rest are frozen.
    """
    def __init__(self, num_classes=10, trainable_layers=1):
        super(CNN_model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 'trainable_layers' layers
        children = list(self.model.children())
     
        if trainable_layers > len(children):
            trainable_layers = len(children)
        
        for layer in children[-trainable_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, num_classes)


    def forward(self, x):
        return self.model(x)

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

def train_CNN_model(model, train_loader, val_loader, learning_rate, epochs=10, device='cuda', patience=3):
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