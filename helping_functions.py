import torch
import torchvision 
from tqdm import tqdm

class CNN_model(torch.nn.Module):
    def __init__(self, input_size, output_size, num_filters, kernel_sizes, pool_kernels, paddings, conv_strides, dense_layer, activation_fn, use_softmax=False, batch_norm=True):
        super(CNN_model, self).__init__()

        self.conv_blocks = torch.nn.ModuleList()

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
        
        def make_tuple(a):
            if isinstance(a, tuple):
                return a
            else:
                return (a,a)

        for i in range(len(num_filters)):
        
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
                torch.nn.MaxPool2d(kernel_size=make_tuple(pool_kernels[i]), stride=make_tuple(pool_kernels[i]))
            )
            self.conv_blocks.append(block)
        
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
        
        self.dense_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=num_filters[-1]*h*w, out_features=dense_layer),
            torch.nn.BatchNorm1d(dense_layer) if batch_norm else torch.nn.Identity(),
            identify_activation(activation_fn),
            torch.nn.Linear(in_features=dense_layer, out_features=output_size),
        )
        self.use_softmax = use_softmax
        if use_softmax:
            self.softmax_layer = torch.nn.Softmax(dim=1)
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        
        x = self.dense_layer(x)

        if self.use_softmax:
            x = self.softmax_layer(x)
        
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_size=(224,224), data_augmentation=False):
        super(Dataset, self).__init__()
        def make_tuple(a):
            if isinstance(a, tuple):
                return a
            else:
                return (a,a)
            
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
    
def train_CNN_model(model, train_loader, learning_rate, epochs, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            scores = model(images)
            loss = criterion(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Update progress bar with current batch loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # for batch_idx, (images, labels) in enumerate(train_loader):
        #     images = images.to(device)
        #     labels = labels.to(device)

        #     scores = model(images)
        #     loss = criterion(scores, labels)

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     running_loss += loss.item()

    print(f"Loss: {running_loss/len(train_loader):.4f}")
    return model

def test_CNN_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")
    return 100 * correct / total