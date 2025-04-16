import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from torchvision import models
import os,sys
import ast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from PIL import Image
from ImageDataset import ImageDataset as ID
import cv2
import matplotlib
import json
import pickle
from TrainEncoder import encoder_training

phase = "test"
# phase = "train"
# phase = "fine-tuning"

# training parameters
batch_size = 32
lr = 1E-4 # learning rate
num_epochs = 300
# Define split sizes
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1


class CNNFeatureExtractor(nn.Module):
    """EfficientNet-based CNN for feature extraction."""
    def __init__(self, output_dim=256):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).features
        self.conv = nn.Conv2d(1280, output_dim, kernel_size=7,stride=5)  # Reduce to manageable dim
    
    def forward(self, x):
        x = self.backbone(x)  # Shape: [B, 1280, H/32, W/32]
        x = self.conv(x)  # Shape: [B, output_dim, H/32, W/32]
        return x


class TransformerEncoder(nn.Module):
    """Simple Transformer Encoder for global feature aggregation."""
    def __init__(self, feature_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(feature_dim, feature_dim)  # Project features
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=512,batch_first=True),
            num_layers=num_layers
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(2, 0, 1)  # Reshape to [seq_len, batch, features]
        x = self.embedding(x)
        x = self.transformer(x)  # Global attention over the image tokens
        x = x.permute(1, 2, 0).view(B, C, H, W)  # Reshape back
        return x

class SegmentationHead(nn.Module):
    """U-Net style segmentation head."""
    def __init__(self, input_dim=256, output_channels=1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(input_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(64, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x):
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x = torch.sigmoid(self.up3(x))  # Output: Segmentation mask
        return x

class PositionRegressionHead(nn.Module):
    """MLP to estimate robot position in the image."""
    def __init__(self, input_dim=256, output_dim=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Output: 3D position (x, y, z)
        )

    
    def forward(self, x):
        x = self.pool(x).view(x.shape[0], -1)  # Global pooling
        x = self.fc(x)  # Predict (x, y)
        return x

class HybridCNNTransformer(nn.Module):
    """Full model combining CNN, Transformer, and Task-Specific Heads."""
    def __init__(self, feature_dim=256):
        super().__init__()
        self.cnn = CNNFeatureExtractor(output_dim=feature_dim)
        self.transformer = TransformerEncoder(feature_dim=feature_dim)
        # self.segmentation_head = SegmentationHead(input_dim=feature_dim, output_channels=1)
        self.regression_head = PositionRegressionHead(input_dim=feature_dim, output_dim=5)

    def forward(self, x):
        features = self.cnn(x)  # CNN feature extraction
        transformed_features = self.transformer(features)  # Transformer for global context
        # segmentation_output = self.segmentation_head(transformed_features)  # Object mask
        position_output = self.regression_head(transformed_features)  # (x, y) position
        return  position_output #,segmentation_output

print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model
model = HybridCNNTransformer()
model.to(device)
criterion = nn.MSELoss()  # Mean squared error loss
optimizer = optim.Adam(model.parameters(), lr=lr)

# load data
# Define transformations for images
# first option is to just increase contrast:
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert image to tensor
#     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (adjust as needed)
# ])

# second option includes downscaling
transform = transforms.Compose([
    transforms.Resize((640, 360), interpolation=transforms.InterpolationMode.BICUBIC),  # Resize with minimal detail loss
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
recordings_dir = os.path.join(project_root,"recordings")

if phase == "train":
    # generate list of data directories
    sequence_dirs = []
    log_file = []
    if os.path.exists(recordings_dir):
        dataset_dirs = [d for d in os.listdir(recordings_dir) if d.startswith("dataset_0_sequence_")]
        for sequence in dataset_dirs:
            sequence_dirs.append(os.path.join(recordings_dir, sequence))
    # Load dataset
    dataset = ID(sequence_dirs, transform=transform)
    # if no split is needed
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # otherwise
    # Perform random splitting
    train_set, val_set, test_set = dataset.data_split(train_ratio, val_ratio)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # saving loss data
    train_losses = []
    val_losses = []
    model.train()  # Set model to training mode
    scaler = torch.amp.GradScaler('cuda')
    # Iterate through batches
    for epoch in range(num_epochs):
        train_loss = 0.0
        
        for images, metadata in train_loader:
            images, metadata = images.to(device), metadata.to(device)  # Move batch to GPU
            # Display one image from the batch using cv2
            # img = images[0].permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array and change dimensions
            # img = (img * 255).astype('uint8')  # Denormalize and convert to uint8
            # cv2.imshow('Image', img)
            # cv2.waitKey(2000)  # Display the image for 1 ms
            optimizer.zero_grad()  # Reset gradients
            
            # Forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                # Compute loss (mean squared error over the first 5 values of metadata)
                loss = criterion(outputs, metadata)

            # Backpropagation
            # loss.backward()
            # optimizer.step()

            # backpropagate with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        model.eval()  # Set model to evaluation mode    
        val_loss = 0.0
        with torch.no_grad():
            for images, metadata in val_loader:
                images, metadata = images.to(device), metadata.to(device)  # Move batch to GPU
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    # Compute loss (mean squared error over the first 5 values of metadata)
                    loss = criterion(outputs, metadata)
            
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_losses[-1]:.4f}, Validation loss: {val_losses[-1]:.4f}")

        if epoch % 10 == 0:
            # Save model checkpoint
            torch.save(model.state_dict(), f"model_checkpoint_{epoch}.pth")

    print("Training complete.")

    # Save the trained model
    torch.save(model.state_dict(), "model_weights.pth")

    # Save loss history
    with open("training_losses.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    # Save dataset indices for reproducibility
    split_indices = {
        "train": train_set.indices,
        "val": val_set.indices,
        "test": test_set.indices
    }

    with open("dataset_splits.pkl", "wb") as f:
        pickle.dump(split_indices, f)

if phase == "test":
    # Load the model
    model.load_state_dict(torch.load("xyz_model_weights.pth"))
    model.eval()  # Set model to evaluation mode

    # generate list of data directories
    sequence_dirs = []
    log_file = []
    if os.path.exists(recordings_dir):
        dataset_dirs = [d for d in os.listdir(recordings_dir) if d.startswith("dataset_0_sequence_")]
        for sequence in dataset_dirs:
            sequence_dirs.append(os.path.join(recordings_dir, sequence))
    
    # Load dataset
    dataset = ID(sequence_dirs, transform=transform)

    with open("dataset_splits.pkl", "rb") as f:
        split_indices = pickle.load(f)
    test_indices = torch.tensor(split_indices["test"])
    test_set = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    test_loss = 0.0
    with torch.no_grad():
        for images, metadata in test_loader:
            images, metadata = images.to(device), metadata.to(device)  # Move batch to GPU
            outputs = model(images)
            # metadata[:, 3] = outputs[:, 3] # remove the angle data from loss computation
            print((metadata-outputs)/metadata) #normalized error
            loss = criterion(outputs[:,:3], metadata[:,:3])
            test_loss += loss.item()    
    print(f"Test loss: {test_loss / len(test_loader):.4f}")

if phase == "fine-tuning":
    # Load the model
    model.load_state_dict(torch.load("model_weights.pth"))

    # generate list of data directories
    sequence_dirs = []
    log_file = []
    if os.path.exists(recordings_dir):
        dataset_dirs = [d for d in os.listdir(recordings_dir) if d.startswith("dataset_0_sequence_")]
        for sequence in dataset_dirs:
            sequence_dirs.append(os.path.join(recordings_dir, sequence))
    
    # Load dataset
    dataset = ID(sequence_dirs, transform=transform)

    with open("dataset_splits.pkl", "rb") as f:
        split_indices = pickle.load(f)
    train_indices = torch.tensor(split_indices["train"])
    train_set = torch.utils.data.Subset(dataset, train_indices)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    val_indices = torch.tensor(split_indices["val"])
    val_set = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    train_losses, val_losses = encoder_training(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, weight_path="xyz_model_weights.pth")







