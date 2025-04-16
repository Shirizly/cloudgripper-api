import os, ast
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_dirs, transform=None):
        self.image_dirs = image_dirs
        self.transform = transform
        self.image_paths = []
        self.metadata = {}
        for image_dir in image_dirs:
            log_file = os.path.join(image_dir, "image_log.txt") 
            # Load metadata from log_file.txt
            with open(log_file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)  # Separate filename and dictionary
                    if len(parts) == 2:
                        image_name, metadata_str = parts
                        data = ast.literal_eval(metadata_str)  # Convert string to dict
                        if isinstance(data,tuple):
                            data = data[0]
                        full_image_path = os.path.join(image_dir, os.path.join("top_images",image_name))
                        self.image_paths.append(full_image_path)
                        robot_config = list(data.values())[:5]
                        robot_config[3] = robot_config[3]/180 #normalize angle
                        self.metadata[full_image_path] = robot_config
        
        # Shuffle data initially
        self.image_paths = sorted(self.image_paths)  # Sorting ensures consistent ordering
        self.metadata = {k: self.metadata[k] for k in self.image_paths}  # Maintain order


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path) #.convert("RGB")  # Convert to RGB if needed        
        
        if self.transform:
            image = self.transform(image)
        
        # Convert metadata dictionary to tensor (modify based on actual data structure)
        metadata_values = torch.tensor(self.metadata[image_path], dtype=torch.float32)

        return image, metadata_values
    
    def data_split(self, train_ratio, val_ratio):
        # Compute sizes
        dataset_size = len(self)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size  # Ensure all samples are used
        # Perform random splitting
        train_set, val_set, test_set = random_split(self, [train_size, val_size, test_size])
        return train_set, val_set, test_set