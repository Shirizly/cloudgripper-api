import os, ast
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from ultralytics import YOLO
import numpy as np

# auxiliary functions
def save_colored_masks(masks, save_path, image_size):
    """
    Save multiple segmentation masks as a single color-coded image.

    Args:
        masks (torch.Tensor or np.ndarray): A tensor of shape (num_masks, H, W)
        save_path (str): Path to save the image
        image_size (tuple): (H, W) - size of the original image
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    num_masks, H, W = masks.shape  # Number of masks

    # Resize masks if needed
    if (H, W) != image_size:
        masks_resized = np.zeros((num_masks, *image_size), dtype=np.uint8)
        for i in range(num_masks):
            masks_resized[i] = np.array(Image.fromarray(masks[i]).resize(image_size[::-1], resample=Image.NEAREST))
        masks = masks_resized

    # Generate random colors for each mask
    np.random.seed(42)  # For consistent colors
    colors = np.random.randint(0, 255, size=(num_masks, 3), dtype=np.uint8)  # (num_masks, 3)
    colors = [[((i+1)*60+20)%255,((i)*50+80)%255,((i+2)*71+20)%255] for i in range(num_masks)] #Alon's attempt at more distinct colors

    # Create an RGB image with the same size as the original image
    mask_image = np.zeros((*image_size, 3), dtype=np.uint8)

    # Assign a color to each mask
    for i in range(num_masks):
        mask_image[masks[i] > 0] = colors[i]

    # Convert to PIL image and save
    mask_pil = Image.fromarray(mask_image)
    mask_pil.save(save_path)

    print(f"Saved colored masks to {save_path}")

def save_masks(masks, save_path):
    """
    Save multiple segmentation masks as a single (np) tensor file.

    Args:
        masks (torch.Tensor or np.ndarray): A tensor of shape (num_masks, H, W)
        save_path (str): Path to save the image
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    np.save(save_path, masks)
    print(f"Saved masks to {save_path}")

def filter_top_masks(masks, num_objects=6):
        # Compute area of each mask (counting nonzero pixels)
        masks_with_area = [(mask, np.count_nonzero(mask)) for mask in masks]
        
        # Sort by area (largest first)
        masks_with_area.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the top 'max_objects' masks
        filtered_masks = [m[0] for m in masks_with_area[:num_objects]]
        
        if len(filtered_masks) < num_objects:
            # Pad with empty masks
            num_empty_masks = num_objects - len(filtered_masks)
            empty_mask = np.zeros_like(filtered_masks[0])
            filtered_masks.extend([empty_mask] * num_empty_masks)

        return filtered_masks


class ImageDataset(Dataset):
    def __init__(self, image_dirs, transform=None, mask_transform=None,preproc_model=None):
        """
        Args:
            image_dirs (list): List of directories containing images and metadata
            transform (callable, optional): Optional transform to be applied to the image
            mask_transform (callable, optional): Optional transform to be applied to the mask
            preproc_model (YOLO): YOLOv11-Seg model
        """ 
        self.image_dirs = image_dirs
        self.transform = transform
        self.mask_transform = mask_transform  # Separate transform for masks
        self.image_paths = []
        self.metadata = {}
        
        # Load YOLOv11-Seg model (Pretrained)
        self.preproc_model = preproc_model

        for image_dir in image_dirs:
            log_file = os.path.join(image_dir, "image_log.txt") 
            mask_dir = os.path.join(image_dir, "masks")
            # self.masks_paths = [os.path.join(mask_dir, f"{os.path.basename(image_path).replace('.png', '.pt')}") for image_path in self.image_paths]
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            with open(log_file, "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        image_name, metadata_str = parts
                        data = ast.literal_eval(metadata_str)
                        if isinstance(data, tuple):
                            data = data[0]
                        full_image_path = os.path.join(image_dir, image_name)
                        self.image_paths.append(full_image_path)
                        robot_config = list(data.values())[:5]
                        robot_config[3] = robot_config[3]/180 #normalize angle
                        self.metadata[full_image_path] = robot_config
        
        self.masks_paths = [None] * len(self.image_paths) #initialized empty for simplicity in checking whether mask exists

        # Sort for consistency
        self.image_paths = sorted(self.image_paths)
        self.metadata = {k: self.metadata[k] for k in self.image_paths}

    def __len__(self):
        return len(self.image_paths)

    def extract_segmentation_mask(self, image):
        """Runs YOLOv11-Seg to extract object segmentation masks."""
        results = self.preproc_model(image)
        for result in results:
            masks = result.masks
            if masks is not None: 
                mask_tensor = masks.data  # (num_objects, H, W)
                # Convert masks to list and filter
                masks_list = [mask.data for mask in masks]
                masks_filtered = filter_top_masks(masks_list, num_objects=6)
                if 0: # Save example masks
                    image_size = image.size[::-1]  # (H, W)
                    index = 1
                    save_colored_masks(torch.cat(masks_filtered, dim=0), f"mask_example{index}f.png", image_size)
                    save_colored_masks(torch.cat(masks_list, dim=0), f"mask_example{index}.png", image_size)
                    image.save(f"mask_example{index}i.png")
                return torch.cat(masks_filtered, dim=0)
        return None  # If no mask is detected

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        # Apply image transformations
        if self.transform:
            imageT = self.transform(image)

        # Get segmentation mask
        if self.masks_paths[idx] is not None:
            maskT_tensor = torch.load(self.masks_paths[idx])
        else:
            mask_tensor = self.extract_segmentation_mask(image)  # Shape: (num_objects, H, W)
            if mask_tensor is None:
                mask_tensor = torch.zeros((6, 128, 128))  # Default empty mask

            # Apply transformations to mask
            if self.mask_transform:
                maskT_tensor = self.mask_transform(mask_tensor)
            mask_path = os.path.join(os.path.dirname(self.image_paths[idx]),f"masks/{os.path.basename(image_path).replace('.png', '.pt')}")
            torch.save(maskT_tensor, mask_path)
            self.masks_paths[idx] = mask_path
            save_masks(mask_tensor, mask_path.replace('.pt', '.npy'))  # Save masks as .npy file

        # Load metadata
        metadata_values = torch.tensor(self.metadata[image_path], dtype=torch.float32)

        return imageT, maskT_tensor, metadata_values

    def data_split(self, train_ratio, val_ratio):
        dataset_size = len(self)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        return random_split(self, [train_size, val_size, test_size])

