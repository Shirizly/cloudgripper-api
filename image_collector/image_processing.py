import os
import numpy as np
from PIL import Image
from pathlib import Path

def compute_median_image(image_dir: str, output_path: str, pattern: str = "*.png") -> None:
    """
    Load a set of images, compute median image per pixel, and save result.
    
    Args:
        image_dir: Directory containing input images
        output_path: Path to save the median image
        pattern: File pattern to match (default: "*.png")
    """
    image_files = sorted(Path(image_dir).glob(pattern))
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    # Load all images
    images = []
    for img_path in image_files:
        img = Image.open(img_path)
        images.append(np.array(img))
    
    # Stack images and compute median
    image_stack = np.stack(images, axis=0)
    median_image = np.median(image_stack, axis=0).astype(np.uint8)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(median_image).save(output_path)
    print(f"Median image saved to {output_path}")

# Example usage:
# compute_median_image("/home/shirizly/Code/CloudGripper/cloudgripper-api/recordings/dataset_0_sequence_20260122_142055", "./output/median.png", "image_base_*.png")

def subtract_images(image_path: str, median_path: str, output_dir: str, threshold: int = 10) -> None:
    """
    Remove pixels from an image that are close to the median image.
    
    Args:
        image_path: Path to the input image
        median_path: Path to the median image
        output_dir: Directory to save the result and mask
        threshold: Pixel value difference threshold (default: 10)
    """
    # Load images
    image = np.array(Image.open(image_path))
    median = np.array(Image.open(median_path))
    
    # Compute pixel-wise difference
    diff = np.abs(image.astype(int) - median.astype(int))
    
    # Create mask where difference is greater than threshold
    mask = (diff > threshold).any(axis=2) if len(diff.shape) == 3 else (diff > threshold)
    
    # Apply mask to image
    result = image.copy()
    result[~mask] = 0
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    Image.fromarray(result).save(os.path.join(output_dir, "subtracted.png"))
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(output_dir, "mask.png"))
    print(f"Subtracted image and mask saved to {output_dir}")

# Example usage:
subtract_images(
    "/home/shirizly/Code/CloudGripper/cloudgripper-api/recordings/dataset_0_sequence_20260122_142055/image_base_0010.png",
    "./output/median100.png",
    "./output/subtraction",
    threshold=50
)