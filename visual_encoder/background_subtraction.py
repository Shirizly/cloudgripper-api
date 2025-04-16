import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import binary_opening, binary_closing
import os
from PIL import Image
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset as ID
from skimage.metrics import structural_similarity as ssim

# mode = "only robot"
mode = "with objects"

# Load an image dataset from a PyTorch DataLoader
def load_image_dataset(dataloader,num_batches=1):
    """
    Load an image dataset from a PyTorch DataLoader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader object

    Returns:
        list: List of images loaded from the DataLoader
    """
    images = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        images.append(batch[0])
    images = torch.cat(images, dim=0)
    return images
# Convert image to PyTorch tensor
def to_tensor(image):
    return transforms.ToTensor()(image)

# Convert tensor back to NumPy for OpenCV processing
def to_numpy(tensor):
    return (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

# Step 1: Align the background using feature matching and homography
def align_images(image, background):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(500)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_background, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    aligned_background = cv2.warpPerspective(background, H, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)

    return aligned_background

#  Align background locally, tile by tile
def align_images_piecewise(image, background, grid_size=(4, 4)):
    H_img, W_img = image.shape[:2]
    h_step, w_step = H_img // grid_size[0], W_img // grid_size[1]

    aligned_background = np.zeros_like(image)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y1, y2 = i * h_step, (i + 1) * h_step
            x1, x2 = j * w_step, (j + 1) * w_step

            img_tile = image[y1:y2, x1:x2]
            bg_tile = background[y1:y2, x1:x2]

            aligned_bg_tile = align_tile(img_tile, bg_tile)
            aligned_background[y1:y2, x1:x2] = aligned_bg_tile

    return aligned_background

# Align a single tile using ORB feature matching
def align_tile(image_tile, background_tile):
    gray_image = cv2.cvtColor(image_tile, cv2.COLOR_RGB2GRAY)
    gray_background = cv2.cvtColor(background_tile, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(500)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_background, None)

    if descriptors1 is None or descriptors2 is None:
        return background_tile  # If no features found, return original tile

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        return background_tile  # Not enough matches for homography

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    if H is None:
        return background_tile  # Homography failed, return original tile

    aligned_tile = cv2.warpPerspective(background_tile, H, (image_tile.shape[1], image_tile.shape[0]), flags=cv2.INTER_NEAREST)
    return aligned_tile

# Step 2: Multi-scale background subtraction (handling focus differences)
def multi_scale_subtraction2(image, background, scales=[1.0, 0.5, 0.25]):
    image_tensor = to_tensor(image)
    background_tensor = to_tensor(background)

    diff_maps = []
    for scale in scales:
        h, w = int(image_tensor.shape[1] * scale), int(image_tensor.shape[2] * scale)
        img_resized = F.interpolate(image_tensor.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        bg_resized = F.interpolate(background_tensor.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

        diff = torch.abs(img_resized - bg_resized)
        diff_maps.append(F.interpolate(diff.unsqueeze(0), size=(image_tensor.shape[1], image_tensor.shape[2]), mode='bilinear').squeeze(0))

    final_diff = torch.stack(diff_maps).mean(dim=0)  # Averaging multi-scale differences

    return final_diff

def multi_scale_subtraction(image, background, threshold=0.1):
    """Perform background subtraction after matching blur levels."""
    # Apply different levels of Gaussian blur
    blur_levels = [1, 3, 5]  # Kernel sizes for Gaussian blur

    masks = []
    for k in blur_levels:
        blurred_image = cv2.GaussianBlur(image, (k, k), 0)
        blurred_background = cv2.GaussianBlur(background, (k, k), 0)

        diff = np.abs(blurred_image.astype(np.float32) - blurred_background.astype(np.float32)) / 255.0
        mask = np.mean(diff, axis=2) > threshold  # Threshold in grayscale
        masks.append(mask)

    # Combine masks (if all scale detects a difference, keep it)
    final_mask = np.logical_and.reduce(masks)

    return final_mask.astype(np.uint8)

def euclidean_subtraction(image, background):
    image_tensor = to_tensor(image)
    # Convert PIL images to PyTorch tensors
    # Compute Euclidean color distance
    # Convert PIL images to NumPy
    image_np = np.array(image)
    background_np = np.array(background)
    # Align background before subtraction
    # aligned_background = align_images(image_np, background_np)
    aligned_background_tensor = transforms.ToTensor()(background)
    # Compute Euclidean distance
    distance = torch.sqrt(torch.sum((image_tensor - aligned_background_tensor) ** 2, dim=0))
    # Thresholding
    threshold = 0.15  # Adjust based on your dataset
    mask = distance > threshold
    return mask

def compute_ssim_mask(image, background, threshold=0.6):
    """Computes an SSIM-based mask to filter out background regions"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    ssim_map = ssim(gray_image, gray_background, full=True)[1]  # Get SSIM similarity map
    mask = (ssim_map < threshold).astype(np.float32)  # Mark pixels where structure changed

    return torch.tensor(mask, dtype=torch.float32)

# Step 3: Morphological processing (removes noise & fills small gaps)
def morphological_refinement(mask_np, structure_size=3):
    # mask_np = mask > 0.1  # Thresholding to get binary mask
    refined_mask = binary_opening(mask_np, structure=np.ones((structure_size, structure_size)))
    for i in range(2):
        refined_mask = binary_closing(refined_mask, structure=np.ones((structure_size*2, 2*structure_size)))

    return torch.tensor(refined_mask, dtype=torch.float32)

# step 3.5: reverse the mask and filter out small components
def filter_small_components_reverse(mask, min_size=1000):
    # Reverse the mask (assuming it's binary: 0s and 1s)
    mask_np = (1 - mask.numpy()).astype(np.uint8) * 255
    
    # Apply connected components analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=4)
    
    # Initialize a blank mask to reconstruct the filtered mask
    filtered_mask = np.zeros_like(mask_np, dtype=np.uint8)

    # Keep only components larger than min_size
    for i in range(1, num_labels):  # Start from 1 to ignore the background (label 0)
        if stats[i, cv2.CC_STAT_AREA] > min_size:
            filtered_mask[labels == i] = 255  # Add large components back

    # Reverse the mask back to original foreground-background logic
    filtered_mask = (1 - (filtered_mask // 255))  # Convert 255 back to 1s, 0 stays 0

    return torch.tensor(filtered_mask, dtype=torch.uint8)  # Convert back to PyTorch tensor


# step 4: use the filtered image to get segmentation mask by connected components
def get_segmentation_mask(mask):
    mask_np = mask.numpy().astype(np.uint8)*255
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=4)

    # Get a list of masks for each connected component
    masks = [(labels == i).astype(np.uint8) for i in range(1, num_labels)]
    # sort masks by size
    masks.sort(key=lambda x: x.sum(), reverse=True)
    masks = np.array(masks)
    return torch.tensor(masks, dtype=torch.uint8)

# Step 5: Post-processing (filter out small components)
def filter_small_components(masks, min_size=1000):
    filtered_masks = [mask for mask in masks if mask.sum() > min_size] 
    return torch.stack(filtered_masks)

# step 5.5: unite the masks into a single mask
def unite_masks(masks):
    mask = torch.zeros_like(masks[0])
    for m in masks:
        mask += m
    mask = torch.clamp(mask, 0, 1)
    return mask

# Step 6: create an image with the filtered masks with each mask as a distinct color
def create_mask_image(masks):
    # Create an empty RGB image
    mask_image = torch.zeros((3,masks[0].shape[0], masks[0].shape[1]), dtype=torch.uint8)
    
    # Define a list of distinct colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    # Assign each mask a distinct color
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        for c in range(3):
            mask_image[c, :, :] += mask * color[c]
    
    return mask_image





# Full pipeline execution
def process_image(image, background):
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    background = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)

    # aligned_background = align_images_piecewise(image_bgr, background) 
    aligned_background = background # align_images(image_bgr, background) 
    # diff = multi_scale_subtraction(image_bgr, aligned_background)
    diff = euclidean_subtraction(image_bgr, background)
    # diff = compute_ssim_mask(image_bgr, aligned_background)
    mask = morphological_refinement(diff)
    mask = filter_small_components_reverse(mask)

    segmentation_masks = get_segmentation_mask(mask)
    filtered_masks = filter_small_components(segmentation_masks, min_size=3000)
    mask_image = create_mask_image(filtered_masks)

    # Apply mask to original image
    mask = unite_masks(filtered_masks)
    image_tensor = to_tensor(image)
    filtered_image = image_tensor * mask

    # Convert back to PIL image (and save(optional))
    filtered_pil = transforms.ToPILImage()(filtered_image)
    mask_pil = transforms.ToPILImage()(mask_image)
    # filtered_pil.save("filtered_image.png")


    return filtered_pil, mask_pil, filtered_masks



def main():

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    recordings_dir = os.path.join(project_root,"recordings")
    # generate list of data directories
    sequence_dirs = []
    log_file = []
    if os.path.exists(recordings_dir):
        dataset_dirs = [d for d in os.listdir(recordings_dir) if d.startswith("dataset_0_sequence_")]
        for sequence in dataset_dirs:
            sequence_dirs.append(os.path.join(os.path.join(recordings_dir, sequence)))
    if 1:
        # Load dataset
        if mode == "only robot":
            dataset = ID([sequence_dirs[0]],transform=transforms.ToTensor())
        if mode == "with objects":
            dataset = ID(sequence_dirs,transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True)
        image_tensors = load_image_dataset(loader, num_batches=4)
        background_tensor = torch.median(image_tensors, dim=0).values
        background = transforms.ToPILImage()(background_tensor)
        background.save("median_image.png")
  
    else:
        background = Image.open("median_image.png")
        background_tensor = transforms.ToTensor()(background)
    # load image, filter and save
    image_path = os.path.join(sequence_dirs[2],os.path.join("top_images","image_top_0600.png"))
    image = Image.open(image_path)
    image.save("query_image.png")
    image_tensor = transforms.ToTensor()(image)
    
    image_filtered, mask_image, _ = process_image(image, background)

    # Save result
    image_filtered.save("filtered_image_pipeline.png")
    mask_image.save("mask_image_pipeline.png")
if __name__ == "__main__":
    main()