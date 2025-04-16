from PIL import Image
from torchvision import transforms
import os,sys
from torchvision import datasets
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset as ID
import torch
import pickle
import cv2
import numpy as np

def resize_image(image_path, output_path, output_image_size=None):
    """
    Preprocess an image by resizing it and saving it to a new location.

    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the output image
        output_image_size (tuple): (H, W) - size of the output image
    """
    
    # Load the image
    image = Image.open(image_path)
    if output_image_size is None:
        output_image_size = image.size

    # Resize the image
    transform = transforms.Resize(output_image_size)
    image_resized = transform(image)


    # Save the resized image
    image_resized.save(output_path)

    print(f"Saved resized image to {output_path}")

# def average_image(image_tensors):
#     """
#     Average multiple images and save the result.

#     Args:
#         images (list): List of PIL images
#         output_path (str): Path to save the output image
#     """
#     # Convert images to tensors
#     # image_tensors = [transforms.ToTensor()(image) for image in images]

#     # Compute the average image
#     average_tensor = torch.mean(image_tensors, dim=0)
#     average_image = transforms.ToPILImage()(average_tensor)

#     return average_image

    

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

def multiscale_background_subtraction(image, background, threshold=0.1):
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

    # Combine masks (if any scale detects a difference, keep it)
    final_mask = np.logical_and.reduce(masks)

    return final_mask.astype(np.uint8)


def align_images(image, background):
    """Aligns background and image using feature matching."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(500)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_background, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp background to align with the image
    background_bgr = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    aligned_background_bgr = cv2.warpPerspective(background_bgr, H, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
    aligned_background = cv2.cvtColor(aligned_background_bgr, cv2.COLOR_BGR2RGB)  # Convert back to RGB

    return aligned_background

def edge_based_subtraction(image, background, threshold=50):
    """Uses Canny edge detection to find differences between image and background."""
    edges_image = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
    edges_background = cv2.Canny(cv2.cvtColor(background, cv2.COLOR_RGB2GRAY), 50, 150)

    # Compute absolute difference
    edge_diff = cv2.absdiff(edges_image, edges_background)

    # Threshold edge difference
    _, mask = cv2.threshold(edge_diff, threshold, 255, cv2.THRESH_BINARY)

    return mask



def main():

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    recordings_dir = os.path.join(project_root,"recordings")
    # generate list of data directories
    sequence_dirs = []
    log_file = []
    if os.path.exists(recordings_dir):
        dataset_dirs = [d for d in os.listdir(recordings_dir) if d.startswith("dataset_0_sequence_")]
        for sequence in dataset_dirs:
            sequence_dirs.append(os.path.join(recordings_dir, sequence))
    if 0:
        # Load dataset
        dataset = ID([sequence_dirs[1]],transform=transforms.ToTensor())

        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        image_tensors = load_image_dataset(loader, num_batches=1)
        
        avg_image_tensor = torch.mean(image_tensors, dim=0)
        avg_image = transforms.ToPILImage()(avg_image_tensor)
        avg_image.save("average_image.png")

        med_image_tensor = torch.median(image_tensors, dim=0).values
        med_image = transforms.ToPILImage()(med_image_tensor)
        med_image.save("median_image.png")
  
    else:
        avg_image = Image.open("average_image.png")
        avg_image_tensor = transforms.ToTensor()(avg_image)
        med_image = Image.open("median_image.png")
        med_image_tensor = transforms.ToTensor()(med_image)
    # load image, filter and save
    image_path = os.path.join(sequence_dirs[1],"image_top_0100.png")
    image = Image.open(image_path)
    image.save("query_image.png")
    image_tensor = transforms.ToTensor()(image)
    
    # first filter - simple subtraction
    image_filt_tensor = image_tensor-med_image_tensor
    image_filt = transforms.ToPILImage()(image_filt_tensor)
    image_filt.save("filtered_image.png")
    
    # second filter - thresholding
    image_filt2_tensor = image_tensor*(image_filt_tensor>0.1)
    image_filt2 = transforms.ToPILImage()(image_filt2_tensor)
    image_filt2.save("filtered_image2.png")
    
    # third filter - euclidean distance
    # Compute Euclidean color distance
    # Convert PIL images to NumPy
    image_np = np.array(image)
    background_np = np.array(med_image)
    # Align background before subtraction
    aligned_background = align_images(image_np, background_np)
    aligned_background_tensor = transforms.ToTensor()(aligned_background)
    transforms.ToPILImage()(aligned_background_tensor).save("aligned_background.png")
    # Compute Euclidean distance
    distance = torch.sqrt(torch.sum((image_tensor - aligned_background_tensor) ** 2, dim=0))
    # Thresholding
    threshold = 0.15  # Adjust based on your dataset
    mask = distance > threshold
    # Apply mask
    image_filt_tensor = image_tensor * mask.unsqueeze(0)
    image_filt = transforms.ToPILImage()(image_filt_tensor)
    image_filt.save("filtered_image_euclidean.png")

    # # fourth filter - adaptive thresholding
    # # Convert to NumPy
    # distance_np = distance.numpy()
    # # Convert to 8-bit grayscale for Otsu’s method
    # distance_8bit = (distance_np * 255).astype(np.uint8)
    # # Compute Otsu’s threshold
    # _, threshold_value = cv2.threshold(distance_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # Apply mask based on Otsu’s threshold
    # mask = distance > (torch.tensor(threshold_value) / 255.0)
    # image_filt_tensor = image_tensor * mask.unsqueeze(0)
    # # Save result
    # image_filt = transforms.ToPILImage()(image_filt_tensor)
    # image_filt.save("filtered_image_otsu.png")
    
    # fifth filter - morphological operations
    from scipy.ndimage import binary_opening, binary_closing
    # Convert mask to NumPy
    mask_np = mask.numpy()
    # Morphological processing
    for i in range(3):
        mask_np = binary_opening(mask_np, structure=np.ones((3, 3)))
        mask_np = binary_closing(mask_np, structure=np.ones((7, 7)))
    # Convert back to tensor
    mask_cleaned = torch.tensor(mask_np, dtype=torch.float32)
    # Apply mask
    image_filt_tensor = image_tensor * mask_cleaned.unsqueeze(0)
    # Save result
    image_filt = transforms.ToPILImage()(image_filt_tensor)
    image_filt.save("filtered_image_morph.png")

    
    
    # # sixth filter - semantic segmentation
    # from torchvision.models.segmentation import deeplabv3_resnet101

    # # Load pre-trained DeepLabV3 model
    # model = deeplabv3_resnet101(pretrained=True)
    # model.eval()

    # # Preprocess image
    # transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    # input_tensor = transform(image_filt).unsqueeze(0)

    # # Run model
    # with torch.no_grad():
    #     output = model(input_tensor)['out']
    #     segmentation_mask = torch.argmax(output.squeeze(), dim=0).numpy()

    # # Assume class 0 is background; keep everything else
    # robot_mask = segmentation_mask > 0

    # # Convert mask to tensor
    # robot_mask_tensor = torch.tensor(robot_mask, dtype=torch.float32)

    # # Apply mask
    # image_filt_tensor = (input_tensor * robot_mask_tensor.unsqueeze(0)).squeeze(0)

    # # Save result
    # image_filt = transforms.ToPILImage()(image_filt_tensor)
    # image_filt.save("filtered_image_deeplab.png")
    


    # seventh filter - multiscale background subtraction
    # Convert PIL images to NumPy
    image_np = np.array(image)
    background_np = np.array(med_image)
    # Align background before subtraction
    aligned_background = align_images(image_np, background_np)

    # Compute mask
    mask = multiscale_background_subtraction(image_np, background_np,threshold=0.15)
    # Apply mask
    # Apply mask
    image_filtered = image_np * mask[..., np.newaxis]
    # Save result
    cv2.imwrite("filtered_image_multiscale2.png", cv2.cvtColor(image_filtered, cv2.COLOR_BGR2RGB))



    # eighth filter - edge-based subtraction
    # Compute edge-based mask
    edge_mask = edge_based_subtraction(image_np, background_np)

    # Apply mask
    image_filtered = image_np * (edge_mask[..., np.newaxis] > 0)

    # Save result
    cv2.imwrite("filtered_image_edges.png", cv2.cvtColor(image_filtered, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    main()