import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from object_tracker.granular_utils import crop_center_region,create_occupancy_mask


def find_thin_tool_center(
    image,
    angle_deg,
    tool_width_px,
    tool_length_px,
    search_margin=20,
    roi=None,
    center_guess=None,
    search_radius=None
):
    """
    Find the center of a thin rectangular tool in an image.
    
    Args:
        image: grayscale or color image
        angle_deg: tool orientation in image (degrees, CCW)
        tool_width_px: expected width in pixels
        tool_length_px: expected length in pixels
        search_margin: margin for y-center search (default 20)
        roi: tuple (x_min, y_min, x_max, y_max) for region of interest, or None to use full image
        center_guess: tuple (x, y) approximate center position to search around
        search_radius: radius around center_guess to search (pixels). If None, uses 2*tool_length_px
    
    Returns:
        (x_center, y_center): center of tool in original image coordinates
    """

    # --- 1. Grayscale ---
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    
    # --- 2. Extract region of interest or use full image ---
    if roi is not None:
        x_min, y_min, x_max, y_max = roi
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        working_image = gray[y_min:y_max, x_min:x_max].copy()
        roi_offset = np.array([x_min, y_min])
    elif center_guess is not None:
        # Define search radius
        if search_radius is None:
            search_radius = int(2 * tool_length_px)
        
        cx, cy = center_guess
        x_min = max(0, cx - search_radius)
        x_max = min(w, cx + search_radius)
        y_min = max(0, cy - search_radius)
        y_max = min(h, cy + search_radius)
        
        working_image = gray[y_min:y_max, x_min:x_max].copy()
        roi_offset = np.array([x_min, y_min])
    else:
        working_image = gray.copy()
        roi_offset = np.array([0, 0])
    
    work_h, work_w = working_image.shape

    # --- 3. Rotate image so tool is horizontal ---
    center = (work_w // 2, work_h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated = cv2.warpAffine(working_image, M, (work_w, work_h), borderMode=cv2.BORDER_REPLICATE)
    
    # --- 4. Apply morphological erosion with tool-sized kernel ---
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(tool_length_px), int(tool_width_px))
    )
    response = cv2.erode(rotated.astype(np.uint8), kernel)
    
    # --- 5. Find best segment along x-axis (tool length direction) ---
    profile_x = np.sum(response, axis=0)
    
    # Smooth profile
    profile_smooth = cv2.GaussianBlur(
        profile_x.astype(np.float32),
        (int(tool_length_px // 2) | 1, 1),  # Odd kernel size
        0
    )
    
    # Find best segment of length ~tool_length using convolution
    L = int(tool_length_px)
    profile_1d = profile_smooth.squeeze()
    
    # Convolve with rectangular window
    conv = np.convolve(profile_1d, np.ones(L), mode="same")
    
    # Find peak, avoiding edges
    valid_range = slice(int(L//2), len(conv) - int(L//2))
    x_center_local = np.argmax(conv[valid_range]) + int(L//2)
    
    # --- 6. Find y-center by summing around x_center ---
    search_x_min = max(0, x_center_local - search_margin)
    search_x_max = min(work_w, x_center_local + search_margin)
    
    y_profile = np.sum(rotated[:, search_x_min:search_x_max], axis=1)
    y_center_local = np.argmax(y_profile)
    
    # --- 7. Refine with local max filtering (optional, for robustness) ---
    # Apply local maximum filter to find sub-pixel accuracy
    kernel_size = int(tool_width_px // 2) | 1
    local_max = cv2.morphologyEx(
        response.astype(np.uint8), 
        cv2.MORPH_CLOSE, 
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    )
    
    # Re-refine x_center using morphology response
    profile_x_refined = np.sum(local_max, axis=0)
    profile_smooth_refined = cv2.GaussianBlur(
        profile_x_refined.astype(np.float32),
        (int(tool_length_px // 2) | 1, 1),
        0
    )
    conv_refined = np.convolve(profile_smooth_refined.squeeze(), np.ones(L), mode="same")
    x_center_local = np.argmax(conv_refined[valid_range]) + int(L//2)
    
    # Re-refine y_center
    search_x_min = max(0, x_center_local - search_margin)
    search_x_max = min(work_w, x_center_local + search_margin)
    y_profile = np.sum(local_max[:, search_x_min:search_x_max], axis=1)
    y_center_local = np.argmax(y_profile)
    
    # --- 8. Transform back to original image ---
    Minv = cv2.invertAffineTransform(M)
    center_img_local = np.dot(Minv, np.array([x_center_local, y_center_local, 1]))
    
    # Add ROI offset to get coordinates in original image
    center_img = center_img_local + roi_offset
    
    return (int(center_img[0]), int(center_img[1])), response, y_profile


# Example usage with ROI
if __name__ == "__main__":
    ref = cv2.imread("reference_empty_base.jpg")
    cropped_ref = crop_center_region(ref)
    cv2.imwrite("reference_empty_base_cropped.jpg",cropped_ref)

    print("trying to run tool detection on saved image")
    image = cv2.imread("latest_manual_base_cropped.png")

    mask = create_occupancy_mask(image,cropped_ref,10)

    masked_image = image.copy()
    masked_image[mask==0] = 0
    image = masked_image
    
    # Example 1: Using full image (old behavior)
    center1, resp1, prof1 = find_thin_tool_center(image, 0, 8, 112, 20)
    print(f"Full image result: {center1}")
    
    # Example 2: Using ROI (if you know approximate region)
    roi = (50, 50, 300, 300)  # (x_min, y_min, x_max, y_max)
    center2, resp2, prof2 = find_thin_tool_center(image, 0, 8, 112, 20, roi=roi)
    print(f"ROI result: {center2}")
    
    # Example 3: Using center guess (if you have approximate center)
    center_guess = (165, 117)
    center3, resp3, prof3 = find_thin_tool_center(image, 0, 8, 112, 20, center_guess=center_guess, search_radius=100)
    print(f"Center guess result: {center3}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].plot(center1[0], center1[1], 'r*', markersize=15, label='Detected')
    axes[0, 0].set_title("Full Image Detection")
    axes[0, 0].legend()
    
    axes[0, 1].imshow(resp1, cmap='gray')
    axes[0, 1].set_title("Erosion Response")
    
    axes[1, 0].plot(prof1)
    axes[1, 0].set_title("Y-axis Profile")
    axes[1, 0].grid(True)
    
    axes[1, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[1, 1].plot(center3[0], center3[1], 'g*', markersize=15, label='With center guess')
    axes[1, 1].set_title("Detection with Center Guess")
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()