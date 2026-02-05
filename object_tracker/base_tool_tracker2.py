import cv2
import numpy as np
import matplotlib.pyplot as plt
import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from object_tracker.granular_utils import crop_center_region,create_occupancy_mask
from scipy import signal

def find_thin_tool_center(
    image,
    angle_deg,
    tool_width_px,
    tool_length_px,
    search_margin=20,
    roi=None,
    center_guess=None,
    search_radius=None,
    debug=False
):
    """
    Find the center of a thin rectangular tool using intensity-based search.
    Works best with center_guess to narrow down the search region.
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
        roi_offset = np.array([x_min, y_min], dtype=np.float32)
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
        roi_offset = np.array([x_min, y_min], dtype=np.float32)
        
        print(f"Extracted ROI from ({x_min}, {y_min}) to ({x_max}, {y_max})")
    else:
        working_image = gray.copy()
        roi_offset = np.array([0, 0], dtype=np.float32)
    
    work_h, work_w = working_image.shape
    print(f"Working image shape: {work_h}h x {work_w}w")

    # --- 3. Rotate image so tool is horizontal ---
    center_rot = (work_w / 2.0, work_h / 2.0)
    M = cv2.getRotationMatrix2D(center_rot, -angle_deg, 1.0)
    rotated = cv2.warpAffine(working_image, M, (work_w, work_h), borderMode=cv2.BORDER_REPLICATE)
    
    print(f"Working image intensity range: [{np.min(working_image)}, {np.max(working_image)}]")
    print(f"Rotated image intensity range: [{np.min(rotated)}, {np.max(rotated)}]")
    
    # --- 4. Find tool using intensity peaks ---
    # Smooth to reduce noise
    blurred = cv2.GaussianBlur(rotated.astype(np.float32), (5, 5), 1.0)
    
    # Find x-center by summing intensity across rows
    profile_x = np.sum(blurred, axis=0)
    
    print(f"Profile X: min={np.min(profile_x)}, max={np.max(profile_x)}, mean={np.mean(profile_x)}")
    
    # Smooth the profile
    profile_x_smooth = cv2.GaussianBlur(
        profile_x.reshape(-1, 1),
        (int(tool_length_px // 2) * 2 + 1, 1),
        tool_length_px / 4
    ).squeeze()
    
    # Find the brightest segment using convolution with tool-length window
    L = int(tool_length_px)
    if len(profile_x_smooth) >= L:
        conv = np.convolve(profile_x_smooth, np.ones(L) / L, mode='same')
        # Avoid edges
        margin = int(L * 0.4)
        valid_conv = conv[margin:-margin] if margin < len(conv)//2 else conv
        x_center_work = np.argmax(valid_conv) + margin
    else:
        x_center_work = np.argmax(profile_x_smooth)
    
    print(f"X center in working image: {x_center_work} (out of {work_w})")
    
    # --- 5. Find y-center with peak detection ---
    # Sum intensity across columns near x_center
    search_x_min = max(0, int(x_center_work - search_margin))
    search_x_max = min(work_w, int(x_center_work + search_margin))
    
    profile_y = np.sum(blurred[:, search_x_min:search_x_max], axis=1)
    
    print(f"Profile Y: min={np.min(profile_y)}, max={np.max(profile_y)}, mean={np.mean(profile_y)}")
    print(f"Summing x range: [{search_x_min}, {search_x_max}]")
    
    # Smooth the profile
    profile_y_smooth = cv2.GaussianBlur(
        profile_y.reshape(-1, 1),
        (int(tool_width_px) * 2 + 1, 1),
        tool_width_px / 2
    ).squeeze()
    
    # Find peaks in the profile
    from scipy import signal
    
    # Use a prominence-based peak detector to avoid picking up noise
    # Look for peaks with a minimum height relative to background
    threshold = np.mean(profile_y_smooth) + 0.3 * (np.max(profile_y_smooth) - np.mean(profile_y_smooth))
    peaks, properties = signal.find_peaks(profile_y_smooth, height=threshold, distance=int(tool_width_px * 0.5))
    
    print(f"Found {len(peaks)} peaks: {peaks}")
    print(f"Peak heights: {properties['peak_heights']}")
    
    if len(peaks) > 0:
        # Pick the peak with maximum height
        best_peak_idx = np.argmax(properties['peak_heights'])
        y_center_work = peaks[best_peak_idx]
        print(f"Selected peak at y={y_center_work} with height={properties['peak_heights'][best_peak_idx]}")
    else:
        # Fallback: just use the max
        y_center_work = np.argmax(profile_y_smooth)
        print(f"No peaks found, using global max at y={y_center_work}")
    
    print(f"Y center in working image: {y_center_work} (out of {work_h})")
    
    # --- 6. Transform back ---
    point_rotated = np.array([x_center_work, y_center_work], dtype=np.float32)
    
    # For angle_deg = 0, this should be identity, but let's be explicit
    Minv = cv2.invertAffineTransform(M)
    point_unrotated = Minv[:, :2] @ point_rotated + Minv[:, 2]
    
    center_orig = point_unrotated + roi_offset
    
    print(f"Final center in original image: {center_orig}")
    
    if debug:
        return (int(center_orig[0]), int(center_orig[1])), blurred, profile_y_smooth, profile_x_smooth, (x_center_work, y_center_work), roi_offset, peaks
    else:
        return (int(center_orig[0]), int(center_orig[1])), blurred, profile_y_smooth


if __name__ == "__main__":
    ref = cv2.imread("reference_empty_base.jpg")
    cropped_ref = crop_center_region(ref)
    cv2.imwrite("reference_empty_base_cropped.jpg", cropped_ref)

    print("trying to run tool detection on saved image")
    image = cv2.imread("latest_manual_base_cropped.png")

    # mask = create_occupancy_mask(image, cropped_ref, 10)

    # masked_image = image.copy()
    # masked_image[mask == 0] = 0
    # image = masked_image
    
    # Example 3: Using center guess (RECOMMENDED)
    print("\n=== Center guess detection (most robust) ===")
    center_guess = (165, 121)
    center3, resp3, prof3, conv3, work_coords3, offset3, peaks = find_thin_tool_center(
        image, 0, 8, 112, 20, 
        center_guess=center_guess, 
        search_radius=100,
        debug=True
    )
    print(f"Center guess result: {center3}")
    print(f"Working coords: {work_coords3}, offset: {offset3}")
    
    # Visualization - FIXED
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image with detected center
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].plot(center_guess[0], center_guess[1], 'b+', markersize=20, markeredgewidth=2, label='Guess')
    axes[0].plot(center3[0], center3[1], 'g*', markersize=15, label='Detected')
    axes[0].set_title("Detection with Center Guess (Original Image)")
    axes[0].legend()
    
    # Erosion response with working image coordinates
    axes[1].imshow(resp3, cmap='gray')
    # Plot the detected center in WORKING IMAGE coordinates
    axes[1].plot(work_coords3[0], work_coords3[1], 'g*', markersize=15, label='Detected (working space)')
    axes[1].set_title("Erosion Response (Working Image Space)")
    axes[1].axvline(work_coords3[0], color='g', linestyle='--', alpha=0.5)
    axes[1].axhline(work_coords3[1], color='g', linestyle='--', alpha=0.5)
    axes[1].legend()
    
    # Y-profile
    axes[2].plot(prof3, label='Y-profile')
    axes[2].axvline(work_coords3[1], color='g', linestyle='--', alpha=0.5, label=f'Peak at {work_coords3[1]}')
    axes[2].set_title("Y-axis Profile (Working Image Space)")
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()