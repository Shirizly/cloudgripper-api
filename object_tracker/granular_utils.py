import cv2
import numpy as np
from typing import Tuple, List, Dict

def crop_center_region(image: np.ndarray, crop_size: Tuple[int, int] = (360, 360), crop_center: Tuple[int, int] = (275, 200)) -> np.ndarray:
    """
    Crop a region around a center point.
    
    Args:
        image: Input image
        crop_size: Size of the crop (height, width)
        crop_center: Center point of the crop (y, x)
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size
    center_y, center_x = crop_center
    
    y1 = max(0, center_y - crop_h // 2)
    y2 = min(h, center_y + crop_h // 2)
    x1 = max(0, center_x - crop_w // 2)
    x2 = min(w, center_x + crop_w // 2)
    
    return image[y1:y2, x1:x2]

def create_occupancy_mask(image: np.ndarray, reference_empty: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Create a binary occupancy mask by comparing image to empty reference.
    
    Args:
        image: Current image from camera
        reference_empty: Reference image of empty glass plate
        threshold: Difference threshold to consider as occupied
        
    Returns:
        Binary occupancy mask (1 = occupied, 0 = empty)
    """
    
    # Compute color difference and create mask for pixels NOT matching reference

    color_diff = np.abs(image.astype(np.float32) - reference_empty.astype(np.float32))
    max_diff = 255 * 0.11  # 10% threshold
    diff = np.max(color_diff, axis=2) if len(color_diff.shape) == 3 else color_diff
    
    # Create binary mask
    mask = (diff > max_diff).astype(np.uint8) * 255

    # remove from mask pixels in strong primary colors (likely robot parts)
    lower_red = np.array([0, 0, 100], dtype=np.uint8)
    upper_red = np.array([80, 80, 255], dtype=np.uint8)
    lower_green = np.array([0, 100, 0], dtype=np.uint8)
    upper_green = np.array([80, 255, 80], dtype=np.uint8)
    lower_blue = np.array([100, 0, 0], dtype=np.uint8)
    upper_blue = np.array([255, 80, 80], dtype=np.uint8)
    red_mask = cv2.inRange(image, lower_red, upper_red)
    green_mask = cv2.inRange(image, lower_green, upper_green)
    blue_mask = cv2.inRange(image, lower_blue, upper_blue)
    lower_yellow = np.array([0, 150, 150], dtype=np.uint8)
    upper_yellow = np.array([120, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)
    primary_color_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask, green_mask), cv2.bitwise_or(blue_mask, yellow_mask))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(primary_color_mask))

    # also remove background solid color
    lower_gray = np.array([170, 170, 170], dtype=np.uint8)
    upper_gray = np.array([240, 240, 240], dtype=np.uint8)
    gray_mask = cv2.inRange(image, lower_gray, upper_gray)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([40, 40, 40], dtype=np.uint8)
    black_mask = cv2.inRange(image, lower_black, upper_black)
    gray_mask = cv2.bitwise_or(gray_mask, black_mask)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(gray_mask))
    
    return mask

def create_chickpea_mask(image: np.ndarray, threshold: int = 0.000001) -> np.ndarray:
    """
    Create a binary mask specifically for chickpeas by comparing image to a distribution of colors.
    
    Args:
        image: Current image from camera
        reference_empty: Reference image of empty glass plate
        """
    # Define multiple gaussian distributions for different shades of chickpeas
    # These values would ideally be determined empirically by analyzing sample images
    Mean_colors = [[54.15659 , 60.272285 ,68.80524 ],
                   [ 74.6377 ,   85.772385 ,101.56367 ],
                   [136.10983, 143.45859, 153.23396],
                   [122.02655 , 122.256775 ,133.88206]]
    covariances = [[[121.12808 , 120.948555 ,135.46205 ],
                    [120.948555 ,171.10521  ,238.90141 ],
                    [135.46205 , 238.90141,  380.1807  ]],
                    [[345.65894 ,336.87213 ,323.93698],
                    [336.87213 ,384.97235 , 425.1901 ],
                    [323.93698 ,425.1901  ,542.6201 ]],
                    [[532.7858  ,436.9837  ,328.4652 ],
                    [436.9837  ,435.40802 ,404.0879 ],
                    [328.46527 ,404.0879  ,466.86096]],
                    [[2864.9705 ,2975.4531 ,2728.6943],
                    [2975.453 , 3776.381 , 3765.613 ],
                    [2728.6943, 3765.613 , 4020.4883]]]
    weights = [0.3, 0.3, 0.3, 0.1]  # relative weights of each gaussian

    chickpea_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mean, cov, weight in zip(Mean_colors, covariances, weights):
        mean = np.array(mean)
        cov = np.array(cov)
        inv_cov = np.linalg.inv(cov)
        diff = image.astype(np.float32) - mean.astype(np.float32)
        mahalanobis_dist = np.einsum('...i,ij,...j->...', diff, inv_cov, diff)
        gaussian_prob = np.exp(-0.5 * mahalanobis_dist) / np.sqrt((2 * np.pi) ** 3 * np.linalg.det(cov))
        # threshold_value = 0.00001  # This threshold may need tuning
        # chickpea_mask |= (gaussian_prob > threshold_value).astype(np.uint8) * 255
        # Alternatively, we can use the probabilities as weights to create a soft mask
        chickpea_mask += ((gaussian_prob * weight)*2100000).astype(np.uint8)
    # chickpea_mask = (chickpea_mask>threshold).astype(np.uint8)*255
    return chickpea_mask
        

def clean_occupancy_mask(mask: np.ndarray, kernel_size: int = 3, min_size: int = 300) -> tuple[np.ndarray, int, cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Clean occupancy mask using morphological operations.
    
    Args:
        mask: Input binary mask
        kernel_size: Size of the morphological kernel
        
    Returns:
        Cleaned binary mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Morphological opening to remove noise
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Morphological closing to fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # compute connected compnonents to filter small regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < min_size:
            closed[labels == i] = 0  # Remove small regions
        # also remove very elongated regions
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        if (width / height > 3 or height / width > 3) and area < min_size * 2: # so as to not remove clumps, only small elongated regions
            closed[labels == i] = 0  # Remove elongated regions
        if width<np.sqrt(min_size) or height<np.sqrt(min_size): # probably pieces of the wall or robot, can't be granules
            closed[labels == i] = 0  #
        # also if area is too small compared to bounding box
        if area < (width * height) * 0.25 and area < min_size * 2:
            closed[labels == i] = 0  # Remove sparse regions
        # print(area,centroids[i], width, height, area / (width * height))


    return closed, num_labels-1, stats, centroids


def downscale_mask(mask: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """
    Downscale occupancy mask to lower resolution.
    
    Args:
        mask: Input binary mask
        scale_factor: Downscaling factor
        
    Returns:
        Downscaled binary mask
    """
    h, w = mask.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    
    downscaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    return downscaled


def find_clumps(mask: np.ndarray, min_size: int = 5, max_size: int = 500) -> List[Dict]:
    """
    Identify connected components (clumps) in occupancy mask.
    
    Args:
        mask: Binary occupancy mask
        min_size: Minimum clump size in pixels
        max_size: Maximum clump size in pixels
        
    Returns:
        List of clump dictionaries with properties
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    clumps = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        if min_size <= area <= max_size:
            clump = {
                'id': i,
                'area': area,
                'centroid': centroids[i],
                'bbox': {
                    'x': stats[i, cv2.CC_STAT_LEFT],
                    'y': stats[i, cv2.CC_STAT_TOP],
                    'width': stats[i, cv2.CC_STAT_WIDTH],
                    'height': stats[i, cv2.CC_STAT_HEIGHT]
                }
            }
            clumps.append(clump)
    
    return clumps


def generate_clump_hierarchy(mask: np.ndarray, scales: List[int] = None) -> List[List[Dict]]:
    """
    Generate clump detections at multiple resolution scales.
    
    Args:
        mask: Binary occupancy mask
        scales: List of scale factors for analysis
        
    Returns:
        List of clump lists, one per scale
    """
    if scales is None:
        scales = [1, 2, 4, 8]
    
    hierarchy = []
    for scale in scales:
        scaled_mask = downscale_mask(mask, scale)
        clumps = find_clumps(scaled_mask)
        hierarchy.append(clumps)
    
    return hierarchy


def process_image(image: np.ndarray, reference_empty: np.ndarray, 
                  scale_factor: int = 4, min_size: int = 300, crop_size = (360, 360), crop_center=(275, 200)) -> Tuple[np.ndarray, int ,cv2.typing.MatLike, cv2.typing.MatLike]:
    """
    Complete pipeline: mask, downscale, and detect clumps.
    
    Args:
        image: Current image
        reference_empty: Empty reference image
        scale_factor: Downscaling factor
        
    Returns:
        Tuple of (downscaled_mask, clumps_list, mask)
    """
    # cv2.imshow("input and reference", np.hstack([image, reference_empty]))
    # cv2.waitKey(10000)
    # cv2.destroyAllWindows()
    cropped_image = crop_center_region(image, crop_size=crop_size, crop_center=crop_center)
    cropped_reference = crop_center_region(reference_empty, crop_size=crop_size, crop_center=crop_center)
    # cv2.imshow("input and reference", np.hstack([cropped_image, cropped_reference]))
    # cv2.waitKey(100000)
    # cv2.destroyAllWindows()
    # cv2.imwrite("cropped_image.png", cropped_image)
    # cv2.imwrite("cropped_reference.png", cropped_reference)
    # mask = create_occupancy_mask(cropped_image, cropped_reference)
    mask = create_chickpea_mask(cropped_image)
    masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)
    mask = create_occupancy_mask(masked_image, cropped_reference)
    # cv2.imwrite("raw_mask.png", mask)
    clean_mask, num_labels, stats, centroids = clean_occupancy_mask(mask, min_size=min_size)


    # downscaled = downscale_mask(mask, scale_factor)
    # clumps = find_clumps(downscaled)
    # cv2.imshow("Occupancy Mask", clean_mask)

    # cv2.waitKey(100000)
    # cv2.destroyAllWindows()
    cv2.imwrite("occupancy_mask_for_debug.png", clean_mask)

    return clean_mask, num_labels, stats, centroids

def check_reset_needed(mask):
    try:
        mask_area = cv2.countNonZero(mask) if mask is not None else 0
        h, w = mask.shape
        # define main workspace area (inside manipulation boundaries) as a mask and calculate occupancy
        # here we assume the main workspace is the central square region
        x_start = int(w * 0.3)
        x_end = int(w * 0.7)
        y_start = int(h * 0.3)
        y_end = int(h * 0.7)
        workspace_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.rectangle(workspace_mask, (x_start, y_start), (x_end, y_end), 255, thickness=-1)
        workspace_area = cv2.countNonZero(workspace_mask)
        if workspace_area == 0:
            return False
        occupied_area = cv2.countNonZero(cv2.bitwise_and(mask, workspace_mask))

        occupancy_ratio = occupied_area / mask_area
        threshold = 0.3  # e.g., if less than 30% of mask is in the main workspace, needs resetting
        if occupancy_ratio < threshold:
            print(f"Main workspace needs reset: occupancy ratio {occupancy_ratio:.2f}")
            return True
        print(f"Main workspace doesn't need reset: occupancy ratio {occupancy_ratio:.2f}")
        return False
    except Exception as e:
        print(f"Error checking if reset is needed: {e}, {repr(e)}, Type: {type(e)}")
        return False