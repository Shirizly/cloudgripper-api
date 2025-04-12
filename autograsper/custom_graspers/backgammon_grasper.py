from grasper import AutograsperBase
from library.utils import OrderType
from library.rgb_object_tracker import get_object_pos
import time
import numpy as np
from typing import List, Tuple

import cv2

import cv2
import numpy as np
import cv2
import numpy as np

def find_object(image, lower_color, upper_color, shape="any", min_size=100, max_size=None, 
                circularity_threshold=0.8, output_path=None, show_mask=False):
    """
    Find an object in an image based on color range, size constraints, and shape,
    mark it with a red dot, and optionally save the result.
    
    Parameters:
    -----------
    image : numpy.ndarray
        OpenCV image object (BGR format)
    lower_color : tuple
        Lower bound of color range in HSV format (hue, saturation, value)
    upper_color : tuple
        Upper bound of color range in HSV format (hue, saturation, value)
    shape : str
        Shape to detect: "circle", "rectangle", or "any"
    min_size : int
        Minimum area of the object in pixels
    max_size : int or None
        Maximum area of the object in pixels, if None, no maximum size limit
    circularity_threshold : float
        Threshold for circle detection (0-1, higher is more strict)
    output_path : str or None
        Path to save the output image with marked object
    show_mask : bool
        If True, returns a masked image showing only colors in the specified range
    
    Returns:
    --------
    tuple
        (center_coordinates, output_image, mask_image)
        center_coordinates: (x, y) coordinates or None if no object found
        output_image: image with red dot at object center
        mask_image: image showing only the colors in range (if show_mask=True, else None)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image input")
    
    # Make a copy for drawing
    output_image = image.copy()
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask based on the color range
    mask = cv2.inRange(hsv_image, np.array(lower_color), np.array(upper_color))
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center_coordinates = None
    
    # Filter contours based on size and shape constraints
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Size filter
        if area < min_size or (max_size is not None and area > max_size):
            continue
        
        # Shape filter
        if shape.lower() == "circle":
            # Calculate circularity (4*pi*area/perimeter^2)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Circles have circularity close to 1
            if circularity < circularity_threshold:
                continue
                
        elif shape.lower() == "rectangle":
            # Calculate how rectangular the shape is
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            extent = float(area) / rect_area
            
            # Rectangles have high extent (area ratio)
            if extent < 0.7:  # Threshold for rectangularity
                continue
                
        # If we got here, the contour passes all filters
        valid_contours.append(contour)
    
    # Process if valid contours found
    if valid_contours:
        # Find the largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Calculate the center of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center_coordinates = (cx, cy)
            
            # Draw a red dot at the center of the object
            cv2.circle(output_image, center_coordinates, 10, (0, 0, 255), -1)
            
            # Optionally draw the contour
            cv2.drawContours(output_image, [largest_contour], 0, (0, 255, 0), 2)
    
    # Create mask visualization if requested
    mask_image = None
    if show_mask:
        # Apply the mask to the original image
        mask_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Save the output image if path provided
    if output_path is not None and center_coordinates is not None:
        cv2.imwrite(output_path, output_image)
        print(f"Image with marked object saved to {output_path}")
    
    return (center_coordinates, output_image, mask_image)

def create_color_mask_image(image, lower_color, upper_color, output_path=None, binary_output=True):
    """
    Create an image showing only the colors within the specified range.
    
    Parameters:
    -----------
    image : numpy.ndarray
        OpenCV image object (BGR format)
    lower_color : tuple
        Lower bound of color range in HSV format (hue, saturation, value)
    upper_color : tuple
        Upper bound of color range in HSV format (hue, saturation, value)
    output_path : str or None
        Path to save the mask image
        If None, no image will be saved
    binary_output : bool
        If True, converts the masked area to white (255,255,255) for better visibility
    
    Returns:
    --------
    numpy.ndarray
        Image showing only the colors in the specified range
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image input")
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask based on the color range
    mask = cv2.inRange(hsv_image, np.array(lower_color), np.array(upper_color))
    
    if binary_output:
        # Create a white image with the same dimensions as the input
        result = np.zeros_like(image)
        result[:] = (255, 255, 255)  # Fill with white
        
        # Apply the mask to show white only where the mask is positive
        mask_image = cv2.bitwise_and(result, result, mask=mask)
    else:
        # Apply the mask to the original image (original behavior)
        mask_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Save the mask image if path provided
    if output_path is not None:
        cv2.imwrite(output_path, mask_image)
        print(f"Color mask image saved to {output_path}")
    
    return mask_image

class BackgammonGrasper(AutograsperBase):
    def __init__(self, config, shutdown_event):
        super().__init__(config, shutdown_event=shutdown_event)



    def center_sweep(self):
        orders = [
            (OrderType.GRIPPER_CLOSE, [0]),
            (OrderType.MOVE_Z, [0]),
            (OrderType.MOVE_XY, [0.5, 0.0]),
            (OrderType.MOVE_XY, [0.5, 0.4]),
            (OrderType.MOVE_XY, [0.5, 0.5]),
            (OrderType.MOVE_XY, [0.5, 0.6]),
            (OrderType.MOVE_XY, [0.4, 0.6]),
            (OrderType.MOVE_XY, [0.4, 0.4]),
            (OrderType.MOVE_XY, [0.6, 0.4]),
        ]
        self.queue_robot_orders(orders, delay=self.time_between_orders)

    def get_color_pos(self, color):
        return get_object_pos(self.bottom_image, self.robot_idx, color)

    def check_grasping_success(self):
        state = self.robot_state

        gripper_pos = [state["x_norm"], state["y_norm"]]

        object_position = self.get_colod_pos("green")

        if object_position is None:
            return False

        gripper_is_close_enough = (
            np.linalg.norm(np.array(gripper_pos) - np.array(object_position)) < 0.10
        )

        return gripper_is_close_enough

    def perform_task(self):
        time.sleep(3)
        lower= (0, 0, 0)
        upper= (180, 50, 50)

        image = self.bottom_image

        center, marked_image, mask_image = find_object(
            image,
            lower,
            upper,
            shape="circle",           # Specify we want to find circles
            circularity_threshold=0.7, # Adjust based on how perfect your circles are
            min_size=100,
            output_path="circular_object.jpg",
            show_mask=True
        )

        # Save the mask image showing only the colors in range
        if mask_image is not None:
            cv2.imwrite("color_mask.jpg", mask_image)

        # Or use the dedicated function for just the color mask
        color_mask = create_color_mask_image(
            image,
            lower,
            upper,
            output_path="color_mask_only.jpg"
        )

        if center:
            print(f"Object center found at coordinates: {center}")
            self.pick_and_place_object(
                object_position= center,
                object_height= 0,
                target_height= 0,
            )

        else:
            print("No object found matching the criteria")

        time.sleep(20)
        return

    def reset_task(self):
        return

    def startup(self):
        return

    def pick_and_place_object(
        self,
        object_position: Tuple[float, float],
        object_height: float,
        target_height: float,
        target_position: List[float] = [0.5, 0.5],
    ):
        orders = [
            (OrderType.GRIPPER_OPEN, []),
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, object_position),
            (OrderType.MOVE_Z, [object_height]),
            (OrderType.GRIPPER_CLOSE, []),
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, target_position),
            (OrderType.MOVE_Z, [target_height]),
            (OrderType.GRIPPER_OPEN, []),
        ]
        self.queue_orders(orders, time_between_orders=self.time_between_orders)
