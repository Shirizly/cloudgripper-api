from client.cloudgripper_client import GripperRobot
import time
import sys
import cv2
import os
from dotenv import load_dotenv
import numpy as np
import logging
import random
from datetime import datetime

# some settings for the session
save_img_top = True
save_img_base = True


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the directory structure
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
recordings_dir = os.path.join(base_dir, "recordings")
os.makedirs(recordings_dir, exist_ok=True)

# Create a unique directory for the sequence
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sequence_dir = os.path.join(recordings_dir, f"sequence_{timestamp}")
os.makedirs(sequence_dir)

# File to store image metadata (image file name and robot configuration)
log_file = os.path.join(sequence_dir, "image_log.txt")

# Load environment variables
load_dotenv()

# Get the CloudGripper API token from environment variables
token = os.getenv("CLOUDGRIPPER_TOKEN")

# Create a GripperRobot instance
robotName = "robot24"
robot = GripperRobot(robotName, token)

# Function to display multiple images using OpenCV

def display_images(images, window_name="Robot Images"):
    # Get dimensions of the first image
    height, width = images[0].shape[:2]

    # Resize all images to the same dimensions
    resized_images = [cv2.resize(image, (width, height)) for image in images]

    # Concatenate images horizontally
    concatenated_image = np.concatenate(resized_images, axis=1)

    # Display the image
    cv2.imshow(window_name, concatenated_image)
    while True:
    # Check if the window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed!")
            break

        # Press any key to exit manually
        key = cv2.waitKey(100)  # Wait for a key press (100ms delay)
        if key != -1:  # Any key except -1 (no key press)
            print("Quit key pressed!")
            break

    cv2.destroyAllWindows()


# List to store images for display
images = []

img_base, timestamp = robot.get_image_base()
img_top, timestamp = robot.get_image_top()


# Convert images to proper format and append to list
images.append(cv2.cvtColor(np.array(img_base), cv2.COLOR_RGB2BGR))
images.append(cv2.cvtColor(np.array(img_top), cv2.COLOR_RGB2BGR))

# initialize robot position
def initialize_robot_configuration(robot):
    actions = [
        lambda: robot.move_z(0.8),
        lambda: robot.gripper_close(),
        lambda: robot.rotate(0),
        lambda: robot.move_xy(0.5, 0.5),
    ]
    for action in actions:
        action()
        time.sleep(1)

# Get initial images
initialize_robot_configuration(robot)
img_base, timestamp = robot.get_image_base()
img_top, timestamp = robot.get_image_top()


# Convert images to proper format and append to list
images.append(cv2.cvtColor(np.array(img_base), cv2.COLOR_RGB2BGR))
images.append(cv2.cvtColor(np.array(img_top), cv2.COLOR_RGB2BGR))

# generate grid for robot to move, or load from memory if exists
import pickle
file_path = "recordings/sub_grids.txt" # File path for sub_grids storage

# Check if the file exists
if os.path.exists(file_path):
    # Load sub_grids from file
    with open(file_path, "rb") as f:
        sub_grids = pickle.load(f)
    
    if sub_grids:
        sub_grid = sub_grids.pop(0)  # Take the first sub-grid
    
        if sub_grids:
            with open(file_path, "wb") as f:
                pickle.dump(sub_grids, f)  # Update the file
        else:
            os.remove(file_path)  # Delete file if no sub-grids left
else:
    # Generate sub_grids and store them in the file
    from grid_gen import generate_modular_nd_grid_random_order
    bounds = [(0.1, 0.9), (0.1, 0.9),(0.3, 0.9)]  # Define the space
    base_resolution = 4
    N = 1  # Number of sub-grids
    sub_grids = generate_modular_nd_grid_random_order(bounds, base_resolution, N, seed=42)
    sub_grid = sub_grids.pop(0)  # Take the first sub-grid
    
    with open(file_path, "wb") as f:
        pickle.dump(sub_grids, f)




# Open the log file for writing
with open(log_file, "w") as file:


    # Execute actions and capture images after each action

    for img_count, position in enumerate(sub_grid):
        
        robot.move_xy(position[0].item(),position[1].item()) # Perform the action
        time.sleep(1)  # Wait a bit for the action to complete
        robot.move_z(position[2].item())
        # robot.rotate(position[3])
        if random.random() > 0.5:
            robot.gripper_close()
        else:    
            robot.gripper_open()
        time.sleep(1)  # Wait a bit for the action to complete
        # image_top, image_base, state, time_state = robot.get_all_states()  # Get new images and robot configuration
        image_top = robot.get_image_top()
        image_base = robot.get_image_base()
        state = robot.get_state()
        current_config = list(state[0].values())[:5]
        img_top_name = f"image_top_{img_count:04d}.png"
        img_base_name = f"image_base_{img_count:04d}.png"
        img_top_path = os.path.join(sequence_dir, img_top_name)
        img_base_path = os.path.join(sequence_dir, img_base_name)

        # Save the image
        if save_img_top:
            cv2.imwrite(img_top_path, cv2.cvtColor(np.array(img_top), cv2.COLOR_RGB2BGR))
        if save_img_base:
            cv2.imwrite(img_base_path, cv2.cvtColor(np.array(img_base), cv2.COLOR_RGB2BGR))    
        
        # Write image metadata
        file.write(f"{img_top_name} {state}\n")

        # if you want to show the images
        # images.append(cv2.cvtColor(np.array(img_base), cv2.COLOR_RGB2BGR))

        print(f"image {img_count} out of {len(sub_grids[0])} taken")

# Display all images
# display_images(images)
