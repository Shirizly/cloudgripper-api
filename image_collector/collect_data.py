import time
import sys
import cv2
import os
from dotenv import load_dotenv
import numpy as np
import logging
import random
from datetime import datetime
from session_log import session_initializer,update_sub_grids
from cv2displayer import display_images

# Ensure the project root is in the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from client.cloudgripper_client import GripperRobot

# some settings for the session
save_image_top = True
save_image_base = True

bounds = [(0.1, 0.9), (0.1, 0.9),(0.3, 0.9)]  # Define the space
base_resolution = 30 # Base resolution of the grid
N = 30  # Number of sub-grids to generate

# load session variables from session_initializer.py
sub_grid, log_file, sequence_dir = session_initializer(bounds,base_resolution,N) # Initialize the session

# Load environment variables
load_dotenv()

# Get the CloudGripper API token from environment variables
token = os.getenv("CLOUDGRIPPER_TOKEN")

# Create a GripperRobot instance
robotName = "robot24"
robot = GripperRobot(robotName, token)
state = robot.get_state()
current_config = list(state[0].values())[:5]  # x, y, z, rotation, gripper
print(f"Initial state: {current_config}")
# some utility functions
def initialize_robot_configuration(robot): # function for initializing robot position
    actions = [
        lambda: robot.move_z(0.8),
        lambda: robot.gripper_close(),
        lambda: robot.rotate(0),
        lambda: robot.move_xy(0, 0),
    ]
    for action in actions:
        action()
        time.sleep(1)

def show_current_state(robot): # just easy way to check the current state of the robot for debugging
    image_top,  _ = robot.get_image_top()
    image_base, _ = robot.get_image_base()
    state = robot.get_state()
    print(f"state: {state}")
    cv2.imshow("image_top", np.array(image_top))
    cv2.imshow("image_base", np.array(image_base))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#setting for retries, because the robot sometimes fails to respond
max_retries = 5  # Maximum times to retry before giving up
base_wait_time = 2  # Initial wait time in seconds


sleep_time = 0.5

# Open the log file for writing
with open(log_file, "a") as file:
    # Execute actions and capture images after each action
    bias = base_resolution**len(bounds)/N - len(sub_grid)
    for image_count, position in enumerate(sub_grid):
        robot.move_xy(position[0].item(),position[1].item()) # Perform the action
        time.sleep(sleep_time)  # Wait a bit for the action to complete
        robot.move_z(position[2].item())
        time.sleep(sleep_time)
        robot.rotate(random.randint(0, 360))
        time.sleep(sleep_time)
        if random.random() > 0.5: # Randomly switch gripper state half the time (on average)
            if current_config[4] > 0.5: # Check if the gripper is open
                robot.gripper_close()
            else:
                robot.gripper_open()
            time.sleep(sleep_time)  # Wait a bit for the actions to complete

        retry_count = 0  # Track retry attempts
        while retry_count < max_retries:
            # Try to get data
            image_top, image_base, state, time_state = robot.get_all_states()  # Get new images and robot configuration - if doesn't work, switch to separate calls
            # image_top, _ = robot.get_image_top()
            # time.sleep(sleep_time)  # Wait a bit for the api to chill
            # image_base, _ = robot.get_image_base()
            # time.sleep(sleep_time)  # Wait a bit for the api to chill
            # state = robot.get_state()
            # time.sleep(sleep_time)  # Wait a bit for the api to chill
            
            # Check if any of the retrieved data is empty
            if (
                image_top is not None and len(image_top) > 0 and
                image_base is not None and len(image_base) > 0 and
                state is not None
            ):
                time.sleep(sleep_time)  # Wait a bit for the api to chill
                break  # Success, continue to the next iteration
            else:
                retry_count += 1
                wait_time = base_wait_time * (2 ** (retry_count-1))  # Exponential backoff
                print(f"API request failed. Retrying {retry_count}/{max_retries} after {wait_time} sec...")
                time.sleep(wait_time)

        if retry_count == max_retries:
            print("Max retries reached.")
            break

        current_config = list(state.values())[:5] # use this if get_all_states() is used
        # current_config = list(state[0].values())[:5]  # use this if get_state() is used

        image_top_name = f"image_top_{image_count+int(bias):04d}.png"
        image_base_name = f"image_base_{image_count+int(bias):04d}.png"
        image_top_path = os.path.join(sequence_dir, image_top_name)
        image_base_path = os.path.join(sequence_dir, image_base_name)

        
        # Save the image
        if save_image_top:
            cv2.imwrite(image_top_path, image_top)
        if save_image_base:
            cv2.imwrite(image_base_path, image_base)
        
        # Write image metadata
        file.write(f"{image_top_name} {state}\n")

        print(f"image {image_count+1} out of {len(sub_grid)} taken")

if image_count == len(sub_grid)-1:
    print("Session complete.")
    update_sub_grids()


