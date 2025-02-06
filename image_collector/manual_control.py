import cv2
import numpy as np
import threading
import time
import random
import sys
import os
from pynput import keyboard  # Replace 'keyboard' with 'pynput'
from dotenv import load_dotenv
from cv2displayer import update_images

# Ensure the project root is in the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from client.cloudgripper_client import GripperRobot

# Load environment variables
load_dotenv()

# Get the CloudGripper API token from environment variables
token = os.getenv("CLOUDGRIPPER_TOKEN")

# Create a GripperRobot instance
robotName = "robot24"
robot = GripperRobot(robotName, token)

# Shared state
running = True
lock = threading.Lock()  # Ensure thread safety

# Function to update camera feed
def update_camera():
    global running, current_config
    while running:
        with lock:
            # Get images from both cameras
            image_top, image_base, state, time_state = robot.get_all_states()  # Get new images and robot configuration - if doesn't work, switch to separate calls
            time.sleep(0.1)
            # image_top, _ = robot.get_image_top()
            # time.sleep(0.5)
            # image_base, _ = robot.get_image_base()
            # time.sleep(0.5) 

        # Convert images to NumPy arrays for OpenCV
        img_top = np.array(image_top)
        img_base = np.array(image_base)
        img_base = np.transpose(img_base, (1, 0, 2))  # Rotate base image

        # Display images
        update_images([img_top, img_base], window_name="Robot Cameras")  # Display images side by side

        # Print current configuration
        prev_config = current_config
        current_config = list(state.values())[:5]  # x, y, z, rotation, gripper
        if current_config != prev_config:
            print(f"Current configuration: {[ '%.2f' % elem for elem in current_config]}")
        # Check if window is closed
        if cv2.waitKey(1) == 27:  # Escape key to exit
            running = False
            break

# Function to handle keyboard input
def on_press(key):
    global running, current_config
    step_size = 0.025  # Step size for robot movement
    try:
        if key.char == "a":  # Move left (X-)
            current_config[0] -= step_size
            robot.move_xy(current_config[0], current_config[1])
        elif key.char == "d":  # Move right (X+)
            current_config[0] += step_size
            robot.move_xy(current_config[0], current_config[1])
        elif key.char == "w":  # Move forward (Y+)
            current_config[1] += step_size
            robot.move_xy(current_config[0], current_config[1])
        elif key.char == "s":  # Move backward (Y-)
            current_config[1] -= step_size
            robot.move_xy(current_config[0], current_config[1])
        elif key.char == "z":  # Move down (Z-)
            current_config[2] -= step_size
            robot.move_z(current_config[2])
        elif key.char == "c":  # Move up (Z+)
            current_config[2] += step_size
            robot.move_z(current_config[2])
        elif key.char == "q":  # Rotate counterclockwise
            current_config[3] -= 10
            robot.rotate(current_config[3])
        elif key.char == "e":  # Rotate clockwise
            current_config[3] += 10
            robot.rotate(current_config[3])
        elif key.char == "x":  # Toggle gripper
            if current_config[4]<0.5: 
                robot.gripper_open()
            else:
                robot.gripper_close()
        elif key.char == "p":  # Stop robot
            running = False
            return False  # Stop listener
        
        # print(f"Current configuration: {current_config}")
        time.sleep(0.1)  # Delay to prevent rapid key presses
    except AttributeError:
        pass  # Handle non-character keys safely

def on_release(key):
    global running
    if key == keyboard.Key.backspace:  # Quit on backspace
        running = False
        return False  # Stop listener

# Get initial robot state
state = robot.get_state()
current_config = list(state[0].values())[:5]  # x, y, z, rotation, gripper
print(f"Current configuration: {current_config}")
# Start camera thread
camera_thread = threading.Thread(target=update_camera)
camera_thread.start()

# Start keyboard listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()  # Keep listening until backspace is pressed

# Cleanup
running = False
camera_thread.join()
cv2.destroyAllWindows()