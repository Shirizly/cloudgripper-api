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
from queue import Queue
print("Starting manual control...", flush=True)
state_queue = Queue()
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

# video recording setup
save_video = True        # turn on/off saving
video_path = "robot_cam_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = None                # will initialize once we know frame size
frame_times = []          # store timestamps to estimate frame rate later

# Function to update camera feed
def update_camera():
    global running, current_config
    out = None
    while running:
        with lock:
            # Get images from both cameras
            data = robot.get_all_states()  # Get new images and robot configuration - if doesn't work, switch to separate calls
            time.sleep(0.15)
            image_top = data[0]
            image_base = data[1]
            state = data[2]
            # image_top, _ = robot.get_image_top()
            # time.sleep(0.5)
            # image_base, _ = robot.get_image_base()
            # time.sleep(0.5) 

        if image_top is None or image_base is None:
            print("Failed to retrieve images from robot.")
            continue

        # Convert images to NumPy arrays for OpenCV
        img_top = np.array(image_top)
        img_base = np.array(image_base)
        img_base = np.transpose(img_base, (1, 0, 2))  # Rotate base image

        # combine images side by side
        def combine_side_by_side(img_left, img_right, common_height=480):
            """
            Resize both images to the same height and concatenate horizontally.
            """
            # Compute aspect-preserving scaling
            hL, wL = img_left.shape[:2]
            hR, wR = img_right.shape[:2]

            scaleL = common_height / hL
            scaleR = common_height / hR

            new_wL = int(wL * scaleL)
            new_wR = int(wR * scaleR)

            # Resize both
            left_resized = cv2.resize(img_left, (new_wL, common_height))
            right_resized = cv2.resize(img_right, (new_wR, common_height))

            # Ensure same height
            if left_resized.shape[0] != right_resized.shape[0]:
                common_height = min(left_resized.shape[0], right_resized.shape[0])
                left_resized = cv2.resize(left_resized, (new_wL, common_height))
                right_resized = cv2.resize(right_resized, (new_wR, common_height))

            combined = np.hstack((left_resized, right_resized))
            return combined

        frame = combine_side_by_side(img_top, img_base, common_height=480)
        now = time.time()
        frame_times.append(now)
        # Initialize video writer if needed
        if save_video and out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))  
        # Display images
        update_images([img_top,img_base], window_name="Robot Cameras")  # Display images side by side
        # Print current configuration
        prev_config = current_config
        current_config = list(state.values())[:5]  # x, y, z, rotation, gripper
        if current_config != prev_config:
            # state_queue.put(current_config)
            # sys.stdout.write(f"Current configuration: {[ '%.2f' % elem for elem in current_config]}\n")
            # sys.stdout.flush()
            print(f"Current configuration: {[ '%.2f' % elem for elem in current_config]}", flush=True)
        
        # Write frame
        if save_video and out is not None:
            out.write(frame)

        # Check if window is closed
        if cv2.waitKey(1) == 27:  # Escape key to exit
            running = False
            break

# Function to handle keyboard input
def on_press(key):
    global running, current_config, step_size
    # step_size = 0.01  # Step size for robot movement
    # current_config = np.clip(current_config, [0,0,0,-180,0], [1,1,1,180,1])  # Ensure within bounds
    current_config[0] = max(0, min(1, current_config[0]))
    current_config[1] = max(0, min(1, current_config[1]))
    current_config[2] = max(0, min(1, current_config[2]))
    current_config[3] = max(-180, min(180, current_config[3]))
    current_config[4] = max(0, min(1, current_config[4]))
    try:
        if key.char == "a":  # Move left (X-)
            current_config[0] -= step_size
            current_config[0] = max(0, current_config[0])  # Prevent going below 0
            print(f"requesting move to {current_config[0]}, {current_config[1]}")
            robot.move_xy(current_config[0], current_config[1])
            # robot.step_left()
        elif key.char == "d":  # Move right (X+)
            current_config[0] += step_size
            current_config[0] = min(1, current_config[0])  # Prevent going above 1
            print(f"requesting move to {current_config[0]}, {current_config[1]}")
            robot.move_xy(current_config[0], current_config[1])
            # robot.step_right()
        elif key.char == "w":  # Move forward (Y+)
            current_config[1] += step_size
            current_config[1] = min(1, current_config[1])  # Prevent going above 1
            print(f"requesting move to {current_config[0]}, {current_config[1]}")
            robot.move_xy(current_config[0], current_config[1])
            # robot.step_forward()
        elif key.char == "s":  # Move backward (Y-)
            current_config[1] -= step_size
            current_config[1] = max(0, current_config[1])  # Prevent going below 0
            print(f"requesting move to {current_config[0]}, {current_config[1]}")
            robot.move_xy(current_config[0], current_config[1])
            # robot.step_backward()
        elif key.char == "z":  # Move down (Z-)
            current_config[2] -= step_size
            current_config[2] = max(0, current_config[2])  # Prevent going below 0
            robot.move_z(current_config[2])
        elif key.char == "c":  # Move up (Z+)
            current_config[2] += step_size
            current_config[2] = min(1, current_config[2])  # Prevent going above 1
            robot.move_z(current_config[2])
        elif key.char == "q":  # Rotate counterclockwise
            current_config[3] -= 1
            current_config[3] = max(-180, current_config[3])
            robot.rotate(current_config[3])
        elif key.char == "e":  # Rotate clockwise
            current_config[3] += 10
            current_config[3] = min(360, current_config[3])
            robot.rotate(current_config[3])
        elif key.char == "x":  # close gripper
            robot.gripper_close()
        elif key.char == "u":  # Open gripper fully
            robot.gripper_open()
        elif key.char == "j":  # Open gripper a little
            current_config[4] -= 0.1
            current_config[4] = max(0, current_config[4])
            robot.move_gripper(current_config[4])
        elif key.char == "k":  # Close gripper a little
            current_config[4] += 0.1
            current_config[4] = min(1, current_config[4])
            robot.move_gripper(current_config[4])
        elif key.char == "n": # increase step size
            step_size = step_size*2
        elif key.char == "m": # decrease step size
            step_size = step_size/2
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
# robot.calibrate()
state = robot.get_state()
# robot.move_xy(0.5, 0.5)  # Move to a safe starting position
time.sleep(1)
state = robot.get_state()
current_config = list(state[0].values())[:5]  # x, y, z, rotation, gripper
global step_size
step_size = 0.01  # Step size for robot movement
print(f"Current configuration: {[ '%.2f' % elem for elem in current_config]}\n", flush=True)
# Start camera thread and keyboard listener - new safety version
try:
    # Start camera + keyboard
    camera_thread = threading.Thread(target=update_camera, daemon=True)
    camera_thread.start()

    with keyboard.Listener(on_press=on_press) as listener: #, on_release=on_release
        listener.join()

except KeyboardInterrupt:
    print("Keyboard interrupt: stopping...")

finally:
    running = False
    if camera_thread.is_alive():
        camera_thread.join()
    cv2.destroyAllWindows()
    print("Cleanup done, exiting.")



# # Start camera thread - old version
# camera_thread = threading.Thread(target=update_camera)
# camera_thread.start()

# # Start keyboard listener
# with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
#     listener.join()  # Keep listening until backspace is pressed

# # Cleanup
# running = False
# camera_thread.join()
# cv2.destroyAllWindows()

