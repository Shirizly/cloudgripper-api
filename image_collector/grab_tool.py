import cv2
import numpy as np
import threading
import time
import sys
import os
from dotenv import load_dotenv
from cv2displayer import update_images
print("Starting tool grab", flush=True)
check_tool_grasp = True
# Ensure the project root is in the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from client.cloudgripper_client import GripperRobot
from object_tracker.tool_user_utils import analyze_tool_grip

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
video_path = "robot_grab_tool.mp4"

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
            data = robot.get_all_states()
            time.sleep(0.15)
            image_top = data[0]
            image_base = data[1]
            state = data[2]

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
        update_images([img_base, img_top], window_name="Robot Cameras")
        # Print current configuration
        prev_config = current_config
        current_config = list(state.values())[:5]  # x, y, z, rotation, gripper
        if current_config != prev_config:
            print(f"Current configuration: {[ '%.2f' % elem for elem in current_config]}", flush=True)
        
        # Write frame
        if save_video and out is not None:
            out.write(frame)

        # Check if window is closed
        if cv2.waitKey(1) == 27:  # Escape key to exit
            running = False
            break

# Get initial robot state
state = robot.get_state()
time.sleep(1)
state = robot.get_state()
current_config = list(state[0].values())[:5]  # x, y, z, rotation, gripper
print(f"Current configuration: {[ '%.2f' % elem for elem in current_config]}\n", flush=True)
robot_bias = {"robot24": 5.0, "robot15": -10.0}


def perform_grab_tool(robot: GripperRobot, check_tool_grasp=False,bias = 0.0) -> float:
    robot.move_z(1.0)
    time.sleep(1.0)
    robot.gripper_open()
    time.sleep(1.0)
    robot.rotate(bias)
    time.sleep(1.0)
    print("Moving to tool position...", flush=True)
    robot.move_xy(0.033, 0.461)
    time.sleep(1)
    robot.move_z(0.277)
    time.sleep(1)
    robot.gripper_close()

    try: 
        robot.move_z(1.0)
        time.sleep(1.0)
        robot.move_xy(0.5,0.41)
        time.sleep(1.0)
        robot.rotate(90)
        time.sleep(1.0)
        if check_tool_grasp:
            top_img, _ = robot.get_image_top()
            roi_cfg = {'x': [0.46, 0.58], 'y': [0.54, 0.63]}
            x_range = roi_cfg.get('x',[0.4,0.7])
            y_range = roi_cfg.get('y',[0.1,0.8])
            h, w, _ = top_img.shape
            x_min = int(x_range[0]*w)
            x_max = int(x_range[1]*w)
            y_min = int(y_range[0]*h)
            y_max = int(y_range[1]*h)
            roi_img = top_img[y_min:y_max, x_min:x_max]
            # Define color range for tool detection (example: white tool in BGR)
            lower_bgr = (160, 160, 120)
            upper_bgr = (190, 210, 190)
            if check_tool_grasp:
                grip_quality = analyze_tool_grip(roi_img, (lower_bgr, upper_bgr))
            else:
                grip_quality = 0.0
            return grip_quality
    except Exception as e:
        print(f"Error during tool grasping: {e}")
    return 0.0

# Move robot to specific position and grab tool
try:
    # Start camera thread to visualize and record
    time.sleep(1)
    camera_thread = threading.Thread(target=update_camera, daemon=True)
    camera_thread.start()
    grip_quality = perform_grab_tool(robot, check_tool_grasp=True, bias=robot_bias.get(robotName, 0.0))
    print(f"Tool grip quality: {grip_quality:.2f}")
    threshold = 0.55
    
    if grip_quality < threshold:
        print("Tool not detected properly in hand. Please adjust tool and restart.")
    else:
        print("Tool grip verified successfully!", flush=True)    
    

except KeyboardInterrupt:
    print("Keyboard interrupt: stopping...")

finally:
    running = False
    if 'camera_thread' in locals() and camera_thread.is_alive():
        camera_thread.join()
    cv2.destroyAllWindows()
    print("Cleanup done, exiting.")

