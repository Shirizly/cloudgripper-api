import time
import sys
import cv2
import os
from dotenv import load_dotenv
import numpy as np
import logging
import random
from datetime import datetime
from session_log import session_initializer_new
from cv2displayer import display_images

# Assume these exist
from object_tracker import ShapeAwarePoseEstimator

def get_tool_position(pose_estimator=None,object_points=None):
    """Use vision to locate the tool."""
    image_top, _ = robot.get_image_top()
    image_base, _ = robot.get_image_base()
    if image_top is None or len(image_top) == 0:
        return None
    if image_base is None or len(image_base) == 0:
        return None
    if object_points is None and pose_estimator is None:
        return None
    if pose_estimator is None:
        pose_estimator = ShapeAwarePoseEstimator(object_points) # need to add the tool points
    tool_pos = pose_estimator.locate_tool(np.array(image_top), np.array(image_base))

    return tool_pos  # (x, y, z) or None

def get_free_space():
    """Use vision to find free space on the table."""
    image_top, _ = robot.get_image_top()
    image_base, _ = robot.get_image_base()
    if image_top is None or len(image_top) == 0:
        return None
    if image_base is None or len(image_base) == 0:
        return None

    
    return free_pos  # (x, y, z) or None

# Ensure the project root is in the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from client.cloudgripper_client import GripperRobot

# session settings
save_image_top = True
save_image_base = True

# grid/session init
bounds = [(0.1, 0.9), (0.1, 0.9), (0.3, 0.9)]  
base_resolution = 30
N = 30
sub_grid, log_file, sequence_dir = session_initializer_new(bounds, base_resolution, N)

# load env
load_dotenv()
token = os.getenv("CLOUDGRIPPER_TOKEN")

# robot setup
robotName = "robot24"
robot = GripperRobot(robotName, token)
state = robot.get_state()
current_config = list(state[0].values())[:5]  
print(f"Initial state: {current_config}")

# retry settings
max_retries = 5
base_wait_time = 2
sleep_time = 0.5

def grab_tool(robot):
    """Locate tool, move to it, and grab it."""
    tool_pos = get_tool_position()
    if tool_pos is None:
        raise RuntimeError("No tool found by locator.")
    print(f"Tool located at {tool_pos}")

    # approach tool
    robot.move_xy(tool_pos[0], tool_pos[1])
    time.sleep(sleep_time)
    robot.move_z(tool_pos[2] + 0.1)  # above tool
    time.sleep(sleep_time)
    robot.move_z(tool_pos[2])        # descend
    time.sleep(sleep_time)
    robot.gripper_close()
    time.sleep(sleep_time)
    return tool_pos

def move_tool_to_free_space(robot):
    """Find free space and move tool there."""
    free_pos = get_free_space()
    if free_pos is None:
        raise RuntimeError("No free space found by locator.")
    print(f"Free space at {free_pos}")

    robot.move_z(free_pos[2] + 0.1)  # lift
    time.sleep(sleep_time)
    robot.move_xy(free_pos[0], free_pos[1])
    time.sleep(sleep_time)
    robot.move_z(free_pos[2])
    time.sleep(sleep_time)
    return free_pos

def translate_random(robot, start_pos, step_size=0.1):
    """Move tool in a random translational direction."""
    angle = random.uniform(0, 2*np.pi)
    dx, dy = step_size*np.cos(angle), step_size*np.sin(angle)
    new_x, new_y = start_pos[0] + dx, start_pos[1] + dy

    print(f"Translating to {(new_x, new_y)}")
    robot.move_xy(new_x, new_y)
    time.sleep(sleep_time)
    return (new_x, new_y, start_pos[2])

# logging loop
with open(log_file, "a") as file:
    bias = base_resolution**len(bounds)/N - len(sub_grid)
    for image_count, position in enumerate(sub_grid):

        try:
            # 1. grab tool
            tool_pos = grab_tool(robot)

            # 2. move tool to free space
            free_pos = move_tool_to_free_space(robot)

            # 3. random translation move
            end_pos = translate_random(robot, free_pos)

            # 4. capture with retry/backoff
            retry_count = 0
            while retry_count < max_retries:
                image_top, image_base, state, time_state = robot.get_all_states()
                if (image_top is not None and len(image_top) > 0 and
                    image_base is not None and len(image_base) > 0 and
                    state is not None):
                    time.sleep(sleep_time)
                    break
                retry_count += 1
                wait_time = base_wait_time * (2 ** (retry_count-1))
                print(f"API request failed. Retrying {retry_count}/{max_retries} after {wait_time} sec...")
                time.sleep(wait_time)

            if retry_count == max_retries:
                print("Max retries reached.")
                break

            current_config = list(state.values())[:5]

            # save images
            image_top_name = f"image_top_{image_count+int(bias):04d}.png"
            image_base_name = f"image_base_{image_count+int(bias):04d}.png"
            image_top_path = os.path.join(sequence_dir, image_top_name)
            image_base_path = os.path.join(sequence_dir, image_base_name)

            if save_image_top:
                cv2.imwrite(image_top_path, image_top)
            if save_image_base:
                cv2.imwrite(image_base_path, image_base)

            # log metadata
            file.write(f"{image_top_name} {state}\n")

            print(f"image {image_count+1} out of {len(sub_grid)} taken")

        except Exception as e:
            print(f"Error in loop: {e}")
            continue

if image_count == len(sub_grid)-1:
    print("Session complete.")
    update_sub_grids()

