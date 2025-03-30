# CloudGripper API

Welcome to the CloudGripper API! This project provides a Python interface to remotely interact with the CloudGripper robot.

## Table of Contents

1. [Introduction](#introduction)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methods](#methods)
6. [Color Picker](#color-picker)
7. [Utilities](#utilities)
8. [Example Projects](#example-projects)
   - [Autograsper](#example-project-autograsper)
   - [Recorder](#example-project-recorder)
   - [Large Scale Data Collection](#example-project-large-scale-data-collection)

## Introduction

The CloudGripper API is designed to facilitate communication with the CloudGripper robot. The client includes functions to control the robot's movements, operate its gripper, retrieve images from its cameras, perform color calibration, and manage orders for the robot.

## Installation

To install the CloudGripper API Client, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/cloudgripper-client.git
cd cloudgripper-client

# Install the required dependencies
pip install -r requirements.txt
```

## Usage

Here's an example of how to use the CloudGripper API Client to control a robot:

```python
from client.cloudgripper_client import GripperRobot

# Initialize the robot client
robot = GripperRobot(name='robot1', token='your_api_token_here')

# Get the robot's current state
state, timestamp = robot.get_state()
print(f"State: {state}, Timestamp: {timestamp}")

# Move the robot
robot.step_forward()
robot.step_backward()
robot.step_left()
robot.step_right()

# Operate the gripper
robot.gripper_open()
robot.gripper_close()

# Retrieve images
base_image, base_timestamp, _ = robot.get_image_base()
top_image, top_timestamp = robot.get_image_top()
```

## Methods

### Initialization

```python
robot = GripperRobot(name='robot1', token='your_api_token_here')
```

### Basic Movements

- `step_forward()`: Move the robot one step forward.
- `step_backward()`: Move the robot one step backward.
- `step_left()`: Move the robot one step to the left.
- `step_right()`: Move the robot one step to the right.

### Gripper Operations

- `gripper_open()`: Open the robot's gripper.
- `gripper_close()`: Close the robot's gripper.
- `move_gripper(angle)`: Move the gripper to a specific angle.

### Rotations and Movements

- `rotate(angle)`: Rotate the robot to a specified angle.
- `move_z(z)`: Move the robot along the Z-axis.
- `move_xy(x, y)`: Move the robot along the X and Y axes.

### Image Retrieval

- `get_image_base()`: Get the base image from the robot's camera.
- `get_image_top()`: Get the top image from the robot's camera.
- `get_all_states()`: Get the combined state and images from the robot.

### State Retrieval

- `get_state()`: Get the current state of the robot.
- `calibrate()`: Calibrate the robot.

## Color-Picker

The `rgb_color_picker.py` script provides functionality for color calibration and object tracking within images.

### Usage

You can use the script from the command line as follows:

```bash
python library/rgb_color_picker.py <image_file> <colors>
```

Example:

```bash
python library/rgb_color_picker.py sample_image.jpg red green orange
```

### Functions

#### `test_calibration(image, colors)`

Tests the calibration of specified colors in the given image.

#### `all_objects_are_visible(objects, image, DEBUG=False)`

Checks if all specified objects are visible in the image.

#### `object_tracking(image, color="red", size_threshold=290, DEBUG=False, debug_image_path="debug_image.png")`

Tracks objects of a specified color in the image and returns their positions.

### Example

```python
import cv2
from library.rgb_color_picker import object_tracking

# Load the image
image = cv2.imread("sample_image.jpg")

# Track red objects
position = object_tracking(image, color="red", DEBUG=True, debug_image_path="debug_red.png")
print("Position of red object:", position)
```

## Utilities

The `utils.py` script provides a set of utility functions for managing robot orders, executing complex sequences, and handling image data.

### Functions

#### `write_order(output_dir: str, start_time: float, previous_order: Optional[Tuple[Any, List[float]]] = None)`

Save the previous order to the `orders.json` file.

#### `execute_order(robot: GripperRobot, order: Tuple[OrderType, List[float]], output_dir: str, reverse_xy: bool = False)`

Execute a single order on the robot and save its state.

#### `queue_orders(robot: GripperRobot, order_list: List[Tuple[OrderType, List[float]]], time_between_orders: float, output_dir: str = "", reverse_xy: bool = False)`

Queue a list of orders for the robot to execute sequentially and save state after each order.

#### `queue_orders_with_input(robot: GripperRobot, order_list: List[Tuple[OrderType, List[float]]], output_dir: str = "", start_time: float = -1.0)`

Queue a list of orders for the robot to execute sequentially, waiting for user input between each command, and save state after each order.

#### `snowflake_sweep(robot: GripperRobot)`

Perform a snowflake sweep pattern with the robot.

#### `sweep_straight(robot: GripperRobot)`

Perform a straight sweep pattern with the robot.

#### `recover_gripper(robot: GripperRobot)`

Recover the gripper by fully opening and then closing it.

#### `generate_position_grid() -> np.ndarray`

Generate a grid of positions.

#### `pick_random_positions(position_bank: np.ndarray, n_layers: int, object_size: float, avoid_positions: Optional[List[np.ndarray]] = None) -> List[np.ndarray]`

Pick random positions from the position bank ensuring they are spaced apart by object_size.

#### `get_undistorted_bottom_image(robot: GripperRobot, m: np.ndarray, d: np.ndarray) -> np.ndarray`

Get an undistorted image from the robot's camera.

#### `convert_ndarray_to_list(obj: Any) -> Any`

Convert a numpy ndarray to a Python list.

### Example

```python
from client.cloudgripper_client import GripperRobot
from library.utils import generate_position_grid, pick_random_positions

# Initialize the robot client
robot = GripperRobot(name='robot1', token='your_api_token_here')

# Generate a grid of positions
position_grid = generate_position_grid()

# Pick random positions ensuring minimum distance
positions = pick_random_positions(position_grid, n_layers=5, object_size=0.1)
print("Selected positions:", positions)
```

## Example-projects

### Example Project: Autograsper

See `autograsper/README.md` for more details



---

For more detailed documentation and examples, please refer to the source code and docstrings provided in each method. If you encounter any issues or have questions, feel free to open an issue on GitHub.
