# Cloudgripper-Autograsper

  
Cloudgripper-Autograsper is a framework for orchestrating robotic grasping tasks, controlling a CloudGripper robot, and recording data (images, states) from the robot's cameras.


---

  

## Table of Contents

  

1. [Main Components](#main-components)

- [Coordinator](#coordinatorpy)

- [Recording](#recordingpy)

- [Autograsper Base](#grasperpy)

- [Example Grasper](#example_grasperpy)

2. [Prerequisites and Installation](#prerequisites-and-installation)

3. [Configuration](#configuration)
  

---

  

  

## Main Components

  

Below is a summary of the core scripts and their roles. 

Detailed documentation of code in the: `library/`, `client/`, and `custom_graspers/` folders is under development, for now you can take a look in the files directly.

  

### `coordinator.py`

**Key Responsibilities**:

- Manages threads for the grasper (autograsper logic) and recorder.

- Coordinates shared state between threads.

- Handles global error events.

- **Customization**: Generally only changed to specify which custom grasper to use. You should not have to change this code other than which grasper is imported and initialized (unless you find general improvements or bug fixes!).


**Core Classes / Structures**:

- `SharedState`: A dataclass holding references to the current robot activity state and the recorder.

- `GrasperRecorderCoordinator`: The main manager. Initializes the Autograsper, sets up recording, and handles lifecycle events.

  
### `recording.py`

**Key Responsibilities**:

- Encapsulates logic for recording data from the robot.

- Handles camera capture (top and bottom camera), encoding, and saving to disk.

- Records additional robot state info into JSON files.

  
### `grasper.py`

**Key Responsibilities**:

- Defines the base class (`AutograsperBase`) for any custom grasper logic.

- Contains shared utility for robot movement (such as `queue_robot_orders`).

- Manages an enum `RobotActivity` to define states like `STARTUP`, `ACTIVE`, `RESETTING`, and `FINISHED`.

  

**Class**:

- `AutograsperBase`: **Abstract**. Requires subclasses to implement:

- `recover_after_fail()`

- `perform_task()`

- `reset_task()`

- `startup()`

- `go_to_start()`

  

Subclasses must provide the actual logic for picking, placing, or any manipulations needed.

  

### `example_grasper.py`

**Key Responsibilities**:

- Demonstrates a simple pick-and-place task flow.

- Provides a reference for how to implement a custom grasper by extending `AutograsperBase`.

- Example tasks include:

- Moving the robot to a start position.

- Picking an object of a specified color.

- Placing it at a target position.

- Handling success/failure states.

  

---

  

## Prerequisites and Installation

### Option 1: Devcontainer (recommended)

**Prerequisite: Docker**

Using devcontainers to run the code is recommended due to its reproducibility. If all share identical environments, a whole swathe of error reasons can be ruled out when troubleshooting.

Devcontainers are well integrated in **VS Code** (possibly other IDEs) and can be set up in **NVIM** as well.

##### VS Code
Upon opening the project you will be prompted if you wish to reopen it in a devcontainer since a `.devcontainer` folder is present in the root directory. Accept and wait (possibly a few minutes the first time) while the container builds.

[Tutorial if needed](https://code.visualstudio.com/docs/devcontainers/tutorial)

  After the devcontainer is built, run:
`pip install -r requirements.txt`
inside the devcontainer terminal.
### Option 2: Virtual environment

1. **Python 3.7+** (Recommended 3.8 or higher).
2. **pip** for Python package management.

**Step-by-step**:

```bash

# 1. Clone the repository

git clone https://github.com/YourOrganization/Cloudgripper-Autograsper.git

cd Cloudgripper-Autograsper

  

# 2. Create a virtual environment (optional but recommended)

python -m venv venv

source venv/bin/activate # Linux/Mac

# OR for Windows: venv\Scripts\activate

  

# 3. Install required packages

pip install -r requirements.txt
```


## Configuration

Common system configuration can be performed in the `autograsper/config.ini` file. For more custom configuration of recording and coordination, changes have to be performed to their respective classes.

#### **The following parameters are easily configured:**

- **`timeout_between_experiments`**: Optional. The time (in seconds) to wait between experiments. Useful if the API call limit is hit frequently.
- **`default_action_delay`**: The delay (in seconds) between sequential actions to ensure the robot can properly execute each command before the next is given.
- **`robot_idx`**: The identifier for the Cloudgripper robot you intend to use.
- **`name`**: A descriptive name for the current experiment. This will be the folder name of the recorded data, useful for splitting up experiments.

#### Task specific configurations 
You can also put task specific settings in this file. The example file holds information about two blocks and some positions:

- **`colors`**: A list of object colors used in the experiment.
- **`block_heights`**: A list of heights for each object/block to be grasped.
- **`position_bank`**: A set of pre-defined positions where objects may be placed.
- **`stack_position`**: The target position for stacking objects.
- **`object_size`**: The size of the objects to be grasped.

#### Camera settings

- **`record`**: Whether to enable recording of video data (`True` or `False`). When set to false, real time continuous viewing of the robot bottom camera will still be active, however no data will be stored on file. 
- **`fps`**: The frame rate (frames per second) at which camera and robot state data are queried.
- **`m`**: The camera's intrinsic matrix, used for calibrating image capture. **Should not be changed**.
- **`d`**: The distortion coefficients for correcting lens distortion. **Should not be changed**.
- **`clip_length`**: This setting is optional. If set to a positive integer `x`, the `recorder` module will split the video data into files containing `x` frames. This could be useful when performing a very long (possibly endless) task and not wanting to store the whole video in memory.


## Configuring Task and Gripper Behavior

To define a custom task or change the robot's behavior, you must implement a new class that extends the abstract base class `AutograsperBase`. This allows you to customize task logic such as picking, placing, and resetting objects during experiments.

### Steps to Define a Custom Grasper

1. **Create a new file** in the `custom_graspers/` directory (e.g., `my_custom_grasper.py`).
2. **Implement the required methods** by extending `AutograsperBase`. For example:

```python
from grasper import AutograsperBase, RobotActivity
from library.utils import OrderType, get_object_pos, queue_orders
from typing import List, Tuple

class MyCustomGrasper(AutograsperBase):

   # this example is missing some details, for a working example see custom_graspers/example_grasper.py
    
    def startup(self):
        """Initialize anything needed for the task, e.g., calibration."""
        print("Starting up custom task.")

    def go_to_start(self):
        """Move the robot to a safe start position."""
        orders = [
            (OrderType.MOVE_Z, [1]), 
            (OrderType.MOVE_XY, [0, 0.7]),
            (OrderType.GRIPPER_CLOSE, [])
        ]
        self.queue_robot_orders(orders)

    def perform_task(self):
        """Define the core task logic, e.g., pick and place."""
        object_position = get_object_pos(self.bottom_image, self.robot_idx, "green")
        target_position = [0.5, 0.5]

        # Generate a series of orders to pick and place the object
        orders = [
            (OrderType.GRIPPER_OPEN, []),
            (OrderType.MOVE_XY, object_position),
            (OrderType.MOVE_Z, [0.1]),
            (OrderType.GRIPPER_CLOSE, []),
            (OrderType.MOVE_Z, [1]),
            (OrderType.MOVE_XY, target_position),
            (OrderType.GRIPPER_OPEN, []),
        ]

        self.queue_robot_orders(orders)

    def recover_after_fail(self):
        """Handle errors and move the robot to a safe state."""
        print("Recovering from task failure...")

    def reset_task(self):
        """Reset the task environment to prepare for the next run."""
        self.go_to_start()

```

1. **Update the Coordinator** (`coordinator.py`) to use your new grasper class. Modify the `_initialize_autograsper` method to return an instance of your custom class:
```python
from custom_graspers.my_custom_grasper import MyCustomGrasper

def _initialize_autograsper(self) -> AutograsperBase:
    return MyCustomGrasper(self.config)
```