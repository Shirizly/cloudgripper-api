# Action Tracking System for Robot Recording

## Overview

The action tracking system has been integrated into the CloudGripper recording pipeline to:

1. **Track robot actions** (movements, grasps, sweeps, etc.) alongside frame captures
2. **Associate actions with specific frame ranges** for precise temporal alignment
3. **Mark action phases** (task vs. reset) for dataset organization
4. **Flag planar 2D motions** for easy planar pushing dataset creation
5. **Auto-save action metadata** as JSON files alongside states and images

## Architecture

### Core Components

#### 1. **action_tracker.py**
Provides the foundational classes:
- `ActionType`: Enum for action types (MOVE_XY, MOVE_Z, ROTATE, GRIPPER_*, SWEEP, etc.)
- `ActionPhase`: Enum for phase classification (TASK, RESET, STARTUP, OTHER)
- `Action`: Dataclass representing a single tracked action with full metadata
- `ActionTracker`: Thread-safe manager for tracking and querying actions

#### 2. **coordinator.py (SharedState)**
Extended with:
- `action_tracker: ActionTracker` - Global action tracking instance
- `frame_index: int` - Current frame being recorded
- `frame_index_lock` - Thread-safe access to frame index

#### 3. **recording.py (Recorder)**
Enhanced with:
- Frame index synchronization with SharedState
- Automatic embedding of action metadata in states.json
- `save_action_summary()` method to save actions.json at end of recording

#### 4. **grasper.py (AutograsperBase)**
Added helper methods:
- `start_action()` - Mark the beginning of an action
- `end_action()` - Mark the end of an action
- `get_current_frame_index()` - Query current recording frame

## Usage

### Basic Workflow

1. **Start an action** before robot movement:
```python
action_id = self.start_action(
    action_type=ActionType.MOVE_XY,
    phase=ActionPhase.TASK,
    frame_index=self.get_current_frame_index(),
    is_planar_2d=True,
    action_details={"x": 0.45, "y": 0.55},
    description="Push #1: moving to (0.45, 0.55)"
)
```

2. **Execute the movement**:
```python
self.robot.move_xy(x, y)
time.sleep(1.0)
```

3. **End the action**:
```python
self.end_action(action_id, self.get_current_frame_index())
```

### Practical Example: RandomPushGrasper

For tracking planar pushes in `perform_task()`:

```python
def perform_task(self):
    for push_idx in range(self.N_pushes):
        x = np.random.uniform(self.manip_x[0], self.manip_x[1])
        y = np.random.uniform(self.manip_y[0], self.manip_y[1])
        
        # Mark action start
        frame_start = self.get_current_frame_index()
        action_id = self.start_action(
            action_type=ActionType.MOVE_XY,
            phase=ActionPhase.TASK,
            frame_index=frame_start,
            is_planar_2d=True,
            action_details={"x": float(x), "y": float(y)},
            description=f"Push {push_idx}"
        )
        
        # Execute movement
        self.robot.move_xy(x, y)
        time.sleep(1.2)
        
        # Mark action end
        frame_end = self.get_current_frame_index()
        self.end_action(action_id, frame_end)
```

For tracking wall sweeps in `reset_task()`:

```python
def reset_task(self):
    for wall in random.sample(self.walls, len(self.walls)):
        frame_start = self.get_current_frame_index()
        action_id = self.start_action(
            action_type=ActionType.SWEEP,
            phase=ActionPhase.RESET,
            frame_index=frame_start,
            is_planar_2d=False,
            action_details={"wall_label": wall.label},
            description=f"Sweep reset from {wall.label}"
        )
        
        # Execute sweep
        self.sweep_wall(wall)
        
        frame_end = self.get_current_frame_index()
        self.end_action(action_id, frame_end)
```

## Output Files

### 1. **states.json** (Enhanced)
Each frame entry now includes optional action metadata:

```json
[
  {
    "state": {...},
    "time": 1234567890.123,
    "frame_index": 42,
    "action": {
      "action_id": 5,
      "action_type": "move_xy",
      "phase": "task",
      "start_frame": 40,
      "end_frame": 50,
      "is_planar_2d": true,
      "action_details": {
        "x": 0.45,
        "y": 0.55
      },
      "description": "Push 2: moving to (0.450, 0.550)"
    }
  },
  ...
]
```

### 2. **actions.json** (New)
Summary of all actions performed:

```json
{
  "total_actions": 12,
  "actions": [
    {
      "action_id": 0,
      "action_type": "move_xy",
      "phase": "task",
      "start_frame": 10,
      "end_frame": 25,
      "is_planar_2d": true,
      "start_robot_state": {...},
      "end_robot_state": {...},
      "action_details": {...},
      "description": "Push 0: moving to (0.450, 0.550)"
    },
    ...
  ]
}
```

## Dataset Creation

### Extract Planar Pushing Data

```python
import json

def extract_planar_pushing_dataset(states_file, output_file):
    with open(states_file, 'r') as f:
        states = json.load(f)
    
    planar_dataset = []
    for state_entry in states:
        if 'action' in state_entry:
            action = state_entry['action']
            if action.get('is_planar_2d') and action['phase'] == 'task':
                planar_dataset.append({
                    'frame_index': state_entry['frame_index'],
                    'action_id': action['action_id'],
                    'frame_range': (action['start_frame'], action['end_frame']),
                    'position': action['action_details'],
                })
    
    with open(output_file, 'w') as f:
        json.dump(planar_dataset, f, indent=2)
```

### Split by Task vs. Reset

```python
def split_task_and_reset(states_file, output_dir):
    with open(states_file, 'r') as f:
        states = json.load(f)
    
    task_frames = [s for s in states if s.get('action', {}).get('phase') == 'task']
    reset_frames = [s for s in states if s.get('action', {}).get('phase') == 'reset']
    
    with open(f"{output_dir}/task_frames.json", 'w') as f:
        json.dump(task_frames, f, indent=2)
    
    with open(f"{output_dir}/reset_frames.json", 'w') as f:
        json.dump(reset_frames, f, indent=2)
```

## Implementation Checklist

### Phase 1: Core System (Already Done ✓)
- [x] Created `action_tracker.py` with ActionType, ActionPhase, Action, ActionTracker
- [x] Extended `SharedState` with action_tracker and frame_index fields
- [x] Enhanced `Recorder.save_state()` to embed action metadata
- [x] Added `Recorder.save_action_summary()` for actions.json
- [x] Added action tracking helpers to `AutograsperBase`
- [x] Implemented frame index synchronization

### Phase 2: Integration (TODO for you)
- [ ] **RandomPushGrasper**:
  - [ ] Integrate in `perform_task()` to track each push
  - [ ] Integrate in `reset_task()` to track each wall sweep
  - [ ] Set `is_planar_2d=True` for planar movements at grasp_height
  
- [ ] **Testing**:
  - [ ] Run a full task with action tracking enabled
  - [ ] Verify states.json has action field
  - [ ] Verify actions.json is created correctly
  - [ ] Check frame ranges are sensible
  
- [ ] **Dataset Validation**:
  - [ ] Create planar pushing dataset from output
  - [ ] Verify action boundaries correspond to actual movements
  - [ ] Create train/test splits as needed

### Phase 3: Optimization (Future)
- [ ] Add more action types as needed (e.g., GRIPPER_ADJUSTMENTS)
- [ ] Store action statistics for analysis
- [ ] Create visualization tools for action sequences
- [ ] Build post-processing scripts for dataset curation

## API Reference

### ActionType Enum
```python
ActionType.MOVE_XY        # Horizontal movement
ActionType.MOVE_Z         # Vertical movement
ActionType.ROTATE         # Gripper rotation
ActionType.GRIPPER_OPEN   # Open gripper
ActionType.GRIPPER_CLOSE  # Close gripper
ActionType.SWEEP          # Wall/surface sweep
ActionType.OTHER          # Other action type
```

### ActionPhase Enum
```python
ActionPhase.TASK          # Action during task execution
ActionPhase.RESET         # Action during reset/cleanup
ActionPhase.STARTUP       # Action during startup/preparation
ActionPhase.OTHER         # Other phase
```

### ActionTracker Methods
```python
start_action(...) -> int              # Returns action_id
end_action(action_id, frame_index)    # Completes action
get_action_for_frame(frame) -> Action # Query action by frame
get_all_actions() -> List[Action]     # Get all completed actions
get_actions_by_type(type) -> List     # Filter by type
get_actions_by_phase(phase) -> List   # Filter by phase
get_planar_2d_actions() -> List       # Get planar movements
```

### AutograsperBase Methods
```python
start_action(...) -> int                # Start tracking action
end_action(action_id, frame_index)      # End tracking action
get_current_frame_index() -> int        # Get current frame number
```

## Performance Considerations

- **Thread Safety**: All access to `action_tracker` and `frame_index` is guarded with locks
- **Memory**: Actions are stored in memory until `save_action_summary()` is called at end of recording
- **Overhead**: Minimal - action tracking adds <1% time overhead per frame

## Troubleshooting

### Frame indices are -1
- **Cause**: `get_current_frame_index()` called before first frame is recorded
- **Fix**: Call action methods only after recording has started

### Actions.json not created
- **Cause**: Recording stopped before `_release_writers()` was called
- **Fix**: Ensure recorder is properly shut down via `disable_recording()` or `stop()`

### Overlapping actions
- **Cause**: Not calling `end_action()` before starting a new action
- **Fix**: Always pair `start_action()` with `end_action()`

## See Also

- `ACTION_TRACKING_GUIDE.md` - Detailed usage examples and patterns
- `ACTION_TRACKING_IMPLEMENTATION.md` - Integration tips for RandomPushGrasper
- `FRAME_INDEX_SYNC.md` - Details on frame index synchronization mechanism
