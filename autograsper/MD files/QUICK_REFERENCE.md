# Action Tracking Quick Reference

## One-Minute Overview

**What**: Track robot movements alongside recorded frames  
**Why**: Create annotated datasets for planar pushing and other behaviors  
**How**: Call `start_action()` before movement, `end_action()` after  
**Where**: In `perform_task()` and `reset_task()`  
**Output**: `states.json` (with action metadata) + `actions.json` (summary)

---

## Essential APIs

### Starting an Action
```python
action_id = self.start_action(
    action_type=ActionType.MOVE_XY,      # What type of movement
    phase=ActionPhase.TASK,              # or ActionPhase.RESET
    frame_index=self.get_current_frame_index(),
    is_planar_2d=True,                   # Planar pushing? True/False
    action_details={"x": 0.45, "y": 0.55},
    description="Push #1"
)
```

### Ending an Action
```python
self.end_action(action_id, self.get_current_frame_index())
```

### Getting Frame Index
```python
frame = self.get_current_frame_index()  # Returns int or None
```

---

## Action Types (Use These)

```python
ActionType.MOVE_XY          # Use for planar pushing
ActionType.MOVE_Z           # Vertical movement
ActionType.ROTATE           # Gripper rotation
ActionType.GRIPPER_OPEN     # Opening gripper
ActionType.GRIPPER_CLOSE    # Closing gripper
ActionType.SWEEP            # Wall sweep (in reset)
ActionType.OTHER            # Anything else
```

---

## Action Phases

```python
ActionPhase.TASK            # During perform_task()
ActionPhase.RESET           # During reset_task()
ActionPhase.STARTUP         # During initialization
ActionPhase.OTHER           # Misc
```

---

## Imports Needed

Add to top of random_push_grasper.py:
```python
from action_tracker import ActionType, ActionPhase
```

---

## Template: Tracking Pushes in perform_task()

```python
def perform_task(self):
    for i in range(self.N_pushes):
        x = np.random.uniform(self.manip_x[0], self.manip_x[1])
        y = np.random.uniform(self.manip_y[0], self.manip_y[1])
        
        # Before movement
        frame_start = self.get_current_frame_index()
        action_id = self.start_action(
            action_type=ActionType.MOVE_XY,
            phase=ActionPhase.TASK,
            frame_index=frame_start,
            is_planar_2d=True,
            action_details={"x": float(x), "y": float(y)},
            description=f"Push {i}"
        )
        
        # Execute movement
        self.robot.move_xy(x, y)
        time.sleep(1.2)
        
        # After movement
        frame_end = self.get_current_frame_index()
        self.end_action(action_id, frame_end)
```

---

## Template: Tracking Sweeps in reset_task()

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
            description=f"Sweep {wall.label}"
        )
        
        # Execute sweep
        self.sweep_wall(wall)
        
        frame_end = self.get_current_frame_index()
        self.end_action(action_id, frame_end)
```

---

## Returned Files (After Recording)

### states.json
```json
[
  {
    "frame_index": 42,
    "time": 1234567890.123,
    "state": {...},
    "action": {
      "action_id": 5,
      "action_type": "move_xy",
      "phase": "task",
      "start_frame": 40,
      "end_frame": 50,
      "is_planar_2d": true,
      "action_details": {"x": 0.45, "y": 0.55}
    }
  }
]
```

### actions.json
```json
{
  "total_actions": 10,
  "actions": [
    {
      "action_id": 0,
      "action_type": "move_xy",
      ...
    }
  ]
}
```

---

## Validation

After recording, verify outputs:
```bash
python validate_action_tracking.py /path/to/output
```

Should show:
- ✓ states.json loaded
- ✓ actions.json loaded
- ✓ Frame ranges valid
- ✓ ALL VALIDATIONS PASSED

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Frame indices are -1 | Called before recording | Call after 1st frame |
| Missing action field | get_current_frame_index() returns None | Check SharedState init |
| actions.json not created | Recording didn't complete | Ensure proper shutdown |
| Overlapping actions | Didn't call end_action() | Always pair start/end |

---

## Simple Testing

Minimal working example:
```python
def test_action_tracking(self):
    # Single test action
    start = self.get_current_frame_index()
    aid = self.start_action(
        ActionType.MOVE_XY, ActionPhase.TASK, start,
        action_details={"x": 0.5, "y": 0.5}
    )
    self.robot.move_xy(0.5, 0.5)
    end = self.get_current_frame_index()
    self.end_action(aid, end)
    
    # Check output files created/updated
```

---

## Integration Checklist

- [ ] Add imports: `from action_tracker import ActionType, ActionPhase`
- [ ] Wrap pushes in perform_task() with start/end action
- [ ] Set is_planar_2d=True for planar movements
- [ ] Wrap sweeps in reset_task() with start/end action
- [ ] Set phase=ActionPhase.RESET for reset sweeps
- [ ] Test with one recording session
- [ ] Verify states.json has "action" field
- [ ] Verify actions.json is created
- [ ] Use validate_action_tracking.py to check
- [ ] Check frame ranges are sensible (usually 10-50 frames per action)

---

## Key Files to Read

1. **ACTION_TRACKING_README.md** - Full documentation
2. **ACTION_TRACKING_IMPLEMENTATION.md** - RandomPushGrasper-specific
3. **ARCHITECTURE_SUMMARY.md** - How it all fits together
4. This file - Quick reference while coding

---

## Pro Tips

1. **Always pair** start_action() with end_action()
2. **Call right before** movement starts, **call right after** it ends
3. **Set is_planar_2d=True** only for XY movements at grasp_height
4. **Include details** in action_details for later dataset extraction
5. **Test early** with single action before doing full integration

---

**Status**: System ready to use  
**Last Updated**: 2026-02-11
