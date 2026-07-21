"""Tests for `autograsper.execution.actions` (Wave 4): `ActionTracker`'s parent/child nesting,
`get_action_for_frame`'s child-preference, `to_dict`/`from_dict` round-tripping `parent_id`, and
the completion-callback registry running outside the lock.
"""

from __future__ import annotations

import threading

from autograsper.execution.actions import Action, ActionPhase, ActionTracker, ActionType


def test_start_action_top_level_and_child_simultaneously():
    tracker = ActionTracker()
    top_id = tracker.start_action(ActionType.MOVE_XY, ActionPhase.TASK, start_frame=0)
    child_id = tracker.start_action(
        ActionType.MOVE_XY, ActionPhase.TASK, start_frame=1, parent_id=top_id
    )
    assert tracker.get_current_action().action_id == top_id
    assert tracker.get_current_child_action().action_id == child_id
    assert tracker.get_current_action().parent_id is None
    assert tracker.get_current_child_action().parent_id == top_id


def test_end_action_matches_child_before_top_level():
    tracker = ActionTracker()
    top_id = tracker.start_action(ActionType.SWEEP, ActionPhase.RESET, start_frame=0)
    child_id = tracker.start_action(
        ActionType.MOVE_XY, ActionPhase.RESET, start_frame=1, parent_id=top_id
    )
    completed_child = tracker.end_action(child_id, end_frame=2)
    assert completed_child.action_id == child_id
    assert completed_child.parent_id == top_id
    assert tracker.get_current_child_action() is None
    assert tracker.get_current_action().action_id == top_id  # top-level still open

    completed_top = tracker.end_action(top_id, end_frame=3)
    assert completed_top.action_id == top_id
    assert completed_top.parent_id is None
    assert tracker.get_current_action() is None

    all_actions = tracker.get_all_actions()
    assert [a.action_id for a in all_actions] == [child_id, top_id]


def test_end_action_unknown_id_returns_none_and_warns(caplog):
    tracker = ActionTracker()
    tracker.start_action(ActionType.MOVE_Z, ActionPhase.TASK, start_frame=0)
    with caplog.at_level("WARNING"):
        result = tracker.end_action(action_id=999, end_frame=1)
    assert result is None
    assert any("matches neither" in r.message for r in caplog.records)


def test_get_action_for_frame_prefers_deepest_in_flight_match():
    tracker = ActionTracker()
    top_id = tracker.start_action(ActionType.SWEEP, ActionPhase.RESET, start_frame=0)
    child_id = tracker.start_action(
        ActionType.MOVE_XY, ActionPhase.RESET, start_frame=2, parent_id=top_id
    )
    # frame 2 is within both the (open) top-level [0, None] and child [2, None] ranges.
    action = tracker.get_action_for_frame(2)
    assert action.action_id == child_id

    # frame 1 is only within the top-level range.
    action = tracker.get_action_for_frame(1)
    assert action.action_id == top_id


def test_get_action_for_frame_prefers_completed_child_over_completed_top_level():
    tracker = ActionTracker()
    top_id = tracker.start_action(ActionType.SWEEP, ActionPhase.RESET, start_frame=0)
    child_id = tracker.start_action(
        ActionType.MOVE_XY, ActionPhase.RESET, start_frame=2, parent_id=top_id
    )
    tracker.end_action(child_id, end_frame=4)
    tracker.end_action(top_id, end_frame=10)

    # frame 3 is inside both completed ranges (child [2,4], top [0,10]) -> prefer child.
    action = tracker.get_action_for_frame(3)
    assert action.action_id == child_id
    assert action.parent_id == top_id

    # frame 7 is only inside the top-level range.
    action = tracker.get_action_for_frame(7)
    assert action.action_id == top_id

    assert tracker.get_action_for_frame(100) is None


def test_to_dict_includes_parent_id_and_legacy_keys_unchanged():
    tracker = ActionTracker()
    top_id = tracker.start_action(
        ActionType.MOVE_XY,
        ActionPhase.TASK,
        start_frame=0,
        action_details={"x": 0.1, "y": 0.2},
        description="Push(...)",
    )
    child_id = tracker.start_action(
        ActionType.ROTATE, ActionPhase.TASK, start_frame=1, parent_id=top_id, action_details={"angle": 45}
    )
    tracker.end_action(child_id, end_frame=2)
    tracker.end_action(top_id, end_frame=3)

    data = tracker.to_dict()
    assert set(data.keys()) == {"actions", "current_action", "current_child_action"}
    child_dict, top_dict = data["actions"]

    legacy_keys = {
        "action_id",
        "action_type",
        "phase",
        "start_frame",
        "end_frame",
        "is_planar_2d",
        "start_robot_state",
        "end_robot_state",
        "action_details",
        "description",
        "extra_metadata",
    }
    assert legacy_keys.issubset(top_dict.keys())
    assert top_dict["parent_id"] is None
    assert child_dict["parent_id"] == top_id
    assert top_dict["action_type"] == "move_xy"
    assert child_dict["action_type"] == "rotate"


def test_from_dict_tolerates_missing_parent_id_key():
    # Simulates loading an actions.json row recorded before this wave (no "parent_id" key).
    legacy_row = {
        "action_id": 0,
        "action_type": "move_xy",
        "phase": "task",
        "start_frame": 0,
        "end_frame": 5,
        "is_planar_2d": True,
        "start_robot_state": None,
        "end_robot_state": None,
        "action_details": {"x": 0.5, "y": 0.5},
        "description": None,
        "extra_metadata": {},
    }
    action = Action.from_dict(legacy_row)
    assert action.parent_id is None
    assert action.action_type is ActionType.MOVE_XY


def test_from_dict_round_trips_parent_id():
    tracker = ActionTracker()
    top_id = tracker.start_action(ActionType.MOVE_XY, ActionPhase.TASK, start_frame=0)
    child_id = tracker.start_action(
        ActionType.MOVE_Z, ActionPhase.TASK, start_frame=1, parent_id=top_id
    )
    tracker.end_action(child_id, end_frame=2)
    child = tracker.get_all_actions()[0]
    restored = Action.from_dict(child.to_dict())
    assert restored.parent_id == top_id
    assert restored.action_type is ActionType.MOVE_Z


def test_completion_callback_invoked_outside_lock_and_receives_completed_action():
    tracker = ActionTracker()
    received = []
    reentrant_ok = threading.Event()

    def on_complete(action):
        received.append(action)
        # Prove the lock is not held during the callback: a reentrant call must not deadlock.
        with tracker.lock:
            reentrant_ok.set()

    tracker.register_completion_callback(on_complete)

    action_id = tracker.start_action(ActionType.MOVE_XY, ActionPhase.TASK, start_frame=0)
    completed = tracker.end_action(action_id, end_frame=1)

    assert reentrant_ok.is_set()
    assert len(received) == 1
    assert received[0] is completed


def test_completion_callback_exception_is_swallowed_and_logged(caplog):
    tracker = ActionTracker()

    def bad_callback(action):
        raise RuntimeError("boom")

    good_calls = []
    tracker.register_completion_callback(bad_callback)
    tracker.register_completion_callback(lambda a: good_calls.append(a))

    action_id = tracker.start_action(ActionType.MOVE_XY, ActionPhase.TASK, start_frame=0)
    with caplog.at_level("ERROR"):
        completed = tracker.end_action(action_id, end_frame=1)

    assert completed is not None
    assert len(good_calls) == 1  # the second callback still ran despite the first raising
    assert any("completion callback raised" in r.message for r in caplog.records)


def test_clear_resets_child_slot_too():
    tracker = ActionTracker()
    top_id = tracker.start_action(ActionType.MOVE_XY, ActionPhase.TASK, start_frame=0)
    tracker.start_action(ActionType.MOVE_Z, ActionPhase.TASK, start_frame=1, parent_id=top_id)
    tracker.clear()
    assert tracker.get_current_action() is None
    assert tracker.get_current_child_action() is None
    assert tracker.get_all_actions() == []
    assert tracker.next_action_id == 0
