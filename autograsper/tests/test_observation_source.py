"""Tests for `autograsper.observation.source` (design 02 §3.2, Wave 2).

Uses `DryRunRobot` only (hard safety rule, CONVENTIONS.md). `camera` is a tiny pass-through fake
(duck-typed against `perception.frames.CameraPipeline`'s `process_bottom` method) so these tests
exercise `ObservationSource`'s threading/pub-sub contract independent of the actual undistort/
homography math, which `test_frames.py` covers separately.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from autograsper.hardware.dryrun import DryRunRobot
from autograsper.hardware.robot_interface import RawObservation
from autograsper.observation.source import ObservationSource


class _PassthroughCamera:
    """Fake `CameraPipeline`: returns the raw bottom image unchanged."""

    def process_bottom(self, raw_bottom):
        return raw_bottom


class _FailingRobot:
    """Fake robot whose `get_all_states()` always raises, for the failure-path test."""

    def get_all_states(self):
        raise RuntimeError("simulated robot failure")


@pytest.fixture
def shutdown_event():
    return threading.Event()


def _make_source(shutdown_event, fps=50.0, robot=None, **kwargs):
    robot = robot if robot is not None else DryRunRobot()
    return ObservationSource(robot, _PassthroughCamera(), fps, shutdown_event, **kwargs)


# --- basic lifecycle / monotonic seq -----------------------------------------


def test_latest_is_none_before_start(shutdown_event):
    source = _make_source(shutdown_event)
    assert source.latest() is None


def test_monotonic_seq_increases(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    try:
        obs1 = source.await_next(after_seq=0, timeout=2.0)
        assert obs1 is not None
        assert obs1.seq == 1

        obs2 = source.await_next(after_seq=obs1.seq, timeout=2.0)
        assert obs2 is not None
        assert obs2.seq == obs1.seq + 1
        assert obs2.timestamp >= obs1.timestamp
    finally:
        source.stop(timeout=2.0)


def test_stop_actually_stops_the_thread(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    assert source.is_running()
    source.await_next(after_seq=0, timeout=2.0)
    source.stop(timeout=2.0)
    assert not source.is_running()


# --- latest() / await_next() semantics ----------------------------------------


def test_await_next_wakes_promptly_on_new_observation(shutdown_event):
    source = _make_source(shutdown_event, fps=20.0)  # 50ms period
    source.start()
    try:
        first = source.await_next(after_seq=0, timeout=2.0)
        assert first is not None

        start = time.perf_counter()
        second = source.await_next(after_seq=first.seq, timeout=2.0)
        elapsed = time.perf_counter() - start

        assert second is not None
        assert second.seq > first.seq
        # Should wake within ~a couple of frame periods, not the full 2s timeout.
        assert elapsed < 1.0
    finally:
        source.stop(timeout=2.0)


def test_await_next_timeout_returns_none(shutdown_event):
    # Very low fps: the next observation would take ~10s, far past our short timeout.
    source = _make_source(shutdown_event, fps=0.1)
    source.start()
    try:
        first = source.await_next(after_seq=0, timeout=2.0)
        assert first is not None
        result = source.await_next(after_seq=first.seq, timeout=0.1)
        assert result is None
    finally:
        source.stop(timeout=1.0)


def test_latest_matches_most_recent_await_next_result(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    try:
        obs = source.await_next(after_seq=0, timeout=2.0)
        assert obs is not None
        assert source.latest().seq >= obs.seq
    finally:
        source.stop(timeout=2.0)


# --- subscriber latest-wins drop-oldest ----------------------------------------


def test_subscriber_drop_oldest_never_exceeds_maxsize(shutdown_event):
    source = _make_source(shutdown_event, fps=200.0)  # fast, to accumulate backlog
    sub = source.subscribe("test-subscriber", maxsize=2)
    source.start()
    try:
        # Let several cycles pass without draining the subscriber queue.
        time.sleep(0.2)
        assert sub.qsize() <= 2
    finally:
        source.stop(timeout=2.0)

    # After stopping, drain and check contents are valid, monotonic Observations, and the last
    # one drained is the most recent thing the source ever published (nothing "stuck" behind it).
    drained = []
    while True:
        item = sub.get(timeout=0.01)
        if item is None:
            break
        drained.append(item)
    assert len(drained) <= 2
    if len(drained) == 2:
        assert drained[0].seq < drained[1].seq
    assert source.latest() is not None
    if drained:
        assert drained[-1].seq <= source.latest().seq


def test_unsubscribe_stops_delivery(shutdown_event):
    source = _make_source(shutdown_event, fps=100.0)
    sub = source.subscribe("temp", maxsize=2)
    source.start()
    try:
        source.await_next(after_seq=0, timeout=2.0)
        source.unsubscribe("temp")
        # Drain whatever it already has.
        while sub.get(timeout=0.01) is not None:
            pass
        time.sleep(0.2)
        # Nothing new should have arrived after unsubscribing.
        assert sub.get(timeout=0.01) is None
    finally:
        source.stop(timeout=2.0)


# --- array writeability ---------------------------------------------------------


def test_observation_images_are_read_only(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    try:
        obs = source.await_next(after_seq=0, timeout=2.0)
        assert obs is not None
        assert obs.top_image.flags.writeable is False
        assert obs.bottom_image.flags.writeable is False
        with pytest.raises(ValueError):
            obs.top_image[0, 0, 0] = 1
        with pytest.raises(ValueError):
            obs.bottom_image[0, 0, 0] = 1
    finally:
        source.stop(timeout=2.0)


# --- frame_index provider ----------------------------------------------------------


def test_frame_index_provider_stamps_observation(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    counter = {"n": None}

    def provider():
        return counter["n"]

    source.set_frame_index_provider(provider)
    source.start()
    try:
        obs_before = source.await_next(after_seq=0, timeout=2.0)
        assert obs_before is not None
        assert obs_before.frame_index is None

        counter["n"] = 42
        obs_after = source.await_next(after_seq=obs_before.seq, timeout=2.0)
        assert obs_after is not None
        assert obs_after.frame_index == 42
    finally:
        source.stop(timeout=2.0)

    source2 = _make_source(shutdown_event, fps=50.0)
    source2.set_frame_index_provider(None)


def test_frame_index_none_when_no_provider_set(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    try:
        obs = source.await_next(after_seq=0, timeout=2.0)
        assert obs is not None
        assert obs.frame_index is None
    finally:
        source.stop(timeout=2.0)


# --- consecutive-failure path --------------------------------------------------------


def test_consecutive_failures_marks_source_failed_without_touching_shutdown_event(shutdown_event):
    source = _make_source(
        shutdown_event, fps=100.0, robot=_FailingRobot(), max_consecutive_failures=3
    )
    source.start()
    try:
        assert source.failed_event.wait(timeout=2.0), "source did not report failure in time"
        assert source.failed is True
        # Session layer owns shutdown_event; the source must never set it itself.
        assert not shutdown_event.is_set()
        # The poll thread should have exited on its own after giving up.
        source._thread.join(timeout=1.0)
        assert not source.is_running()
    finally:
        source.stop(timeout=1.0)


def test_missing_images_count_as_a_failure(shutdown_event):
    class _NoImagesRobot:
        def get_all_states(self):
            return RawObservation(
                top_image=None, bottom_image_raw=None, robot_state_dict={}, timestamp=0.0
            )

    source = _make_source(
        shutdown_event, fps=100.0, robot=_NoImagesRobot(), max_consecutive_failures=2
    )
    source.start()
    try:
        assert source.failed_event.wait(timeout=2.0)
        assert source.failed is True
        assert not shutdown_event.is_set()
    finally:
        source.stop(timeout=1.0)
