"""Tests for `autograsper.perception.yolo_segmenter.SegmentationWorker` (Wave 3a).

Uses `DryRunRobot` + a stub `OccupancyProvider` (no YOLO/GPU involved), per CONVENTIONS.md.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from autograsper.hardware.dryrun import DryRunRobot
from autograsper.observation.source import ObservationSource
from autograsper.observation.types import Observation
from autograsper.perception.occupancy import ClumpStats, OccupancyResult
from autograsper.perception.yolo_segmenter import SegmentationWorker


class _PassthroughCamera:
    """Fake `CameraPipeline`: returns the raw bottom image unchanged (same fake used by
    test_observation_source.py)."""

    def process_bottom(self, raw_bottom):
        return raw_bottom


def _make_result(obs: Observation, num_clumps: int = 1) -> OccupancyResult:
    grid_mask = np.zeros((4, 4), dtype=np.uint8)
    crop_mask = np.zeros((8, 8), dtype=np.uint8)
    stats = ClumpStats(
        num_clumps=num_clumps,
        total_area_px=num_clumps * 10,
        areas=tuple([10] * num_clumps),
        centroids=tuple([(1.0, 1.0)] * num_clumps),
    )
    return OccupancyResult(
        source_seq=obs.seq,
        frame_index=obs.frame_index,
        timestamp=obs.timestamp,
        grid_mask=grid_mask,
        crop_mask=crop_mask,
        instances=(),
        stats=stats,
    )


class _StubProvider:
    """Records every observation it's asked to compute on (for latest-wins assertions); optional
    artificial delay to simulate a slow model call."""

    def __init__(self, delay: float = 0.0, num_clumps: int = 1):
        self.delay = delay
        self.num_clumps = num_clumps
        self.seen_seqs = []
        self._lock = threading.Lock()

    def compute(self, obs: Observation) -> OccupancyResult:
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.seen_seqs.append(obs.seq)
        return _make_result(obs, num_clumps=self.num_clumps)


class _SwitchingProvider:
    """Returns `num_clumps=1` for the first `warmup` calls, then `num_clumps=0` forever after --
    for exercising the "detections collapse to 0 after having seen occupancy" degraded trigger."""

    def __init__(self, warmup: int = 1):
        self.warmup = warmup
        self.calls = 0

    def compute(self, obs: Observation) -> OccupancyResult:
        self.calls += 1
        n = 1 if self.calls <= self.warmup else 0
        return _make_result(obs, num_clumps=n)


class _RaisingProvider:
    def compute(self, obs: Observation) -> OccupancyResult:
        raise RuntimeError("simulated provider failure")


@pytest.fixture
def shutdown_event():
    return threading.Event()


def _make_source(shutdown_event, fps=50.0):
    return ObservationSource(DryRunRobot(), _PassthroughCamera(), fps, shutdown_event)


# --- lifecycle / basic API ---------------------------------------------------------------


def test_latest_is_none_before_start(shutdown_event):
    source = _make_source(shutdown_event)
    worker = SegmentationWorker(_StubProvider(), source, shutdown_event)
    assert worker.latest() is None
    assert worker.is_running() is False
    assert worker.degraded is False


def test_start_twice_raises(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    worker = SegmentationWorker(_StubProvider(), source, shutdown_event)
    worker.start()
    try:
        with pytest.raises(RuntimeError):
            worker.start()
    finally:
        worker.stop(timeout=2.0)
        source.stop(timeout=2.0)


# --- latest() / await_result() semantics -------------------------------------------------


def test_worker_publishes_results(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    worker = SegmentationWorker(_StubProvider(), source, shutdown_event)
    worker.start()
    try:
        result = worker.await_result(min_source_seq=1, timeout=3.0)
        assert result is not None
        assert result.source_seq >= 1
        assert worker.latest() is result or worker.latest().source_seq >= result.source_seq
    finally:
        worker.stop(timeout=2.0)
        source.stop(timeout=2.0)


def test_await_result_min_seq_semantics(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    worker = SegmentationWorker(_StubProvider(), source, shutdown_event)
    worker.start()
    try:
        first = worker.await_result(min_source_seq=1, timeout=3.0)
        assert first is not None
        target = first.source_seq + 5
        later = worker.await_result(min_source_seq=target, timeout=3.0)
        assert later is not None
        assert later.source_seq >= target
    finally:
        worker.stop(timeout=2.0)
        source.stop(timeout=2.0)


def test_await_result_times_out_when_unreachable(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    worker = SegmentationWorker(_StubProvider(), source, shutdown_event)
    worker.start()
    try:
        result = worker.await_result(min_source_seq=10_000_000, timeout=0.3)
        assert result is None
    finally:
        worker.stop(timeout=2.0)
        source.stop(timeout=2.0)


# --- latest-wins skipping -----------------------------------------------------------------


def test_latest_wins_skips_backlog(shutdown_event):
    # Fast observation source + slow provider: the worker must never work through a backlog of
    # every queued observation -- once free, it should jump straight to the newest, skipping
    # whatever piled up while compute() was running (design 03 §2.1).
    source = _make_source(shutdown_event, fps=200.0)
    source.start()
    provider = _StubProvider(delay=0.2)
    worker = SegmentationWorker(provider, source, shutdown_event)
    worker.start()
    try:
        time.sleep(1.0)
        latest_obs = source.latest()
        assert latest_obs is not None
        # far more observations were produced in 1s at 200fps than a 0.2s-per-call provider could
        # have processed one-by-one
        assert len(provider.seen_seqs) < latest_obs.seq
        # and strictly fewer than half of them, generously
        assert len(provider.seen_seqs) * 2 < latest_obs.seq
    finally:
        worker.stop(timeout=2.0)
        source.stop(timeout=2.0)


# --- degraded flag ------------------------------------------------------------------------


def test_degraded_on_repeated_compute_failures(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    worker = SegmentationWorker(
        _RaisingProvider(), source, shutdown_event, max_consecutive_failures=3
    )
    worker.start()
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline and not worker.degraded:
            time.sleep(0.05)
        assert worker.degraded is True
        assert worker.latest() is None  # never published a result
        assert worker.is_running() is True  # keeps running despite being degraded
    finally:
        worker.stop(timeout=2.0)
        source.stop(timeout=2.0)


def test_degraded_on_zero_detection_streak_after_seeing_occupancy(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    provider = _SwitchingProvider(warmup=1)
    worker = SegmentationWorker(
        provider, source, shutdown_event, zero_detection_streak_for_degraded=3
    )
    worker.start()
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline and not worker.degraded:
            time.sleep(0.05)
        assert worker.degraded is True
    finally:
        worker.stop(timeout=2.0)
        source.stop(timeout=2.0)


def test_never_seen_occupancy_does_not_count_as_degraded(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    provider = _StubProvider(num_clumps=0)
    worker = SegmentationWorker(
        provider, source, shutdown_event, zero_detection_streak_for_degraded=2
    )
    worker.start()
    try:
        time.sleep(0.6)
        assert worker.degraded is False
    finally:
        worker.stop(timeout=2.0)
        source.stop(timeout=2.0)


def test_zero_streak_trigger_can_be_disabled(shutdown_event):
    source = _make_source(shutdown_event, fps=50.0)
    source.start()
    provider = _SwitchingProvider(warmup=1)
    worker = SegmentationWorker(
        provider, source, shutdown_event, zero_detection_streak_for_degraded=None
    )
    worker.start()
    try:
        time.sleep(0.6)
        assert worker.degraded is False
    finally:
        worker.stop(timeout=2.0)
        source.stop(timeout=2.0)
