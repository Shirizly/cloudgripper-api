"""`DebugSink` — no-op-by-default debug artifact writer.

Design reference: `autograsper/design/02_proposed_architecture.md` §3.3 ("debug artifacts go
through an injected `DebugSink` that writes into the session directory when enabled") and
`autograsper/docs/CONVENTIONS.md` ("No `cv2.imwrite`/`cv2.imshow` side effects in library code
... Debug artifacts go through the `DebugSink`").

Porting notes: this replaces every ad hoc `cv2.imwrite("<name>.png", ...)` call to the current
working directory found in legacy library code, e.g.
`autograsper/custom_graspers/fence_utils.py::make_tool_mask`
(`cv2.imwrite(f'tool_mask_{w_px}x{h_px}_angle{angle_deg}.png', mask*255)`) and
`check_wall_reset_needed` (`cv2.imwrite(f'band mask and mask at {wall.label}.png', ...)`), and
`object_tracker/granular_utils.py::process_image`
(`cv2.imwrite("occupancy_mask_for_debug.png", clean_mask)`). Perception/planning code written
against `DebugSink` in later waves should call `.save(name, image)` / `.save_json(name, obj)`
instead of `cv2.imwrite`/manual `json.dump`, so debug output only appears when explicitly enabled,
and only under a directory the session layer controls (never the CWD).

Threading: thread-safe (internal lock guards the enabled directory and the filename counter);
one `DebugSink` instance is meant to be shared across the observation source, perception workers,
and planner.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DebugSink:
    """No-op unless `enable(directory)` has been called; then writes numbered artifacts there."""

    def __init__(self) -> None:
        self._dir: Optional[str] = None
        self._lock = threading.Lock()
        self._counter = 0

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._dir is not None

    def enable(self, directory: str) -> None:
        """Start writing debug artifacts under `directory` (created if missing). Resets the
        filename counter."""
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            self._dir = directory
            self._counter = 0
        logger.info("DebugSink enabled: writing to %s", directory)

    def disable(self) -> None:
        """Return to no-op behavior. Does not delete previously written files."""
        with self._lock:
            self._dir = None
        logger.info("DebugSink disabled")

    def _claim(self) -> Optional[str]:
        """Return the directory to write into plus a fresh numbered prefix, or None if disabled."""
        with self._lock:
            if self._dir is None:
                return None
            n = self._counter
            self._counter += 1
            return os.path.join(self._dir, f"{n:06d}_")

    def save(self, name: str, image: np.ndarray) -> Optional[str]:
        """Write `image` as `<dir>/<counter>_<name>.png`. No-op (returns None) when disabled."""
        prefix = self._claim()
        if prefix is None:
            return None
        path = f"{prefix}{name}.png"
        try:
            cv2.imwrite(path, image)
        except Exception:
            logger.exception("DebugSink.save: failed to write %s", path)
            return None
        return path

    def save_json(self, name: str, obj: Any) -> Optional[str]:
        """Write `obj` as `<dir>/<counter>_<name>.json`. No-op (returns None) when disabled."""
        prefix = self._claim()
        if prefix is None:
            return None
        path = f"{prefix}{name}.json"
        try:
            with open(path, "w") as f:
                json.dump(obj, f, indent=2, default=str)
        except Exception:
            logger.exception("DebugSink.save_json: failed to write %s", path)
            return None
        return path
