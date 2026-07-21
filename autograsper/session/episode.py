"""`EpisodeState` + `EpisodeStateMachine` — the explicit episode lifecycle (Wave 5).

Design reference: `autograsper/design/02_proposed_architecture.md` §3.5 (the mermaid state diagram
this module encodes verbatim):

```mermaid
stateDiagram-v2
    [*] --> Startup
    Startup --> Resetting: needs_reset
    Startup --> Active: ready
    Startup --> Intervention: tool check failed
    Intervention --> Startup: human confirmed / regrasp ok
    Active --> Evaluating: plan executed
    Evaluating --> Resetting: needs_reset
    Evaluating --> Startup: next episode
    Resetting --> Active
    Active --> Aborted: unrecoverable error
    Resetting --> Aborted: unrecoverable error
    [*] --> Finished: episode budget reached / shutdown
```

Replaces `grasper.py::RobotActivity` (the legacy 10 Hz-polled activity enum) with an explicit,
validated transition function — `session.coordinator.SessionRunner` is the only caller
(design 02 §3.5: "Single-threaded control loop ... Transitions are function calls in one loop —
no message queue, no polling, no missed states").

Every state is reachable from `FINISHED`/`ABORTED` only via a fresh `EpisodeStateMachine` (both are
terminal — no transition is defined out of them); `FINISHED`/`ABORTED` are themselves reachable
from *every* other state (the external shutdown/unrecoverable-error channels the diagram's `[*] -->
Finished` and the two `--> Aborted` edges represent — generalized here to every state, not just
`Active`/`Resetting`, since `SessionRunner`'s `shutdown_event` can be observed, and an unexpected
exception can occur, at any point in the loop, not only during those two).

Threading: `EpisodeStateMachine` is a plain object with no internal locking — it is driven by
exactly one thread (`SessionRunner`'s control loop), matching every other single-threaded-control
component in this wave (`Executor`, per its own docs).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class EpisodeState(Enum):
    STARTUP = "startup"
    ACTIVE = "active"
    EVALUATING = "evaluating"
    RESETTING = "resetting"
    INTERVENTION = "intervention"
    FINISHED = "finished"
    ABORTED = "aborted"


class InvalidTransition(Exception):
    """Raised by `EpisodeStateMachine.transition` when the requested transition is not one of the
    edges in design 02 §3.5's diagram (see module docstring for the two terminal-state and
    reach-from-anywhere generalizations)."""


_TERMINAL = (EpisodeState.FINISHED, EpisodeState.ABORTED)

# Explicit allowed-edge table (design 02 §3.5's diagram, plus "FINISHED/ABORTED reachable from any
# non-terminal state" per module docstring). Self-transitions (a state "transitioning" to itself)
# are always allowed unconditionally by `transition()` below and are not listed here.
_ALLOWED = {
    EpisodeState.STARTUP: {EpisodeState.RESETTING, EpisodeState.ACTIVE, EpisodeState.INTERVENTION},
    EpisodeState.INTERVENTION: {EpisodeState.STARTUP},
    EpisodeState.ACTIVE: {EpisodeState.EVALUATING},
    EpisodeState.EVALUATING: {EpisodeState.RESETTING, EpisodeState.STARTUP},
    EpisodeState.RESETTING: {EpisodeState.ACTIVE},
}


class EpisodeStateMachine:
    """Validated episode-state transitions with an optional callback fired on every successful
    (non-self) transition, `(old_state, new_state) -> None`. A raising callback is logged and does
    not affect the transition itself."""

    def __init__(
        self,
        initial: EpisodeState = EpisodeState.STARTUP,
        on_transition: Optional[Callable[[EpisodeState, EpisodeState], None]] = None,
    ) -> None:
        self._state = initial
        self._on_transition = on_transition

    @property
    def state(self) -> EpisodeState:
        return self._state

    def is_terminal(self) -> bool:
        return self._state in _TERMINAL

    def transition(self, new_state: EpisodeState) -> None:
        """Move to `new_state`. Raises `InvalidTransition` if there is no edge for
        `self.state -> new_state` (self-transitions and moves into `FINISHED`/`ABORTED` from any
        non-terminal state are always allowed; nothing is allowed out of a terminal state except
        another self-transition)."""
        old_state = self._state
        if new_state == old_state:
            return
        if old_state in _TERMINAL:
            raise InvalidTransition(
                f"EpisodeStateMachine: {old_state.name} is terminal, cannot transition to "
                f"{new_state.name}"
            )
        allowed = _ALLOWED.get(old_state, set()) | set(_TERMINAL)
        if new_state not in allowed:
            raise InvalidTransition(f"EpisodeStateMachine: {old_state.name} -> {new_state.name} is not allowed")

        self._state = new_state
        logger.info("EpisodeStateMachine: %s -> %s", old_state.name, new_state.name)
        if self._on_transition is not None:
            try:
                self._on_transition(old_state, new_state)
            except Exception:
                logger.exception("EpisodeStateMachine: on_transition callback raised")
