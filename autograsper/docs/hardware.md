# `autograsper/hardware/` — RobotInterface, CloudGripperRobot, DryRunRobot

Status: **implemented** (Wave 1). Design reference:
[`../design/02_proposed_architecture.md`](../design/02_proposed_architecture.md) §3.1.

## Purpose

The single seam between "code that decides what the robot should do" and "code that actually
moves it." Every layer above this one (observation, perception, planning, execution, session —
Waves 2+) talks to the robot only through the `RobotInterface` protocol and the `Order` /
`CommandReceipt` / `RawObservation` value types defined here. This:

- lets any component be tested by swapping in `DryRunRobot` — never `CloudGripperRobot` — per the
  hard safety rule in [CONVENTIONS.md](CONVENTIONS.md) ("never send commands to a real robot" in
  tests/examples/`__main__` blocks/default configs);
- collapses the legacy "two independent `GripperRobot` connections, synchronized only by
  wall-clock sleeps" pattern (design 01 §2) into one typed interface;
- puts the rotation-bias correction in exactly one place (see below), fixing design 01 §5 defect
  #16 ("rotation bias applied in two places with opposite signs").

## Public API

`autograsper/hardware/robot_interface.py`:

- `OrderType` — enum `{MOVE_XY, MOVE_Z, ROTATE, GRIPPER}`. One canonical order type per
  `RobotInterface` method (see "Porting notes" for why this differs from legacy's 5-member enum).
- `Order(type: OrderType, values: Tuple[float, ...])` — frozen dataclass. `.validate()` returns a
  normalized/clipped copy or raises `OrderValidationError`:
  - **Raises** on wrong arity (`MOVE_XY` needs 2 values, others need 1), non-numeric values
    (including `bool`, which is rejected even though `isinstance(True, int)` is true in Python),
    or a `ROTATE` value that isn't `int`-castable.
  - **Does not raise**, but clips-and-warns, for out-of-range `MOVE_XY`/`MOVE_Z`/`GRIPPER` values
    (clipped to `[0, 1]`, `logging.warning` emitted — legacy `np.clip` was silent).
  - **Normalizes** `ROTATE` values: `int()`-cast (truncates, matching legacy
    `int(order_value[0])`), then `% 360`. Never raises for being "out of range" — rotation has no
    upper bound to clip to, it wraps.
  - Idempotent: validating an already-valid order is a no-op (no duplicate warnings).
- `OrderValidationError(ValueError)` — the only exception this module raises.
- `CommandReceipt(order: Order, send_time: float, robot_reported_time: Optional[str])` — result
  of one command. `send_time` is a local clock reading; `robot_reported_time` is the CloudGripper
  API's own `"time"` field for that command, passed through unparsed (`None` for `DryRunRobot`).
- `RawObservation(top_image, bottom_image_raw, robot_state_dict, timestamp)` — one
  `get_all_states()` snapshot, **before** any undistortion/homography rectification (that pipeline
  is `observation/source.py`'s job, Wave 2). `robot_state_dict` uses the legacy key set —
  `x_norm`, `y_norm`, `z_norm`, `rotation`, `claw_norm` — see
  `autograsper/legacy/library/utils.py::manual_control` for the reference shape. `timestamp` is a local
  wall-clock float, not the API's raw `time_state` string.
- `RobotInterface(Protocol)` — `move_xy`, `move_z`, `rotate`, `set_gripper`, `get_all_states`.
  Structural typing: any class implementing these five methods satisfies it, no explicit
  inheritance required.
- `OrderDispatchMixin` — provides `execute(order: Order) -> CommandReceipt`, a convenience
  dispatcher that validates then routes to the matching method. Mixed into both concrete
  implementations below so callers can use either the typed methods or the generic `execute()`.

`autograsper/hardware/cloudgripper.py`:

- `CloudGripperRobot(config: RobotConfig, *, client: Optional[GripperRobot] = None)` — wraps
  `client.cloudgripper_client.GripperRobot`. Reads the API token from the environment variable
  named by `config.token_env_var` **at construction time**; raises `RobotTokenError` if unset.
  Accepts an already-constructed `client` for advanced composition (e.g. sharing one HTTP client
  across instances per design 02 §3.1), otherwise builds its own.
- `set_gripper(opening: float, *, stepped_close: bool = False)` — `stepped_close=True` reproduces
  the legacy cautious-close sequence (see Porting notes). This kwarg is additive; the Protocol is
  still satisfied via the required 1-positional-arg signature.
- `RobotTokenError(RuntimeError)` — raised when the token env var is unset/empty.

`autograsper/hardware/dryrun.py`:

- `DryRunRobot(*, robot_idx="dryrun", initial_state=None, top_image_shape=(480,640,3),
  bottom_image_shape=(480,640,3), frame_source_dir=None, latency=0.0, assume_success=True)` —
  hardware-free implementation. Validates every order through the exact same
  `Order.validate()` path as `CloudGripperRobot` (called from within each of
  `move_xy`/`move_z`/`rotate`/`set_gripper`, so validation happens whether the caller uses the
  typed methods directly or goes through `execute()`).
  - Maintains an in-memory state dict (`x_norm`, `y_norm`, `z_norm`, `rotation`, `claw_norm`);
    with `assume_success=True` (the default) each command updates state immediately/unconditionally
    — there is no notion of a command failing or of the physical robot lagging behind the command.
  - `command_log: List[CommandReceipt]` — every command issued, in call order, for test
    assertions. A raised `OrderValidationError` is never logged (the order is invalid, no command
    was "sent").
  - Fake clock: a deterministic counter (`+0.1` per command, plus configured `latency`), not
    `time.time()` — keeps tests fast and their timestamp-ordering assertions independent of
    scheduler jitter.
  - `get_all_states()` returns synthetic all-zero `uint8` images of the configured shape by
    default, or replays image files (sorted by filename, cycling) from `frame_source_dir` if
    given.
  - `DEFAULT_TOP_SHAPE = DEFAULT_BOTTOM_SHAPE = (480, 640, 3)`.

## Data flow

```
Planner/Executor (Wave 2+)
    │  Order(type, values)
    ▼
RobotInterface.execute(order) / .move_xy(...) / .rotate(...) / ...
    │  Order.validate()  (raise OrderValidationError | clip+warn | normalize)
    ▼
CloudGripperRobot ─────────────────► client.cloudgripper_client.GripperRobot ──► HTTP API
    (rotation bias, token, stepped-close)
        or
DryRunRobot ───────────────────────► in-memory state dict + command_log
    │
    ▼
CommandReceipt(order, send_time, robot_reported_time)
```

`get_all_states()` flows the other direction: HTTP (or synthetic/replay) → `RawObservation`.

## Threading

Both implementations are plain synchronous objects with no internal locking. `CloudGripperRobot`
is intended to be constructed once and shared (design 02 §3.1: "the HTTP client is stateless, so
sharing is safe") by the executor and the observation source in later waves; those callers are
responsible for their own concurrency control (not addressed until Wave 2/3). `DryRunRobot` is
intended for single-threaded test/dev use; `command_log` is a plain list, not thread-safe.

## Config keys consumed

`CloudGripperRobot` consumes `config_schema.RobotConfig`: `idx`, `token_env_var`,
`rotation_bias`. See [configuration.md](configuration.md) for the full schema.

## Porting notes

- `autograsper/legacy/library/utils.py::OrderType` (5 members: `MOVE_XY`, `MOVE_Z`, `GRIPPER_CLOSE`,
  `GRIPPER_OPEN`, `ROTATE`) → collapsed to 4 (`MOVE_XY`, `MOVE_Z`, `ROTATE`, `GRIPPER`), matching
  the `set_gripper(opening: float)` signature already specified by design 02 §3.1. **Intentional
  change**, not explicitly dictated by the design doc (which shows the method signature but not an
  order enum) — logged in
  [`../design/IMPLEMENTATION_LOG.md`](../design/IMPLEMENTATION_LOG.md).
- `autograsper/legacy/library/utils.py::execute_order` — `np.clip(order_value, 0, 1)` and
  `int(order_value[0])` for rotation → `Order.validate()`. Clipping is now logged (legacy was
  silent).
- `autograsper/legacy/library/utils.py::execute_order`, `OrderType.GRIPPER_CLOSE` empty-value branch
  (the 0.30→0.24 step-by-0.04, 0.1s-wait staged close) → `CloudGripperRobot.set_gripper(...,
  stepped_close=True)`, ported verbatim (same constants, same loop condition `>=`).
- `autograsper/legacy/grasper.py` lines ~302-306 (`execute_order`: `local_order[1][0] +=
  self.rotation_bias` on the outgoing rotate value) + `recording.py` lines ~186-188
  (`Recorder._update`: `state['rotation'] -= angle_bias; state['rotation'] %= 180`) → both folded
  into `CloudGripperRobot` (`rotate()` and `get_all_states()` respectively). **Fixes design 01 §5
  defect #16** ("rotation bias applied in two places with opposite signs... easy to break"): now
  one class, one config field (`RobotConfig.rotation_bias`), both directions of the correction.
  (`recording.py` was deleted in the Wave 6 cleanup, superseded by `recording/`+`session/`; this
  citation is now historical provenance only.)
- `recording.py` lines ~56-61 (`CLOUDGRIPPER_TOKEN` env var, hardcoded name; this file was deleted
  in the Wave 6 cleanup, superseded by `recording/`+`session/` — historical citation) →
  `CloudGripperRobot.__init__` reads `os.environ[config.token_env_var]`, configurable name,
  defaulting to the same `"CLOUDGRIPPER_TOKEN"`, still raised at construction (moved earlier:
  legacy raised in `Recorder.__init__`, a different class from the one issuing commands).
- `autograsper/legacy/library/utils.py::manual_control` (`state["x_norm"]`, `state["y_norm"]`,
  `state["z_norm"]`, `state["rotation"]`, `state["claw_norm"]`) → `DryRunRobot`'s simulated state
  dict keys, so anything written against the legacy shape works unmodified against
  `RawObservation.robot_state_dict`.
- `client/cloudgripper_client.py::GripperRobot` — wrapped, not modified, per the hard rule in
  `../design/02_proposed_architecture.md` §6 ("CloudGripper HTTP client — wrapped, not
  rewritten").

## Testing

`autograsper/tests/test_hardware_dryrun.py` — order validation (arity/type/range/rotation),
`DryRunRobot` state evolution across a command sequence, `command_log` contents, and
`get_all_states()` shapes/timestamps/replay. Run:

```
cd /home/alon/Code/cloudgripper-api
/home/alon/anaconda3/envs/cge/bin/python -m pytest autograsper/tests/test_hardware_dryrun.py -q
```

`CloudGripperRobot` has no dedicated unit tests (would require network/hardware) and is never
instantiated by the test suite, per the hard safety rule.
