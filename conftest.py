"""Root pytest conftest.

Ensures the repository root is importable as the base for package-absolute imports
(`from autograsper.hardware.dryrun import DryRunRobot`, `from client.cloudgripper_client import
GripperRobot`, etc.) regardless of the directory pytest is invoked from.

This is the ONLY place in the codebase allowed to touch `sys.path` (see
`autograsper/docs/CONVENTIONS.md`). Library code must use package-absolute imports only.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
