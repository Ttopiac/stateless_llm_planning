"""Example agents that can interact with most PDDLSIM simulations.

> [!NOTE]
> To use this module, the `agents` extra **must** be enabled.
"""

import importlib
import importlib.util

try:
    importlib.util.find_spec("unified_planning")
except ImportError as error:
    raise ValueError(
        "to use this module, activate the `agents` extra"
    ) from error
