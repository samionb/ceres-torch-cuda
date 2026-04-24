"""Compatibility shim for the old torch_ceres import path.

The project was renamed to `ceres-torch`; new code should use
`import ceres_torch`.
"""

from ceres_torch import *  # noqa: F401,F403

