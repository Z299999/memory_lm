"""Core modules for TOOLS/SimplexNet.

This package provides the core Simplex Memory Network implementation:
- SimplexMemoryGraph: Simplicial lattice DAG structure
- SMNmodule: PyTorch nn.Module for SMN
- SMN_RL: High-level RL wrapper

Usage::

    from TOOLS.SimplexNet.core import SimplexMemoryGraph, SMNmodule, SMN_RL
"""

from .SimplexMemoryGraph import SimplexMemoryGraph
from .SMNmodule import SMNmodule

# Lazy import SMN_RL to avoid circular dependency
def __getattr__(name):
    if name == 'SMN_RL':
        from .SMN_RL import SMN_RL as _SMN_RL
        return _SMN_RL
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['SimplexMemoryGraph', 'SMNmodule', 'SMN_RL']
