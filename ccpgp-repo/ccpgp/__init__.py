"""
CCPGP — Constitutionally Constrained Provenance-Gated Plasticity

A novel synaptic learning rule for ethically bounded adaptive systems.
Zero external dependencies. Pure Python.

Copyright (c) 2026 LEAP A.I Industries Ltd.
"""

from .core import (
    PROVENANCE_GATES,
    CCANeuron,
    CCPGPSynapse,
    CCPGPNetwork,
)
from .hetero import (
    PyramidalCCA,
    FastPVCCA,
    SlowSSTCCA,
    VIPCCA,
    TRNCCA,
    HeteroSynapse,
    HeteroNetwork,
)

__version__ = "1.0.0"
__all__ = [
    "PROVENANCE_GATES",
    "CCANeuron", "CCPGPSynapse", "CCPGPNetwork",
    "PyramidalCCA", "FastPVCCA", "SlowSSTCCA", "VIPCCA", "TRNCCA",
    "HeteroSynapse", "HeteroNetwork",
]
