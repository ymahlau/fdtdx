"""FDTD simulation module providing various FDTD propagation implementations.

This module provides different implementations of the Finite-Difference Time-Domain (FDTD)
method for electromagnetic wave propagation, including:

- Reversible FDTD that leverages time-reversibility for memory-efficient gradient computation
- Checkpointed FDTD that saves field states at specific timesteps for gradient computation
- Custom forward FDTD that allows customizing the propagation behavior
"""

from fdtdx.fdtd.fdtd import checkpointed_fdtd, custom_fdtd_forward, reversible_fdtd

__all__ = [
    "reversible_fdtd",  # Time-reversible FDTD implementation
    "checkpointed_fdtd",  # FDTD with field state checkpointing
    "custom_fdtd_forward",  # Customizable FDTD forward propagation
]
