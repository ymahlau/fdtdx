"""Electromagnetic mode computation and analysis.

This module provides functionality for computing and analyzing electromagnetic modes
in waveguides and other photonic structures. It uses tidy3d's mode solver under the
hood but provides a simplified interface integrated with the FDTDX framework.

The module supports:
- Computing guided modes for arbitrary permittivity cross-sections
- Handling both TE and TM polarizations
- Flexible mode filtering and sorting
- Support for angled propagation
- Single and double precision computations
"""

from collections import namedtuple
from types import SimpleNamespace
from typing import List, Literal

import numpy as np
from tidy3d.plugins.mode.solver import compute_modes as _compute_modes

from fdtdx.core import constants

ModeTupleType = namedtuple("Mode", ["neff", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
"""A named tuple representing an electromagnetic mode.

Attributes:
    neff: Complex effective refractive index of the mode
    Ex: X component of the electric field
    Ey: Y component of the electric field
    Ez: Z component of the electric field
    Hx: X component of the magnetic field
    Hy: Y component of the magnetic field
    Hz: Z component of the magnetic field
"""


def compute_modes(
    wavelength: float,
    permittivity_cross_section: np.ndarray,
    coords: List[np.ndarray],
    direction: Literal["+", "-"],
    target_neff: float | None = None,
    angle_theta: float = 0.0,
    angle_phi: float = 0.0,
    num_modes: int = 10,
    precision: Literal["single", "double"] = "double",
    filter_pol: Literal["te", "tm"] | None = None,
) -> List[ModeTupleType]:
    """Compute electromagnetic modes for a given cross-section.

    This function uses tidy3d's mode solver to compute the electromagnetic modes
    for a given permittivity cross-section. It returns the modes sorted by their
    real effective refractive index in descending order.

    Args:
        wavelength: Operating wavelength in meters
        permittivity_cross_section: 2D array of relative permittivity values
        coords: List of coordinate arrays [x, y] for the cross-section
        direction: Propagation direction, either "+" or "-"
        target_neff: Optional target effective index to search around
        angle_theta: Polar angle in radians (default: 0.0)
        angle_phi: Azimuthal angle in radians (default: 0.0)
        num_modes: Number of modes to compute (default: 10)
        precision: Numerical precision, either "single" or "double" (default: "double")
        filter_pol: Optional polarization filter, either "te", "tm" or None

    Returns:
        List[ModeTupleType]: List of computed modes sorted by descending real
            effective refractive index
    """
    # see https://docs.flexcompute.com/projects/tidy3d/en/latest/_autosummary/tidy3d.ModeSpec.html#tidy3d.ModeSpec
    mode_spec = SimpleNamespace(
        num_modes=num_modes,
        target_neff=target_neff,
        num_pml=(0, 0),
        filter_pol=filter_pol,
        angle_theta=angle_theta,
        angle_phi=angle_phi,
        bend_radius=None,
        bend_axis=None,
        precision=precision,
        track_freq="central",
        group_index_step=False,
    )
    od = np.zeros_like(permittivity_cross_section)
    eps_cross = [
        permittivity_cross_section,
        od,
        od,
        od,
        permittivity_cross_section,
        od,
        od,
        od,
        permittivity_cross_section,
    ]

    EH, neffs, _ = _compute_modes(
        eps_cross=eps_cross,
        coords=coords,
        freq=constants.c / (wavelength),
        mode_spec=mode_spec,
        direction=direction,
    )
    ((Ex, Ey, Ez), (Hx, Hy, Hz)) = EH.squeeze()

    if num_modes == 1:
        modes = [
            ModeTupleType(
                Ex=Ex,
                Ey=Ey,
                Ez=Ez,
                Hx=Hx,
                Hy=Hy,
                Hz=Hz,
                neff=float(neffs.real) + 1j * float(neffs.imag),
            )
            for _ in range(num_modes)
        ]
    else:
        modes = [
            ModeTupleType(
                Ex=Ex[..., i],
                Ey=Ey[..., i],
                Ez=Ez[..., i],
                Hx=Hx[..., i],
                Hy=Hy[..., i],
                Hz=Hz[..., i],
                neff=neffs[i],
            )
            for i in range(num_modes)
        ]
    modes = sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)
    return modes
