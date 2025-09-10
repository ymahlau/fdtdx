from collections import namedtuple
from types import SimpleNamespace
from typing import List, Literal

import jax
import jax.numpy as jnp
import numpy as np
import tidy3d
from tidy3d.components.mode.solver import compute_modes as _compute_modes

from fdtdx.core.physics.metrics import normalize_by_poynting_flux

ModeTupleType = namedtuple("Mode", ["neff", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
"""A named tuple containing the mode fields and effective index.
"""


def compute_mode_polarization_fraction(
    #: a ModeTupleType instance
    mode: ModeTupleType,
    #: indices of transverse E-field component axes.
    tangential_axes: tuple[int, int],
    #: "te" or "tm" determines which axis is 'E1'
    pol: Literal["te", "tm"],
) -> float:
    """Mode polarization fraction."""

    E_fields = [mode.Ex, mode.Ey, mode.Ez]
    E1 = E_fields[tangential_axes[0]]
    E2 = E_fields[tangential_axes[1]]

    if pol == "te":
        numerator = np.sum(np.abs(E1) ** 2)
    elif pol == "tm":
        numerator = np.sum(np.abs(E2) ** 2)
    else:
        raise ValueError(f"pol must be 'te' or 'tm', but got {pol}")

    denominator = np.sum(np.abs(E1) ** 2 + np.abs(E2) ** 2) + 1e-18

    #: Polarization fraction between 0 and 1.
    return numerator / denominator


def sort_modes(
    #: list of modes.
    modes: list[ModeTupleType],
    #: If not none, sort by polarization specificaton.
    filter_pol: Literal["te", "tm"] | None,
    #: indices of transverse E-field component axes.
    tangential_axes: tuple[int, int],
) -> list[ModeTupleType]:
    """
    Sort modes by polarization.
    """
    if filter_pol is None:
        return sorted(modes, key=lambda m: float(np.real(m.neff)), reverse=True)

    def is_matching(mode):
        frac = compute_mode_polarization_fraction(mode, tangential_axes, filter_pol)
        return frac >= 0.5

    matching = [m for m in modes if is_matching(m)]
    non_matching = [m for m in modes if not is_matching(m)]

    matching_sorted = sorted(matching, key=lambda m: float(np.real(m.neff)), reverse=True)
    non_matching_sorted = sorted(non_matching, key=lambda m: float(np.real(m.neff)), reverse=True)

    #: sorted list of modes.
    return matching_sorted + non_matching_sorted


def compute_mode(
    frequency: float,
    inv_permittivities: jax.Array,  # shape (nx, ny, nz)
    inv_permeabilities: jax.Array | float,
    resolution: float,
    direction: Literal["+", "-"],
    mode_index: int = 0,
    filter_pol: Literal["te", "tm"] | None = None,
) -> tuple[
    jax.Array,  # E
    jax.Array,  # H
    jax.Array,  # complex propagation constant
]:
    # Input validation
    if inv_permittivities.squeeze().ndim != 2:
        raise Exception(f"Invalid shape of inv_permittivities: {inv_permittivities.shape}")
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        if inv_permeabilities.squeeze().ndim != 2:
            raise Exception(f"Invalid shape of inv_permeabilities: {inv_permeabilities.shape}")
        # raise Exception("Mode solver currently does not support metallic materials")

    def mode_helper(permittivity, permeability):
        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency,
            permittivity_cross_section=permittivity,
            permeability_cross_section=permeability,
            coords=coords,
            direction=direction,
            num_modes=2 * (mode_index + 1) + 10,
        )

        # sort modes by polarization
        # tidy3d assumes propagation in the z-direction. The tangential axes are therefore x and y.
        modes = sort_modes(modes, filter_pol, (0, 1))
        mode = modes[mode_index]

        if propagation_axis == 0:
            mode_E, mode_H = (
                np.stack([mode.Ez, mode.Ex, mode.Ey], axis=0).astype(np.complex64),
                np.stack([mode.Hz, mode.Hx, mode.Hy], axis=0).astype(np.complex64),
            )
        elif propagation_axis == 1:
            mode_E, mode_H = (
                np.stack([mode.Ex, mode.Ez, mode.Ey], axis=0).astype(np.complex64),
                -np.stack([mode.Hx, mode.Hz, mode.Hy], axis=0).astype(np.complex64),
            )
        elif propagation_axis == 2:
            mode_E, mode_H = (
                np.stack([mode.Ex, mode.Ey, mode.Ez], axis=0).astype(np.complex64),
                np.stack([mode.Hx, mode.Hy, mode.Hz], axis=0).astype(np.complex64),
            )
        else:
            raise Exception("This should never happen")

        neff = np.asarray(mode.neff).astype(np.complex64)
        return mode_E, mode_H, neff

    # compute input to tidy3d Mode solver
    permittivities = 1 / inv_permittivities
    other_axes = [a for a in range(3) if permittivities.shape[a] != 1]
    propagation_axis = permittivities.shape.index(1)
    coords = [np.arange(permittivities.shape[dim] + 1) * resolution / 1e-6 for dim in other_axes]
    permittivity_squeezed = jnp.take(
        permittivities,
        indices=0,
        axis=propagation_axis,
    )
    result_shape_dtype = (
        jnp.zeros((3, *permittivity_squeezed.shape), dtype=jnp.complex64),
        jnp.zeros((3, *permittivity_squeezed.shape), dtype=jnp.complex64),
        jnp.zeros(shape=(), dtype=jnp.complex64),
    )

    permeabilities = 1 / inv_permeabilities
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        permeability_squeezed = jnp.take(
            permeabilities,
            indices=0,
            axis=propagation_axis,
        )
    else:  # float
        permeability_squeezed = permeabilities

    # pure callback to tidy3d is necessary to work in jitted environment
    mode_E_raw, mode_H_raw, eff_idx = jax.pure_callback(
        mode_helper,
        result_shape_dtype,
        jax.lax.stop_gradient(permittivity_squeezed),
        jax.lax.stop_gradient(permeability_squeezed),
    )
    mode_E = jnp.expand_dims(mode_E_raw, axis=propagation_axis + 1)
    mode_H = jnp.expand_dims(mode_H_raw, axis=propagation_axis + 1)

    # Tidy3D uses different scaling internally, so convert back
    mode_H = mode_H * tidy3d.constants.ETA_0

    mode_E_norm, mode_H_norm = normalize_by_poynting_flux(mode_E, mode_H, axis=propagation_axis)

    return mode_E_norm, mode_H_norm, eff_idx


def tidy3d_mode_computation_wrapper(
    #: Operating frequency in Hz
    frequency: float,
    #: 2D array of relative permittivity values
    permittivity_cross_section: np.ndarray,
    #: List of coordinate arrays [x, y] defining the grid
    coords: List[np.ndarray],
    #: Propagation direction, either "+" or "-"
    direction: Literal["+", "-"],
    #: 2D array of relative permeability values. Defauts to None.
    permeability_cross_section: np.ndarray | None = None,
    #: Target effective index to search around. Defaults to None
    target_neff: float | None = None,
    #: Polar angle in radians. Defaults to 0.0.
    angle_theta: float = 0.0,
    #: Azimuthal angle in radians. Defaults to 0.0
    angle_phi: float = 0.0,
    #: Number of modes to compute. Defaults to 10
    num_modes: int = 10,
    #: Numerical precision. Defaults to "double"
    precision: Literal["single", "double"] = "double",
) -> List[ModeTupleType]:
    """Compute optical modes of a waveguide cross-section.

    This function uses the Tidy3D mode solver to compute the optical modes of a given
    waveguide cross-section defined by its permittivity distribution.

    Notes:
        tidy3d assumes propagation in z-direction. The output fields should be handled accordingly.
    """
    # see https://docs.flexcompute.com/projects/tidy3d/en/latest/_autosummary/tidy3d.ModeSpec.html#tidy3d.ModeSpec
    mode_spec = SimpleNamespace(
        # Note that the filter_pol argument is not used here since it does not work from tidy3d
        num_modes=num_modes,
        target_neff=target_neff,
        num_pml=(0, 0),
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
    mu_cross = None
    if permeability_cross_section is not None:
        mu_cross = [
            permeability_cross_section,
            od,
            od,
            od,
            permeability_cross_section,
            od,
            od,
            od,
            permeability_cross_section,
        ]

    EH, neffs, _ = _compute_modes(
        eps_cross=eps_cross,
        coords=coords,
        freq=frequency,
        precision=precision,
        mode_spec=mode_spec,
        direction=direction,
        mu_cross=mu_cross,
    )
    ((Ex, Ey, Ez), (Hx, Hy, Hz)) = EH.squeeze()

    if num_modes == 1:
        modes = [
            ModeTupleType(
                #: x-component of the electric field
                Ex=Ex,
                #: y-component of the electric field
                Ey=Ey,
                #: z-component of the electric field
                Ez=Ez,
                #: x-component of the magnetic field
                Hx=Hx,
                #: y-component of the magnetic field
                Hy=Hy,
                #: z-component of the magnetic field
                Hz=Hz,
                #: Complex effective refractive index of the mode
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

    #: List of computed modes sorted by decreasing real part of effective index.
    #: Each mode contains the field components and effective index.
    return modes
