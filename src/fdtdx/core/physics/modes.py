from collections import namedtuple
from types import SimpleNamespace
from typing import List, Literal

import jax
import jax.numpy as jnp
import numpy as np
from tidy3d.components.mode.solver import compute_modes as _compute_modes

from fdtdx.core.physics.metrics import normalize_by_energy

ModeTupleType = namedtuple("Mode", ["neff", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"])
"""A named tuple containing the mode fields and effective index.

Attributes:
    neff: Complex effective refractive index of the mode
    Ex: x-component of the electric field
    Ey: y-component of the electric field
    Ez: z-component of the electric field
    Hx: x-component of the magnetic field
    Hy: y-component of the magnetic field
    Hz: z-component of the magnetic field
"""


def compute_accurate_mode(
    frequency: float,
    inv_permittivities: jax.Array,  # shape (nx, ny, nz)
    inv_permeabilities: jax.Array | float,
    resolution: float,
    direction: Literal["+", "-"],
    mode_index: int = 0,
) -> tuple[
    jax.Array,  # E
    jax.Array,  # H
    jax.Array,  # complex propagation constant
]:
    # Input validation
    input_dtype = inv_permittivities.dtype
    if inv_permittivities.squeeze().ndim != 2:
        raise Exception(f"Invalid shape of inv_permittivities: {inv_permittivities.shape}")
    if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
        raise Exception(f"Mode solver currently does not support metallic materials")
    # if inv_permeabilities.squeeze().ndim != 2:
    #     raise Exception(f"Invalid shape of inv_permeabilities: {inv_permeabilities.shape}")

    def mode_helper(permittivity):
        modes = tidy3d_mode_computation_wrapper(
            frequency=frequency,
            permittivity_cross_section=permittivity,  # type: ignore
            coords=coords,
            num_modes=mode_index + 2,
            filter_pol=None,
            direction=direction,
        )
        mode = modes[mode_index]

        if propagation_axis == 0:
            mode_E, mode_H = (
                np.stack([mode.Ez, mode.Ex, mode.Ey], axis=0).astype(np.complex64),
                np.stack([mode.Hz, mode.Hx, mode.Hy], axis=0).astype(np.complex64),
            )
        elif propagation_axis == 1:
            # mode_E, mode_H = (  # untested
            #     np.stack([mode.Ex, mode.Ez, mode.Ey], axis=0).astype(np.complex64),
            #     np.stack([mode.Hx, mode.Hz, mode.Hy], axis=0).astype(np.complex64),
            # )
            raise NotImplementedError()
        elif propagation_axis == 2:
            # mode_E, mode_H = (  # untested
            #     np.stack([mode.Ez, mode.Ex, mode.Ey], axis=0).astype(np.complex64),
            #     np.stack([mode.Hz, mode.Hx, mode.Hy], axis=0).astype(np.complex64),
            # )
            raise NotImplementedError()
        else:
            raise Exception(f"This should never happen")
        
        neff = np.asarray(mode.neff).astype(np.complex64)
        return mode_E, mode_H, neff
    
    # compute input to tidy3d Mode solver
    permittivities = 1 / inv_permittivities
    other_axes = [a for a in range(3) if permittivities.shape[a] != 1]
    propagation_axis = permittivities.shape.index(1)
    coords = [
        np.arange(permittivities.shape[dim] + 1) * resolution / 1e-6 for dim in other_axes
    ]
    permittivity_squeezed = jnp.take(
        permittivities,
        indices=0,
        axis=propagation_axis,
    )
    result_shape_dtype = (
        jnp.zeros((3, *permittivity_squeezed.shape), dtype=jnp.complex64),
        jnp.zeros((3, *permittivity_squeezed.shape), dtype=jnp.complex64),
        jnp.zeros(shape=(), dtype=jnp.complex64)
    )
    
    # pure callback to tidy3d is necessary to work in jitted environment
    mode_E_raw, mode_H_raw, eff_idx = jax.pure_callback(
        mode_helper,
        result_shape_dtype,
        jax.lax.stop_gradient(permittivity_squeezed),
    )
    mode_E = jnp.real(jnp.expand_dims(mode_E_raw, axis=propagation_axis + 1)).astype(input_dtype)
    mode_H = jnp.real(jnp.expand_dims(mode_H_raw, axis=propagation_axis + 1)).astype(input_dtype)
    
    # compute_mode_error(
    #     mode_E=mode_E,
    #     mode_H=mode_H,
    #     eff_idx=eff_idx,
    #     inv_permittivities=inv_permittivities,
    #     inv_permeabilities=inv_permeabilities,
    #     resolution=resolution,
    #     direction=direction,
    # )
    
    mode_E2_norm, mode_H2_norm = normalize_by_energy(
        E=mode_E,
        H=mode_H,
        inv_permittivity=inv_permittivities,
        inv_permeability=inv_permeabilities,
    )
    
    return mode_E2_norm, mode_H2_norm, eff_idx



# def compute_mode_error(
#     mode_E: jax.Array,
#     mode_H: jax.Array,
#     eff_idx: jax.Array,
#     inv_permittivities: jax.Array,  # shape (nx, ny, nz)
#     inv_permeabilities: jax.Array | float,
#     resolution: float,
#     direction: Literal["+", "-"],
# ):
#     factors = jnp.linspace(-1, 1, 20)
#     a = 1
    
    
    
    



def tidy3d_mode_computation_wrapper(
    frequency: float,
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
    """Compute optical modes of a waveguide cross-section.

    This function uses the Tidy3D mode solver to compute the optical modes of a given
    waveguide cross-section defined by its permittivity distribution.

    Args:
        frequency (float): Operating frequency in Hz
        permittivity_cross_section (np.ndarray): 2D array of relative permittivity values
        coords (List[np.ndarray]): List of coordinate arrays [x, y] defining the grid
        direction (Literal["+", "-"]): Propagation direction, either "+" or "-"
        target_neff (float | None, optional): Target effective index to search around. Defaults to None.
        angle_theta (float, optional): Polar angle in radians. Defaults to 0.0.
        angle_phi (float, optional): Azimuthal angle in radians. Defaults to 0.0.
        num_modes (int, optional): Number of modes to compute. Defaults to 10.
        precision (Literal["single", "double"], optional): Numerical precision. Defaults to "double".
        filter_pol (Literal["te", "tm"] | None, optional): Mode polarization filter. Defaults to None.

    Returns:
        List[ModeTupleType]: List of computed modes sorted by decreasing real part of
            effective index. Each mode contains the field components and effective index.
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
        freq=frequency,
        precision=precision,
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
