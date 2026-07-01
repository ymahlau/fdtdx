"""Numerical helpers for homogeneous-medium near-to-far field projection.

The detector classes own FDTDX object state and validation. This module keeps the array math in one place so the
heavy projection paths remain reusable, JAX-friendly, and independent of detector placement details.

The far-field path evaluates surface-equivalence Fourier integrals of the form
``I(r_hat) = int_S J(r_prime) exp(-i k r_hat . r_prime) dS``. The exact finite-distance path evaluates the
homogeneous-medium dyadic Green-function fields from equivalent electric and magnetic surface currents. Both paths
use physical detector coordinates and trapezoidal weights, which is important for rectilinear non-uniform grids.
"""

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.utils import (
    concrete_jax_bool,
    finite_numeric_array,
    finite_scalar_or_1d_array,
    is_jax_tracer,
)

_EDGE_TAPER_DECAY = 15.0
# Aperture-edge amplitude is exp(-0.5 * _EDGE_TAPER_DECAY) ~= 5e-4.


def _positive_projection_parameter(name: str, value: Any, expected_size: int) -> float | np.ndarray:
    """Validate a positive scalar or per-frequency projection-medium parameter."""
    parameter = finite_scalar_or_1d_array(name, value, expected_size)
    if np.any(np.asarray(parameter) <= 0):
        raise ValueError(f"{name} must be positive.")
    return parameter


def _projection_parameter_value(value: float | Sequence[float], index: int) -> float:
    """Return a scalar projection parameter value for one wave-character index."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return float(array[index])


def _projection_parameter_is_default(name: str, value: Any, default: float, expected_size: int) -> bool:
    """Return whether a scalar/per-frequency parameter equals its default value."""
    parameter = _positive_projection_parameter(name, value, expected_size)
    return bool(np.allclose(np.asarray(parameter, dtype=float), default))


def _passive_complex_sqrt(value: complex) -> complex:
    """Return the square-root branch with non-negative imaginary part for passive waves."""
    root = complex(np.sqrt(complex(value)))
    if root.imag < 0.0 or (np.isclose(root.imag, 0.0) and root.real < 0.0):
        root = -root
    return root


def _positive_impedance_sqrt(value: complex) -> complex:
    """Return the square-root branch with non-negative real impedance."""
    root = complex(np.sqrt(complex(value)))
    if root.real < 0.0 or (np.isclose(root.real, 0.0) and root.imag < 0.0):
        root = -root
    return root


def _validate_theta_range(theta: jax.Array | np.ndarray) -> None:
    """Validate that eager angle coordinates lie in the global spherical interval ``[0, pi]``."""
    if is_jax_tracer(theta):
        return
    if isinstance(theta, jax.Array):
        if concrete_jax_bool(jnp.any((theta < 0.0) | (theta > jnp.pi))):
            raise ValueError("theta values must be in the interval [0, pi].")
        return
    try:
        theta_array = np.asarray(theta)
    except jax.errors.TracerArrayConversionError:
        return
    if np.any((theta_array < 0.0) | (theta_array > np.pi)):
        raise ValueError("theta values must be in the interval [0, pi].")


def _validate_positive_values(name: str, values: jax.Array | np.ndarray) -> None:
    """Validate positive eager numeric arrays such as finite projection distances."""
    if is_jax_tracer(values):
        return
    if isinstance(values, jax.Array):
        if concrete_jax_bool(jnp.any(values <= 0.0)):
            raise ValueError(f"{name} values must be positive.")
        return
    if np.any(np.asarray(values) <= 0.0):
        raise ValueError(f"{name} values must be positive.")


def _validate_1d_coordinate_pair(
    first_name: str,
    first: jax.Array | np.ndarray,
    second_name: str,
    second: jax.Array | np.ndarray,
) -> tuple[jax.Array, jax.Array]:
    """Validate two non-empty one-dimensional coordinate arrays for projection grids."""
    first_array = finite_numeric_array(first_name, first)
    second_array = finite_numeric_array(second_name, second)
    if first_array.ndim != 1 or second_array.ndim != 1:
        raise ValueError(f"{first_name} and {second_name} must be one-dimensional arrays.")
    if first_array.size == 0 or second_array.size == 0:
        raise ValueError(f"{first_name} and {second_name} must be non-empty.")
    return first_array, second_array


def trapezoidal_weights_1d(points: jax.Array | np.ndarray) -> jax.Array:
    """Return one-dimensional trapezoidal integration weights for physical sample coordinates.

    Endpoint samples receive half of their adjacent interval. Interior samples receive half the sum of the
    neighboring intervals, so integrating a constant over non-uniform coordinates gives the physical span.
    """
    points = jnp.asarray(points, dtype=float)
    if points.size <= 1:
        return jnp.ones(points.shape, dtype=float)
    deltas = jnp.abs(jnp.diff(points))
    interior = 0.5 * (deltas[:-1] + deltas[1:])
    return jnp.concatenate((jnp.asarray([0.5 * deltas[0]]), interior, jnp.asarray([0.5 * deltas[-1]])))


def subsample_indices(num_points: int, interval: int) -> np.ndarray:
    """Return a regular subsampling pattern that always keeps the physical aperture endpoints.

    The last endpoint is retained even when ``interval`` does not divide ``num_points - 1``. This avoids changing
    the integration aperture when interval-space downsampling is enabled.
    """
    if interval == 1 or num_points <= 1:
        return np.arange(num_points)
    indices = np.arange(0, num_points, interval)
    if indices[-1] != num_points - 1:
        indices = np.append(indices, num_points - 1)
    return indices


def edge_window_1d(points: jax.Array | np.ndarray, window_size: float) -> jax.Array:
    """Return a Gaussian finite-aperture taper that suppresses only the two edges.

    ``window_size`` is the fractional width of the tapered region over both edges. The central region is left at
    unit weight, which keeps the main aperture amplitude unchanged while reducing edge-diffraction ringing.
    """
    points = jnp.asarray(points, dtype=float)
    if window_size <= 0:
        return jnp.ones(points.shape, dtype=float)
    bound_min = jnp.min(points)
    bound_max = jnp.max(points)
    size = bound_max - bound_min
    transition = float(window_size) * size / 2.0
    window_minus = bound_min + transition
    window_plus = bound_max - transition
    window = jnp.ones(points.shape, dtype=float)
    lower = points < window_minus
    upper = points > window_plus
    lower_window = jnp.exp(-0.5 * _EDGE_TAPER_DECAY * ((points - window_minus) / transition) ** 2)
    upper_window = jnp.exp(-0.5 * _EDGE_TAPER_DECAY * ((points - window_plus) / transition) ** 2)
    window = jnp.where(lower, lower_window, window)
    window = jnp.where(upper, upper_window, window)
    return window


def direct_project_component(
    current: jax.Array | np.ndarray,
    u_coords: jax.Array | np.ndarray,
    v_coords: jax.Array | np.ndarray,
    u_direction: jax.Array | np.ndarray,
    v_direction: jax.Array | np.ndarray,
    normal_direction: jax.Array | np.ndarray,
    *,
    wavenumber: complex,
    normal_offset: float | jax.Array,
) -> jax.Array:
    """Project one equivalent-current component with the far-field separable phase factor.

    For observation direction ``r_hat`` this computes ``int J(u, v) exp(-i k r_hat . r_prime) du dv``. The phase
    separates into u, v, and normal-offset factors for planar detector surfaces, avoiding a large dense
    ``num_sources x num_observations`` temporary in the common far-field path.
    """
    k = jnp.asarray(wavenumber)
    observation_axes = (1,) * u_direction.ndim
    phase_u = jnp.exp(jnp.reshape(-1j * k * u_coords, (u_coords.size, *observation_axes)) * u_direction[None, ...])
    phase_v = jnp.exp(jnp.reshape(-1j * k * v_coords, (v_coords.size, *observation_axes)) * v_direction[None, ...])
    phase_normal = jnp.exp((-1j * k * normal_offset) * normal_direction)
    tmp = jnp.tensordot(current, phase_v, axes=((1,), (0,)))
    return phase_normal * jnp.sum(tmp * phase_u, axis=0)


def _spherical_basis_from_trig(
    sin_theta: jax.Array | np.ndarray,
    cos_theta: jax.Array | np.ndarray,
    sin_phi: jax.Array | np.ndarray,
    cos_phi: jax.Array | np.ndarray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build global spherical basis vectors from precomputed sine and cosine arrays.

    Both tensor-product angle grids and paired angle arrays share this helper. The inputs may already contain
    broadcast dimensions; the output component axis is always first.
    """
    zeros = jnp.zeros_like(sin_theta * cos_phi, dtype=float)
    radial = jnp.stack((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta + zeros), axis=0)
    theta_hat = jnp.stack((cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta + zeros), axis=0)
    phi_hat = jnp.stack((-sin_phi + zeros, cos_phi + zeros, zeros), axis=0)
    return radial, theta_hat, phi_hat


def spherical_basis_grid(
    theta: jax.Array | np.ndarray, phi: jax.Array | np.ndarray
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return ``r_hat``, ``theta_hat``, and ``phi_hat`` on a tensor-product ``theta x phi`` grid.

    The returned arrays have shape ``(3, len(theta), len(phi))`` and use the global spherical convention where
    ``theta`` is measured from +z and ``phi`` from +x in the x-y plane.
    """
    theta = jnp.asarray(theta, dtype=float)
    phi = jnp.asarray(phi, dtype=float)
    return _spherical_basis_from_trig(
        jnp.sin(theta)[:, None], jnp.cos(theta)[:, None], jnp.sin(phi)[None, :], jnp.cos(phi)[None, :]
    )


def spherical_basis_paired(
    theta: jax.Array | np.ndarray, phi: jax.Array | np.ndarray
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return spherical basis vectors for paired ``theta`` and ``phi`` arrays with matching shapes.

    This is used by Cartesian and k-space projections after their observation points are converted to global
    spherical angles.
    """
    theta = jnp.asarray(theta, dtype=float)
    phi = jnp.asarray(phi, dtype=float)
    return _spherical_basis_from_trig(jnp.sin(theta), jnp.cos(theta), jnp.sin(phi), jnp.cos(phi))


def cartesian_to_spherical_angles(points: jax.Array | np.ndarray) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Convert Cartesian observation points to radial distance and global spherical angles.

    ``points`` has component axis first, with shape ``(3, ...)``. The returned arrays keep the trailing observation
    shape so they can feed the paired-angle projection path directly.
    """
    points = jnp.asarray(points, dtype=float)
    radial_distance = jnp.linalg.norm(points, axis=0)
    safe_distance = jnp.maximum(radial_distance, jnp.finfo(radial_distance.dtype).tiny)
    theta = jnp.arccos(jnp.clip(points[2] / safe_distance, -1.0, 1.0))
    phi = jnp.arctan2(points[1], points[0])
    return radial_distance, theta, phi


def exact_cartesian_fields_for_observation(
    *,
    observation_point: jax.Array | np.ndarray,
    electric_current: jax.Array | np.ndarray,
    magnetic_current: jax.Array | np.ndarray,
    source_coordinates: jax.Array | np.ndarray,
    weights: jax.Array | np.ndarray,
    angular_frequency: float,
    wavenumber: complex,
    epsilon: complex,
    permeability: complex,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate finite-distance fields from equivalent surface currents in a homogeneous medium.

    This is the exact projection path used when ``far_field_approx`` is false. It applies the dyadic Green-function
    relations to electric and magnetic surface currents and keeps the near-, intermediate-, and far-field terms via
    ``G``, ``dG/dr``, and ``d2G/dr2``. The caller supplies quadrature weights and source coordinates in metres.
    """
    displacement = observation_point[:, None, None] - source_coordinates
    radius = jnp.linalg.norm(displacement, axis=0)
    radial_hat = displacement / radius[None, :, :]

    ikr = 1j * wavenumber * radius
    green = jnp.exp(ikr) / (4.0 * jnp.pi * radius)
    d_green = green * (ikr - 1.0) / radius
    d2_green = d_green * (ikr - 1.0) / radius + green / (radius**2)

    def grad_dg_dot_current(current: jax.Array | np.ndarray) -> jax.Array:
        """Apply the dyadic Green-function gradient-divergence term to one current sheet."""
        radial_dot_current = jnp.sum(radial_hat * current, axis=0)
        transverse_current = current - radial_hat * radial_dot_current[None, :, :]
        return (
            radial_hat * d2_green[None, :, :] * radial_dot_current[None, :, :]
            + transverse_current * (d_green / radius)[None, :, :]
        )

    radial_cross_electric = jnp.cross(radial_hat, electric_current, axisa=0, axisb=0, axisc=0)
    radial_cross_magnetic = jnp.cross(radial_hat, magnetic_current, axisa=0, axisb=0, axisc=0)

    vector_potential_a = permeability * electric_current * green[None, :, :]
    curl_a = permeability * radial_cross_electric * d_green[None, :, :]
    grad_div_a = permeability * grad_dg_dot_current(electric_current)

    vector_potential_f = epsilon * magnetic_current * green[None, :, :]
    curl_f = epsilon * radial_cross_magnetic * d_green[None, :, :]
    grad_div_f = epsilon * grad_dg_dot_current(magnetic_current)

    electric_integrand = 1j * angular_frequency * (vector_potential_a + grad_div_a / (wavenumber**2)) - curl_f / epsilon
    magnetic_integrand = (
        1j * angular_frequency * (vector_potential_f + grad_div_f / (wavenumber**2)) + curl_a / permeability
    )

    electric_field = jnp.sum(electric_integrand * weights[None, :, :], axis=(1, 2))
    magnetic_field = jnp.sum(magnetic_integrand * weights[None, :, :], axis=(1, 2))
    return electric_field, magnetic_field


def raise_if_exact_observations_overlap_sources(
    *,
    source_coordinates: jax.Array | np.ndarray,
    radial_flat: jax.Array | np.ndarray,
    distance_flat: jax.Array | np.ndarray,
    batch_size: int | None,
) -> None:
    """Reject exact-projection observation points that coincide with source samples in eager mode.

    The homogeneous Green function is singular at zero source-observer distance. Under JAX tracing the Python-side
    value check is skipped, matching the rest of the detector validation strategy for traced coordinate arrays.
    """
    observation_points = distance_flat[None, :] * radial_flat
    coordinate_scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.linalg.norm(observation_points, axis=0)), jnp.max(jnp.abs(source_coordinates))),
        jnp.finfo(float).tiny,
    )
    source_coordinates_flat = source_coordinates.reshape((3, -1))
    tolerance = 64.0 * jnp.finfo(source_coordinates.dtype).eps * coordinate_scale
    if batch_size is None:
        observation_batches = [(0, observation_points.shape[1])]
    else:
        observation_batches = [
            (start, min(start + batch_size, observation_points.shape[1]))
            for start in range(0, observation_points.shape[1], batch_size)
        ]
    try:
        for start, stop in observation_batches:
            observation_batch = observation_points[:, start:stop]
            displacement = observation_batch[:, None, :] - source_coordinates_flat[:, :, None]
            radius = jnp.linalg.norm(displacement, axis=0)
            overlap = np.asarray(radius <= tolerance)
            if bool(np.any(overlap)):
                raise ValueError("exact field projection observation points must not coincide with source points.")
    except jax.errors.TracerArrayConversionError:
        return
