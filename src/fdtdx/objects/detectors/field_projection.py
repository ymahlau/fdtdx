from numbers import Integral
from typing import Any, Literal, Mapping, Self, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core

from fdtdx import constants
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import Material
from fdtdx.objects.detectors.detector import DetectorState
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.typing import SliceTuple3D

ProjectionSurface = Literal["x-", "x+", "y-", "y+", "z-", "z+"]
ProjectionArray = jax.Array | np.ndarray
ProjectionScalar = float | jax.Array

_PROJECTION_FIELD_KEYS = ("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi")
_PROJECTION_RESULT_KEYS = (*_PROJECTION_FIELD_KEYS, "power")
_PROJECTION_SURFACES: tuple[ProjectionSurface, ...] = ("x-", "x+", "y-", "y+", "z-", "z+")
_SURFACE_KEY_SUFFIX = {"-": "minus", "+": "plus"}
_MATERIAL_DIAGONAL_INDICES = (0, 4, 8)
_MATERIAL_OFF_DIAGONAL_INDICES = (1, 2, 3, 5, 6, 7)
_SURFACE_AXIS_DIRECTIONS: dict[ProjectionSurface, tuple[int, Literal["+", "-"]]] = {
    "x-": (0, "-"),
    "x+": (0, "+"),
    "y-": (1, "-"),
    "y+": (1, "+"),
    "z-": (2, "-"),
    "z+": (2, "+"),
}
_SURFACE_NAMES_BY_AXIS_DIRECTION: dict[tuple[int, Literal["+", "-"]], ProjectionSurface] = {
    value: key for key, value in _SURFACE_AXIS_DIRECTIONS.items()
}


def _surface_axis_direction(surface: ProjectionSurface) -> tuple[int, Literal["+", "-"]]:
    return _SURFACE_AXIS_DIRECTIONS[surface]


def _surface_name(axis: int, direction: Literal["+", "-"]) -> ProjectionSurface:
    return _SURFACE_NAMES_BY_AXIS_DIRECTION[(axis, direction)]


def _surface_state_key(surface: ProjectionSurface) -> str:
    return f"phasor_{surface[0]}_{_SURFACE_KEY_SUFFIX[surface[1]]}"


def _contains_bool(values: Any) -> bool:
    if isinstance(values, (bool, np.bool_)):
        return True
    if isinstance(values, jax.Array):
        return bool(np.issubdtype(values.dtype, np.bool_))
    if isinstance(values, jax_core.Tracer):
        return False
    if isinstance(values, np.ndarray):
        if np.issubdtype(values.dtype, np.bool_):
            return True
        if values.dtype != object:
            return False
    if isinstance(values, (str, bytes)):
        return False
    try:
        iterator = iter(values)
    except TypeError:
        return False
    return any(_contains_bool(value) for value in iterator)


def _is_jax_tracer(values: Any) -> bool:
    return isinstance(values, jax_core.Tracer)


def _is_real_numeric_dtype(dtype: Any) -> bool:
    return np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.complexfloating)


def _concrete_jax_bool(value: jax.Array) -> bool:
    # Eager concrete JAX arrays can be value-checked; traced values cannot raise Python validation errors.
    try:
        return bool(value)
    except jax.errors.TracerBoolConversionError:
        return False


def _concrete_jax_array_has_nonfinite(values: jax.Array) -> bool:
    return _concrete_jax_bool(jnp.any(~jnp.isfinite(values)))


def _finite_1d_array(name: str, values: Any, expected_size: int) -> np.ndarray:
    if _contains_bool(values):
        raise ValueError(f"{name} must contain finite numeric values.")
    try:
        raw_array = np.asarray(values)
    except jax.errors.TracerArrayConversionError as err:
        raise ValueError(f"{name} must contain finite numeric values.") from err
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must contain finite numeric values.") from err
    if not _is_real_numeric_dtype(raw_array.dtype):
        raise ValueError(f"{name} must contain finite numeric values.")
    array = raw_array.astype(float)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must contain {expected_size} values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values.")
    return array


def _finite_numeric_array(name: str, values: Any) -> jax.Array:
    if _contains_bool(values):
        raise ValueError(f"{name} must contain finite numeric values.")
    if _is_jax_tracer(values):
        if not _is_real_numeric_dtype(values.dtype):
            raise ValueError(f"{name} must contain finite numeric values.")
        return jnp.asarray(values, dtype=float)
    if isinstance(values, jax.Array):
        if not _is_real_numeric_dtype(values.dtype):
            raise ValueError(f"{name} must contain finite numeric values.")
        array = jnp.asarray(values, dtype=float)
        if _concrete_jax_array_has_nonfinite(array):
            raise ValueError(f"{name} must contain finite numeric values.")
        return array
    try:
        raw_array = np.asarray(values)
    except jax.errors.TracerArrayConversionError:
        return jnp.asarray(values)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must contain finite numeric values.") from err
    if not _is_real_numeric_dtype(raw_array.dtype):
        raise ValueError(f"{name} must contain finite numeric values.")
    array = raw_array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite numeric values.")
    return jnp.asarray(array)


def _finite_scalar(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite numeric value.")
    if not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite numeric value.")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a finite numeric value.") from err
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite numeric value.")
    return scalar


def _finite_scalar_or_1d_array(name: str, value: Any, expected_size: int) -> float | np.ndarray:
    if _contains_bool(value):
        raise ValueError(f"{name} must be a finite numeric scalar or contain {expected_size} values.")
    try:
        raw_array = np.asarray(value)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{name} must be a finite numeric scalar or contain {expected_size} values.") from err
    if not _is_real_numeric_dtype(raw_array.dtype):
        raise ValueError(f"{name} must be a finite numeric scalar or contain {expected_size} values.")
    array = raw_array.astype(float)
    if array.ndim == 0:
        scalar = float(array)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be a finite numeric scalar or contain {expected_size} values.")
        return scalar
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must be a finite numeric scalar or contain {expected_size} values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite numeric scalar or contain {expected_size} values.")
    return array


def _positive_projection_parameter(name: str, value: Any, expected_size: int) -> float | np.ndarray:
    parameter = _finite_scalar_or_1d_array(name, value, expected_size)
    if np.any(np.asarray(parameter) <= 0):
        raise ValueError(f"{name} must be positive.")
    return parameter


def _projection_parameter_value(value: float | Sequence[float], index: int) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return float(array[index])


def _projection_parameter_is_default(name: str, value: Any, default: float, expected_size: int) -> bool:
    parameter = _positive_projection_parameter(name, value, expected_size)
    return bool(np.allclose(np.asarray(parameter, dtype=float), default))


def _isotropic_material_property_scalar(name: str, values: Any) -> float:
    if _contains_bool(values):
        raise ValueError(f"projection_medium {name} must be a finite real isotropic value.")
    raw_array = np.asarray(values)
    if raw_array.shape != (9,) or not np.issubdtype(raw_array.dtype, np.number):
        raise ValueError(f"projection_medium {name} must be a finite real isotropic value.")
    if np.iscomplexobj(raw_array) and np.any(np.imag(raw_array) != 0.0):
        raise ValueError(f"projection_medium {name} must be real.")
    array = raw_array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"projection_medium {name} must contain finite values.")
    if not np.allclose(array[list(_MATERIAL_OFF_DIAGONAL_INDICES)], 0.0):
        raise ValueError(f"projection_medium {name} must be isotropic.")
    diagonal = array[list(_MATERIAL_DIAGONAL_INDICES)]
    if not np.allclose(diagonal, diagonal[0]):
        raise ValueError(f"projection_medium {name} must be isotropic.")
    return float(diagonal[0])


def _passive_complex_sqrt(value: complex) -> complex:
    root = complex(np.sqrt(complex(value)))
    if root.imag < 0.0 or (np.isclose(root.imag, 0.0) and root.real < 0.0):
        root = -root
    return root


def _positive_impedance_sqrt(value: complex) -> complex:
    root = complex(np.sqrt(complex(value)))
    if root.real < 0.0 or (np.isclose(root.real, 0.0) and root.imag < 0.0):
        root = -root
    return root


def _trapz_weights_1d(points: ProjectionArray) -> jax.Array:
    points = jnp.asarray(points, dtype=float)
    if points.size <= 1:
        return jnp.ones(points.shape, dtype=float)
    deltas = jnp.abs(jnp.diff(points))
    interior = 0.5 * (deltas[:-1] + deltas[1:])
    return jnp.concatenate((jnp.asarray([0.5 * deltas[0]]), interior, jnp.asarray([0.5 * deltas[-1]])))


def _subsample_indices(num_points: int, interval: int) -> np.ndarray:
    if interval == 1 or num_points <= 1:
        return np.arange(num_points)
    indices = np.arange(0, num_points, interval)
    if indices[-1] != num_points - 1:
        indices = np.append(indices, num_points - 1)
    return indices


def _edge_window_1d(points: ProjectionArray, window_size: float) -> jax.Array:
    points = jnp.asarray(points, dtype=float)
    if window_size <= 0:
        return jnp.ones(points.shape, dtype=float)
    window_factor = 15.0
    bound_min = jnp.min(points)
    bound_max = jnp.max(points)
    size = bound_max - bound_min
    transition = float(window_size) * size / 2.0
    window_minus = bound_min + transition
    window_plus = bound_max - transition
    window = jnp.ones(points.shape, dtype=float)
    lower = points < window_minus
    upper = points > window_plus
    lower_window = jnp.exp(-0.5 * window_factor * ((points - window_minus) / transition) ** 2)
    upper_window = jnp.exp(-0.5 * window_factor * ((points - window_plus) / transition) ** 2)
    window = jnp.where(lower, lower_window, window)
    window = jnp.where(upper, upper_window, window)
    return window


def _direct_project_component(
    current: ProjectionArray,
    u_coords: ProjectionArray,
    v_coords: ProjectionArray,
    u_direction: ProjectionArray,
    v_direction: ProjectionArray,
    normal_direction: ProjectionArray,
    *,
    wavenumber: complex,
    normal_offset: ProjectionScalar,
) -> jax.Array:
    k = jnp.asarray(wavenumber)
    observation_axes = (1,) * u_direction.ndim
    phase_u = jnp.exp(jnp.reshape(-1j * k * u_coords, (u_coords.size, *observation_axes)) * u_direction[None, ...])
    phase_v = jnp.exp(jnp.reshape(-1j * k * v_coords, (v_coords.size, *observation_axes)) * v_direction[None, ...])
    phase_normal = jnp.exp((-1j * k * normal_offset) * normal_direction)
    tmp = jnp.tensordot(current, phase_v, axes=((1,), (0,)))
    return phase_normal * jnp.sum(tmp * phase_u, axis=0)


def _global_spherical_basis(theta: ProjectionArray, phi: ProjectionArray) -> tuple[jax.Array, jax.Array, jax.Array]:
    theta = jnp.asarray(theta, dtype=float)
    phi = jnp.asarray(phi, dtype=float)
    sin_theta = jnp.sin(theta)[:, None]
    cos_theta = jnp.cos(theta)[:, None]
    sin_phi = jnp.sin(phi)[None, :]
    cos_phi = jnp.cos(phi)[None, :]
    grid_shape = (theta.size, phi.size)
    ones = jnp.ones(grid_shape, dtype=float)
    radial = jnp.stack(
        (
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta * ones,
        ),
        axis=0,
    )
    theta_hat = jnp.stack(
        (
            cos_theta * cos_phi,
            cos_theta * sin_phi,
            -sin_theta * ones,
        ),
        axis=0,
    )
    phi_hat = jnp.stack(
        (
            -ones * sin_phi,
            ones * cos_phi,
            jnp.zeros(grid_shape, dtype=float),
        ),
        axis=0,
    )
    return radial, theta_hat, phi_hat


def _global_spherical_basis_paired(
    theta: ProjectionArray,
    phi: ProjectionArray,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    theta = jnp.asarray(theta, dtype=float)
    phi = jnp.asarray(phi, dtype=float)
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    sin_phi = jnp.sin(phi)
    cos_phi = jnp.cos(phi)
    radial = jnp.stack(
        (
            sin_theta * cos_phi,
            sin_theta * sin_phi,
            cos_theta,
        ),
        axis=0,
    )
    theta_hat = jnp.stack(
        (
            cos_theta * cos_phi,
            cos_theta * sin_phi,
            -sin_theta,
        ),
        axis=0,
    )
    phi_hat = jnp.stack(
        (
            -sin_phi,
            cos_phi,
            jnp.zeros_like(theta, dtype=float),
        ),
        axis=0,
    )
    return radial, theta_hat, phi_hat


def _cartesian_to_spherical_angles(points: ProjectionArray) -> tuple[jax.Array, jax.Array, jax.Array]:
    points = jnp.asarray(points, dtype=float)
    radial_distance = jnp.linalg.norm(points, axis=0)
    theta = jnp.arccos(jnp.clip(points[2] / radial_distance, -1.0, 1.0))
    phi = jnp.arctan2(points[1], points[0])
    return radial_distance, theta, phi


def _projection_transverse_axes(projection_axis: int) -> tuple[int, int]:
    transverse_axes = [axis for axis in range(3) if axis != projection_axis]
    return transverse_axes[0], transverse_axes[1]


def _validate_theta_range(theta: ProjectionArray) -> None:
    if _is_jax_tracer(theta):
        return
    if isinstance(theta, jax.Array):
        if _concrete_jax_bool(jnp.any((theta < 0.0) | (theta > jnp.pi))):
            raise ValueError("theta values must be in the interval [0, pi].")
        return
    try:
        theta_array = np.asarray(theta)
    except jax.errors.TracerArrayConversionError:
        return
    if np.any((theta_array < 0.0) | (theta_array > np.pi)):
        raise ValueError("theta values must be in the interval [0, pi].")


def _validate_positive_values(name: str, values: ProjectionArray) -> None:
    if _is_jax_tracer(values):
        return
    if isinstance(values, jax.Array):
        if _concrete_jax_bool(jnp.any(values <= 0.0)):
            raise ValueError(f"{name} values must be positive.")
        return
    if np.any(np.asarray(values) <= 0.0):
        raise ValueError(f"{name} values must be positive.")


def _validate_projection_axis(projection_axis: Any) -> int:
    if (
        not isinstance(projection_axis, Integral)
        or isinstance(projection_axis, bool)
        or projection_axis not in (0, 1, 2)
    ):
        raise ValueError("projection_axis must be 0, 1, or 2.")
    return int(projection_axis)


@autoinit
class FieldProjectionDetectorBase(PhasorDetector):
    """Shared base for frequency-domain near-to-far field projection detectors.

    The detector records time-interpolated complex E/H phasors using the same
    accumulation path as :class:`~fdtdx.objects.detectors.phasor.PhasorDetector`.
    After the FDTD run, concrete subclasses convert those phasors into
    equivalent surface currents and project them to their requested observation
    coordinates. This base owns the shared surface/box placement, equivalent
    current construction, homogeneous medium model, far-field kernel, exact
    Green's-function kernel, and coherent multi-surface summation machinery.

    Field projection detectors can be placed either on a single planar surface
    or on a box volume. A planar detector has exactly one grid dimension of
    size one and requires ``direction`` to select the outward normal. A box
    detector has three non-singleton grid dimensions, records the six outer
    surfaces, and may omit selected surfaces with ``exclude_surfaces``.
    """

    #: Direction of the outward detector normal for a single planar detector
    #: surface. Must be ``None`` for a box-volume projection.
    direction: Literal["+", "-"] | None = frozen_field(default=None)

    #: Box surfaces to exclude from a box-volume projection.
    #: Valid entries are ``"x-"``, ``"x+"``, ``"y-"``, ``"y+"``, ``"z-"``,
    #: and ``"z+"``. Ignored for planar detectors.
    exclude_surfaces: tuple[ProjectionSurface, ...] = frozen_field(default=())

    #: Origin used for the projection phase reference. If ``None``, the center
    #: of the placed detector region is used.
    origin: tuple[float, float, float] | None = frozen_field(default=None)

    #: Projection distance from ``origin`` to the observation points, in meters.
    #: For angle and k-space projections this is a radial distance. For
    #: Cartesian projections this is the offset of the observation plane along
    #: ``projection_axis``.
    projection_distance: float = frozen_field(default=1.0)

    #: Whether to use the far-field approximation. When False, the detector
    #: evaluates the full homogeneous-medium Green's function at finite distance.
    far_field_approx: bool = frozen_field(default=True)

    #: Observation points per XLA batch for exact finite-distance projection.
    #: The default keeps peak temporary memory bounded for large observation
    #: grids. Set to ``None`` to project all observation points in one vectorized
    #: operation.
    exact_projection_batch_size: int | None = frozen_field(default=128)

    #: Relative Gaussian edge-window size along the two transverse detector axes.
    #: This can reduce finite-aperture ringing for single planar detectors. Box
    #: projections require the default ``(0.0, 0.0)``.
    window_size: tuple[float, float] = frozen_field(default=(0.0, 0.0))

    #: Spatial sampling interval along x, y, and z.
    #: The first and last points are always retained.
    interval_space: tuple[int, int, int] = frozen_field(default=(1, 1, 1))

    #: Homogeneous projection medium. Only uniform isotropic materials can be
    #: represented by the scalar Green's function used by this detector.
    projection_medium: Material | None = frozen_field(default=None)

    #: Refractive index of the homogeneous non-magnetic projection medium.
    #: May be a scalar or one value per wave character.
    #: Ignored when ``projection_medium`` is set.
    projection_medium_refractive_index: float | Sequence[float] = frozen_field(default=1.0)

    #: Wave impedance of the homogeneous projection medium.
    #: May be a scalar or one value per wave character.
    #: Ignored when ``projection_medium`` is set.
    projection_medium_impedance: float | Sequence[float] | None = frozen_field(default=None)

    #: Far-field projection needs all six phasor components on the full detector plane.
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        init=False,
    )

    #: Field projection always keeps the full recorded surface data.
    reduce_volume: bool = frozen_field(default=False, init=False)

    #: Field projection results are returned by ``project`` and ``project_all``.
    plot: bool = frozen_field(default=False, init=False)

    #: Field projection always uses FDTDX's exact E/H detector interpolation.
    exact_interpolation: bool = frozen_field(default=True, init=False)

    def __post_init__(self):
        super().__post_init__()
        if len(self.wave_characters) == 0:
            raise ValueError("wave_characters must contain at least one wave character.")
        if self.direction is not None and self.direction not in ["+", "-"]:
            raise ValueError("direction must be '+', '-', or None.")
        self._validate_exclude_surfaces()
        if self.origin is not None:
            _finite_1d_array("origin", self.origin, 3)
        projection_distance = _finite_scalar("projection_distance", self.projection_distance)
        if projection_distance <= 0:
            raise ValueError("projection_distance must be positive.")
        if not isinstance(self.far_field_approx, bool):
            raise ValueError("far_field_approx must be a boolean.")
        if self.exact_projection_batch_size is not None and (
            not isinstance(self.exact_projection_batch_size, Integral)
            or isinstance(self.exact_projection_batch_size, bool)
            or self.exact_projection_batch_size <= 0
        ):
            raise ValueError("exact_projection_batch_size must be a positive integer or None.")

        window_size = _finite_1d_array("window_size", self.window_size, 2)
        if np.any((window_size < 0) | (window_size > 1)):
            raise ValueError("window_size values must be in the interval [0, 1].")
        if len(self.interval_space) != 3:
            raise ValueError("interval_space must contain one value for each spatial axis.")
        if any(
            not isinstance(interval, Integral) or isinstance(interval, bool) or interval <= 0
            for interval in self.interval_space
        ):
            raise ValueError("interval_space values must be positive integers.")
        if self.projection_medium is not None:
            self._validate_projection_medium()
            if not _projection_parameter_is_default(
                "projection_medium_refractive_index",
                self.projection_medium_refractive_index,
                1.0,
                len(self.wave_characters),
            ):
                raise ValueError(
                    "projection_medium_refractive_index must not be set when projection_medium is provided."
                )
            if self.projection_medium_impedance is not None:
                raise ValueError("projection_medium_impedance must not be set when projection_medium is provided.")
        else:
            _positive_projection_parameter(
                "projection_medium_refractive_index",
                self.projection_medium_refractive_index,
                len(self.wave_characters),
            )
            if self.projection_medium_impedance is not None:
                _positive_projection_parameter(
                    "projection_medium_impedance",
                    self.projection_medium_impedance,
                    len(self.wave_characters),
                )

    def _validate_projection_medium(self) -> None:
        if not isinstance(self.projection_medium, Material):
            raise ValueError("projection_medium must be a Material.")
        eps_inf = _isotropic_material_property_scalar("permittivity", self.projection_medium.permittivity)
        permeability = _isotropic_material_property_scalar("permeability", self.projection_medium.permeability)
        electric_conductivity = _isotropic_material_property_scalar(
            "electric_conductivity",
            self.projection_medium.electric_conductivity,
        )
        magnetic_conductivity = _isotropic_material_property_scalar(
            "magnetic_conductivity",
            self.projection_medium.magnetic_conductivity,
        )
        if eps_inf == 0:
            raise ValueError("projection_medium permittivity must be non-zero.")
        if permeability == 0:
            raise ValueError("projection_medium permeability must be non-zero.")
        if electric_conductivity < 0:
            raise ValueError("projection_medium electric_conductivity must be non-negative.")
        if magnetic_conductivity < 0:
            raise ValueError("projection_medium magnetic_conductivity must be non-negative.")

    def _validate_exclude_surfaces(self) -> None:
        if isinstance(self.exclude_surfaces, (str, bytes)):
            raise ValueError("exclude_surfaces must contain valid box surface names.")
        try:
            exclude_surfaces = tuple(self.exclude_surfaces)
        except TypeError as err:
            raise ValueError("exclude_surfaces must contain valid box surface names.") from err
        invalid_surfaces = [surface for surface in exclude_surfaces if surface not in _PROJECTION_SURFACES]
        if invalid_surfaces:
            raise ValueError(f"exclude_surfaces contains invalid surfaces: {invalid_surfaces}.")
        if len(set(exclude_surfaces)) != len(exclude_surfaces):
            raise ValueError("exclude_surfaces must not contain duplicate surfaces.")
        if len(exclude_surfaces) == len(_PROJECTION_SURFACES):
            raise ValueError("exclude_surfaces must not exclude every box surface.")

    @property
    def _projection_mode(self) -> Literal["surface", "box"]:
        unit_axes = sum(axis_size == 1 for axis_size in self.grid_shape)
        if unit_axes == 1:
            return "surface"
        if unit_axes == 0:
            return "box"
        raise Exception(
            "Invalid field projection detector shape: "
            f"{self.grid_shape}. Expected a single planar surface or a box volume."
        )

    def _included_box_surfaces(self) -> tuple[ProjectionSurface, ...]:
        excluded_surfaces = set(self.exclude_surfaces)
        surfaces = tuple(surface for surface in _PROJECTION_SURFACES if surface not in excluded_surfaces)
        if len(surfaces) == 0:
            raise ValueError("exclude_surfaces must not exclude every box surface.")
        return surfaces

    @property
    def propagation_axis(self) -> int:
        """Return the detector-plane normal axis."""
        if self._projection_mode != "surface":
            raise Exception(f"Invalid field projection detector surface shape: {self.grid_shape}")
        return self.grid_shape.index(1)

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(grid_slice_tuple=grid_slice_tuple, config=config, key=key)
        if self._projection_mode == "surface":
            if self.direction is None:
                raise ValueError("direction must be specified for a planar field projection detector.")
            if len(self.exclude_surfaces) > 0:
                raise ValueError("exclude_surfaces is valid only for box field projection detector placement.")
        else:
            if self.direction is not None:
                raise ValueError("direction must be None for box field projection detector placement.")
            if np.any(np.asarray(self.window_size, dtype=float) != 0.0):
                raise ValueError("window_size must be (0, 0) for box field projection detector placement.")
            _ = self._included_box_surfaces()
        return self

    def _surface_grid_shape(self, surface: ProjectionSurface) -> tuple[int, int, int]:
        axis, _ = _surface_axis_direction(surface)
        shape = list(self.grid_shape)
        shape[axis] = 1
        return shape[0], shape[1], shape[2]

    def _shape_dtype_single_time_step(self) -> dict[str, jax.ShapeDtypeStruct]:
        field_dtype = jnp.complex128 if self.dtype == jnp.complex128 else jnp.complex64
        num_components = len(self.components)
        num_frequencies = len(self._angular_frequencies)
        if self._projection_mode == "surface":
            return {
                "phasor": jax.ShapeDtypeStruct(
                    shape=(num_frequencies, num_components, *self.grid_shape),
                    dtype=field_dtype,
                )
            }
        return {
            _surface_state_key(surface): jax.ShapeDtypeStruct(
                shape=(num_frequencies, num_components, *self._surface_grid_shape(surface)),
                dtype=field_dtype,
            )
            for surface in self._included_box_surfaces()
        }

    def update(
        self,
        time_step: jax.Array,
        E: jax.Array,
        H: jax.Array,
        state: DetectorState,
        inv_permittivity: jax.Array,
        inv_permeability: jax.Array | float,
    ) -> DetectorState:
        if self._projection_mode == "surface":
            return super().update(
                time_step=time_step,
                E=E,
                H=H,
                state=state,
                inv_permittivity=inv_permittivity,
                inv_permeability=inv_permeability,
            )

        del inv_permeability, inv_permittivity
        time_passed = time_step * self._config.time_step_duration
        if self.scaling_mode == "continuous":
            static_scale = 2 / self.num_time_steps_recorded
        elif self.scaling_mode == "pulse":
            static_scale = 1
        else:
            raise Exception(f"Invalid scaling mode: {self.scaling_mode=}")

        e_local = E[:, *self.grid_slice]
        h_local = H[:, *self.grid_slice]
        fields = jnp.stack((e_local[0], e_local[1], e_local[2], h_local[0], h_local[1], h_local[2]), axis=0)
        phase_angles = self._angular_frequencies * time_passed
        phasors = jnp.exp(1j * phase_angles)
        phasors = phasors.reshape((len(self._angular_frequencies),) + (1,) * fields.ndim)

        new_state: DetectorState = {}
        for surface in self._included_box_surfaces():
            axis, direction = _surface_axis_direction(surface)
            face_slices: list[slice] = [slice(None), slice(None), slice(None), slice(None)]
            face_slices[axis + 1] = slice(0, 1) if direction == "-" else slice(self.grid_shape[axis] - 1, None)
            face_fields = fields[tuple(face_slices)]
            new_phasors = face_fields * phasors * static_scale
            state_key = _surface_state_key(surface)
            if self.inverse:
                result = state[state_key] - new_phasors[None, ...]
            else:
                result = state[state_key] + new_phasors[None, ...]
            new_state[state_key] = result.astype(self.dtype)
        return new_state

    def _axis_centers(self, axis: int) -> jax.Array:
        grid = self._config.resolved_grid
        lower, upper = self.grid_slice_tuple[axis]
        if grid is not None:
            return jnp.asarray(grid.centers(axis)[lower:upper], dtype=float)
        spacing = self._config.uniform_spacing()
        return (jnp.arange(lower, upper, dtype=float) + 0.5) * spacing

    def _local_axes_for_surface(
        self,
        axis: int,
        direction: Literal["+", "-"],
    ) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
        axes = ((axis + 1) % 3, (axis + 2) % 3, axis)
        if direction == "+":
            return axes, (1.0, 1.0, 1.0)
        return axes, (1.0, -1.0, -1.0)

    def _local_axes(self) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
        if self.direction is None:
            raise ValueError("direction must be specified for a planar field projection detector.")
        return self._local_axes_for_surface(self.propagation_axis, self.direction)

    def _state_phasor(self, state: DetectorState) -> jax.Array:
        if "phasor" not in state:
            raise ValueError("state must contain a 'phasor' entry.")
        phasor = jnp.asarray(state["phasor"])
        expected_shape = (1, len(self.wave_characters), 6, *self.grid_shape)
        if phasor.shape != expected_shape:
            raise ValueError(f"state['phasor'] must have shape {expected_shape}, got {phasor.shape}.")
        return phasor

    def _state_surface_phasor(self, state: DetectorState, surface: ProjectionSurface) -> jax.Array:
        state_key = _surface_state_key(surface)
        if state_key not in state:
            raise ValueError(f"state must contain a '{state_key}' entry.")
        phasor = jnp.asarray(state[state_key])
        expected_shape = (1, len(self.wave_characters), 6, *self._surface_grid_shape(surface))
        if phasor.shape != expected_shape:
            raise ValueError(f"state['{state_key}'] must have shape {expected_shape}, got {phasor.shape}.")
        return phasor

    def _local_plane_fields(
        self,
        state: DetectorState,
        wave_character_index: int,
        surface: ProjectionSurface | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if surface is None:
            phasor = self._state_phasor(state)[0, wave_character_index]
            propagation_axis = self.propagation_axis
            if self.direction is None:
                raise ValueError("direction must be specified for a planar field projection detector.")
            direction = self.direction
        else:
            phasor = self._state_surface_phasor(state, surface)[0, wave_character_index]
            propagation_axis, direction = _surface_axis_direction(surface)
        e_global = constants.eta0 * phasor[:3]
        h_global = phasor[3:6]

        remaining_axes = [axis for axis in range(3) if axis != propagation_axis]
        local_axes, signs = self._local_axes_for_surface(propagation_axis, direction)
        spatial_order = [remaining_axes.index(axis) for axis in local_axes[:2]]

        e_plane = jnp.squeeze(e_global, axis=propagation_axis + 1)
        h_plane = jnp.squeeze(h_global, axis=propagation_axis + 1)
        e_plane = jnp.transpose(e_plane, (0, spatial_order[0] + 1, spatial_order[1] + 1))
        h_plane = jnp.transpose(h_plane, (0, spatial_order[0] + 1, spatial_order[1] + 1))

        e_local = jnp.stack([signs[i] * e_plane[axis] for i, axis in enumerate(local_axes)], axis=0)
        h_local = jnp.stack([signs[i] * h_plane[axis] for i, axis in enumerate(local_axes)], axis=0)
        return e_local, h_local

    def _local_coordinates_for_surface(
        self,
        axis: int,
        direction: Literal["+", "-"],
    ) -> tuple[jax.Array, jax.Array, ProjectionScalar]:
        local_axes, signs = self._local_axes_for_surface(axis, direction)
        axis_u = self._axis_centers(local_axes[0])
        axis_v = self._axis_centers(local_axes[1])
        axis_n = self._axis_centers(local_axes[2])
        surface_n = axis_n[0] if direction == "-" else axis_n[-1]
        if self.origin is None:
            origin_u = 0.5 * (axis_u[0] + axis_u[-1])
            origin_v = 0.5 * (axis_v[0] + axis_v[-1])
            origin_n = 0.5 * (axis_n[0] + axis_n[-1])
        else:
            origin_u = self.origin[local_axes[0]]
            origin_v = self.origin[local_axes[1]]
            origin_n = self.origin[local_axes[2]]
        u_coords = signs[0] * (axis_u - origin_u)
        v_coords = signs[1] * (axis_v - origin_v)
        normal_offset = signs[2] * (surface_n - origin_n)
        return u_coords, v_coords, normal_offset

    def _local_coordinates(self) -> tuple[jax.Array, jax.Array, ProjectionScalar]:
        if self.direction is None:
            raise ValueError("direction must be specified for a planar field projection detector.")
        return self._local_coordinates_for_surface(self.propagation_axis, self.direction)

    def _local_transverse_coordinates(self) -> tuple[jax.Array, jax.Array]:
        u_coords, v_coords, _ = self._local_coordinates()
        return u_coords, v_coords

    def _projection_material_relative_permittivity_permeability(
        self, wave_character_index: int
    ) -> tuple[complex, complex]:
        if self.projection_medium is None:
            raise ValueError("projection_medium must be provided.")
        frequency = float(self.wave_characters[wave_character_index].get_frequency())
        omega = 2.0 * np.pi * frequency
        if omega <= 0 or not np.isfinite(omega):
            raise ValueError("wave character frequency must be finite and positive.")

        eps_inf = _isotropic_material_property_scalar("permittivity", self.projection_medium.permittivity)
        mu_r = _isotropic_material_property_scalar("permeability", self.projection_medium.permeability)
        sigma_e = _isotropic_material_property_scalar(
            "electric_conductivity",
            self.projection_medium.electric_conductivity,
        )
        sigma_m = _isotropic_material_property_scalar(
            "magnetic_conductivity",
            self.projection_medium.magnetic_conductivity,
        )

        if self.projection_medium.dispersion is None:
            eps_complex = complex(eps_inf)
        else:
            eps_complex = complex(self.projection_medium.dispersion.permittivity(omega, eps_inf=eps_inf))
        eps_complex = eps_complex + 1j * sigma_e / (omega * constants.eps0)
        mu_complex = complex(mu_r) + 1j * sigma_m / (omega * constants.mu0)
        if eps_complex == 0:
            raise ValueError("projection_medium permittivity is zero at the selected frequency.")
        if mu_complex == 0:
            raise ValueError("projection_medium permeability is zero at the selected frequency.")
        return eps_complex, mu_complex

    def _projection_relative_permittivity_permeability(self, wave_character_index: int) -> tuple[complex, complex]:
        if self.projection_medium is not None:
            return self._projection_material_relative_permittivity_permeability(wave_character_index)

        refractive_index = _projection_parameter_value(
            self.projection_medium_refractive_index,
            wave_character_index,
        )
        if self.projection_medium_impedance is None:
            impedance = constants.eta0 / refractive_index
        else:
            impedance = _projection_parameter_value(self.projection_medium_impedance, wave_character_index)
        eps_complex = constants.eta0 * refractive_index / impedance
        mu_complex = refractive_index * impedance / constants.eta0
        return complex(eps_complex), complex(mu_complex)

    def _projection_material_parameters(self, wave_character_index: int) -> tuple[complex, complex, complex]:
        eps_complex, mu_complex = self._projection_material_relative_permittivity_permeability(wave_character_index)
        refractive_index = _passive_complex_sqrt(eps_complex * mu_complex)
        impedance = constants.eta0 * _positive_impedance_sqrt(mu_complex / eps_complex)
        wavenumber = (2.0 * np.pi / self.wave_characters[wave_character_index].get_wavelength()) * refractive_index
        return refractive_index, impedance, wavenumber

    def _projection_refractive_index(self, wave_character_index: int) -> complex:
        if self.projection_medium is not None:
            refractive_index, _, _ = self._projection_material_parameters(wave_character_index)
            return refractive_index
        return _projection_parameter_value(self.projection_medium_refractive_index, wave_character_index)

    def _projection_impedance(self, wave_character_index: int) -> complex:
        if self.projection_medium is not None:
            _, impedance, _ = self._projection_material_parameters(wave_character_index)
            return impedance
        if self.projection_medium_impedance is not None:
            return _projection_parameter_value(self.projection_medium_impedance, wave_character_index)
        return constants.eta0 / self._projection_refractive_index(wave_character_index)

    def _projection_wavenumber(self, wave_character_index: int) -> complex:
        if self.projection_medium is not None:
            _, _, wavenumber = self._projection_material_parameters(wave_character_index)
            return wavenumber
        return (
            2.0
            * np.pi
            * self._projection_refractive_index(wave_character_index)
            / self.wave_characters[wave_character_index].get_wavelength()
        )

    def _projection_wavelength(self, wave_character_index: int) -> complex:
        return self.wave_characters[wave_character_index].get_wavelength() / self._projection_refractive_index(
            wave_character_index
        )

    def _propagation_factor(self, wave_character_index: int) -> jax.Array:
        """Return the far-field propagation prefactor for a 3D homogeneous medium."""
        k = self._projection_wavenumber(wave_character_index)
        return -1j * k * jnp.exp(1j * k * self.projection_distance) / (4.0 * jnp.pi * self.projection_distance)

    def _projection_metadata(self, wave_character_index: int) -> dict[str, float | complex | int]:
        wave_character = self.wave_characters[wave_character_index]
        return {
            "wave_character_index": int(wave_character_index),
            "frequency": float(wave_character.get_frequency()),
            "free_space_wavelength": float(wave_character.get_wavelength()),
            "projection_distance": _finite_scalar("projection_distance", self.projection_distance),
            "far_field_approx": self.far_field_approx,
            "projection_medium_refractive_index": self._projection_refractive_index(wave_character_index),
            "projection_medium_impedance": self._projection_impedance(wave_character_index),
            "projection_wavenumber": self._projection_wavenumber(wave_character_index),
            "projection_wavelength": self._projection_wavelength(wave_character_index),
        }

    def _validate_wave_character_index(self, wave_character_index: int) -> int:
        if not isinstance(wave_character_index, Integral):
            raise ValueError("wave_character_index must be an integer.")
        if wave_character_index < 0 or wave_character_index >= len(self.wave_characters):
            raise ValueError("wave_character_index is out of range for this detector.")
        return int(wave_character_index)

    def _surface_currents_geometry(
        self,
        *,
        e_field: ProjectionArray,
        h_field: ProjectionArray,
        u_coords: ProjectionArray,
        v_coords: ProjectionArray,
        normal_offset: ProjectionScalar,
        propagation_axis: int,
        direction: Literal["+", "-"],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        local_axes, signs = self._local_axes_for_surface(propagation_axis, direction)
        u_indices = _subsample_indices(u_coords.size, int(self.interval_space[local_axes[0]]))
        v_indices = _subsample_indices(v_coords.size, int(self.interval_space[local_axes[1]]))
        e_field = e_field[:, u_indices][:, :, v_indices]
        h_field = h_field[:, u_indices][:, :, v_indices]
        u_coords = u_coords[u_indices]
        v_coords = v_coords[v_indices]

        u_weights = _trapz_weights_1d(u_coords) * _edge_window_1d(u_coords, self.window_size[0])
        v_weights = _trapz_weights_1d(v_coords) * _edge_window_1d(v_coords, self.window_size[1])
        weights = u_weights[:, None] * v_weights[None, :]

        electric_current_local_u = -h_field[1]
        electric_current_local_v = h_field[0]
        magnetic_current_local_u = e_field[1]
        magnetic_current_local_v = -e_field[0]

        current_shape = electric_current_local_u.shape
        zeros_current = jnp.zeros(current_shape, dtype=complex)
        zeros_coordinates = jnp.zeros(current_shape, dtype=float)
        electric_local = (electric_current_local_u, electric_current_local_v, zeros_current)
        magnetic_local = (magnetic_current_local_u, magnetic_current_local_v, zeros_current)
        source_local = (
            u_coords[:, None] * jnp.ones((1, v_coords.size), dtype=float),
            v_coords[None, :] * jnp.ones((u_coords.size, 1), dtype=float),
            normal_offset * jnp.ones(current_shape, dtype=float),
        )

        electric_components = [zeros_current, zeros_current, zeros_current]
        magnetic_components = [zeros_current, zeros_current, zeros_current]
        source_components = [zeros_coordinates, zeros_coordinates, zeros_coordinates]
        for local_index, axis in enumerate(local_axes):
            electric_components[axis] = signs[local_index] * electric_local[local_index]
            magnetic_components[axis] = signs[local_index] * magnetic_local[local_index]
            source_components[axis] = signs[local_index] * source_local[local_index]
        electric_current = jnp.stack(electric_components, axis=0)
        magnetic_current = jnp.stack(magnetic_components, axis=0)
        source_coordinates = jnp.stack(source_components, axis=0)
        return electric_current, magnetic_current, source_coordinates, weights

    @staticmethod
    def _cross_axis0(first: ProjectionArray, second: ProjectionArray) -> jax.Array:
        return jnp.moveaxis(jnp.cross(jnp.moveaxis(first, 0, -1), jnp.moveaxis(second, 0, -1)), -1, 0)

    @staticmethod
    def _exact_cartesian_fields_for_observation(
        *,
        observation_point: ProjectionArray,
        electric_current: ProjectionArray,
        magnetic_current: ProjectionArray,
        source_coordinates: ProjectionArray,
        weights: ProjectionArray,
        angular_frequency: float,
        wavenumber: complex,
        epsilon: complex,
        permeability: complex,
    ) -> tuple[jax.Array, jax.Array]:
        displacement = observation_point[:, None, None] - source_coordinates
        radius = jnp.linalg.norm(displacement, axis=0)
        radial_hat = displacement / radius[None, :, :]

        ikr = 1j * wavenumber * radius
        green = jnp.exp(ikr) / (4.0 * jnp.pi * radius)
        d_green = green * (ikr - 1.0) / radius
        d2_green = d_green * (ikr - 1.0) / radius + green / (radius**2)

        def grad_dg_dot_current(current: ProjectionArray) -> jax.Array:
            radial_dot_current = jnp.sum(radial_hat * current, axis=0)
            transverse_current = current - radial_hat * radial_dot_current[None, :, :]
            return (
                radial_hat * d2_green[None, :, :] * radial_dot_current[None, :, :]
                + transverse_current * (d_green / radius)[None, :, :]
            )

        radial_cross_electric = FieldProjectionDetectorBase._cross_axis0(radial_hat, electric_current)
        radial_cross_magnetic = FieldProjectionDetectorBase._cross_axis0(radial_hat, magnetic_current)

        vector_potential_a = permeability * electric_current * green[None, :, :]
        curl_a = permeability * radial_cross_electric * d_green[None, :, :]
        grad_div_a = permeability * grad_dg_dot_current(electric_current)

        vector_potential_f = epsilon * magnetic_current * green[None, :, :]
        curl_f = epsilon * radial_cross_magnetic * d_green[None, :, :]
        grad_div_f = epsilon * grad_dg_dot_current(magnetic_current)

        electric_integrand = (
            1j * angular_frequency * (vector_potential_a + grad_div_a / (wavenumber**2)) - curl_f / epsilon
        )
        magnetic_integrand = (
            1j * angular_frequency * (vector_potential_f + grad_div_f / (wavenumber**2)) + curl_a / permeability
        )

        electric_field = jnp.sum(electric_integrand * weights[None, :, :], axis=(1, 2))
        magnetic_field = jnp.sum(magnetic_integrand * weights[None, :, :], axis=(1, 2))
        return electric_field, magnetic_field

    @staticmethod
    def _raise_if_exact_observations_overlap_sources(
        *,
        source_coordinates: ProjectionArray,
        radial_flat: ProjectionArray,
        distance_flat: ProjectionArray,
        batch_size: int | None,
    ) -> None:
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

    def _project_surface_exact_contribution(
        self,
        *,
        e_field: ProjectionArray,
        h_field: ProjectionArray,
        u_coords: ProjectionArray,
        v_coords: ProjectionArray,
        normal_offset: ProjectionScalar,
        propagation_axis: int,
        direction: Literal["+", "-"],
        theta: ProjectionArray,
        phi: ProjectionArray,
        wave_character_index: int,
    ) -> dict[str, jax.Array]:
        radial, theta_hat, phi_hat = _global_spherical_basis(theta, phi)
        observation_shape = radial.shape[1:]
        distance_flat = jnp.full((theta.size * phi.size,), self.projection_distance, dtype=float)
        return self._project_surface_exact_contribution_pairs(
            e_field=e_field,
            h_field=h_field,
            u_coords=u_coords,
            v_coords=v_coords,
            normal_offset=normal_offset,
            propagation_axis=propagation_axis,
            direction=direction,
            radial_flat=radial.reshape((3, -1)),
            theta_hat_flat=theta_hat.reshape((3, -1)),
            phi_hat_flat=phi_hat.reshape((3, -1)),
            distance_flat=distance_flat,
            wave_character_index=wave_character_index,
            observation_shape=observation_shape,
        )

    def _project_surface_contribution(
        self,
        *,
        e_field: ProjectionArray,
        h_field: ProjectionArray,
        u_coords: ProjectionArray,
        v_coords: ProjectionArray,
        normal_offset: ProjectionScalar,
        propagation_axis: int,
        direction: Literal["+", "-"],
        theta: ProjectionArray,
        phi: ProjectionArray,
        wavenumber: complex,
        impedance: complex,
    ) -> tuple[jax.Array, jax.Array]:
        radial, theta_hat, phi_hat = _global_spherical_basis(theta, phi)
        return self._project_surface_far_field_grid(
            e_field=e_field,
            h_field=h_field,
            u_coords=u_coords,
            v_coords=v_coords,
            normal_offset=normal_offset,
            propagation_axis=propagation_axis,
            direction=direction,
            radial=radial,
            theta_hat=theta_hat,
            phi_hat=phi_hat,
            wavenumber=wavenumber,
            impedance=impedance,
        )

    def _project_surface_far_field_grid(
        self,
        *,
        e_field: ProjectionArray,
        h_field: ProjectionArray,
        u_coords: ProjectionArray,
        v_coords: ProjectionArray,
        normal_offset: ProjectionScalar,
        propagation_axis: int,
        direction: Literal["+", "-"],
        radial: ProjectionArray,
        theta_hat: ProjectionArray,
        phi_hat: ProjectionArray,
        wavenumber: complex,
        impedance: complex,
    ) -> tuple[jax.Array, jax.Array]:
        local_axes, signs = self._local_axes_for_surface(propagation_axis, direction)
        radial_local = jnp.stack([signs[i] * radial[axis] for i, axis in enumerate(local_axes)], axis=0)
        theta_hat_local = jnp.stack([signs[i] * theta_hat[axis] for i, axis in enumerate(local_axes)], axis=0)
        phi_hat_local = jnp.stack([signs[i] * phi_hat[axis] for i, axis in enumerate(local_axes)], axis=0)

        u_indices = _subsample_indices(u_coords.size, int(self.interval_space[local_axes[0]]))
        v_indices = _subsample_indices(v_coords.size, int(self.interval_space[local_axes[1]]))
        e_field = e_field[:, u_indices][:, :, v_indices]
        h_field = h_field[:, u_indices][:, :, v_indices]
        u_coords = u_coords[u_indices]
        v_coords = v_coords[v_indices]

        u_weights = _trapz_weights_1d(u_coords) * _edge_window_1d(u_coords, self.window_size[0])
        v_weights = _trapz_weights_1d(v_coords) * _edge_window_1d(v_coords, self.window_size[1])
        weights = u_weights[:, None] * v_weights[None, :]

        electric_u = -h_field[1] * weights
        electric_v = h_field[0] * weights
        magnetic_u = e_field[1] * weights
        magnetic_v = -e_field[0] * weights

        projection_kwargs = {
            "u_direction": radial_local[0],
            "v_direction": radial_local[1],
            "normal_direction": radial_local[2],
            "wavenumber": wavenumber,
            "normal_offset": normal_offset,
        }
        n_u = _direct_project_component(electric_u, u_coords, v_coords, **projection_kwargs)
        n_v = _direct_project_component(electric_v, u_coords, v_coords, **projection_kwargs)
        l_u = _direct_project_component(magnetic_u, u_coords, v_coords, **projection_kwargs)
        l_v = _direct_project_component(magnetic_v, u_coords, v_coords, **projection_kwargs)

        n_theta = n_u * theta_hat_local[0] + n_v * theta_hat_local[1]
        n_phi = n_u * phi_hat_local[0] + n_v * phi_hat_local[1]
        l_theta = l_u * theta_hat_local[0] + l_v * theta_hat_local[1]
        l_phi = l_u * phi_hat_local[0] + l_v * phi_hat_local[1]
        e_theta = -(l_phi + impedance * n_theta)
        e_phi = l_theta - impedance * n_phi
        return e_theta, e_phi

    def _project_surface_exact_contribution_pairs(
        self,
        *,
        e_field: ProjectionArray,
        h_field: ProjectionArray,
        u_coords: ProjectionArray,
        v_coords: ProjectionArray,
        normal_offset: ProjectionScalar,
        propagation_axis: int,
        direction: Literal["+", "-"],
        radial_flat: ProjectionArray,
        theta_hat_flat: ProjectionArray,
        phi_hat_flat: ProjectionArray,
        distance_flat: ProjectionArray,
        wave_character_index: int,
        observation_shape: tuple[int, ...],
    ) -> dict[str, jax.Array]:
        electric_current, magnetic_current, source_coordinates, weights = self._surface_currents_geometry(
            e_field=e_field,
            h_field=h_field,
            u_coords=u_coords,
            v_coords=v_coords,
            normal_offset=normal_offset,
            propagation_axis=propagation_axis,
            direction=direction,
        )
        eps_relative, mu_relative = self._projection_relative_permittivity_permeability(wave_character_index)
        epsilon = constants.eps0 * eps_relative
        permeability = constants.mu0 * mu_relative
        wavenumber = self._projection_wavenumber(wave_character_index)
        angular_frequency = 2.0 * np.pi * self.wave_characters[wave_character_index].get_frequency()

        self._raise_if_exact_observations_overlap_sources(
            source_coordinates=source_coordinates,
            radial_flat=radial_flat,
            distance_flat=distance_flat,
            batch_size=self.exact_projection_batch_size,
        )

        def project_observation(radial, theta_hat, phi_hat, distance):
            observation_point = distance * radial
            electric_field, magnetic_field = self._exact_cartesian_fields_for_observation(
                observation_point=observation_point,
                electric_current=electric_current,
                magnetic_current=magnetic_current,
                source_coordinates=source_coordinates,
                weights=weights,
                angular_frequency=angular_frequency,
                wavenumber=wavenumber,
                epsilon=epsilon,
                permeability=permeability,
            )
            return (
                jnp.sum(electric_field * radial, axis=0),
                jnp.sum(electric_field * theta_hat, axis=0),
                jnp.sum(electric_field * phi_hat, axis=0),
                jnp.sum(magnetic_field * radial, axis=0),
                jnp.sum(magnetic_field * theta_hat, axis=0),
                jnp.sum(magnetic_field * phi_hat, axis=0),
            )

        observation_inputs = (
            jnp.moveaxis(radial_flat, 1, 0),
            jnp.moveaxis(theta_hat_flat, 1, 0),
            jnp.moveaxis(phi_hat_flat, 1, 0),
            distance_flat,
        )
        if self.exact_projection_batch_size is None or radial_flat.shape[1] <= self.exact_projection_batch_size:
            projected = jax.vmap(project_observation)(*observation_inputs)
        else:
            projected = jax.lax.map(
                lambda observation: project_observation(*observation),
                observation_inputs,
                batch_size=int(self.exact_projection_batch_size),
            )

        return {key: projected[index].reshape(observation_shape) for index, key in enumerate(_PROJECTION_FIELD_KEYS)}

    def _surface_projection_context(
        self,
        state: DetectorState,
        wave_character_index: int,
        surface: ProjectionSurface | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, ProjectionScalar, int, Literal["+", "-"]]:
        if surface is None:
            if self.direction is None:
                raise ValueError("direction must be specified for a planar field projection detector.")
            propagation_axis = self.propagation_axis
            direction = self.direction
            u_coords, v_coords, normal_offset = self._local_coordinates()
        else:
            propagation_axis, direction = _surface_axis_direction(surface)
            u_coords, v_coords, normal_offset = self._local_coordinates_for_surface(propagation_axis, direction)
        e_field, h_field = self._local_plane_fields(state, wave_character_index, surface=surface)
        return e_field, h_field, u_coords, v_coords, normal_offset, propagation_axis, direction

    def _project_observation_angle_grid(
        self,
        state: DetectorState,
        theta: ProjectionArray,
        phi: ProjectionArray,
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        wave_character_index = self._validate_wave_character_index(wave_character_index)
        if not self.far_field_approx:
            fields = {key: jnp.zeros((theta.size, phi.size), dtype=complex) for key in _PROJECTION_FIELD_KEYS}
            surface_names = []
            for surface in self._surface_projection_sequence():
                e_field, h_field, u_coords, v_coords, normal_offset, propagation_axis, direction = (
                    self._surface_projection_context(state, wave_character_index, surface)
                )
                surface_fields = self._project_surface_exact_contribution(
                    e_field=e_field,
                    h_field=h_field,
                    u_coords=u_coords,
                    v_coords=v_coords,
                    normal_offset=normal_offset,
                    propagation_axis=propagation_axis,
                    direction=direction,
                    theta=theta,
                    phi=phi,
                    wave_character_index=wave_character_index,
                )
                for key in _PROJECTION_FIELD_KEYS:
                    fields[key] = fields[key] + surface_fields[key]
                surface_names.append(self._surface_projection_name(surface))

            power = 0.5 * jnp.real(
                fields["Etheta"] * jnp.conj(fields["Hphi"]) - fields["Ephi"] * jnp.conj(fields["Htheta"])
            )
            return {
                **fields,
                "power": power,
                "theta": theta,
                "phi": phi,
                "direction": self.direction,
                "surfaces": tuple(surface_names),
                **self._projection_metadata(wave_character_index),
            }

        wavenumber = self._projection_wavenumber(wave_character_index)
        impedance = self._projection_impedance(wave_character_index)
        e_theta = jnp.zeros((theta.size, phi.size), dtype=complex)
        e_phi = jnp.zeros((theta.size, phi.size), dtype=complex)
        surface_names = []
        for surface in self._surface_projection_sequence():
            e_field, h_field, u_coords, v_coords, normal_offset, propagation_axis, direction = (
                self._surface_projection_context(state, wave_character_index, surface)
            )
            surface_e_theta, surface_e_phi = self._project_surface_contribution(
                e_field=e_field,
                h_field=h_field,
                u_coords=u_coords,
                v_coords=v_coords,
                normal_offset=normal_offset,
                propagation_axis=propagation_axis,
                direction=direction,
                theta=theta,
                phi=phi,
                wavenumber=wavenumber,
                impedance=impedance,
            )
            e_theta = e_theta + surface_e_theta
            e_phi = e_phi + surface_e_phi
            surface_names.append(self._surface_projection_name(surface))

        h_theta = -e_phi / impedance
        h_phi = e_theta / impedance
        propagation_factor = self._propagation_factor(wave_character_index)
        e_theta = propagation_factor * e_theta
        e_phi = propagation_factor * e_phi
        h_theta = propagation_factor * h_theta
        h_phi = propagation_factor * h_phi
        power = 0.5 * jnp.real(e_theta * jnp.conj(h_phi) - e_phi * jnp.conj(h_theta))
        radial_e = jnp.zeros_like(e_theta)
        radial_h = jnp.zeros_like(h_theta)
        return {
            "Er": radial_e,
            "Etheta": e_theta,
            "Ephi": e_phi,
            "Hr": radial_h,
            "Htheta": h_theta,
            "Hphi": h_phi,
            "power": power,
            "theta": theta,
            "phi": phi,
            "direction": self.direction,
            "surfaces": tuple(surface_names),
            **self._projection_metadata(wave_character_index),
        }

    def _project_paired_observation_angles(
        self,
        state: DetectorState,
        theta: ProjectionArray,
        phi: ProjectionArray,
        distance: ProjectionArray,
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        wave_character_index = self._validate_wave_character_index(wave_character_index)
        theta = _finite_numeric_array("theta", theta)
        phi = _finite_numeric_array("phi", phi)
        distance = _finite_numeric_array("projection distance", distance)
        if theta.shape != phi.shape or theta.shape != distance.shape:
            raise ValueError("theta, phi, and projection distance must have the same shape.")
        if theta.size == 0:
            raise ValueError("projection observation grid must be non-empty.")
        _validate_theta_range(theta)
        _validate_positive_values("projection distance", distance)

        observation_shape = theta.shape
        radial, theta_hat, phi_hat = _global_spherical_basis_paired(theta, phi)
        radial_flat = radial.reshape((3, -1))
        theta_hat_flat = theta_hat.reshape((3, -1))
        phi_hat_flat = phi_hat.reshape((3, -1))
        distance_flat = distance.reshape((-1,))
        surface_names = []

        if not self.far_field_approx:
            fields = {key: jnp.zeros(observation_shape, dtype=complex) for key in _PROJECTION_FIELD_KEYS}
            for surface in self._surface_projection_sequence():
                e_field, h_field, u_coords, v_coords, normal_offset, propagation_axis, direction = (
                    self._surface_projection_context(state, wave_character_index, surface)
                )
                surface_fields = self._project_surface_exact_contribution_pairs(
                    e_field=e_field,
                    h_field=h_field,
                    u_coords=u_coords,
                    v_coords=v_coords,
                    normal_offset=normal_offset,
                    propagation_axis=propagation_axis,
                    direction=direction,
                    radial_flat=radial_flat,
                    theta_hat_flat=theta_hat_flat,
                    phi_hat_flat=phi_hat_flat,
                    distance_flat=distance_flat,
                    wave_character_index=wave_character_index,
                    observation_shape=observation_shape,
                )
                for key in _PROJECTION_FIELD_KEYS:
                    fields[key] = fields[key] + surface_fields[key]
                surface_names.append(self._surface_projection_name(surface))
            power = 0.5 * jnp.real(
                fields["Etheta"] * jnp.conj(fields["Hphi"]) - fields["Ephi"] * jnp.conj(fields["Htheta"])
            )
            return {
                **fields,
                "power": power,
                "theta": theta,
                "phi": phi,
                "direction": self.direction,
                "surfaces": tuple(surface_names),
                **self._projection_metadata(wave_character_index),
            }

        wavenumber = self._projection_wavenumber(wave_character_index)
        impedance = self._projection_impedance(wave_character_index)
        e_theta = jnp.zeros(observation_shape, dtype=complex)
        e_phi = jnp.zeros(observation_shape, dtype=complex)
        for surface in self._surface_projection_sequence():
            e_field, h_field, u_coords, v_coords, normal_offset, propagation_axis, direction = (
                self._surface_projection_context(state, wave_character_index, surface)
            )
            surface_e_theta, surface_e_phi = self._project_surface_far_field_grid(
                e_field=e_field,
                h_field=h_field,
                u_coords=u_coords,
                v_coords=v_coords,
                normal_offset=normal_offset,
                propagation_axis=propagation_axis,
                direction=direction,
                radial=radial,
                theta_hat=theta_hat,
                phi_hat=phi_hat,
                wavenumber=wavenumber,
                impedance=impedance,
            )
            e_theta = e_theta + surface_e_theta
            e_phi = e_phi + surface_e_phi
            surface_names.append(self._surface_projection_name(surface))

        h_theta = -e_phi / impedance
        h_phi = e_theta / impedance
        propagation_factor = -1j * wavenumber * jnp.exp(1j * wavenumber * distance) / (4.0 * jnp.pi * distance)
        e_theta = propagation_factor * e_theta
        e_phi = propagation_factor * e_phi
        h_theta = propagation_factor * h_theta
        h_phi = propagation_factor * h_phi
        power = 0.5 * jnp.real(e_theta * jnp.conj(h_phi) - e_phi * jnp.conj(h_theta))
        radial_e = jnp.zeros_like(e_theta)
        radial_h = jnp.zeros_like(h_theta)
        return {
            "Er": radial_e,
            "Etheta": e_theta,
            "Ephi": e_phi,
            "Hr": radial_h,
            "Htheta": h_theta,
            "Hphi": h_phi,
            "power": power,
            "theta": theta,
            "phi": phi,
            "direction": self.direction,
            "surfaces": tuple(surface_names),
            **self._projection_metadata(wave_character_index),
        }

    def _stack_frequency_results(
        self,
        results: Sequence[Mapping[str, Any]],
        coordinate_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        projected: dict[str, Any] = {
            key: jnp.stack([result[key] for result in results], axis=0) for key in _PROJECTION_RESULT_KEYS
        }
        projected.update(coordinate_metadata)
        projected.update(
            {
                "wave_character_indices": np.arange(len(self.wave_characters)),
                "frequencies": np.asarray([result["frequency"] for result in results]),
                "free_space_wavelengths": np.asarray([result["free_space_wavelength"] for result in results]),
                "projection_distance": _finite_scalar("projection_distance", self.projection_distance),
                "far_field_approx": self.far_field_approx,
                "projection_medium_refractive_indices": jnp.asarray(
                    [result["projection_medium_refractive_index"] for result in results]
                ),
                "projection_medium_impedances": jnp.asarray(
                    [result["projection_medium_impedance"] for result in results]
                ),
                "projection_wavenumbers": jnp.asarray([result["projection_wavenumber"] for result in results]),
                "projection_wavelengths": jnp.asarray([result["projection_wavelength"] for result in results]),
                "direction": self.direction,
                "surfaces": results[0]["surfaces"],
            }
        )
        return projected

    def _surface_projection_sequence(self) -> tuple[ProjectionSurface | None, ...]:
        if self._projection_mode == "surface":
            return (None,)
        return self._included_box_surfaces()

    def _surface_projection_name(self, surface: ProjectionSurface | None) -> ProjectionSurface:
        if surface is not None:
            return surface
        if self.direction is None:
            raise ValueError("direction must be specified for a planar field projection detector.")
        return _surface_name(self.propagation_axis, self.direction)


@autoinit
class FieldProjectionAngleDetector(FieldProjectionDetectorBase):
    """Frequency-domain detector for projecting a phasor plane to observation angles.

    This detector records the complex frequency-domain E/H fields on a surface
    or box and projects them to a spherical angular grid after the FDTD run.
    ``theta`` and ``phi`` follow the global spherical-coordinate convention:
    ``theta`` is measured from the positive z-axis and ``phi`` from the positive
    x-axis in the x-y plane.

    For planar placement, ``direction`` selects the outward normal used to form
    the equivalent surface currents. For box placement, all included outer
    surfaces are projected coherently, using the detector ``origin`` as the
    common phase reference.
    """

    def __post_init__(self):
        super().__post_init__()

    def _validate_projection_inputs(
        self,
        theta: ProjectionArray,
        phi: ProjectionArray,
        wave_character_index: int,
    ) -> tuple[jax.Array, jax.Array, int]:
        theta = _finite_numeric_array("theta", theta)
        phi = _finite_numeric_array("phi", phi)
        if theta.ndim != 1 or phi.ndim != 1:
            raise ValueError("theta and phi must be one-dimensional arrays.")
        if theta.size == 0 or phi.size == 0:
            raise ValueError("theta and phi must be non-empty.")
        _validate_theta_range(theta)
        return theta, phi, self._validate_wave_character_index(wave_character_index)

    def project(
        self,
        state: DetectorState,
        theta: ProjectionArray,
        phi: ProjectionArray,
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        """Project the recorded phasor plane to a far-field angular grid.

        Args:
            state: Detector state holding the ``phasor`` array.
            theta: Global polar angles in radians, measured from the positive z-axis.
            phi: Global azimuthal angles in radians, measured from the positive x-axis.
            wave_character_index: Frequency index to project.

        Returns:
            Dictionary containing ``Er``, ``Etheta``, ``Ephi``, ``Hr``,
            ``Htheta``, ``Hphi``, and radial ``power`` on the requested
            ``(theta, phi)`` grid, together with coordinate and projection
            metadata. In the far-field approximation the radial field
            components are zero.
        """
        theta, phi, wave_character_index = self._validate_projection_inputs(theta, phi, wave_character_index)
        return self._project_observation_angle_grid(
            state,
            theta,
            phi,
            wave_character_index=wave_character_index,
        )

    def project_all(
        self,
        state: DetectorState,
        theta: ProjectionArray,
        phi: ProjectionArray,
    ) -> dict[str, Any]:
        """Project the recorded phasor plane for every wave character.

        Args:
            state: Detector state holding the ``phasor`` array.
            theta: Global polar angles in radians, measured from the positive z-axis.
            phi: Global azimuthal angles in radians, measured from the positive x-axis.

        Returns:
            Dictionary containing the same projected field keys as :meth:`project`,
            with a leading frequency axis ordered like ``self.wave_characters``.
        """
        results = [
            self.project(
                state=state,
                theta=theta,
                phi=phi,
                wave_character_index=wave_character_index,
            )
            for wave_character_index in range(len(self.wave_characters))
        ]
        return self._stack_frequency_results(
            results,
            {
                "theta": results[0]["theta"],
                "phi": results[0]["phi"],
            },
        )


@autoinit
class FieldProjectionCartesianDetector(FieldProjectionDetectorBase):
    """Frequency-domain detector for projecting phasors to a Cartesian observation plane.

    ``x`` and ``y`` passed to :meth:`project` are local coordinates on the
    observation plane. The plane normal is selected by ``projection_axis`` and
    the plane is located ``projection_distance`` meters from ``origin`` along
    that axis. The detector returns the same projected field components as
    :class:`FieldProjectionAngleDetector`, evaluated at the observation
    directions corresponding to the Cartesian plane points.
    """

    #: Axis normal to the Cartesian observation plane, where 0=x, 1=y, and 2=z.
    projection_axis: int = frozen_field(default=2)

    def __post_init__(self):
        super().__post_init__()
        _validate_projection_axis(self.projection_axis)

    def _validate_cartesian_inputs(self, x: ProjectionArray, y: ProjectionArray) -> tuple[jax.Array, jax.Array]:
        x = _finite_numeric_array("x", x)
        y = _finite_numeric_array("y", y)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be one-dimensional arrays.")
        if x.size == 0 or y.size == 0:
            raise ValueError("x and y must be non-empty.")
        return x, y

    def _cartesian_observation_grid(
        self,
        x: ProjectionArray,
        y: ProjectionArray,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        x_grid, y_grid = jnp.meshgrid(x, y, indexing="ij")
        points = jnp.zeros((3, x.size, y.size), dtype=float)
        transverse_axis_x, transverse_axis_y = _projection_transverse_axes(
            _validate_projection_axis(self.projection_axis)
        )
        points = points.at[transverse_axis_x].set(x_grid)
        points = points.at[transverse_axis_y].set(y_grid)
        points = points.at[self.projection_axis].set(self.projection_distance)
        radial_distance, theta, phi = _cartesian_to_spherical_angles(points)
        return radial_distance, theta, phi

    def project(
        self,
        state: DetectorState,
        x: ProjectionArray,
        y: ProjectionArray,
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        """Project recorded phasors to a Cartesian observation plane.

        Args:
            state: Detector state holding the recorded phasor array.
            x: First local coordinate array on the observation plane, in meters.
            y: Second local coordinate array on the observation plane, in meters.
            wave_character_index: Frequency index to project.

        Returns:
            Dictionary containing projected field components, radial ``power``,
            the derived ``theta`` and ``phi`` observation angles, the input
            ``x`` and ``y`` coordinates, and projection metadata.
        """
        x, y = self._validate_cartesian_inputs(x, y)
        radial_distance, theta, phi = self._cartesian_observation_grid(x, y)
        result = self._project_paired_observation_angles(
            state,
            theta,
            phi,
            radial_distance,
            wave_character_index=wave_character_index,
        )
        result.update(
            {
                "x": x,
                "y": y,
                "projection_axis": int(self.projection_axis),
            }
        )
        return result

    def project_all(
        self,
        state: DetectorState,
        x: ProjectionArray,
        y: ProjectionArray,
    ) -> dict[str, Any]:
        """Project the recorded phasors to a Cartesian plane for every wave character.

        Args:
            state: Detector state holding the recorded phasor array.
            x: First local coordinate array on the observation plane, in meters.
            y: Second local coordinate array on the observation plane, in meters.

        Returns:
            Dictionary containing the same projected field keys as :meth:`project`,
            with a leading frequency axis ordered like ``self.wave_characters``.
        """
        x, y = self._validate_cartesian_inputs(x, y)
        radial_distance, theta, phi = self._cartesian_observation_grid(x, y)
        results = [
            self._project_paired_observation_angles(
                state,
                theta,
                phi,
                radial_distance,
                wave_character_index=wave_character_index,
            )
            for wave_character_index in range(len(self.wave_characters))
        ]
        return self._stack_frequency_results(
            results,
            {
                "x": x,
                "y": y,
                "theta": theta,
                "phi": phi,
                "projection_axis": int(self.projection_axis),
            },
        )


@autoinit
class FieldProjectionKSpaceDetector(FieldProjectionDetectorBase):
    """Frequency-domain detector for projecting phasors to a k-space direction grid.

    ``ux`` and ``uy`` are direction cosines in the local transverse coordinates
    of the projection axis. Only propagating directions satisfying
    ``ux**2 + uy**2 <= 1`` are accepted. The detector returns the same projected
    field components as :class:`FieldProjectionAngleDetector`, evaluated at the
    corresponding spherical directions.
    """

    #: Axis defining the local k-space propagation direction, where 0=x, 1=y, and 2=z.
    projection_axis: int = frozen_field(default=2)

    def __post_init__(self):
        super().__post_init__()
        _validate_projection_axis(self.projection_axis)

    def _validate_kspace_inputs(self, ux: ProjectionArray, uy: ProjectionArray) -> tuple[jax.Array, jax.Array]:
        ux = _finite_numeric_array("ux", ux)
        uy = _finite_numeric_array("uy", uy)
        if ux.ndim != 1 or uy.ndim != 1:
            raise ValueError("ux and uy must be one-dimensional arrays.")
        if ux.size == 0 or uy.size == 0:
            raise ValueError("ux and uy must be non-empty.")
        if _is_jax_tracer(ux) or _is_jax_tracer(uy):
            return ux, uy
        if _concrete_jax_bool(jnp.any(jnp.abs(ux) > 1.0)) or _concrete_jax_bool(jnp.any(jnp.abs(uy) > 1.0)):
            raise ValueError("ux and uy values must lie in the interval [-1, 1].")
        ux_grid, uy_grid = jnp.meshgrid(ux, uy, indexing="ij")
        if _concrete_jax_bool(jnp.any(ux_grid**2 + uy_grid**2 > 1.0)):
            raise ValueError("ux^2 + uy^2 must not exceed 1 for propagating k-space directions.")
        return ux, uy

    def _kspace_observation_grid(
        self, ux: ProjectionArray, uy: ProjectionArray
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        ux_grid, uy_grid = jnp.meshgrid(ux, uy, indexing="ij")
        transverse_radius = jnp.sqrt(ux_grid**2 + uy_grid**2)
        theta_local = jnp.arcsin(transverse_radius)
        phi_local = jnp.arctan2(uy_grid, ux_grid)

        if self.projection_axis == 2:
            theta = theta_local
            phi = phi_local
        else:
            x = jnp.cos(theta_local)
            y = jnp.sin(theta_local) * jnp.cos(phi_local)
            z = jnp.sin(theta_local) * jnp.sin(phi_local)
            if self.projection_axis == 1:
                x, y, z = y, x, z
            theta = jnp.arccos(jnp.clip(z, -1.0, 1.0))
            phi = jnp.arctan2(y, x)
        distance = jnp.full(theta.shape, _finite_scalar("projection_distance", self.projection_distance), dtype=float)
        return theta, phi, distance

    def project(
        self,
        state: DetectorState,
        ux: ProjectionArray,
        uy: ProjectionArray,
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        """Project recorded phasors to a k-space direction-cosine grid.

        Args:
            state: Detector state holding the recorded phasor array.
            ux: First transverse direction-cosine array.
            uy: Second transverse direction-cosine array.
            wave_character_index: Frequency index to project.

        Returns:
            Dictionary containing projected field components, radial ``power``,
            the derived ``theta`` and ``phi`` observation angles, the input
            ``ux`` and ``uy`` direction-cosine arrays, and projection metadata.
        """
        ux, uy = self._validate_kspace_inputs(ux, uy)
        theta, phi, distance = self._kspace_observation_grid(ux, uy)
        result = self._project_paired_observation_angles(
            state,
            theta,
            phi,
            distance,
            wave_character_index=wave_character_index,
        )
        result.update(
            {
                "ux": ux,
                "uy": uy,
                "projection_axis": int(self.projection_axis),
            }
        )
        return result

    def project_all(
        self,
        state: DetectorState,
        ux: ProjectionArray,
        uy: ProjectionArray,
    ) -> dict[str, Any]:
        """Project the recorded phasors to a k-space grid for every wave character.

        Args:
            state: Detector state holding the recorded phasor array.
            ux: First transverse direction-cosine array.
            uy: Second transverse direction-cosine array.

        Returns:
            Dictionary containing the same projected field keys as :meth:`project`,
            with a leading frequency axis ordered like ``self.wave_characters``.
        """
        ux, uy = self._validate_kspace_inputs(ux, uy)
        theta, phi, distance = self._kspace_observation_grid(ux, uy)
        results = [
            self._project_paired_observation_angles(
                state,
                theta,
                phi,
                distance,
                wave_character_index=wave_character_index,
            )
            for wave_character_index in range(len(self.wave_characters))
        ]
        return self._stack_frequency_results(
            results,
            {
                "ux": ux,
                "uy": uy,
                "theta": theta,
                "phi": phi,
                "projection_axis": int(self.projection_axis),
            },
        )
