"""Near-to-far field projection detectors.

The public detector classes differ only in how users specify observation points: angular coordinates, Cartesian
observation-plane coordinates, or k-space direction cosines. Internally they share one projection pipeline.

During the simulation the detector records frequency-domain E/H phasors on either one planar surface or the
included faces of a box. Post-processing applies the surface-equivalence method: each surface is converted to
equivalent electric and magnetic surface currents, all included surfaces are summed coherently as complex fields,
and the final power is computed from projected E_theta/E_phi and H_theta/H_phi components.

Two propagation models are available. The default far-field path evaluates separable Fourier-type surface
integrals and derives H from the homogeneous-medium impedance. The exact path evaluates finite-distance
homogeneous Green-function fields and therefore can return non-zero radial components.
"""

from numbers import Integral
from typing import Any, Literal, Mapping, Self, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx import constants
from fdtdx.config import SimulationConfig
from fdtdx.core.axis import get_oriented_transverse_axes, get_transverse_axes, validate_axis
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.core.jax.utils import (
    concrete_jax_bool,
    finite_1d_array,
    finite_numeric_array,
    finite_scalar,
    is_jax_tracer,
)
from fdtdx.core.physics.field_projection import (
    _passive_complex_sqrt,
    _positive_impedance_sqrt,
    _positive_projection_parameter,
    _projection_parameter_is_default,
    _projection_parameter_value,
    _validate_1d_coordinate_pair,
    _validate_positive_values,
    _validate_theta_range,
    cartesian_to_spherical_angles,
    direct_project_component,
    edge_window_1d,
    exact_cartesian_fields_for_observation,
    raise_if_exact_observations_overlap_sources,
    spherical_basis_grid,
    spherical_basis_paired,
    subsample_indices,
    trapezoidal_weights_1d,
)
from fdtdx.materials import Material, isotropic_property_value
from fdtdx.objects.detectors.detector import DetectorState
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.typing import ProjectionSurface, SliceTuple3D

_PROJECTION_FIELD_KEYS = ("Er", "Etheta", "Ephi", "Hr", "Htheta", "Hphi")
_PROJECTION_RESULT_KEYS = (*_PROJECTION_FIELD_KEYS, "power")
_PROJECTION_SURFACES: tuple[ProjectionSurface, ...] = ("x-", "x+", "y-", "y+", "z-", "z+")
_SURFACE_KEY_SUFFIX = {"-": "minus", "+": "plus"}
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
    """Return the physical axis and sign encoded by a surface name such as ``"z+"``."""
    return _SURFACE_AXIS_DIRECTIONS[surface]


def _surface_name(axis: int, direction: Literal["+", "-"]) -> ProjectionSurface:
    """Return the canonical surface name for a physical axis and outward sign."""
    return _SURFACE_NAMES_BY_AXIS_DIRECTION[(axis, direction)]


def _surface_state_key(surface: ProjectionSurface) -> str:
    """Return the state-dictionary key used to store a box-face phasor."""
    return f"phasor_{surface[0]}_{_SURFACE_KEY_SUFFIX[surface[1]]}"


@autoinit
class FieldProjectionDetectorBase(PhasorDetector):
    """Shared base for frequency-domain near-to-far field projection detectors.

    The detector records complex E/H phasors on a planar surface or box, then
    projects equivalent surface currents to observation coordinates supplied by
    concrete subclasses.
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
        """Validate immutable projection-detector configuration after ``autoinit`` construction."""
        super().__post_init__()
        if len(self.wave_characters) == 0:
            raise ValueError("wave_characters must contain at least one wave character.")
        if self.direction is not None and self.direction not in ["+", "-"]:
            raise ValueError("direction must be '+', '-', or None.")
        self._validate_exclude_surfaces()
        if self.origin is not None:
            finite_1d_array("origin", self.origin, 3)
        projection_distance = finite_scalar("projection_distance", self.projection_distance)
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

        window_size = finite_1d_array("window_size", self.window_size, 2)
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
        """Validate that ``projection_medium`` is supported by the scalar homogeneous projection model."""
        if not isinstance(self.projection_medium, Material):
            raise ValueError("projection_medium must be a Material.")
        eps_inf, permeability, electric_conductivity, magnetic_conductivity = self._projection_medium_scalars()
        if eps_inf == 0:
            raise ValueError("projection_medium permittivity must be non-zero.")
        if permeability == 0:
            raise ValueError("projection_medium permeability must be non-zero.")
        if electric_conductivity < 0:
            raise ValueError("projection_medium electric_conductivity must be non-negative.")
        if magnetic_conductivity < 0:
            raise ValueError("projection_medium magnetic_conductivity must be non-negative.")

    def _validate_exclude_surfaces(self) -> None:
        """Validate box-face exclusion names before detector placement."""
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
        """Classify the placed detector geometry as a single surface or a box volume."""
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
        """Return box faces that participate in the coherent projection sum."""
        excluded_surfaces = set(self.exclude_surfaces)
        surfaces = tuple(surface for surface in _PROJECTION_SURFACES if surface not in excluded_surfaces)
        if len(surfaces) == 0:
            raise ValueError("exclude_surfaces must not exclude every box surface.")
        return surfaces

    @property
    def propagation_axis(self) -> int:
        """Return the normal axis for a single planar detector surface."""
        if self._projection_mode != "surface":
            raise Exception(f"Invalid field projection detector surface shape: {self.grid_shape}")
        return self.grid_shape.index(1)

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        """Place the detector and validate surface-only versus box-only options."""
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
        """Return the grid shape of one box face while retaining its singleton normal axis."""
        axis, _ = _surface_axis_direction(surface)
        shape = list(self.grid_shape)
        shape[axis] = 1
        return shape[0], shape[1], shape[2]

    def _shape_dtype_single_time_step(self) -> dict[str, jax.ShapeDtypeStruct]:
        """Return phasor state shapes for the planar or box recording mode."""
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
        """Record frequency-domain phasors for every included detector face.

        Planar detectors use the inherited ``PhasorDetector`` update. Box detectors slice each included boundary face
        from the detector volume and store one phasor array per surface so post-processing can apply each outward normal
        with the correct phase reference.
        """
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

        fields = jnp.concatenate((E[:, *self.grid_slice], H[:, *self.grid_slice]), axis=0)
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
        """Return physical cell-center coordinates for one detector axis."""
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
        """Return local ``(u, v, n)`` axes and signs for one outward surface normal.

        The local normal always points outward from the projection surface. The sign tuple converts global field and
        coordinate components into this local basis before equivalent surface currents are formed.
        """
        axes = (*get_oriented_transverse_axes(axis), axis)
        if direction == "+":
            return axes, (1.0, 1.0, 1.0)
        return axes, (1.0, -1.0, -1.0)

    def _local_axes(self) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
        """Return local axes and signs for a planar detector surface."""
        if self.direction is None:
            raise ValueError("direction must be specified for a planar field projection detector.")
        return self._local_axes_for_surface(self.propagation_axis, self.direction)

    def _state_phasor(self, state: DetectorState) -> jax.Array:
        """Read and shape-check the planar detector phasor state."""
        if "phasor" not in state:
            raise ValueError("state must contain a 'phasor' entry.")
        phasor = jnp.asarray(state["phasor"])
        expected_shape = (1, len(self.wave_characters), 6, *self.grid_shape)
        if phasor.shape != expected_shape:
            raise ValueError(f"state['phasor'] must have shape {expected_shape}, got {phasor.shape}.")
        return phasor

    def _state_surface_phasor(self, state: DetectorState, surface: ProjectionSurface) -> jax.Array:
        """Read and shape-check one box-face phasor state."""
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
        """Return recorded E/H phasors rotated into the local surface basis.

        The projection formulas operate on local tangential components. This method squeezes the singleton normal axis,
        orders the transverse dimensions as ``u, v``, and applies the surface-orientation signs.
        """
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

        remaining_axes = get_transverse_axes(propagation_axis)
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
    ) -> tuple[jax.Array, jax.Array, float | jax.Array]:
        """Return local physical coordinates and normal offset relative to the projection origin.

        The origin controls the phase reference. For boxes, all faces use a common origin so their projected complex
        fields can be summed coherently.
        """
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

    def _local_coordinates(self) -> tuple[jax.Array, jax.Array, float | jax.Array]:
        """Return local coordinates for a planar detector surface."""
        if self.direction is None:
            raise ValueError("direction must be specified for a planar field projection detector.")
        return self._local_coordinates_for_surface(self.propagation_axis, self.direction)

    def _local_transverse_coordinates(self) -> tuple[jax.Array, jax.Array]:
        """Return only the local ``u`` and ``v`` coordinate arrays for planar tests and utilities."""
        u_coords, v_coords, _ = self._local_coordinates()
        return u_coords, v_coords

    def _projection_medium_scalars(self) -> tuple[float, float, float, float]:
        """Return scalar material properties used by the homogeneous projection medium."""
        projection_medium = self.projection_medium
        if projection_medium is None:
            raise ValueError("projection_medium must be provided.")
        return (
            isotropic_property_value(projection_medium.permittivity, "projection_medium permittivity"),
            isotropic_property_value(projection_medium.permeability, "projection_medium permeability"),
            isotropic_property_value(
                projection_medium.electric_conductivity, "projection_medium electric_conductivity"
            ),
            isotropic_property_value(
                projection_medium.magnetic_conductivity, "projection_medium magnetic_conductivity"
            ),
        )

    def _projection_material_relative_permittivity_permeability(
        self, wave_character_index: int
    ) -> tuple[complex, complex]:
        """Return complex relative permittivity and permeability for ``projection_medium`` at one frequency.

        Electric and magnetic conductivity are folded into the complex material response using the selected angular
        frequency. Dispersive permittivity models are evaluated at the same frequency.
        """
        frequency = float(self.wave_characters[wave_character_index].get_frequency())
        omega = 2.0 * np.pi * frequency
        if omega <= 0 or not np.isfinite(omega):
            raise ValueError("wave character frequency must be finite and positive.")

        projection_medium = self.projection_medium
        if projection_medium is None:
            raise ValueError("projection_medium must be provided.")
        eps_inf, mu_r, sigma_e, sigma_m = self._projection_medium_scalars()
        if projection_medium.dispersion is None:
            eps_complex = complex(eps_inf)
        else:
            eps_complex = complex(projection_medium.dispersion.permittivity(omega, eps_inf=eps_inf))
        eps_complex = eps_complex + 1j * sigma_e / (omega * constants.eps0)
        mu_complex = complex(mu_r) + 1j * sigma_m / (omega * constants.mu0)
        if eps_complex == 0:
            raise ValueError("projection_medium permittivity is zero at the selected frequency.")
        if mu_complex == 0:
            raise ValueError("projection_medium permeability is zero at the selected frequency.")
        return eps_complex, mu_complex

    def _projection_relative_permittivity_permeability(self, wave_character_index: int) -> tuple[complex, complex]:
        """Return homogeneous-medium relative material parameters for one wave character."""
        if self.projection_medium is not None:
            return self._projection_material_relative_permittivity_permeability(wave_character_index)

        refractive_index, impedance, _ = self._projection_parameters(wave_character_index)
        eps_complex = constants.eta0 * refractive_index / impedance
        mu_complex = refractive_index * impedance / constants.eta0
        return complex(eps_complex), complex(mu_complex)

    def _projection_parameters(self, wave_character_index: int) -> tuple[complex, complex, complex]:
        """Return refractive index, wave impedance, and wavenumber for one projection frequency."""
        wavelength = self.wave_characters[wave_character_index].get_wavelength()
        if self.projection_medium is not None:
            eps_complex, mu_complex = self._projection_material_relative_permittivity_permeability(wave_character_index)
            refractive_index = _passive_complex_sqrt(eps_complex * mu_complex)
            impedance = constants.eta0 * _positive_impedance_sqrt(mu_complex / eps_complex)
        else:
            refractive_index = _projection_parameter_value(
                self.projection_medium_refractive_index, wave_character_index
            )
            impedance = (
                constants.eta0 / refractive_index
                if self.projection_medium_impedance is None
                else _projection_parameter_value(self.projection_medium_impedance, wave_character_index)
            )
        return refractive_index, impedance, (2.0 * np.pi / wavelength) * refractive_index

    def _propagation_factor(self, wave_character_index: int) -> jax.Array:
        """Return the scalar far-field Green-function prefactor."""
        _, _, k = self._projection_parameters(wave_character_index)
        return -1j * k * jnp.exp(1j * k * self.projection_distance) / (4.0 * jnp.pi * self.projection_distance)

    def _projection_metadata(self, wave_character_index: int) -> dict[str, float | complex | int]:
        """Return scalar metadata shared by single-frequency projection results."""
        wave_character = self.wave_characters[wave_character_index]
        refractive_index, impedance, wavenumber = self._projection_parameters(wave_character_index)
        return {
            "wave_character_index": int(wave_character_index),
            "frequency": float(wave_character.get_frequency()),
            "free_space_wavelength": float(wave_character.get_wavelength()),
            "projection_distance": finite_scalar("projection_distance", self.projection_distance),
            "far_field_approx": self.far_field_approx,
            "projection_medium_refractive_index": refractive_index,
            "projection_medium_impedance": impedance,
            "projection_wavenumber": wavenumber,
            "projection_wavelength": wave_character.get_wavelength() / refractive_index,
        }

    def _validate_wave_character_index(self, wave_character_index: int) -> int:
        """Validate and normalize a wave-character index."""
        if not isinstance(wave_character_index, Integral):
            raise ValueError("wave_character_index must be an integer.")
        if wave_character_index < 0 or wave_character_index >= len(self.wave_characters):
            raise ValueError("wave_character_index is out of range for this detector.")
        return int(wave_character_index)

    def _subsample_surface_quadrature(
        self,
        *,
        e_field: jax.Array | np.ndarray,
        h_field: jax.Array | np.ndarray,
        u_coords: jax.Array | np.ndarray,
        v_coords: jax.Array | np.ndarray,
        propagation_axis: int,
        direction: Literal["+", "-"],
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        tuple[int, int, int],
        tuple[float, float, float],
    ]:
        """Subsample a local surface and build matching quadrature/window weights.

        Both projection paths integrate the same local E/H samples over the same physical surface. Keeping the
        interval-space slicing, coordinate slicing, trapezoidal rule, and edge taper in one place prevents exact and
        far-field projections from drifting apart when the detector grid is downsampled.
        """
        local_axes, signs = self._local_axes_for_surface(propagation_axis, direction)
        e_field = jnp.asarray(e_field)
        h_field = jnp.asarray(h_field)
        u_coords = jnp.asarray(u_coords)
        v_coords = jnp.asarray(v_coords)
        u_indices = subsample_indices(u_coords.size, int(self.interval_space[local_axes[0]]))
        v_indices = subsample_indices(v_coords.size, int(self.interval_space[local_axes[1]]))
        e_field = e_field[:, u_indices][:, :, v_indices]
        h_field = h_field[:, u_indices][:, :, v_indices]
        u_coords = u_coords[u_indices]
        v_coords = v_coords[v_indices]

        u_weights = trapezoidal_weights_1d(u_coords) * edge_window_1d(u_coords, self.window_size[0])
        v_weights = trapezoidal_weights_1d(v_coords) * edge_window_1d(v_coords, self.window_size[1])
        weights = u_weights[:, None] * v_weights[None, :]
        return e_field, h_field, u_coords, v_coords, weights, local_axes, signs

    def _surface_currents_geometry(
        self,
        *,
        e_field: jax.Array | np.ndarray,
        h_field: jax.Array | np.ndarray,
        u_coords: jax.Array | np.ndarray,
        v_coords: jax.Array | np.ndarray,
        normal_offset: float | jax.Array,
        propagation_axis: int,
        direction: Literal["+", "-"],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Build equivalent surface currents, source coordinates, and quadrature weights.

        For a surface with outward normal ``n_hat``, the implementation uses the local tangential components to form
        electric current ``J_s = n_hat x H`` and magnetic current ``M_s = E x n_hat``. Coordinates and weights are
        subsampled together so interval-space downsampling preserves a consistent quadrature rule.
        """
        e_field, h_field, u_coords, v_coords, weights, local_axes, signs = self._subsample_surface_quadrature(
            e_field=e_field,
            h_field=h_field,
            u_coords=u_coords,
            v_coords=v_coords,
            propagation_axis=propagation_axis,
            direction=direction,
        )

        current_shape = h_field[0].shape
        zeros_current = jnp.zeros(current_shape, dtype=complex)
        zeros_coordinates = jnp.zeros(current_shape, dtype=float)
        electric_local = (-h_field[1], h_field[0], zeros_current)
        magnetic_local = (e_field[1], -e_field[0], zeros_current)
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

    def _project_surface_far_field_grid(
        self,
        *,
        e_field: jax.Array | np.ndarray,
        h_field: jax.Array | np.ndarray,
        u_coords: jax.Array | np.ndarray,
        v_coords: jax.Array | np.ndarray,
        normal_offset: float | jax.Array,
        propagation_axis: int,
        direction: Literal["+", "-"],
        radial: jax.Array | np.ndarray,
        theta_hat: jax.Array | np.ndarray,
        phi_hat: jax.Array | np.ndarray,
        wavenumber: complex,
        impedance: complex,
    ) -> tuple[jax.Array, jax.Array]:
        """Project one surface to a far-field observation grid.

        This evaluates the Fourier integrals of the tangential equivalent currents and combines them into transverse
        ``E_theta`` and ``E_phi`` components. The common propagation factor and impedance-derived H field are applied
        after all included surfaces have been coherently summed.
        """
        e_field, h_field, u_coords, v_coords, weights, local_axes, signs = self._subsample_surface_quadrature(
            e_field=e_field,
            h_field=h_field,
            u_coords=u_coords,
            v_coords=v_coords,
            propagation_axis=propagation_axis,
            direction=direction,
        )
        radial_local = jnp.stack([signs[i] * radial[axis] for i, axis in enumerate(local_axes)], axis=0)
        theta_hat_local = jnp.stack([signs[i] * theta_hat[axis] for i, axis in enumerate(local_axes)], axis=0)
        phi_hat_local = jnp.stack([signs[i] * phi_hat[axis] for i, axis in enumerate(local_axes)], axis=0)

        electric_u = -h_field[1] * weights
        electric_v = h_field[0] * weights
        magnetic_u = e_field[1] * weights
        magnetic_v = -e_field[0] * weights

        def project_component(current: jax.Array | np.ndarray) -> jax.Array:
            """Apply the same far-field phase integral to each tangential current component."""
            return direct_project_component(
                current,
                u_coords,
                v_coords,
                u_direction=radial_local[0],
                v_direction=radial_local[1],
                normal_direction=radial_local[2],
                wavenumber=wavenumber,
                normal_offset=normal_offset,
            )

        n_u, n_v, l_u, l_v = (
            project_component(current) for current in (electric_u, electric_v, magnetic_u, magnetic_v)
        )

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
        e_field: jax.Array | np.ndarray,
        h_field: jax.Array | np.ndarray,
        u_coords: jax.Array | np.ndarray,
        v_coords: jax.Array | np.ndarray,
        normal_offset: float | jax.Array,
        propagation_axis: int,
        direction: Literal["+", "-"],
        radial_flat: jax.Array | np.ndarray,
        theta_hat_flat: jax.Array | np.ndarray,
        phi_hat_flat: jax.Array | np.ndarray,
        distance_flat: jax.Array | np.ndarray,
        wave_character_index: int,
        observation_shape: tuple[int, ...],
    ) -> dict[str, jax.Array]:
        """Project one surface with the finite-distance homogeneous Green-function method.

        Unlike the far-field path, this evaluates each observation point from the full Cartesian E/H fields and then
        resolves those fields onto the local spherical basis. This keeps radial field components and near-field terms.
        """
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
        _, _, wavenumber = self._projection_parameters(wave_character_index)
        angular_frequency = 2.0 * np.pi * self.wave_characters[wave_character_index].get_frequency()

        raise_if_exact_observations_overlap_sources(
            source_coordinates=source_coordinates,
            radial_flat=radial_flat,
            distance_flat=distance_flat,
            batch_size=self.exact_projection_batch_size,
        )

        def project_observation(radial, theta_hat, phi_hat, distance):
            """Evaluate exact E/H fields at one observation direction and resolve spherical components."""
            observation_point = distance * radial
            electric_field, magnetic_field = exact_cartesian_fields_for_observation(
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
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, float | jax.Array, int, Literal["+", "-"]]:
        """Collect local fields, coordinates, normal, and direction for one projected surface."""
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

    def _finish_projection_result(
        self,
        fields: Mapping[str, jax.Array],
        coordinate_metadata: Mapping[str, Any],
        surface_names: Sequence[ProjectionSurface],
        *,
        wave_character_index: int,
    ) -> dict[str, Any]:
        """Assemble projected fields, Poynting power, coordinates, surfaces, and frequency metadata."""
        power = 0.5 * jnp.real(
            fields["Etheta"] * jnp.conj(fields["Hphi"]) - fields["Ephi"] * jnp.conj(fields["Htheta"])
        )
        return {
            **fields,
            "power": power,
            **coordinate_metadata,
            "direction": self.direction,
            "surfaces": tuple(surface_names),
            **self._projection_metadata(wave_character_index),
        }

    def _project_observations(
        self,
        state: DetectorState,
        radial: jax.Array,
        theta_hat: jax.Array,
        phi_hat: jax.Array,
        distance: jax.Array,
        *,
        wave_character_index: int,
        coordinate_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Project recorded phasors to a prepared observation grid.

        This is the central shared projection path. It iterates over the planar surface or included box faces, sums
        complex field amplitudes coherently, and computes power only after the field sum is complete.
        """
        wave_character_index = self._validate_wave_character_index(wave_character_index)
        observation_shape = radial.shape[1:]
        surface_names = []
        if not self.far_field_approx:
            fields = {key: jnp.zeros(observation_shape, dtype=complex) for key in _PROJECTION_FIELD_KEYS}
            radial_flat = radial.reshape((3, -1))
            theta_hat_flat = theta_hat.reshape((3, -1))
            phi_hat_flat = phi_hat.reshape((3, -1))
            distance_flat = jnp.broadcast_to(distance, observation_shape).reshape((-1,))
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
            return self._finish_projection_result(
                fields, coordinate_metadata, surface_names, wave_character_index=wave_character_index
            )

        _, impedance, wavenumber = self._projection_parameters(wave_character_index)
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
        fields = {
            "Er": jnp.zeros_like(e_theta),
            "Etheta": propagation_factor * e_theta,
            "Ephi": propagation_factor * e_phi,
            "Hr": jnp.zeros_like(h_theta),
            "Htheta": propagation_factor * h_theta,
            "Hphi": propagation_factor * h_phi,
        }
        return self._finish_projection_result(
            fields, coordinate_metadata, surface_names, wave_character_index=wave_character_index
        )

    def _project_observation_angle_grid(
        self,
        state: DetectorState,
        theta: jax.Array | np.ndarray,
        phi: jax.Array | np.ndarray,
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        """Project to a tensor-product global ``theta x phi`` angle grid."""
        radial, theta_hat, phi_hat = spherical_basis_grid(theta, phi)
        return self._project_observations(
            state,
            radial,
            theta_hat,
            phi_hat,
            jnp.asarray(self.projection_distance, dtype=float),
            wave_character_index=wave_character_index,
            coordinate_metadata={"theta": theta, "phi": phi},
        )

    def _project_paired_observation_angles(
        self,
        state: DetectorState,
        theta: jax.Array | np.ndarray,
        phi: jax.Array | np.ndarray,
        distance: jax.Array | np.ndarray,
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        """Project to paired observation angles and distances with matching array shapes."""
        theta = finite_numeric_array("theta", theta)
        phi = finite_numeric_array("phi", phi)
        distance = finite_numeric_array("projection distance", distance)
        if theta.shape != phi.shape or theta.shape != distance.shape:
            raise ValueError("theta, phi, and projection distance must have the same shape.")
        if theta.size == 0:
            raise ValueError("projection observation grid must be non-empty.")
        _validate_theta_range(theta)
        _validate_positive_values("projection distance", distance)
        radial, theta_hat, phi_hat = spherical_basis_paired(theta, phi)
        return self._project_observations(
            state,
            radial,
            theta_hat,
            phi_hat,
            distance,
            wave_character_index=wave_character_index,
            coordinate_metadata={"theta": theta, "phi": phi},
        )

    def _project_paired_coordinates(
        self,
        state: DetectorState,
        theta: jax.Array | np.ndarray,
        phi: jax.Array | np.ndarray,
        distance: jax.Array | np.ndarray,
        coordinate_metadata: Mapping[str, Any],
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        """Project paired coordinates and merge detector-specific coordinate metadata."""
        result = self._project_paired_observation_angles(
            state, theta, phi, distance, wave_character_index=wave_character_index
        )
        result.update(coordinate_metadata)
        return result

    def _project_all_paired_coordinates(
        self,
        state: DetectorState,
        theta: jax.Array | np.ndarray,
        phi: jax.Array | np.ndarray,
        distance: jax.Array | np.ndarray,
        coordinate_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run paired-coordinate projection for all wave characters and stack the results."""
        results = [
            self._project_paired_observation_angles(
                state, theta, phi, distance, wave_character_index=wave_character_index
            )
            for wave_character_index in range(len(self.wave_characters))
        ]
        return self._stack_frequency_results(results, {**coordinate_metadata, "theta": theta, "phi": phi})

    def _stack_frequency_results(
        self,
        results: Sequence[Mapping[str, Any]],
        coordinate_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Stack single-frequency projection dictionaries into the ``project_all`` result format."""

        def stack_metadata(key: str) -> jax.Array:
            return jnp.asarray([result[key] for result in results])

        projected: dict[str, Any] = {
            key: jnp.stack([result[key] for result in results], axis=0) for key in _PROJECTION_RESULT_KEYS
        }
        projected.update(coordinate_metadata)
        projected.update(
            {
                "wave_character_indices": np.arange(len(self.wave_characters)),
                "frequencies": np.asarray([result["frequency"] for result in results]),
                "free_space_wavelengths": np.asarray([result["free_space_wavelength"] for result in results]),
                "projection_distance": finite_scalar("projection_distance", self.projection_distance),
                "far_field_approx": self.far_field_approx,
                "projection_medium_refractive_indices": stack_metadata("projection_medium_refractive_index"),
                "projection_medium_impedances": stack_metadata("projection_medium_impedance"),
                "projection_wavenumbers": stack_metadata("projection_wavenumber"),
                "projection_wavelengths": stack_metadata("projection_wavelength"),
                "direction": self.direction,
                "surfaces": results[0]["surfaces"],
            }
        )
        return projected

    def _surface_projection_sequence(self) -> tuple[ProjectionSurface | None, ...]:
        """Return the ordered planar or box surfaces used for coherent projection."""
        if self._projection_mode == "surface":
            return (None,)
        return self._included_box_surfaces()

    def _surface_projection_name(self, surface: ProjectionSurface | None) -> ProjectionSurface:
        """Return the public surface name corresponding to a projected internal surface entry."""
        if surface is not None:
            return surface
        if self.direction is None:
            raise ValueError("direction must be specified for a planar field projection detector.")
        return _surface_name(self.propagation_axis, self.direction)


@autoinit
class FieldProjectionAngleDetector(FieldProjectionDetectorBase):
    """Frequency-domain detector for projecting a phasor plane to observation angles.

    ``theta`` and ``phi`` follow the global spherical-coordinate convention:
    ``theta`` is measured from the positive z-axis and ``phi`` from the positive
    x-axis in the x-y plane.
    """

    def __post_init__(self):
        """Validate angle-detector configuration through the shared base class."""
        super().__post_init__()

    def _validate_projection_inputs(
        self,
        theta: jax.Array | np.ndarray,
        phi: jax.Array | np.ndarray,
        wave_character_index: int,
    ) -> tuple[jax.Array, jax.Array, int]:
        """Validate angular projection coordinates and selected wave-character index."""
        theta, phi = _validate_1d_coordinate_pair("theta", theta, "phi", phi)
        _validate_theta_range(theta)
        return theta, phi, self._validate_wave_character_index(wave_character_index)

    def project(
        self,
        state: DetectorState,
        theta: jax.Array | np.ndarray,
        phi: jax.Array | np.ndarray,
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        """Project recorded phasors to the requested ``(theta, phi)`` angular grid."""
        theta, phi, wave_character_index = self._validate_projection_inputs(theta, phi, wave_character_index)
        return self._project_observation_angle_grid(state, theta, phi, wave_character_index=wave_character_index)

    def project_all(
        self,
        state: DetectorState,
        theta: jax.Array | np.ndarray,
        phi: jax.Array | np.ndarray,
    ) -> dict[str, Any]:
        """Project recorded phasors for every wave character."""
        theta, phi, _ = self._validate_projection_inputs(theta, phi, 0)
        results = [
            self._project_observation_angle_grid(state, theta, phi, wave_character_index=wave_character_index)
            for wave_character_index in range(len(self.wave_characters))
        ]
        coordinate_metadata = {"theta": results[0]["theta"], "phi": results[0]["phi"]}
        return self._stack_frequency_results(results, coordinate_metadata)


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
        """Validate Cartesian-detector configuration including the observation-plane axis."""
        super().__post_init__()
        validate_axis(self.projection_axis, name="projection_axis")

    def _validate_cartesian_inputs(
        self, x: jax.Array | np.ndarray, y: jax.Array | np.ndarray
    ) -> tuple[jax.Array, jax.Array]:
        """Validate Cartesian observation-plane coordinate arrays."""
        return _validate_1d_coordinate_pair("x", x, "y", y)

    def _cartesian_observation_grid(
        self,
        x: jax.Array | np.ndarray,
        y: jax.Array | np.ndarray,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Convert Cartesian observation-plane coordinates to global spherical projection inputs.

        The observation plane is placed at ``projection_distance`` along ``projection_axis``. The shared paired-angle
        projection path then handles both far-field and exact finite-distance propagation.
        """
        x_grid, y_grid = jnp.meshgrid(x, y, indexing="ij")
        points = jnp.zeros((3, x.size, y.size), dtype=float)
        transverse_axis_x, transverse_axis_y = get_transverse_axes(
            validate_axis(self.projection_axis, name="projection_axis")
        )
        points = points.at[transverse_axis_x].set(x_grid)
        points = points.at[transverse_axis_y].set(y_grid)
        points = points.at[self.projection_axis].set(self.projection_distance)
        radial_distance, theta, phi = cartesian_to_spherical_angles(points)
        return radial_distance, theta, phi

    def project(
        self,
        state: DetectorState,
        x: jax.Array | np.ndarray,
        y: jax.Array | np.ndarray,
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        """Project recorded phasors to a Cartesian observation plane."""
        x, y = self._validate_cartesian_inputs(x, y)
        radial_distance, theta, phi = self._cartesian_observation_grid(x, y)
        coordinate_metadata = {"x": x, "y": y, "projection_axis": int(self.projection_axis)}
        return self._project_paired_coordinates(
            state, theta, phi, radial_distance, coordinate_metadata, wave_character_index=wave_character_index
        )

    def project_all(
        self,
        state: DetectorState,
        x: jax.Array | np.ndarray,
        y: jax.Array | np.ndarray,
    ) -> dict[str, Any]:
        """Project recorded phasors to a Cartesian observation plane for every wave character."""
        x, y = self._validate_cartesian_inputs(x, y)
        radial_distance, theta, phi = self._cartesian_observation_grid(x, y)
        coordinate_metadata = {"x": x, "y": y, "projection_axis": int(self.projection_axis)}
        return self._project_all_paired_coordinates(state, theta, phi, radial_distance, coordinate_metadata)


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
        """Validate k-space detector configuration including the projection axis."""
        super().__post_init__()
        validate_axis(self.projection_axis, name="projection_axis")

    def _validate_kspace_inputs(
        self, ux: jax.Array | np.ndarray, uy: jax.Array | np.ndarray
    ) -> tuple[jax.Array, jax.Array]:
        """Validate k-space direction cosines and reject evanescent directions in eager mode."""
        ux, uy = _validate_1d_coordinate_pair("ux", ux, "uy", uy)
        if is_jax_tracer(ux) or is_jax_tracer(uy):
            return ux, uy
        if concrete_jax_bool(jnp.any(jnp.abs(ux) > 1.0)) or concrete_jax_bool(jnp.any(jnp.abs(uy) > 1.0)):
            raise ValueError("ux and uy values must lie in the interval [-1, 1].")
        ux_grid, uy_grid = jnp.meshgrid(ux, uy, indexing="ij")
        if concrete_jax_bool(jnp.any(ux_grid**2 + uy_grid**2 > 1.0)):
            raise ValueError("ux^2 + uy^2 must not exceed 1 for propagating k-space directions.")
        return ux, uy

    def _kspace_observation_grid(
        self, ux: jax.Array | np.ndarray, uy: jax.Array | np.ndarray
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Convert local k-space direction cosines to global spherical projection inputs.

        ``ux`` and ``uy`` define transverse direction cosines relative to ``projection_axis``. Only propagating points
        inside the unit disk are projected.
        """
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
        distance = jnp.full(theta.shape, finite_scalar("projection_distance", self.projection_distance), dtype=float)
        return theta, phi, distance

    def project(
        self,
        state: DetectorState,
        ux: jax.Array | np.ndarray,
        uy: jax.Array | np.ndarray,
        *,
        wave_character_index: int = 0,
    ) -> dict[str, Any]:
        """Project recorded phasors to a k-space direction-cosine grid."""
        ux, uy = self._validate_kspace_inputs(ux, uy)
        theta, phi, distance = self._kspace_observation_grid(ux, uy)
        coordinate_metadata = {"ux": ux, "uy": uy, "projection_axis": int(self.projection_axis)}
        return self._project_paired_coordinates(
            state, theta, phi, distance, coordinate_metadata, wave_character_index=wave_character_index
        )

    def project_all(
        self,
        state: DetectorState,
        ux: jax.Array | np.ndarray,
        uy: jax.Array | np.ndarray,
    ) -> dict[str, Any]:
        """Project recorded phasors to a k-space grid for every wave character."""
        ux, uy = self._validate_kspace_inputs(ux, uy)
        theta, phi, distance = self._kspace_observation_grid(ux, uy)
        coordinate_metadata = {"ux": ux, "uy": uy, "projection_axis": int(self.projection_axis)}
        return self._project_all_paired_coordinates(state, theta, phi, distance, coordinate_metadata)
