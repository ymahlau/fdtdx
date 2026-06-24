from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal, Self, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.config import SimulationConfig
from fdtdx.constants import c
from fdtdx.core.axis import get_transverse_axes
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.misc import tilted_polarization_vectors
from fdtdx.core.null import Null
from fdtdx.core.physics.metrics import normalize_by_poynting_flux
from fdtdx.core.physics.modes import compute_mode
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.dispersion import effective_inv_permittivity
from fdtdx.objects.detectors.detector import DetectorState
from fdtdx.objects.detectors.phasor import PhasorDetector
from fdtdx.typing import SliceTuple3D


def gaussian_mode_fields(
    coordinates: Sequence[jax.Array],
    propagation_axis: int,
    *,
    radius: float,
    direction: Literal["+", "-"],
    polarization_axis: int | None = None,
    fixed_E_polarization_vector: tuple[float, float, float] | None = None,
    fixed_H_polarization_vector: tuple[float, float, float] | None = None,
    azimuth_angle: float = 0.0,
    elevation_angle: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
    refractive_index: float | jax.Array = 1.0,
    wavenumber: float | jax.Array = 0.0,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    """Build an analytic Gaussian beam profile on a detector / source plane.

    The transverse amplitude is a Gaussian ``exp(-r^2 / radius^2)``. Polarization and the
    propagation direction are handled exactly like :class:`~fdtdx.GaussianPlaneSource`:
    the raw E/H polarization vectors come from
    :func:`~fdtdx.core.misc.normalize_polarization_for_source`, and a non-zero
    ``azimuth_angle`` / ``elevation_angle`` tilts the wave vector and both polarization
    vectors via :func:`~fdtdx.core.linalg.rotate_vector` (degrees, same convention as the
    source). In FDTDX's :math:`\\eta_0`-normalized units a plane wave satisfies
    ``|H| = n * |E|``, so ``H`` is scaled by ``refractive_index``.

    Under a tilt, a transverse phase ramp ``exp(i k_t . r)`` (with ``k = wavenumber`` the
    medium wavenumber) is applied so the reference wavefront matches a beam propagating at
    that angle; at normal incidence no ramp is applied and the mode stays real-valued
    (identical to the un-tilted case).

    Args:
        coordinates: ``(X, Y, Z)`` cell-center coordinate meshgrids, each of shape
            ``grid_shape`` (singleton on ``propagation_axis``). Same physical convention
            as the grid (center-origin once the grid is resolved, see #363).
        propagation_axis: Physical propagation axis (0=x, 1=y, 2=z).
        radius: Gaussian ``1/e`` amplitude radius in metres.
        direction: ``"+"`` (forward) or ``"-"`` (backward) along ``propagation_axis``.
        polarization_axis: Convenience selector — transverse axis the E field points
            along. Mutually exclusive with ``fixed_E/H_polarization_vector``. Defaults to
            the first transverse axis (ascending index order).
        fixed_E_polarization_vector: Explicit E polarization 3-vector (mirrors the source).
        fixed_H_polarization_vector: Explicit H polarization 3-vector (mirrors the source).
        azimuth_angle: Propagation tilt around the vertical axis, in degrees.
        elevation_angle: Propagation tilt around the horizontal axis, in degrees.
        center: Transverse center offset ``(off_t0, off_t1)`` in metres for the two
            transverse axes in ascending index order.
        refractive_index: Local medium index used for the ``|H| = n |E|`` ratio.
        wavenumber: Medium wavenumber ``k = n * omega / c`` used for the tilt phase ramp.
        dtype: Float dtype used to build polarization/rotation vectors.

    Returns:
        Tuple ``(mode_E, mode_H)``, each of shape ``(3, *grid_shape)`` (complex under tilt).
    """
    if polarization_axis is not None and (
        fixed_E_polarization_vector is not None or fixed_H_polarization_vector is not None
    ):
        raise ValueError("Specify either polarization_axis or fixed_E/H_polarization_vector, not both")
    if fixed_E_polarization_vector is None and fixed_H_polarization_vector is None:
        pol_axis = get_transverse_axes(propagation_axis)[0] if polarization_axis is None else polarization_axis
        if pol_axis == propagation_axis:
            raise ValueError(
                f"polarization_axis ({pol_axis}) must be transverse to the propagation axis ({propagation_axis})"
            )
        e_vec = [0.0, 0.0, 0.0]
        e_vec[pol_axis] = 1.0
        fixed_E_polarization_vector = (e_vec[0], e_vec[1], e_vec[2])

    # E/H polarization unit vectors and the (tilted) wave vector — same derivation as the
    # plane sources (degrees -> radians; a zero angle is the identity rotation).
    e_pol, h_pol, wave_vector = tilted_polarization_vectors(
        direction=direction,
        propagation_axis=propagation_axis,
        fixed_E_polarization_vector=fixed_E_polarization_vector,
        fixed_H_polarization_vector=fixed_H_polarization_vector,
        azimuth_radians=jnp.asarray(np.deg2rad(azimuth_angle), dtype=dtype),
        elevation_radians=jnp.asarray(np.deg2rad(elevation_angle), dtype=dtype),
        dtype=dtype,
    )
    is_tilted = azimuth_angle != 0.0 or elevation_angle != 0.0

    t0, t1 = get_transverse_axes(propagation_axis)
    transverse_0 = coordinates[t0] - center[0]
    transverse_1 = coordinates[t1] - center[1]
    amplitude = jnp.exp(-(transverse_0**2 + transverse_1**2) / (radius**2))

    if is_tilted:
        # Transverse phase ramp matching a tilted wavefront (phase 0 at the plane center).
        phase_arg = wavenumber * (wave_vector[t0] * transverse_0 + wave_vector[t1] * transverse_1)
        amplitude = amplitude * jnp.exp(1j * phase_arg)

    mode_E = amplitude[None, ...] * e_pol[:, None, None, None]
    mode_H = amplitude[None, ...] * (refractive_index * h_pol)[:, None, None, None]
    return mode_E, mode_H


def gaussian_mode_function(
    *,
    radius: float,
    direction: Literal["+", "-"],
    polarization_axis: int | None = None,
    fixed_E_polarization_vector: tuple[float, float, float] | None = None,
    fixed_H_polarization_vector: tuple[float, float, float] | None = None,
    azimuth_angle: float = 0.0,
    elevation_angle: float = 0.0,
    center: tuple[float, float] = (0.0, 0.0),
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    """Return a ``mode_function`` for :class:`CustomModeOverlapDetector` (analytic Gaussian).

    The returned callable derives the local refractive index (and the medium wavenumber
    for tilted beams) from the permittivity slice and delegates to
    :func:`gaussian_mode_fields`. See that function for the meaning of the arguments,
    including polarization selection and ``azimuth_angle`` / ``elevation_angle`` tilt.
    """

    def _mode_function(
        *,
        coordinates: Sequence[jax.Array],
        frequency: float,
        propagation_axis: int,
        inv_permittivity: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        refractive_index = jnp.sqrt(jnp.mean(1.0 / inv_permittivity))
        wavenumber = refractive_index * 2.0 * jnp.pi * frequency / c
        return gaussian_mode_fields(
            coordinates,
            propagation_axis,
            radius=radius,
            direction=direction,
            polarization_axis=polarization_axis,
            fixed_E_polarization_vector=fixed_E_polarization_vector,
            fixed_H_polarization_vector=fixed_H_polarization_vector,
            azimuth_angle=azimuth_angle,
            elevation_angle=elevation_angle,
            center=center,
            refractive_index=refractive_index,
            wavenumber=wavenumber,
            dtype=inv_permittivity.dtype,
        )

    return _mode_function


@autoinit
class BaseModeOverlapDetector(PhasorDetector, ABC):
    """Abstract base for mode-overlap detectors.

    Owns everything that is independent of *how* the reference mode is produced: the
    stored reference mode fields (``_mode_E`` / ``_mode_H`` of shape
    ``(num_freqs, 3, *spatial)``), the detector-plane face-area weights, and the overlap
    integral itself (:meth:`compute_overlap` / :meth:`compute_overlap_to_mode`).

    Subclasses only implement :meth:`_compute_mode_fields`, which returns the reference
    mode for a single frequency on the detector plane. :class:`ModeOverlapDetector` solves
    it with the waveguide mode solver; :class:`CustomModeOverlapDetector` /
    :class:`GaussianModeOverlapDetector` evaluate a user-supplied / analytic mode instead.

    ``compute_overlap()`` returns a complex array of shape ``(num_freqs,)``, where
    ``num_freqs = len(wave_characters)``.
    """

    #: Cannot be specified here since the detector needs all components.
    components: Sequence[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = frozen_field(
        default=("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"),
        init=False,  # in this detector, we always want all components. Do not give user a choice
    )

    #: Cannot be specified here since plotting a single scalar is useless.
    plot: bool = frozen_field(default=False, init=False)  # single scalar is useless for plotting

    _mode_E: jax.Array = private_field()
    _mode_H: jax.Array = private_field()
    _mode_neff: jax.Array = private_field()  # not required for detection, used for inspection
    _cached_face_area_weights: jax.Array = private_field()

    @property
    def propagation_axis(self) -> int:
        if sum([a == 1 for a in self.grid_shape]) != 1:
            raise Exception(f"Invalid ModeOverlapDetector shape: {self.grid_shape}")
        return self.grid_shape.index(1)

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        grid = self._config.resolved_grid
        if grid is not None:
            weights = grid.face_area(axis=self.propagation_axis, slice_tuple=self.grid_slice_tuple)
        else:
            spacing = self._config.uniform_spacing()
            weights = jnp.ones(self.grid_shape, dtype=jnp.float32) * spacing * spacing
        self = self.aset("_cached_face_area_weights", weights, create_new_ok=True)
        return self

    def _face_area_weights(self) -> jax.Array:
        """Return detector-plane face areas for mode-overlap integration."""
        return self._cached_face_area_weights

    def _plane_coordinates(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(X, Y, Z)`` cell-center coordinate meshgrids for the detector plane.

        Each array has shape ``grid_shape`` (singleton on the propagation axis).  Used by
        subclasses that evaluate an analytic mode profile on the actual placed grid. When
        the grid is resolved the coordinates follow the center-origin convention (#363);
        the uniform fallback (unresolved policy, only hit in lightweight tests) is
        corner-relative.
        """
        grid = self._config.resolved_grid
        axis_centers: list[jax.Array] = []
        for axis in range(3):
            lower, upper = self.grid_slice_tuple[axis]
            if grid is not None:
                centers = jnp.asarray(grid.centers(axis))[lower:upper]
            else:
                spacing = self._config.uniform_spacing()
                centers = (jnp.arange(lower, upper) + 0.5) * spacing
            axis_centers.append(centers)
        x_coords, y_coords, z_coords = jnp.meshgrid(*axis_centers, indexing="ij")
        return x_coords, y_coords, z_coords

    @abstractmethod
    def _compute_mode_fields(
        self,
        *,
        wave_character: WaveCharacter,
        inv_permittivity_slice: jax.Array,
        inv_permeability_slice: jax.Array | float,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Produce the reference mode ``(mode_E, mode_H, mode_neff)`` for one frequency.

        ``mode_E`` / ``mode_H`` must have shape ``(3, *grid_shape)`` (singleton on the
        propagation axis); ``mode_neff`` is a complex/real scalar used only for
        inspection. ``inv_permittivity_slice`` is already restricted to the detector plane
        and, in a dispersive medium, corrected to the effective permittivity at the
        frequency's carrier (see :meth:`apply`).
        """
        raise NotImplementedError

    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
    ) -> Self:
        del key
        inv_permittivity_slice = inv_permittivities[:, *self.grid_slice]
        if isinstance(inv_permeabilities, jax.Array) and inv_permeabilities.ndim > 0:
            inv_permeability_slice = inv_permeabilities[:, *self.grid_slice]
        else:
            inv_permeability_slice = inv_permeabilities

        c1_slice = c2_slice = c3_slice = None
        if dispersive_c1 is not None and dispersive_c2 is not None and dispersive_c3 is not None:
            c1_slice = dispersive_c1[:, :, *self.grid_slice]
            c2_slice = dispersive_c2[:, :, *self.grid_slice]
            c3_slice = dispersive_c3[:, :, *self.grid_slice]

        all_mode_Es: list[jax.Array] = []
        all_mode_Hs: list[jax.Array] = []
        all_mode_neffs: list[jax.Array] = []
        for wc in self.wave_characters:
            inv_eps_i = inv_permittivity_slice
            if c1_slice is not None and c2_slice is not None and c3_slice is not None:
                inv_eps_i = effective_inv_permittivity(
                    inv_eps=inv_permittivity_slice,
                    c1=c1_slice,
                    c2=c2_slice,
                    c3=c3_slice,
                    omega=2.0 * np.pi * wc.get_frequency(),
                    dt=self._config.time_step_duration,
                )
            mode_E, mode_H, mode_neff = self._compute_mode_fields(
                wave_character=wc,
                inv_permittivity_slice=inv_eps_i,
                inv_permeability_slice=inv_permeability_slice,
            )
            all_mode_Es.append(mode_E)
            all_mode_Hs.append(mode_H)
            all_mode_neffs.append(mode_neff)

        self = self.aset("_mode_E", jnp.stack(all_mode_Es, axis=0), create_new_ok=True)
        self = self.aset("_mode_H", jnp.stack(all_mode_Hs, axis=0), create_new_ok=True)
        self = self.aset("_mode_neff", jnp.stack(all_mode_neffs, axis=0), create_new_ok=True)
        return self

    def compute_overlap_to_mode(
        self,
        state: DetectorState,
        mode_E: jax.Array,
        mode_H: jax.Array,
        wave_character_index: int = 0,
    ) -> jax.Array:
        """Compute the overlap integral of *one* mode against phasors at ``wave_character_index``.

        Args:
            state: Detector state holding the phasor array of shape
                ``(1, num_freqs, 6, *spatial)``.
            mode_E: Electric mode field of shape ``(3, *spatial)``.
            mode_H: Magnetic mode field of shape ``(3, *spatial)``.
            wave_character_index: Index into the phasor frequency axis to use.

        Returns:
            Complex scalar overlap coefficient.
        """
        # shape (time step, num_freqs, num_components, *spatial)
        # time steps is always 1 and num_components always 6
        phasors = state["phasor"]
        phasors_E, phasors_H = phasors[0, wave_character_index, :3], phasors[0, wave_character_index, 3:]

        E_cross_H_star_sim = jnp.cross(
            mode_E,
            jnp.conj(phasors_H),
            axis=0,
        )[self.propagation_axis]

        E_star_cross_H_sim = jnp.cross(
            jnp.conj(phasors_E),
            mode_H,
            axis=0,
        )[self.propagation_axis]

        integrand = E_cross_H_star_sim + E_star_cross_H_sim
        integrand = integrand * self._face_area_weights()
        alpha_coeff = jnp.sum(integrand)

        # in pulsed mode return unscaled coefficient
        if self.scaling_mode != "pulse":
            alpha_coeff = alpha_coeff / 4.0

        return alpha_coeff

    def compute_overlap(
        self,
        state: DetectorState,
    ) -> jax.Array:
        """Compute mode overlaps at every frequency in ``wave_characters``.

        Returns:
            Complex array of shape ``(num_freqs,)``.
        """
        if isinstance(self._mode_E, Null) or isinstance(self._mode_H, Null):
            raise Exception("Need to call apply on the mode-overlap detector before calling compute_overlap!")
        overlaps = [
            self.compute_overlap_to_mode(
                state=state,
                mode_E=self._mode_E[i],
                mode_H=self._mode_H[i],
                wave_character_index=i,
            )
            for i in range(len(self.wave_characters))
        ]
        return jnp.stack(overlaps, axis=0)


@autoinit
class ModeOverlapDetector(BaseModeOverlapDetector):
    """
    Detector for measuring the overlap of a waveguide mode with the simulation fields.
    This detector computes the overlap integral at every frequency in ``wave_characters``,
    enabling broadband frequency-domain analysis of the electromagnetic fields.

    The reference mode is obtained from the waveguide mode solver (``compute_mode``). For a
    user-supplied or analytic reference mode (e.g. a Gaussian beam) use
    :class:`CustomModeOverlapDetector` or :class:`GaussianModeOverlapDetector` instead;
    both share the same overlap machinery via :class:`BaseModeOverlapDetector`.

    The mode overlap is calculated by integrating the cross product of the mode fields
    with the simulation fields over a cross-sectional plane. This is useful for
    analyzing waveguide coupling efficiency, transmission coefficients, and modal
    decomposition of electromagnetic fields.

    ``compute_overlap()`` returns a complex array of shape ``(num_freqs,)``, where
    ``num_freqs = len(wave_characters)``.
    """

    #: Direction of mode propagation, either "+" (forward) or "-" (backward).
    #: Determines which direction along the waveguide axis the mode is assumed to propagate.
    direction: Literal["+", "-"] = frozen_field()

    #: Index of the waveguide mode to use for overlap calculation.
    #: Defaults to 0 (fundamental mode). Higher indices correspond to higher-order modes.
    mode_index: int = frozen_field(default=0)

    #: Optional polarization filter for the mode calculation.
    #: Can be "te" (transverse electric), "tm" (transverse magnetic), or None (no filtering).
    #: When specified, only modes of the given polarization type are considered. Defaults to None.
    filter_pol: Literal["te", "tm"] | None = frozen_field(default=None)

    #: Bend radius of the waveguide in meters. When set, the mode solver accounts for the conformal
    #: transformation introduced by the bend. Must be set together with bend_axis. Defaults to None
    #: (straight waveguide).
    bend_radius: float | None = frozen_field(default=None)

    #: Physical axis index (0=x, 1=y, 2=z) pointing from the waveguide center toward the center of
    #: curvature. Must differ from the propagation axis. Required when bend_radius is set.
    bend_axis: int | None = frozen_field(default=None)

    #: Symmetry-plane condition at the min edge of each transverse axis (the two non-propagation
    #: physical axes, in increasing-index order): ``0`` = PEC mirror (electric wall, the default),
    #: ``1`` = PMC mirror (magnetic wall). Set this when the detector plane's waveguide lies on a
    #: symmetry plane of a reduced (half/quarter) domain so the reference mode is solved with the
    #: same wall the FDTD uses (e.g. ``(0, 1)`` for PEC at y=0 and PMC at the z Si-mid plane).
    #: Must match the corresponding ModePlaneSource for a consistent overlap.
    symmetry: tuple[int, int] = frozen_field(default=(0, 0))

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        if (self.bend_radius is None) != (self.bend_axis is None):
            raise ValueError("bend_radius and bend_axis must both be set or both be None")
        if self.bend_axis is not None and self.bend_axis == self.propagation_axis:
            raise ValueError(
                f"bend_axis ({self.bend_axis}) must differ from the propagation axis ({self.propagation_axis})"
            )
        return self

    def _transverse_edge_coordinates(self) -> tuple[jax.Array, jax.Array] | None:
        """Return physical transverse edge coordinates for the mode solver.

        Tidy3D can solve modes on rectilinear non-uniform grids when supplied
        with edge-coordinate arrays.  Returning ``None`` keeps the uniform scalar
        spacing path for legacy configurations and older tests.
        """
        grid = self._config.resolved_grid
        if grid is None:
            return None

        transverse_edges = []
        for axis in range(3):
            if axis == self.propagation_axis:
                continue
            lower, upper = self.grid_slice_tuple[axis]
            transverse_edges.append(grid.edges(axis)[lower : upper + 1])
        e0, e1 = transverse_edges
        return e0, e1

    def _mode_solver_resolution(self) -> float:
        """Return scalar resolution only for legacy uniform mode-solver setup.

        ``compute_mode`` ignores this value when explicit transverse coordinates
        are supplied.  For non-uniform grids we pass a harmless finite value so
        the compatibility argument does not force a uniform-grid check.
        """
        if self._config.has_nonuniform_grid:
            assert self._config.resolved_grid is not None
            return self._config.resolved_grid.min_spacing
        return self._config.uniform_spacing()

    def _compute_mode_fields(
        self,
        *,
        wave_character: WaveCharacter,
        inv_permittivity_slice: jax.Array,
        inv_permeability_slice: jax.Array | float,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return compute_mode(
            frequency=wave_character.get_frequency(),
            inv_permittivities=inv_permittivity_slice,
            inv_permeabilities=inv_permeability_slice,
            resolution=self._mode_solver_resolution(),
            direction=self.direction,
            mode_index=self.mode_index,
            filter_pol=self.filter_pol,
            dtype=self._config.dtype,
            bend_radius=self.bend_radius,
            bend_axis=self.bend_axis,
            symmetry=self.symmetry,
            transverse_coords=self._transverse_edge_coordinates(),
        )


@autoinit
class CustomModeOverlapDetector(BaseModeOverlapDetector):
    """Mode-overlap detector using a user-provided reference mode.

    Instead of solving the mode with the waveguide mode solver, the reference mode is
    produced by ``mode_function`` — a callable evaluated on the detector plane during
    :meth:`apply`. This enables overlap against an arbitrary mode (e.g. an analytic
    Gaussian beam, a fiber mode, or a mode imported from another tool) and avoids the
    tidy3d mode-solver dependency.

    ``mode_function`` is called once per frequency with keyword arguments::

        mode_function(
            coordinates,        # (X, Y, Z) cell-center meshgrids, each (3==None) shape grid_shape
            frequency,          # float, Hz
            propagation_axis,   # int, 0/1/2
            inv_permittivity,   # effective eps slice on the plane (n_comp, *grid_shape)
        ) -> (mode_E, mode_H)   # each (3, *grid_shape)

    and must return the E and H fields in FDTDX's :math:`\\eta_0`-normalized convention.
    Use :func:`gaussian_mode_function` for a ready-made analytic Gaussian, or
    :class:`GaussianModeOverlapDetector` for the same as a configurable detector class.

    Dispersive media are handled by the shared :meth:`BaseModeOverlapDetector.apply`: in a
    dispersive simulation ``inv_permittivity`` is the *effective* inverse permittivity
    :math:`1/\\mathrm{Re}(\\varepsilon_\\infty + \\chi(\\omega_c))` at the wave character's
    carrier frequency (same correction as ``ModeOverlapDetector`` / ``ModePlaneSource``),
    not :math:`\\varepsilon_\\infty`. A frequency-aware ``mode_function`` therefore sees the
    true medium index per frequency.
    """

    #: Callable producing the reference ``(mode_E, mode_H)`` on the detector plane.
    #: See the class docstring for the exact keyword-argument signature.
    mode_function: Callable[..., tuple[jax.Array, jax.Array]] = frozen_field()

    #: Whether to renormalize the provided mode to unit Poynting flux over the detector
    #: plane (matching what the mode solver does). Defaults to True.
    normalize: bool = frozen_field(default=True)

    def _compute_mode_fields(
        self,
        *,
        wave_character: WaveCharacter,
        inv_permittivity_slice: jax.Array,
        inv_permeability_slice: jax.Array | float,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del inv_permeability_slice
        coordinates = self._plane_coordinates()
        mode_E, mode_H = self.mode_function(
            coordinates=coordinates,
            frequency=wave_character.get_frequency(),
            propagation_axis=self.propagation_axis,
            inv_permittivity=inv_permittivity_slice,
        )
        # Nominal effective index for inspection only (mean medium index on the plane).
        mode_neff = jnp.sqrt(jnp.mean(1.0 / inv_permittivity_slice))
        if self.normalize:
            mode_E, mode_H = normalize_by_poynting_flux(
                mode_E,
                mode_H,
                axis=self.propagation_axis,
                area_weights=self._face_area_weights(),
            )
        return mode_E, mode_H, mode_neff


@autoinit
class GaussianModeOverlapDetector(BaseModeOverlapDetector):
    """Mode-overlap detector using an analytic Gaussian beam as the reference mode.

    Convenience wrapper around :func:`gaussian_mode_fields`: the reference mode is a
    transverse Gaussian amplitude profile with ``|H| = n |E|`` (local medium index ``n``
    derived from the permittivity slice). Polarization and the propagation angle are
    configured exactly like :class:`~fdtdx.GaussianPlaneSource` — pick an explicit
    polarization vector (or a transverse axis) and tilt the beam with
    ``azimuth_angle`` / ``elevation_angle``. No mode solver is involved.

    Dispersion-aware: in a dispersive medium ``n`` (and the tilt wavenumber ``k = n ω/c``)
    is taken from the effective permittivity :math:`\\mathrm{Re}(\\varepsilon_\\infty +
    \\chi(\\omega_c))` at each wave character's carrier frequency, via the shared
    :meth:`BaseModeOverlapDetector.apply` correction — the same one ``ModeOverlapDetector``
    and ``ModePlaneSource`` use — not :math:`\\varepsilon_\\infty`.
    """

    #: Gaussian ``1/e`` amplitude radius (beam waist) in metres.
    mode_radius: float = frozen_field()

    #: Direction of propagation, "+" (forward) or "-" (backward) along the plane normal.
    direction: Literal["+", "-"] = frozen_field()

    #: Convenience polarization selector — transverse axis the E field points along.
    #: Mutually exclusive with ``fixed_E/H_polarization_vector``. Defaults to the first
    #: transverse axis (ascending index order).
    polarization_axis: int | None = frozen_field(default=None)

    #: Explicit electric polarization 3-vector (mirrors ``GaussianPlaneSource``).
    fixed_E_polarization_vector: tuple[float, float, float] | None = frozen_field(default=None)

    #: Explicit magnetic polarization 3-vector (mirrors ``GaussianPlaneSource``).
    fixed_H_polarization_vector: tuple[float, float, float] | None = frozen_field(default=None)

    #: Propagation tilt around the vertical axis, in degrees (off-normal incidence).
    azimuth_angle: float = frozen_field(default=0.0)

    #: Propagation tilt around the horizontal axis, in degrees (off-normal incidence).
    elevation_angle: float = frozen_field(default=0.0)

    #: Transverse center offset ``(off_t0, off_t1)`` in metres for the two transverse axes
    #: (ascending index order), in the same physical coordinates as the grid.
    center: tuple[float, float] = frozen_field(default=(0.0, 0.0))

    #: Whether to renormalize the mode to unit Poynting flux over the detector plane.
    normalize: bool = frozen_field(default=True)

    def _compute_mode_fields(
        self,
        *,
        wave_character: WaveCharacter,
        inv_permittivity_slice: jax.Array,
        inv_permeability_slice: jax.Array | float,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del inv_permeability_slice
        coordinates = self._plane_coordinates()
        refractive_index = jnp.sqrt(jnp.mean(1.0 / inv_permittivity_slice))
        wavenumber = refractive_index * 2.0 * jnp.pi * wave_character.get_frequency() / c
        mode_E, mode_H = gaussian_mode_fields(
            coordinates,
            self.propagation_axis,
            radius=self.mode_radius,
            direction=self.direction,
            polarization_axis=self.polarization_axis,
            fixed_E_polarization_vector=self.fixed_E_polarization_vector,
            fixed_H_polarization_vector=self.fixed_H_polarization_vector,
            azimuth_angle=self.azimuth_angle,
            elevation_angle=self.elevation_angle,
            center=self.center,
            refractive_index=refractive_index,
            wavenumber=wavenumber,
            dtype=self._config.dtype,
        )
        if self.normalize:
            mode_E, mode_H = normalize_by_poynting_flux(
                mode_E,
                mode_H,
                axis=self.propagation_axis,
                area_weights=self._face_area_weights(),
            )
        return mode_E, mode_H, refractive_index
